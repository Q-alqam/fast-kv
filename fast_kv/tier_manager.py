"""Tier Manager for Fast-KV.

Manages the hot (Tier 1) and cold (Tier 2) caches, handling all
promotion/demotion logic and sub-tier bit-width assignment.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from fast_kv.compression import (
    apply_residual,
    compute_residual,
    dequantize_vector,
    quantize_vector,
    quantize_vector_channelwise_outlier_aware,
    quantize_vector_outlier_aware,
)
from fast_kv.config import FastKVConfig
from fast_kv.importance_scorer import ImportanceScoringEngine

logger = logging.getLogger(__name__)


class TierManager:
    """Manages two-tier KV cache with dynamic promotion and demotion.

    Tokens are assigned to hot (full precision) or cold (compressed) tiers
    based on their importance scores. Cold tokens are further subdivided
    into sub-tiers with different compression levels.

    Args:
        config: FastKVConfig with tier thresholds and compression settings.
        ise: ImportanceScoringEngine instance for scoring tokens.
        kv_dim: Dimension of KV vectors (used for RAM estimation).
    """

    def __init__(
        self, config: FastKVConfig, ise: ImportanceScoringEngine, kv_dim: int = 1024
    ) -> None:
        self.config = config
        self.ise = ise
        self.kv_dim = kv_dim

        # Tier 1 (hot): token_id -> {'kv': np.ndarray, 'layer': int}
        self.hot_cache: Dict[int, Dict] = {}

        # Tier 2 (cold): token_id -> {'quantized': dict, 'residual': optional,
        #                              'layer': int, 'sub_tier': str}
        self.cold_cache: Dict[int, Dict] = {}

        # Quick lookup: token_id -> 'hot' or 'cold'
        self.token_tiers: Dict[int, str] = {}

        # Sub-tier counts for reporting
        self._subtier_counts: Dict[str, int] = {"2A": 0, "2B": 0, "2C": 0}

        # Event counters
        self.n_promotions_total: int = 0
        self.n_demotions_total: int = 0

        # Warmup tracking
        self._current_step: int = 0
        self._warmup_complete: bool = config.warmup_steps <= 0

    def _assign_sub_tier(self, score: float) -> Tuple[str, int]:
        """Determine the cold sub-tier and bit width based on importance score.

        Args:
            score: The token's importance score.

        Returns:
            Tuple of (sub_tier_name, bit_width).
        """
        if score >= 0.35:
            return "2A", self.config.bits_subtier_2a
        elif score >= 0.15:
            return "2B", self.config.bits_subtier_2b
        else:
            return "2C", self.config.bits_subtier_2c

    def _compress_to_cold(
        self, token_id: int, kv_vector: np.ndarray, layer_id: int, score: float
    ) -> None:
        """Compress a KV vector and store it in the cold cache.

        Args:
            token_id: Unique token identifier.
            kv_vector: Full-precision KV vector.
            layer_id: Model layer index.
            score: Current importance score (determines sub-tier).
        """
        sub_tier, bits = self._assign_sub_tier(score)

        # Select compression method
        sigma = {
            "2A": self.config.outlier_sigma_2a,
            "2B": self.config.outlier_sigma_2b,
            "2C": self.config.outlier_sigma_2c,
        }.get(sub_tier, 3.0)

        if self.config.compression_method == "channelwise":
            gs = self.config.channelwise_group_size
            # Use larger groups for the most compressed tier
            if sub_tier == "2C":
                gs = gs * 2 if gs else gs
            quantized = quantize_vector_channelwise_outlier_aware(
                kv_vector, bits, group_size=gs, threshold_sigma=sigma,
            )
        elif self.config.use_outlier_aware:
            quantized = quantize_vector_outlier_aware(kv_vector, bits, sigma)
        else:
            quantized = quantize_vector(kv_vector, bits)

        entry: Dict = {
            "quantized": quantized,
            "residual": None,
            "layer": layer_id,
            "sub_tier": sub_tier,
        }

        # Store residual for sub-tier 2A if enabled
        if sub_tier == "2A" and self.config.use_residuals:
            entry["residual"] = compute_residual(
                kv_vector, quantized, self.config.residual_bits
            )

        self.cold_cache[token_id] = entry
        self.token_tiers[token_id] = "cold"
        self._subtier_counts[sub_tier] += 1

    def add_token(
        self,
        token_id: int,
        kv_vector: np.ndarray,
        token_text: str,
        layer_id: int,
        current_step: int = 0,
    ) -> str:
        """Add a new token to the appropriate cache tier.

        Scores the token immediately and assigns it to hot or cold tier.

        Args:
            token_id: Unique token identifier.
            kv_vector: The key-value vector (concatenation of K and V).
            token_text: Raw token text for static classification.
            layer_id: Model layer index.
            current_step: Current inference step.

        Returns:
            Tier assignment: 'hot' or 'cold'.
        """
        # Register token with ISE if not already done
        self.ise.register_token(token_id, token_text)
        score = self.ise.get_score(token_id, current_step)

        # During warmup, force all tokens to hot tier
        if current_step < self.config.warmup_steps:
            self.hot_cache[token_id] = {"kv": kv_vector.copy(), "layer": layer_id}
            self.token_tiers[token_id] = "hot"
            logger.debug(
                "Token %d ('%s') -> HOT (warmup, score=%.3f)",
                token_id, token_text, score,
            )
            return "hot"

        if score >= self.config.hot_threshold:
            self.hot_cache[token_id] = {"kv": kv_vector.copy(), "layer": layer_id}
            self.token_tiers[token_id] = "hot"
            logger.debug(
                "Token %d ('%s') -> HOT (score=%.3f)", token_id, token_text, score
            )
            return "hot"
        else:
            self._compress_to_cold(token_id, kv_vector, layer_id, score)
            logger.debug(
                "Token %d ('%s') -> COLD/%s (score=%.3f)",
                token_id, token_text, self.cold_cache[token_id]["sub_tier"], score,
            )
            return "cold"

    def check_demotions(self, current_step: int) -> List[int]:
        """Check all hot tokens for demotion to cold tier.

        Called every demotion_check_every_n_steps inference steps.
        Skipped entirely during warmup period.

        Args:
            current_step: Current inference step.

        Returns:
            List of token_ids that were demoted.
        """
        self._current_step = current_step

        # Skip demotions during warmup
        if current_step < self.config.warmup_steps:
            return []

        # Log warmup completion once
        if not self._warmup_complete:
            self._warmup_complete = True
            logger.info(
                "Fast-KV warmup complete at step %d, tiering now active",
                current_step,
            )

        demoted = []
        # Iterate over a copy of keys since we modify the dict
        for token_id in list(self.hot_cache.keys()):
            score = self.ise.get_score(token_id, current_step)
            if score < self.config.cold_threshold:
                entry = self.hot_cache.pop(token_id)
                self._compress_to_cold(
                    token_id, entry["kv"], entry["layer"], score
                )
                demoted.append(token_id)
                self.n_demotions_total += 1
                logger.debug(
                    "DEMOTION: token %d score=%.3f -> COLD/%s",
                    token_id, score, self.cold_cache[token_id]["sub_tier"],
                )
        return demoted

    def check_promotions(
        self, attention_weights: Dict[int, float], current_step: int
    ) -> List[int]:
        """Check cold tokens for promotion based on attention spikes.

        Called every inference step. A cold token is promoted if its
        current attention weight exceeds promotion_spike_threshold.

        Args:
            attention_weights: Current step's attention weights per token.
            current_step: Current inference step.

        Returns:
            List of token_ids that were promoted.
        """
        promoted = []
        for token_id in list(self.cold_cache.keys()):
            attn = attention_weights.get(token_id, 0.0)
            if attn > self.config.promotion_spike_threshold:
                entry = self.cold_cache.pop(token_id)

                # Decompress
                if entry["residual"] is not None:
                    kv_vector = apply_residual(entry["quantized"], entry["residual"])
                else:
                    kv_vector = dequantize_vector(entry["quantized"])

                self.hot_cache[token_id] = {
                    "kv": kv_vector,
                    "layer": entry["layer"],
                }
                self.token_tiers[token_id] = "hot"
                self._subtier_counts[entry["sub_tier"]] -= 1
                self.n_promotions_total += 1
                promoted.append(token_id)
                logger.debug(
                    "PROMOTION: token %d attn=%.3f -> HOT", token_id, attn
                )
        return promoted

    def get_kv_for_attention(self, token_ids: List[int]) -> np.ndarray:
        """Retrieve KV vectors for attention computation.

        Hot tokens are returned directly; cold tokens are decompressed on the fly.

        Args:
            token_ids: List of token_ids to retrieve KV vectors for.

        Returns:
            Stacked KV matrix of shape (len(token_ids), kv_dim).
        """
        vectors = []
        for tid in token_ids:
            if tid in self.hot_cache:
                vectors.append(self.hot_cache[tid]["kv"])
            elif tid in self.cold_cache:
                entry = self.cold_cache[tid]
                if entry["residual"] is not None:
                    vectors.append(
                        apply_residual(entry["quantized"], entry["residual"])
                    )
                else:
                    vectors.append(dequantize_vector(entry["quantized"]))
            else:
                logger.warning("Token %d not found in any cache tier", tid)
                vectors.append(np.zeros(self.kv_dim, dtype=np.float32))
        return np.stack(vectors) if vectors else np.empty((0, self.kv_dim))

    def get_stats(self) -> Dict:
        """Compute current cache statistics and RAM estimates.

        Returns:
            Dictionary with counts, fractions, RAM estimates, and event totals.
        """
        n_hot = len(self.hot_cache)
        n_cold = len(self.cold_cache)
        n_total = n_hot + n_cold

        # RAM estimates (bytes)
        bytes_per_token_full = self.kv_dim * 4  # float32 = 4 bytes
        ram_hot = n_hot * bytes_per_token_full
        ram_uncompressed = n_total * bytes_per_token_full

        # Estimate cold RAM based on sub-tier bit widths
        ram_cold = 0.0
        for entry in self.cold_cache.values():
            st = entry["sub_tier"]
            if st == "2A":
                bits = self.config.bits_subtier_2a
                # Add residual storage if present
                residual_overhead = (
                    self.kv_dim * self.config.residual_bits / 8
                    if entry["residual"] is not None
                    else 0
                )
            elif st == "2B":
                bits = self.config.bits_subtier_2b
                residual_overhead = 0
            else:
                bits = self.config.bits_subtier_2c
                residual_overhead = 0
            # Account for outlier storage overhead
            n_outliers = entry["quantized"].get("outlier_count", 0)
            outlier_bytes = n_outliers * 8  # 4 bytes index + 4 bytes value
            ram_cold += self.kv_dim * bits / 8 + 8 + residual_overhead + outlier_bytes

        # Count total outliers across cold cache
        total_outliers = sum(
            e["quantized"].get("outlier_count", 0)
            for e in self.cold_cache.values()
        )

        hot_fraction = n_hot / n_total if n_total > 0 else 0.0
        total_fkv = ram_hot + ram_cold
        ratio = ram_uncompressed / total_fkv if total_fkv > 0 else 1.0

        return {
            "n_hot": n_hot,
            "n_cold": n_cold,
            "n_total": n_total,
            "hot_fraction": hot_fraction,
            "estimated_ram_hot_mb": ram_hot / (1024 * 1024),
            "estimated_ram_cold_mb": ram_cold / (1024 * 1024),
            "estimated_ram_uncompressed_mb": ram_uncompressed / (1024 * 1024),
            "estimated_ram_total_mb": total_fkv / (1024 * 1024),
            "compression_ratio": ratio,
            "n_promotions_total": self.n_promotions_total,
            "n_demotions_total": self.n_demotions_total,
            "subtier_counts": dict(self._subtier_counts),
            "total_outliers_stored": total_outliers,
            "avg_outliers_per_token": total_outliers / n_cold if n_cold > 0 else 0.0,
        }
