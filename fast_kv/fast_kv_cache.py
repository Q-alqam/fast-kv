"""Main Fast-KV Cache class — drop-in replacement for a standard KV cache.

Wraps the Importance Scoring Engine, Compression Engine, and Tier Manager
into a single unified interface.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from fast_kv.config import FastKVConfig
from fast_kv.importance_scorer import ImportanceScoringEngine
from fast_kv.tier_manager import TierManager

logger = logging.getLogger(__name__)


class FastKVCache:
    """Two-tier KV cache with dynamic importance-aware compression.

    This class is the main entry point for the Fast-KV system. It manages
    per-layer KV caches with automatic tiering, compression, and
    promotion/demotion.

    Args:
        config: FastKVConfig with all tunable parameters.
        model_config: Dictionary with model parameters:
            - n_layers (int): Number of transformer layers.
            - kv_dim (int): Dimension of K/V vectors per layer.
            - dtype (str): Data type name (e.g., 'float32').
    """

    def __init__(self, config: FastKVConfig, model_config: Dict) -> None:
        self.config = config
        self.model_config = model_config
        self.n_layers: int = model_config["n_layers"]
        self.kv_dim: int = model_config["kv_dim"]
        self.dtype: str = model_config.get("dtype", "float32")

        # One ISE shared across all layers (token importance is global)
        self.ise = ImportanceScoringEngine(config)

        # One TierManager per layer
        self.tier_managers: Dict[int, TierManager] = {
            layer_id: TierManager(config, self.ise, self.kv_dim * 2)
            for layer_id in range(self.n_layers)
        }

        self._current_step: int = 0

    def update(
        self,
        layer_id: int,
        token_id: int,
        token_text: str,
        key_vector: np.ndarray,
        value_vector: np.ndarray,
        attention_weights: Dict[int, float],
        current_step: int,
    ) -> Dict:
        """Process a new token through the Fast-KV system.

        Called on every new token for each layer. Updates the ISE, adds
        the token to the appropriate tier, and runs promotion/demotion checks.

        Args:
            layer_id: Transformer layer index.
            token_id: Unique token identifier.
            token_text: Raw text of the token.
            key_vector: Key vector from the attention mechanism.
            value_vector: Value vector from the attention mechanism.
            attention_weights: Attention weights for all tokens this step.
            current_step: Current inference step number.

        Returns:
            Current cache statistics dictionary.
        """
        self._current_step = current_step

        # Update ISE with latest attention (only once per step, not per layer)
        if layer_id == 0:
            self.ise.update_attention_scores(attention_weights, current_step)

        # Concatenate K and V into a single vector for storage
        kv_vector = np.concatenate([key_vector, value_vector])

        # Add token to the appropriate tier
        tm = self.tier_managers[layer_id]
        tm.add_token(token_id, kv_vector, token_text, layer_id, current_step)

        # Periodic demotion check
        if current_step % self.config.demotion_check_every_n_steps == 0:
            tm.check_demotions(current_step)

        # Promotion check every step
        tm.check_promotions(attention_weights, current_step)

        return tm.get_stats()

    def get_kv_cache(
        self, layer_id: int, token_ids: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve key and value matrices for the attention mechanism.

        Args:
            layer_id: Transformer layer index.
            token_ids: List of token_ids to retrieve.

        Returns:
            Tuple of (keys, values) arrays, each of shape (n_tokens, kv_dim).
        """
        tm = self.tier_managers[layer_id]
        kv_matrix = tm.get_kv_for_attention(token_ids)
        # Split concatenated KV back into separate K and V
        keys = kv_matrix[:, : self.kv_dim]
        values = kv_matrix[:, self.kv_dim :]
        return keys, values

    def get_compression_method(self) -> str:
        """Return the current compression method name."""
        return self.config.compression_method

    def get_warmup_status(self) -> Dict:
        """Return the current warmup status.

        Returns:
            Dictionary with 'in_warmup', 'steps_remaining', 'warmup_steps'.
        """
        in_warmup = self._current_step < self.config.warmup_steps
        remaining = max(0, self.config.warmup_steps - self._current_step)
        return {
            "in_warmup": in_warmup,
            "steps_remaining": remaining,
            "warmup_steps": self.config.warmup_steps,
        }

    def reset(self) -> None:
        """Clear all caches and reset ISE state for a new conversation."""
        self.ise.reset()
        for tm in self.tier_managers.values():
            tm.hot_cache.clear()
            tm.cold_cache.clear()
            tm.token_tiers.clear()
            tm._subtier_counts = {"2A": 0, "2B": 0, "2C": 0}
            tm.n_promotions_total = 0
            tm.n_demotions_total = 0
            tm._current_step = 0
            tm._warmup_complete = self.config.warmup_steps <= 0
        self._current_step = 0
        logger.info("Fast-KV cache reset")

    def get_memory_report(self) -> str:
        """Generate a human-readable memory usage report.

        Aggregates statistics across all layers and formats a detailed
        report showing tier distribution, RAM usage, and event counts.

        Returns:
            Formatted multi-line string with the memory report.
        """
        # Aggregate stats across all layers
        total_hot = 0
        total_cold = 0
        total_ram_hot = 0.0
        total_ram_cold = 0.0
        total_ram_uncompressed = 0.0
        total_promotions = 0
        total_demotions = 0
        subtier_totals = {"2A": 0, "2B": 0, "2C": 0}

        for tm in self.tier_managers.values():
            stats = tm.get_stats()
            total_hot += stats["n_hot"]
            total_cold += stats["n_cold"]
            total_ram_hot += stats["estimated_ram_hot_mb"]
            total_ram_cold += stats["estimated_ram_cold_mb"]
            total_ram_uncompressed += stats["estimated_ram_uncompressed_mb"]
            total_promotions += stats["n_promotions_total"]
            total_demotions += stats["n_demotions_total"]
            for st, count in stats["subtier_counts"].items():
                subtier_totals[st] += count

        total_tokens = total_hot + total_cold
        hot_pct = (total_hot / total_tokens * 100) if total_tokens > 0 else 0.0
        cold_pct = 100.0 - hot_pct
        total_fkv = total_ram_hot + total_ram_cold
        savings = total_ram_uncompressed - total_fkv
        savings_pct = (savings / total_ram_uncompressed * 100) if total_ram_uncompressed > 0 else 0.0

        # Per-layer stats use layer 0 token count (same tokens across layers)
        unique_tokens = len(self.tier_managers[0].token_tiers) if self.n_layers > 0 else 0

        report = f"""
=== Fast-KV Memory Report ===
Unique tokens:         {unique_tokens}
Tokens in Tier 1 (hot):    {total_hot:>6}  ({hot_pct:.1f}%)
Tokens in Tier 2 (cold):   {total_cold:>6}  ({cold_pct:.1f}%)
  +-- Sub-tier 2A (4-bit):  {subtier_totals['2A']:>5}
  +-- Sub-tier 2B (2-bit):  {subtier_totals['2B']:>5}
  +-- Sub-tier 2C (1-bit):  {subtier_totals['2C']:>5}

RAM Usage:
  Hot tier:           {total_ram_hot:>8.1f} MB (full precision)
  Cold tier:          {total_ram_cold:>8.1f} MB (compressed)
  Total Fast-KV:      {total_fkv:>8.1f} MB
  Without Fast-KV:    {total_ram_uncompressed:>8.1f} MB
  Savings:            {savings:>8.1f} MB ({savings_pct:.1f}% reduction)

Tier Events:
  Total promotions:   {total_promotions:>6}
  Total demotions:    {total_demotions:>6}
"""
        return report.strip()
