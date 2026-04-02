"""Importance Scoring Engine (ISE) for Fast-KV.

The ISE assigns each token a score in [0.0, 1.0] using three layered signals:
  Layer 1 - Static token classification (zero runtime cost)
  Layer 2 - Attention score EMA (free signal from attention computation)
  Layer 3 - Recency decay (minimal cost, 3 ops per token)
"""

import logging
import re
from typing import Dict, List, Optional

from fast_kv.config import FastKVConfig

logger = logging.getLogger(__name__)

# Comprehensive set of tokens that are always classified as cold (low importance).
# These are function words that carry little semantic weight.
ALWAYS_COLD_TOKENS: set = {
    # Articles
    "the", "a", "an",
    # Prepositions
    "in", "on", "at", "to", "of", "for", "by", "with", "from", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "under", "over", "about", "against", "along", "around", "among",
    "upon", "within", "without", "toward", "towards", "across", "behind",
    "beyond", "near", "beside", "besides", "beneath",
    # Conjunctions
    "and", "or", "but", "so", "yet", "nor", "for", "both", "either",
    "neither", "whether", "although", "though", "because", "since",
    "while", "if", "unless", "until", "than", "that", "which",
    # Auxiliary verbs
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "having",
    "will", "would", "shall", "should", "may", "might", "can", "could",
    "must",
    # Common pronouns (not first-person)
    "it", "its", "he", "she", "him", "her", "his", "hers",
    "they", "them", "their", "theirs", "this", "that", "these", "those",
    # Punctuation tokens
    ".", ",", ";", ":", "!", "?", "-", "(", ")", "[", "]", "{", "}",
    "'", '"', "/", "\\", "|", "&", "#", "@", "^", "~", "`",
    # Common filler words
    "just", "very", "also", "then", "some", "such", "each", "every",
    "any", "all", "most", "much", "many", "few", "other",
}

# Regex patterns for tokens that should always be classified as hot.
_QUESTION_WORDS = {"who", "what", "where", "when", "why", "how"}
_FIRST_PERSON = {"i", "my", "me", "mine", "we", "our", "ours", "us"}

# Pre-compiled patterns for hot token detection
_NUMERIC_PATTERN = re.compile(r"^\d[\d.,]*$")
_VERSION_PATTERN = re.compile(r"^\d+\.\d+")
_CVE_PATTERN = re.compile(r"^CVE-\d{4}-\d+$", re.IGNORECASE)
_IP_PATTERN = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")


def _is_always_hot(token_text: str) -> bool:
    """Check if a token matches any ALWAYS_HOT pattern.

    Detects: all-uppercase acronyms, mid-sentence proper nouns, numeric values,
    version numbers, CVE identifiers, IP addresses, question words, and
    first-person pronouns.

    Args:
        token_text: The raw text of the token.

    Returns:
        True if the token should always be in the hot tier.
    """
    lower = token_text.lower().strip()

    # Question words
    if lower in _QUESTION_WORDS:
        return True

    # First-person pronouns
    if lower in _FIRST_PERSON:
        return True

    stripped = token_text.strip()
    if not stripped:
        return False

    # All-uppercase tokens with length >= 2 (acronyms/identifiers like CVE, SQL)
    if stripped.isupper() and len(stripped) >= 2 and stripped.isalpha():
        return True

    # Capitalized tokens in mid-sentence (proper nouns) — heuristic: first letter
    # upper, rest has lowercase letters
    if stripped[0].isupper() and len(stripped) >= 2 and any(c.islower() for c in stripped):
        return True

    # CVE identifiers
    if _CVE_PATTERN.match(stripped):
        return True

    # IP addresses
    if _IP_PATTERN.match(stripped):
        return True

    # Version numbers (e.g., 3.14, 2.0.1)
    if _VERSION_PATTERN.match(stripped):
        return True

    # Numeric patterns (digits, decimals)
    if _NUMERIC_PATTERN.match(stripped):
        return True

    return False


class ImportanceScoringEngine:
    """Three-layer importance scorer for KV cache tokens.

    Combines static classification, attention EMA, and recency decay into
    a single importance score per token.

    Args:
        config: FastKVConfig with ISE weights and parameters.
    """

    def __init__(self, config: FastKVConfig) -> None:
        self.config = config

        # Layer 2 state: attention EMA per token
        self._attention_ema: Dict[int, float] = {}

        # Layer 3 state: last step where token had high attention
        self._last_hot_step: Dict[int, int] = {}

        # Static scores cache (token_id -> static score)
        self._static_scores: Dict[int, float] = {}

        # Token text lookup (for static scoring)
        self._token_texts: Dict[int, str] = {}

    def register_token(self, token_id: int, token_text: str) -> None:
        """Register a new token's text for static scoring.

        Args:
            token_id: Unique identifier for the token position.
            token_text: The raw text of the token.
        """
        self._token_texts[token_id] = token_text
        self._static_scores[token_id] = self.static_score(token_text)
        # Initialize attention EMA at neutral 0.5
        if token_id not in self._attention_ema:
            self._attention_ema[token_id] = 0.5
        # Initialize last hot step to registration time (step 0)
        if token_id not in self._last_hot_step:
            self._last_hot_step[token_id] = 0

    def static_score(self, token_text: str) -> float:
        """Layer 1: Static token classification.

        Zero-cost lookup that instantly classifies ~40-50% of tokens.

        Args:
            token_text: The raw text of the token.

        Returns:
            0.0 for always-cold tokens, 1.0 for always-hot tokens,
            0.5 for neutral tokens.
        """
        lower = token_text.lower().strip()
        if lower in ALWAYS_COLD_TOKENS:
            return 0.0
        if _is_always_hot(token_text):
            return 1.0
        return 0.5

    def attention_ema_score(self, token_id: int) -> float:
        """Layer 2: Return the current attention EMA for a token.

        Args:
            token_id: The token's unique identifier.

        Returns:
            Current EMA value in [0.0, 1.0].
        """
        return self._attention_ema.get(token_id, 0.5)

    def recency_weight(self, token_id: int, current_step: int) -> float:
        """Layer 3: Recency decay function.

        Penalizes tokens that haven't been attended to recently.
        Formula: 1.0 / (1.0 + lambda * (current_step - last_hot_step))

        Args:
            token_id: The token's unique identifier.
            current_step: The current inference step number.

        Returns:
            Recency weight in (0.0, 1.0].
        """
        last_step = self._last_hot_step.get(token_id, 0)
        return 1.0 / (1.0 + self.config.recency_lambda * (current_step - last_step))

    def update_attention_scores(
        self,
        attention_weights: Dict[int, float],
        current_step: int,
    ) -> None:
        """Update attention EMA and recency tracking for all tokens.

        Called every inference step with the latest attention weights from
        the transformer's attention computation.

        Args:
            attention_weights: Mapping of token_id -> attention weight from
                the current step's softmax(QK^T / sqrt(d_k)).
            current_step: The current inference step number.
        """
        alpha = self.config.attention_ema_alpha

        # Normalize attention relative to uniform distribution so that scores
        # are meaningful regardless of context length. A token at uniform
        # average gets 0.5; tokens above average approach 1.0.
        n_tokens = len(attention_weights)
        if n_tokens > 0:
            uniform = 1.0 / n_tokens
            # Scale so that 2x uniform -> ~1.0, 0x -> ~0.0
            # Using: normalized = min(1.0, attn / (2 * uniform))
            scale_factor = 2.0 * uniform
        else:
            scale_factor = 1.0

        for token_id, new_attn in attention_weights.items():
            # Normalize: token at uniform avg -> 0.5, 2x avg -> 1.0, 0 -> 0.0
            normalized_attn = min(1.0, new_attn / scale_factor) if scale_factor > 0 else 0.0

            # Update EMA with normalized attention
            old_ema = self._attention_ema.get(token_id, 0.5)
            self._attention_ema[token_id] = alpha * normalized_attn + (1.0 - alpha) * old_ema

            # Update last hot step if token receives above-average attention
            if normalized_attn > 0.5:
                self._last_hot_step[token_id] = current_step

    def get_score(self, token_id: int, current_step: Optional[int] = None) -> float:
        """Compute the combined importance score for a single token.

        importance = w_static * static + w_attention * ema + w_recency * recency

        Args:
            token_id: The token's unique identifier.
            current_step: Current inference step (defaults to 0 if not provided).

        Returns:
            Combined importance score clamped to [0.0, 1.0].
        """
        if current_step is None:
            current_step = 0

        static = self._static_scores.get(token_id, 0.5)
        ema = self.attention_ema_score(token_id)
        recency = self.recency_weight(token_id, current_step)

        score = (
            self.config.w_static * static
            + self.config.w_attention * ema
            + self.config.w_recency * recency
        )
        return max(0.0, min(1.0, score))

    def get_scores_batch(
        self, token_ids: List[int], current_step: Optional[int] = None
    ) -> Dict[int, float]:
        """Compute importance scores for a batch of tokens.

        Args:
            token_ids: List of token identifiers to score.
            current_step: Current inference step.

        Returns:
            Dictionary mapping token_id -> importance score.
        """
        return {tid: self.get_score(tid, current_step) for tid in token_ids}

    def reset(self) -> None:
        """Clear all ISE state for a new conversation."""
        self._attention_ema.clear()
        self._last_hot_step.clear()
        self._static_scores.clear()
        self._token_texts.clear()
        logger.debug("ISE state reset")
