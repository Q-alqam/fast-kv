"""Tests for the Importance Scoring Engine."""

import pytest
import numpy as np

from fast_kv.config import FastKVConfig
from fast_kv.importance_scorer import (
    ALWAYS_COLD_TOKENS,
    ImportanceScoringEngine,
    _is_always_hot,
)


@pytest.fixture
def ise():
    """Create an ISE with default config."""
    return ImportanceScoringEngine(FastKVConfig())


class TestStaticScore:
    """Tests for Layer 1: static token classification."""

    def test_stopwords_score_zero(self, ise):
        """All stopwords should get static_score = 0.0."""
        for token in ["the", "a", "an", "in", "on", "and", "or", "is", ".", ","]:
            assert ise.static_score(token) == 0.0, f"'{token}' should be 0.0"

    def test_all_cold_tokens_comprehensive(self, ise):
        """Every token in ALWAYS_COLD_TOKENS should score 0.0."""
        for token in ALWAYS_COLD_TOKENS:
            assert ise.static_score(token) == 0.0, f"'{token}' should be cold"

    def test_proper_nouns_score_one(self, ise):
        """Proper nouns (capitalized mid-sentence tokens) should score 1.0."""
        for token in ["Linux", "Apache", "Windows", "Docker"]:
            assert ise.static_score(token) == 1.0, f"'{token}' should be 1.0"

    def test_acronyms_score_one(self, ise):
        """All-uppercase tokens should score 1.0."""
        for token in ["SQL", "HTTP", "DNS", "TLS", "SSH"]:
            assert ise.static_score(token) == 1.0, f"'{token}' should be 1.0"

    def test_numbers_score_one(self, ise):
        """Numeric tokens should score 1.0."""
        for token in ["123", "3.14", "2024", "0.001"]:
            assert ise.static_score(token) == 1.0, f"'{token}' should be 1.0"

    def test_cve_score_one(self, ise):
        """CVE identifiers should score 1.0."""
        assert ise.static_score("CVE-2024-1234") == 1.0

    def test_ip_address_score_one(self, ise):
        """IP addresses should score 1.0."""
        assert ise.static_score("192.168.1.1") == 1.0

    def test_question_words_score_one(self, ise):
        """Question words should score 1.0."""
        for token in ["who", "what", "where", "when", "why", "how"]:
            assert ise.static_score(token) == 1.0, f"'{token}' should be 1.0"

    def test_first_person_score_one(self, ise):
        """First-person pronouns should score 1.0."""
        for token in ["I", "my", "me", "mine", "we", "our"]:
            assert ise.static_score(token) == 1.0, f"'{token}' should be 1.0"

    def test_neutral_tokens_score_half(self, ise):
        """Tokens matching neither cold nor hot should score 0.5."""
        for token in ["server", "attack", "running", "detected"]:
            assert ise.static_score(token) == 0.5, f"'{token}' should be 0.5"


class TestAttentionEMA:
    """Tests for Layer 2: attention EMA."""

    def test_initial_ema_is_half(self, ise):
        """New tokens should start with EMA of 0.5."""
        ise.register_token(0, "test")
        assert ise.attention_ema_score(0) == 0.5

    def test_ema_increases_with_high_attention(self, ise):
        """EMA should increase when token receives above-average attention."""
        ise.register_token(0, "test")
        ise.register_token(1, "other")
        # With 2 tokens, uniform = 0.5. Token 0 gets 0.9 (well above avg)
        ise.update_attention_scores({0: 0.05, 1: 0.05}, current_step=0)
        low = ise.attention_ema_score(0)
        ise.update_attention_scores({0: 0.9, 1: 0.1}, current_step=1)
        after = ise.attention_ema_score(0)
        assert after > low

    def test_ema_decreases_with_low_attention(self, ise):
        """EMA should decrease when token receives low attention."""
        ise.register_token(0, "test")
        # First give high attention
        ise.update_attention_scores({0: 1.0}, current_step=1)
        high = ise.attention_ema_score(0)
        # Then give zero attention repeatedly
        for step in range(2, 10):
            ise.update_attention_scores({0: 0.0}, current_step=step)
        low = ise.attention_ema_score(0)
        assert low < high

    def test_ema_correctly_decays_over_steps(self, ise):
        """EMA should decay predictably with constant zero attention."""
        ise.register_token(0, "test")
        alpha = ise.config.attention_ema_alpha

        # Manually verify EMA computation
        ema = 0.5  # initial
        for step in range(1, 6):
            ise.update_attention_scores({0: 0.0}, current_step=step)
            ema = alpha * 0.0 + (1 - alpha) * ema  # expected

        assert abs(ise.attention_ema_score(0) - ema) < 1e-6


class TestRecencyWeight:
    """Tests for Layer 3: recency decay."""

    def test_recency_max_at_current_step(self, ise):
        """Recency weight should be 1.0 when last_hot_step == current_step."""
        ise.register_token(0, "test")
        ise._last_hot_step[0] = 100
        weight = ise.recency_weight(0, current_step=100)
        assert abs(weight - 1.0) < 1e-6

    def test_recency_decreases_over_time(self, ise):
        """Recency weight should decrease as steps increase."""
        ise.register_token(0, "test")
        ise._last_hot_step[0] = 0
        w1 = ise.recency_weight(0, current_step=10)
        w2 = ise.recency_weight(0, current_step=100)
        w3 = ise.recency_weight(0, current_step=1000)
        assert w1 > w2 > w3

    def test_recency_never_negative(self, ise):
        """Recency weight should always be positive."""
        ise.register_token(0, "test")
        ise._last_hot_step[0] = 0
        for step in [1, 10, 100, 1000, 100000]:
            w = ise.recency_weight(0, current_step=step)
            assert w > 0.0


class TestCombinedScore:
    """Tests for the combined importance score."""

    def test_score_in_valid_range(self, ise):
        """Combined score should always be in [0.0, 1.0]."""
        for token_text in ["the", "Linux", "server", "CVE-2024-1234", "."]:
            ise.register_token(hash(token_text), token_text)
            score = ise.get_score(hash(token_text), current_step=50)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for '{token_text}'"

    def test_cold_tokens_score_low(self, ise):
        """Stopwords should have low combined scores."""
        ise.register_token(0, "the")
        # Give low attention too
        ise.update_attention_scores({0: 0.0}, current_step=1)
        ise.update_attention_scores({0: 0.0}, current_step=2)
        score = ise.get_score(0, current_step=50)
        assert score < 0.4

    def test_hot_tokens_score_high(self, ise):
        """Important tokens with high attention should score high."""
        ise.register_token(0, "CVE-2024-1234")
        ise.update_attention_scores({0: 0.9}, current_step=1)
        ise.update_attention_scores({0: 0.9}, current_step=2)
        score = ise.get_score(0, current_step=2)
        assert score > 0.7

    def test_batch_scores(self, ise):
        """get_scores_batch should return scores for all requested tokens."""
        tokens = {0: "the", 1: "Linux", 2: "server"}
        for tid, text in tokens.items():
            ise.register_token(tid, text)
        scores = ise.get_scores_batch([0, 1, 2], current_step=0)
        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores.values())

    def test_reset_clears_state(self, ise):
        """reset() should clear all internal state."""
        ise.register_token(0, "test")
        ise.update_attention_scores({0: 0.9}, current_step=1)
        ise.reset()
        assert len(ise._attention_ema) == 0
        assert len(ise._static_scores) == 0
        assert len(ise._last_hot_step) == 0
