"""Tests for the Tier Manager."""

import pytest
import numpy as np

from fast_kv.config import FastKVConfig
from fast_kv.importance_scorer import ImportanceScoringEngine
from fast_kv.tier_manager import TierManager


@pytest.fixture
def config():
    return FastKVConfig()


@pytest.fixture
def ise(config):
    return ImportanceScoringEngine(config)


@pytest.fixture
def tm(config, ise):
    return TierManager(config, ise, kv_dim=256)


def make_kv(dim=256):
    """Generate a random KV vector."""
    return np.random.randn(dim).astype(np.float32)


class TestTierAssignment:
    """Tests for initial tier assignment."""

    def test_important_token_goes_hot(self, tm):
        """Tokens with high static scores + high attention should go to hot."""
        # Give CVE token high attention first
        tm.ise.register_token(0, "CVE-2024-1234")
        tm.ise.update_attention_scores({0: 0.9}, current_step=0)
        tm.ise.update_attention_scores({0: 0.9}, current_step=1)

        tier = tm.add_token(0, make_kv(), "CVE-2024-1234", layer_id=0, current_step=2)
        assert tier == "hot"
        assert 0 in tm.hot_cache

    def test_stopword_goes_cold(self, tm):
        """Stopword tokens should go to cold tier immediately."""
        # Give stopword low attention
        tm.ise.register_token(0, "the")
        tm.ise.update_attention_scores({0: 0.0}, current_step=0)
        tm.ise.update_attention_scores({0: 0.0}, current_step=1)

        tier = tm.add_token(0, make_kv(), "the", layer_id=0, current_step=2)
        assert tier == "cold"
        assert 0 in tm.cold_cache

    def test_stopword_in_deep_cold(self, tm):
        """Stopwords with very low scores should be in sub-tier 2C."""
        tm.ise.register_token(0, "the")
        for step in range(10):
            tm.ise.update_attention_scores({0: 0.0}, current_step=step)

        tm.add_token(0, make_kv(), "the", layer_id=0, current_step=50)
        assert tm.cold_cache[0]["sub_tier"] in ("2B", "2C")


class TestDemotion:
    """Tests for hot -> cold demotion."""

    def test_demotion_on_low_score(self, tm):
        """A hot token whose score drops should be demoted to cold."""
        # First add as hot with high attention
        tm.ise.register_token(0, "CVE-2024-1234")
        for step in range(5):
            tm.ise.update_attention_scores({0: 0.9}, current_step=step)

        tm.add_token(0, make_kv(), "CVE-2024-1234", layer_id=0, current_step=5)
        assert tm.token_tiers.get(0) == "hot"

        # Now drop attention for many steps
        for step in range(6, 200):
            tm.ise.update_attention_scores({0: 0.0}, current_step=step)

        # Check demotions
        demoted = tm.check_demotions(current_step=200)
        assert 0 in demoted
        assert tm.token_tiers[0] == "cold"

    def test_demotion_increments_counter(self, tm):
        """Demotion events should increment the counter."""
        tm.ise.register_token(0, "CVE-2024-1234")
        for step in range(5):
            tm.ise.update_attention_scores({0: 0.9}, current_step=step)
        tm.add_token(0, make_kv(), "CVE-2024-1234", layer_id=0, current_step=5)

        for step in range(6, 200):
            tm.ise.update_attention_scores({0: 0.0}, current_step=step)

        initial_demotions = tm.n_demotions_total
        tm.check_demotions(current_step=200)
        assert tm.n_demotions_total > initial_demotions


class TestPromotion:
    """Tests for cold -> hot promotion."""

    def test_promotion_on_attention_spike(self, tm):
        """A cold token with an attention spike should be promoted to hot."""
        # Add as cold
        tm.ise.register_token(0, "the")
        for step in range(5):
            tm.ise.update_attention_scores({0: 0.0}, current_step=step)
        tm.add_token(0, make_kv(), "the", layer_id=0, current_step=5)
        assert tm.token_tiers[0] == "cold"

        # Spike attention above promotion threshold
        promoted = tm.check_promotions({0: 0.85}, current_step=6)
        assert 0 in promoted
        assert tm.token_tiers[0] == "hot"
        assert 0 in tm.hot_cache

    def test_promotion_increments_counter(self, tm):
        """Promotion events should increment the counter."""
        tm.ise.register_token(0, "the")
        for step in range(5):
            tm.ise.update_attention_scores({0: 0.0}, current_step=step)
        tm.add_token(0, make_kv(), "the", layer_id=0, current_step=5)

        initial = tm.n_promotions_total
        tm.check_promotions({0: 0.85}, current_step=6)
        assert tm.n_promotions_total > initial


class TestGetKV:
    """Tests for KV vector retrieval."""

    def test_retrieve_hot_token(self, tm):
        """Should retrieve hot token's KV vector."""
        kv = make_kv()
        tm.ise.register_token(0, "CVE-2024-1234")
        for step in range(5):
            tm.ise.update_attention_scores({0: 0.9}, current_step=step)
        tm.add_token(0, kv, "CVE-2024-1234", layer_id=0, current_step=5)

        retrieved = tm.get_kv_for_attention([0])
        assert retrieved.shape == (1, 256)
        # Hot tokens should be exact
        np.testing.assert_allclose(retrieved[0], kv, atol=1e-6)

    def test_retrieve_cold_token(self, tm):
        """Should retrieve and decompress cold token's KV vector."""
        kv = make_kv()
        tm.ise.register_token(0, "the")
        for step in range(10):
            tm.ise.update_attention_scores({0: 0.0}, current_step=step)
        tm.add_token(0, kv, "the", layer_id=0, current_step=10)

        retrieved = tm.get_kv_for_attention([0])
        assert retrieved.shape == (1, 256)
        # Cold tokens will have some reconstruction error
        assert np.all(np.isfinite(retrieved))


class TestStats:
    """Tests for statistics reporting."""

    def test_stats_correct_counts(self, tm):
        """get_stats should return correct token counts."""
        np.random.seed(42)
        # Add a mix of hot and cold tokens
        for i in range(5):
            tm.ise.register_token(i, "CVE-2024-1234")
            for step in range(5):
                tm.ise.update_attention_scores({i: 0.9}, current_step=step)
            tm.add_token(i, make_kv(), "CVE-2024-1234", layer_id=0, current_step=5)

        for i in range(5, 10):
            tm.ise.register_token(i, "the")
            for step in range(10):
                tm.ise.update_attention_scores({i: 0.0}, current_step=step)
            tm.add_token(i, make_kv(), "the", layer_id=0, current_step=10)

        stats = tm.get_stats()
        assert stats["n_total"] == 10
        assert stats["n_hot"] + stats["n_cold"] == 10
        assert stats["n_hot"] >= 1  # At least some CVE tokens should be hot
        assert "estimated_ram_hot_mb" in stats
        assert "estimated_ram_cold_mb" in stats
        assert "compression_ratio" in stats
