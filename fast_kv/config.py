"""Configuration dataclass for all Fast-KV tunable parameters."""

from dataclasses import dataclass


@dataclass
class FastKVConfig:
    """All tunable parameters for the Fast-KV system.

    Attributes:
        hot_threshold: Score >= this places token in Tier 1 (hot cache).
        cold_threshold: Score <= this places token in Tier 2 (cold cache).
            Scores between cold_threshold and hot_threshold form a hysteresis
            buffer zone where tokens keep their current tier assignment.
        promotion_spike_threshold: Instant promotion if a cold token's
            attention weight exceeds this value in a single step.
        w_static: Weight for the static token classification score (Layer 1).
        w_attention: Weight for the attention EMA score (Layer 2).
        w_recency: Weight for the recency decay score (Layer 3).
            Note: w_static + w_attention + w_recency must equal 1.0.
        attention_ema_alpha: Smoothing factor for exponential moving average
            of attention scores. Lower values = slower decay.
        recency_lambda: Controls how fast the recency weight decays.
            Higher values = faster decay of old tokens' importance.
        bits_subtier_2a: Bit width for sub-tier 2A (score 0.35-0.50).
        bits_subtier_2b: Bit width for sub-tier 2B (score 0.15-0.35).
        bits_subtier_2c: Bit width for sub-tier 2C (score 0.00-0.15).
        use_residuals: Whether to store quantization residuals for sub-tier 2A.
        residual_bits: Bit width for residual error storage.
        demotion_check_every_n_steps: How often to check for hot->cold demotions.
        hot_tier_max_fraction: Target maximum fraction of tokens in hot tier.
    """

    # Tier thresholds
    hot_threshold: float = 0.65
    cold_threshold: float = 0.35
    promotion_spike_threshold: float = 0.70

    # ISE weights (must sum to 1.0)
    w_static: float = 0.30
    w_attention: float = 0.50
    w_recency: float = 0.20

    # Attention EMA
    attention_ema_alpha: float = 0.3

    # Recency decay
    recency_lambda: float = 0.02

    # Tier 2 sub-tier bit widths
    bits_subtier_2a: int = 4   # Score 0.35-0.50 -> 4-bit
    bits_subtier_2b: int = 2   # Score 0.15-0.35 -> 2-bit
    bits_subtier_2c: int = 1   # Score 0.00-0.15 -> 1-bit

    # Residual storage
    use_residuals: bool = True
    residual_bits: int = 8

    # Demotion schedule
    demotion_check_every_n_steps: int = 10

    # Hot tier target size
    hot_tier_max_fraction: float = 0.25

    # Warmup period
    warmup_steps: int = 50  # All tokens stay in Tier 1 during warmup

    # Outlier-aware quantization
    use_outlier_aware: bool = True    # Master toggle for outlier detection
    outlier_sigma_2a: float = 3.0    # Outlier threshold for 4-bit sub-tier
    outlier_sigma_2b: float = 2.5    # Outlier threshold for 2-bit sub-tier
    outlier_sigma_2c: float = 2.0    # Outlier threshold for 1-bit sub-tier

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        weight_sum = self.w_static + self.w_attention + self.w_recency
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(
                f"ISE weights must sum to 1.0, got {weight_sum:.4f}"
            )
        if self.cold_threshold >= self.hot_threshold:
            raise ValueError(
                "cold_threshold must be less than hot_threshold"
            )
