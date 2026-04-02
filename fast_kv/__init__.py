"""Fast-KV: Tiered KV Cache Compression for Edge LLMs."""

from fast_kv.config import FastKVConfig
from fast_kv.importance_scorer import ImportanceScoringEngine
from fast_kv.compression import quantize_vector, dequantize_vector
from fast_kv.tier_manager import TierManager
from fast_kv.fast_kv_cache import FastKVCache

__all__ = [
    "FastKVConfig",
    "ImportanceScoringEngine",
    "quantize_vector",
    "dequantize_vector",
    "TierManager",
    "FastKVCache",
]
__version__ = "0.1.0"
