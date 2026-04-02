"""Speed benchmark for Fast-KV.

Measures the ISE compute overhead relative to simulated total inference time.
"""

import sys
import time

import numpy as np
from scipy.special import softmax

sys.path.insert(0, ".")
from fast_kv.config import FastKVConfig
from fast_kv.fast_kv_cache import FastKVCache


# Simulated inference latency per token (ms) for consumer hardware
SIMULATED_INFERENCE_MS = 100.0  # Typical for 8B model on consumer CPU


def benchmark_ise_overhead(n_tokens: int = 1000) -> None:
    """Measure ISE compute time vs simulated total inference time.

    Args:
        n_tokens: Number of tokens to simulate.
    """
    np.random.seed(42)
    config = FastKVConfig()
    model_config = {"n_layers": 32, "kv_dim": 1024, "dtype": "float32"}
    cache = FastKVCache(config, model_config)

    stopwords = ["the", "a", "in", "on", "and", "or", "is", ".", ",", "to"]
    important = ["CVE-2024-1234", "Linux", "Apache", "192.168.1.1", "SQL"]
    nouns = ["server", "process", "attack", "network", "exploit"]
    all_tokens = stopwords + important + nouns

    ise_times = []
    tier_times = []

    for step in range(n_tokens):
        token_text = all_tokens[step % len(all_tokens)]
        key_vec = np.random.randn(model_config["kv_dim"]).astype(np.float32)
        val_vec = np.random.randn(model_config["kv_dim"]).astype(np.float32)

        # Generate attention weights
        n_existing = step + 1
        raw_attn = np.random.randn(min(n_existing, 100))
        attn_probs = softmax(raw_attn)
        attention_weights = {
            i: float(attn_probs[i % len(attn_probs)])
            for i in range(n_existing)
        }

        # Time ISE update alone
        t0 = time.perf_counter()
        cache.ise.update_attention_scores(attention_weights, step)
        cache.ise.register_token(step, token_text)
        _ = cache.ise.get_score(step, step)
        ise_time = time.perf_counter() - t0
        ise_times.append(ise_time)

        # Time full tier management (includes compression)
        t0 = time.perf_counter()
        for layer_id in range(model_config["n_layers"]):
            kv = np.random.randn(model_config["kv_dim"]).astype(np.float32)
            vv = np.random.randn(model_config["kv_dim"]).astype(np.float32)
            cache.update(layer_id, step, token_text, kv, vv, attention_weights, step)
        tier_time = time.perf_counter() - t0
        tier_times.append(tier_time)

    total_ise_ms = sum(ise_times) * 1000
    total_tier_ms = sum(tier_times) * 1000
    simulated_inference_ms = n_tokens * SIMULATED_INFERENCE_MS

    avg_ise_us = total_ise_ms / n_tokens * 1000  # microseconds
    avg_tier_us = total_tier_ms / n_tokens * 1000

    print("=" * 70)
    print(f"Fast-KV Speed Benchmark ({n_tokens} tokens)")
    print("=" * 70)
    print()
    print(f"ISE Compute:")
    print(f"  Total ISE time:          {total_ise_ms:>10.2f} ms")
    print(f"  Avg ISE time/token:      {avg_ise_us:>10.1f} us")
    print(f"  ISE % of inference:      {total_ise_ms/simulated_inference_ms*100:>10.4f}%")
    print()
    print(f"Full Tier Management (ISE + compression + tier ops):")
    print(f"  Total tier time:         {total_tier_ms:>10.2f} ms")
    print(f"  Avg tier time/token:     {avg_tier_us:>10.1f} us")
    print(f"  Tier % of inference:     {total_tier_ms/simulated_inference_ms*100:>10.4f}%")
    print()
    print(f"Simulated inference time:  {simulated_inference_ms:>10.0f} ms")
    print(f"  ({SIMULATED_INFERENCE_MS:.0f} ms/token on consumer CPU)")
    print()

    if total_ise_ms / simulated_inference_ms * 100 < 1.0:
        print("PASS: ISE overhead < 1% of inference time")
    else:
        print("WARN: ISE overhead >= 1% of inference time")


def main() -> None:
    """Run the speed benchmark."""
    benchmark_ise_overhead(500)


if __name__ == "__main__":
    main()
