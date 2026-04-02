"""Memory benchmark for Fast-KV.

Simulates realistic LLM conversations at various lengths and measures
RAM usage with and without Fast-KV compression. Uses synthetic KV vectors
and attention weights — no real LLM required.
"""

import sys
import time
from typing import Dict, List, Tuple

import numpy as np
from scipy.special import softmax

sys.path.insert(0, ".")
from fast_kv.config import FastKVConfig
from fast_kv.fast_kv_cache import FastKVCache

# Llama 3 8B parameters
MODEL_CONFIG = {
    "n_layers": 32,
    "kv_dim": 1024,
    "dtype": "float32",
}

# Token vocabulary for realistic simulation
STOPWORDS = [
    "the", "a", "an", "in", "on", "at", "to", "of", "for", "by", "with",
    "and", "or", "but", "is", "are", "was", "were", "be", "do", "has",
    "had", ".", ",", ";", ":", "!", "?",
]
NOUNS = [
    "system", "server", "attack", "network", "process", "file", "memory",
    "kernel", "module", "payload", "exploit", "database", "firewall",
    "token", "cache", "buffer", "socket", "thread", "daemon", "binary",
]
VERBS = [
    "detected", "executed", "scanned", "blocked", "encrypted", "connected",
    "terminated", "injected", "escalated", "resolved", "patched", "deployed",
]
PROPER_NOUNS = [
    "Linux", "Windows", "Apache", "Nginx", "Docker", "Kubernetes",
    "CVE-2024-1234", "CVE-2023-5678", "192.168.1.1", "10.0.0.1",
    "MITRE", "NIST", "SSH", "HTTP", "TLS", "DNS",
]


def generate_token_text(idx: int, n_tokens: int) -> str:
    """Generate a realistic token text based on position.

    Mixes stopwords (~40%), nouns (~25%), verbs (~15%), proper nouns (~20%).
    """
    rng = np.random.RandomState(idx)
    r = rng.random()
    if r < 0.40:
        return rng.choice(STOPWORDS)
    elif r < 0.65:
        return rng.choice(NOUNS)
    elif r < 0.80:
        return rng.choice(VERBS)
    else:
        return rng.choice(PROPER_NOUNS)


def run_benchmark(n_tokens: int, config: FastKVConfig) -> Dict:
    """Run a single benchmark scenario.

    Args:
        n_tokens: Number of tokens in the simulated conversation.
        config: FastKVConfig instance.

    Returns:
        Dictionary with benchmark results.
    """
    np.random.seed(42)
    cache = FastKVCache(config, MODEL_CONFIG)
    kv_dim = MODEL_CONFIG["kv_dim"]

    # Track timing
    total_ise_time = 0.0
    total_time_start = time.perf_counter()

    # Track reconstruction errors
    original_vectors = {}

    for step in range(n_tokens):
        token_text = generate_token_text(step, n_tokens)

        # Synthetic KV vectors
        key_vec = np.random.randn(kv_dim).astype(np.float32)
        value_vec = np.random.randn(kv_dim).astype(np.float32)
        original_vectors[step] = np.concatenate([key_vec, value_vec])

        # Synthetic attention weights (softmax over existing tokens)
        n_existing = step + 1
        raw_attn = np.random.randn(n_existing)
        attn_probs = softmax(raw_attn)
        attention_weights = {i: float(attn_probs[i]) for i in range(n_existing)}

        # Time the ISE + tier management
        t0 = time.perf_counter()
        # Only update layer 0 for ISE timing (ISE is shared)
        cache.update(0, step, token_text, key_vec, value_vec, attention_weights, step)
        total_ise_time += time.perf_counter() - t0

        # Update remaining layers without timing ISE
        for layer_id in range(1, MODEL_CONFIG["n_layers"]):
            key_vec_l = np.random.randn(kv_dim).astype(np.float32)
            value_vec_l = np.random.randn(kv_dim).astype(np.float32)
            cache.update(
                layer_id, step, token_text, key_vec_l, value_vec_l,
                attention_weights, step,
            )

    total_time = time.perf_counter() - total_time_start

    # Measure reconstruction accuracy on layer 0
    tm = cache.tier_managers[0]
    errors = []
    for tid, orig in original_vectors.items():
        reconstructed = tm.get_kv_for_attention([tid])[0]
        errors.append(float(np.abs(orig - reconstructed).mean()))
    mean_error = np.mean(errors) if errors else 0.0

    # Get memory stats
    report = cache.get_memory_report()
    stats = cache.tier_managers[0].get_stats()

    # Baseline RAM: all tokens at full precision across all layers
    bytes_per_token = kv_dim * 2 * 4  # K+V, float32
    baseline_ram_mb = n_tokens * bytes_per_token * MODEL_CONFIG["n_layers"] / (1024 ** 2)

    # Fast-KV RAM (aggregated from report)
    fkv_ram_mb = sum(
        tm.get_stats()["estimated_ram_total_mb"]
        for tm in cache.tier_managers.values()
    )

    savings_pct = (1.0 - fkv_ram_mb / baseline_ram_mb) * 100 if baseline_ram_mb > 0 else 0

    return {
        "n_tokens": n_tokens,
        "baseline_ram_mb": baseline_ram_mb,
        "fkv_ram_mb": fkv_ram_mb,
        "savings_pct": savings_pct,
        "mean_abs_error": mean_error,
        "ise_time_s": total_ise_time,
        "total_time_s": total_time,
        "ise_overhead_pct": (total_ise_time / total_time * 100) if total_time > 0 else 0,
        "hot_fraction": stats["hot_fraction"],
        "report": report,
    }


def main() -> None:
    """Run benchmarks for all conversation lengths and print comparison table."""
    config = FastKVConfig()
    scenarios = [100, 500, 1000, 5000]

    print("=" * 90)
    print("Fast-KV Memory Benchmark")
    print(f"Model: Llama 3 8B (n_layers={MODEL_CONFIG['n_layers']}, kv_dim={MODEL_CONFIG['kv_dim']})")
    print("=" * 90)
    print()

    results = []
    for n_tokens in scenarios:
        print(f"Running {n_tokens}-token scenario...")
        result = run_benchmark(n_tokens, config)
        results.append(result)
        print(f"  Done in {result['total_time_s']:.1f}s")

    # Print comparison table
    print()
    print("-" * 90)
    print(f"{'Conversation':<16} | {'Baseline RAM':>12} | {'Fast-KV RAM':>12} | "
          f"{'Savings':>8} | {'Accuracy Loss':>14} | {'ISE Overhead':>12}")
    print("-" * 90)

    for r in results:
        print(
            f"{r['n_tokens']:>6} tokens    | "
            f"{r['baseline_ram_mb']:>9.1f} MB | "
            f"{r['fkv_ram_mb']:>9.1f} MB | "
            f"{r['savings_pct']:>6.1f}% | "
            f"{r['mean_abs_error']:>12.5f}% | "
            f"{r['ise_overhead_pct']:>10.1f}%"
        )

    print("-" * 90)

    # Print detailed report for 1000-token scenario
    print()
    for r in results:
        if r["n_tokens"] == 1000:
            print(r["report"])
            break


if __name__ == "__main__":
    main()
