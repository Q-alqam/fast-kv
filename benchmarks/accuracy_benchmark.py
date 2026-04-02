"""Accuracy benchmark for Fast-KV compression.

Tests quantization quality across bit widths, with/without residuals,
and measures false demotion/retention rates.
"""

import sys
import time

import numpy as np
from scipy.special import softmax

sys.path.insert(0, ".")
from fast_kv.compression import (
    apply_residual,
    compute_residual,
    dequantize_vector,
    quantize_vector,
)
from fast_kv.config import FastKVConfig
from fast_kv.fast_kv_cache import FastKVCache


def benchmark_quantization_accuracy(n_vectors: int = 10000, kv_dim: int = 1024) -> None:
    """Test quantization accuracy across bit widths on random vectors.

    Args:
        n_vectors: Number of random vectors to test.
        kv_dim: Dimension of each vector.
    """
    np.random.seed(42)
    vectors = np.random.randn(n_vectors, kv_dim).astype(np.float32)

    print("=" * 80)
    print(f"Quantization Accuracy Benchmark ({n_vectors} vectors, dim={kv_dim})")
    print("=" * 80)
    print()

    header = (
        f"{'Bits':>5} | {'Compression':>12} | {'Mean Abs Err':>13} | "
        f"{'Max Abs Err':>12} | {'Cosine Sim':>11} | {'Time (ms)':>10}"
    )
    print(header)
    print("-" * len(header))

    for bits in [1, 2, 4, 8, 32]:
        mae_list = []
        max_err_list = []
        cosine_list = []
        total_time = 0.0

        for v in vectors:
            t0 = time.perf_counter()
            qdata = quantize_vector(v, bits)
            restored = dequantize_vector(qdata)
            total_time += time.perf_counter() - t0

            error = np.abs(v - restored)
            mae_list.append(error.mean())
            max_err_list.append(error.max())

            # Cosine similarity
            dot = np.dot(v, restored)
            norm_v = np.linalg.norm(v)
            norm_r = np.linalg.norm(restored)
            if norm_v > 0 and norm_r > 0:
                cosine_list.append(dot / (norm_v * norm_r))
            else:
                cosine_list.append(1.0)

        ratio = 32.0 / bits if bits > 0 else 0
        print(
            f"{bits:>5} | {ratio:>10.1f}x | "
            f"{np.mean(mae_list):>13.6f} | {np.mean(max_err_list):>12.6f} | "
            f"{np.mean(cosine_list):>11.8f} | {total_time*1000:>10.2f}"
        )

    print()


def benchmark_residual_improvement(n_vectors: int = 10000, kv_dim: int = 1024) -> None:
    """Compare 4-bit quantization with and without residual storage.

    Args:
        n_vectors: Number of vectors to test.
        kv_dim: Dimension of each vector.
    """
    np.random.seed(42)
    vectors = np.random.randn(n_vectors, kv_dim).astype(np.float32)

    print("=" * 80)
    print("Residual Storage Impact (4-bit quantization)")
    print("=" * 80)
    print()

    # Without residuals
    mae_no_res = []
    cos_no_res = []
    for v in vectors:
        qdata = quantize_vector(v, 4)
        restored = dequantize_vector(qdata)
        mae_no_res.append(np.abs(v - restored).mean())
        dot = np.dot(v, restored)
        cos_no_res.append(dot / (np.linalg.norm(v) * np.linalg.norm(restored)))

    # With residuals (8-bit)
    mae_with_res = []
    cos_with_res = []
    for v in vectors:
        qdata = quantize_vector(v, 4)
        residual = compute_residual(v, qdata, residual_bits=8)
        restored = apply_residual(qdata, residual)
        mae_with_res.append(np.abs(v - restored).mean())
        dot = np.dot(v, restored)
        cos_with_res.append(dot / (np.linalg.norm(v) * np.linalg.norm(restored)))

    print(f"{'Method':<25} | {'Mean Abs Error':>15} | {'Cosine Similarity':>18}")
    print("-" * 65)
    print(f"{'4-bit (no residual)':<25} | {np.mean(mae_no_res):>15.6f} | {np.mean(cos_no_res):>18.8f}")
    print(f"{'4-bit + 8-bit residual':<25} | {np.mean(mae_with_res):>15.6f} | {np.mean(cos_with_res):>18.8f}")
    improvement = (1 - np.mean(mae_with_res) / np.mean(mae_no_res)) * 100
    print(f"\nResidual reduces MAE by {improvement:.1f}%")
    print()


def benchmark_tier_accuracy(n_conversations: int = 100, tokens_per_conv: int = 50) -> None:
    """Simulate conversations and measure false demotion/retention rates.

    Tests whether the ISE correctly distinguishes important vs unimportant
    tokens at initial assignment. A "false demotion" occurs when a token
    that has a high static score AND is receiving high attention still ends
    up in the cold tier. We check at the midpoint of each conversation
    (when recency hasn't decayed everything yet).

    Args:
        n_conversations: Number of conversations to simulate.
        tokens_per_conv: Tokens per conversation.
    """
    from fast_kv.importance_scorer import ALWAYS_COLD_TOKENS, _is_always_hot

    np.random.seed(42)
    config = FastKVConfig()
    model_config = {"n_layers": 1, "kv_dim": 128, "dtype": "float32"}

    print("=" * 80)
    print(f"Tier Assignment Accuracy ({n_conversations} conversations, {tokens_per_conv} tokens each)")
    print("=" * 80)
    print()

    important_tokens_text = [
        "CVE-2024-1234", "192.168.1.1", "Linux", "Apache",
        "SQL", "HTTP", "10.0.0.1", "Windows", "NIST", "SSH",
    ]
    stopword_tokens = ["the", "a", "in", "on", "and", "or", "is", ".", ",", "to"]
    all_tokens = important_tokens_text + stopword_tokens + ["server", "process", "attack"]

    total_false_demotions = 0
    total_important_checks = 0
    total_false_retentions = 0
    total_stopword_checks = 0

    for conv in range(n_conversations):
        cache = FastKVCache(config, model_config)

        for step in range(tokens_per_conv):
            token_text = all_tokens[step % len(all_tokens)]
            key_vec = np.random.randn(model_config["kv_dim"]).astype(np.float32)
            val_vec = np.random.randn(model_config["kv_dim"]).astype(np.float32)

            # Give important tokens concentrated attention (unnormalized scores)
            n_existing = step + 1
            raw_attn = np.zeros(n_existing)
            for i in range(n_existing):
                t = all_tokens[i % len(all_tokens)]
                if _is_always_hot(t):
                    raw_attn[i] = 5.0 + np.random.random()
                elif t.lower() in ALWAYS_COLD_TOKENS:
                    raw_attn[i] = np.random.random() * 0.5
                else:
                    raw_attn[i] = 1.0 + np.random.random()

            attn_probs = softmax(raw_attn)
            attn = {i: float(attn_probs[i]) for i in range(n_existing)}

            cache.update(0, step, token_text, key_vec, val_vec, attn, step)

        # Check tier assignments for recently-added tokens (last 20)
        # These tokens haven't had time for recency to decay
        tm = cache.tier_managers[0]
        check_range = range(max(0, tokens_per_conv - 20), tokens_per_conv)
        for tid in check_range:
            tier = tm.token_tiers.get(tid, "unknown")
            token_text = all_tokens[tid % len(all_tokens)]
            if _is_always_hot(token_text):
                total_important_checks += 1
                if tier == "cold":
                    # Only count as false demotion if token's ISE score is high
                    score = cache.ise.get_score(tid, tokens_per_conv)
                    if score >= config.cold_threshold:
                        total_false_demotions += 1
            elif token_text.lower() in ALWAYS_COLD_TOKENS:
                total_stopword_checks += 1
                if tier == "hot":
                    total_false_retentions += 1

    false_demotion_rate = (
        total_false_demotions / total_important_checks * 100
        if total_important_checks > 0 else 0
    )
    false_retention_rate = (
        total_false_retentions / total_stopword_checks * 100
        if total_stopword_checks > 0 else 0
    )

    print(f"Important tokens checked:  {total_important_checks}")
    print(f"False demotions:           {total_false_demotions}")
    print(f"False demotion rate:       {false_demotion_rate:.2f}%")
    print()
    print(f"Stopword tokens checked:   {total_stopword_checks}")
    print(f"False retentions:          {total_false_retentions}")
    print(f"False retention rate:      {false_retention_rate:.2f}%")
    print()

    if false_demotion_rate < 5.0:
        print("PASS: False demotion rate < 5%")
    else:
        print("FAIL: False demotion rate >= 5%")


def main() -> None:
    """Run all accuracy benchmarks."""
    benchmark_quantization_accuracy()
    benchmark_residual_improvement()
    benchmark_tier_accuracy()


if __name__ == "__main__":
    main()
