"""Outlier-aware quantization benchmark for Fast-KV.

Validates the improvement of outlier-aware quantization on LLM-like vectors
and measures the impact on real model compression.
"""

import gc
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, ".")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np

from fast_kv.compression import (
    benchmark_outlier_aware,
    dequantize_vector,
    quantize_vector,
    quantize_vector_outlier_aware,
)
from fast_kv.config import FastKVConfig


def generate_llm_like_vector(dim: int = 1024) -> np.ndarray:
    """Generate a vector with outlier patterns similar to real LLM KV vectors.

    Real transformer KV vectors have ~2% of dimensions with values
    dramatically larger than the rest.

    Args:
        dim: Vector dimension.

    Returns:
        Float32 vector with injected outliers.
    """
    vector = np.random.randn(dim).astype(np.float32) * 0.3
    n_outliers = max(1, int(dim * 0.02))
    outlier_indices = np.random.choice(dim, n_outliers, replace=False)
    vector[outlier_indices] = np.random.randn(n_outliers).astype(np.float32) * 50.0
    return vector


def benchmark_quantization_comparison(n_vectors: int = 10000, dim: int = 1024) -> None:
    """Compare standard vs outlier-aware quantization on LLM-like vectors."""
    np.random.seed(42)
    vectors = [generate_llm_like_vector(dim) for _ in range(n_vectors)]

    print("=" * 75)
    print(f"Outlier-Aware Quantization Benchmark ({n_vectors} vectors, dim={dim})")
    print("=" * 75)
    print()

    for bits in [4, 2, 1]:
        std_maes = []
        oa_maes = []
        outlier_counts = []

        for v in vectors:
            result = benchmark_outlier_aware(v, bits)
            std_maes.append(result["standard_mae"])
            oa_maes.append(result["outlier_aware_mae"])
            outlier_counts.append(result["outlier_count"])

        avg_std = np.mean(std_maes)
        avg_oa = np.mean(oa_maes)
        improvement = (avg_std - avg_oa) / avg_std * 100 if avg_std > 0 else 0
        avg_outliers = np.mean(outlier_counts)

        print(f"{bits}-bit quantization:")
        print(f"  Standard MAE:       {avg_std:.6f}")
        print(f"  Outlier-Aware MAE:  {avg_oa:.6f}")
        print(f"  Improvement:        {improvement:.1f}%")
        print(f"  Avg outliers/vec:   {avg_outliers:.1f} ({avg_outliers/dim*100:.1f}%)")
        print()


def benchmark_real_model() -> None:
    """Run real model benchmark with outlier-aware quantization."""
    try:
        from fast_kv.model_hook import FastKVModelHook
    except ImportError:
        print("Skipping real model benchmark (torch/transformers not available)")
        return

    # Conversations
    conversations = [
        {
            "name": "Short (~150 tok)",
            "max_new_tokens": 60,
            "turns": [
                {"role": "user", "content": (
                    "We got an alert for CVE-2024-3094 on our edge gateway "
                    "at 10.0.12.5. The XZ Utils backdoor is flagged. "
                    "What containment steps should we take?"
                )},
                {"role": "assistant", "content": (
                    "Isolate 10.0.12.5 by moving it to a quarantine VLAN. "
                    "Check if SSH is exposed on port 22."
                )},
                {"role": "user", "content": "SSH is exposed. Is 198.51.100.23:443 C2?"},
            ],
        },
        {
            "name": "Medium (~400 tok)",
            "max_new_tokens": 80,
            "turns": [
                {"role": "user", "content": "Explain how black holes form."},
                {"role": "assistant", "content": (
                    "When a star over 25 solar masses exhausts fuel, the core "
                    "collapses. If mass exceeds the TOV limit, collapse continues "
                    "to a singularity."
                )},
                {"role": "user", "content": "What about the information paradox?"},
                {"role": "assistant", "content": (
                    "Quantum mechanics says information can't be destroyed, but "
                    "general relativity says it's lost past the event horizon."
                )},
                {"role": "user", "content": "Has the paradox been resolved?"},
            ],
        },
        {
            "name": "Long (~800 tok)",
            "max_new_tokens": 100,
            "turns": [
                {"role": "user", "content": (
                    "I'm building a REST API with FastAPI. Need rate limiting "
                    "at 100 requests/minute per API key. Best approach?"
                )},
                {"role": "assistant", "content": (
                    "Use Redis sorted sets. Store timestamps per key, remove "
                    "old entries, count remaining, reject if >= 100."
                )},
                {"role": "user", "content": "Show me the middleware implementation."},
                {"role": "assistant", "content": (
                    "Create a dependency checking Redis. Use ZADD and "
                    "ZRANGEBYSCORE for atomic counting."
                )},
                {"role": "user", "content": "What about distributed rate limiting?"},
                {"role": "assistant", "content": (
                    "Redis is shared state across instances. Use Redis Cluster "
                    "for HA. Consider lua scripting for atomicity."
                )},
                {"role": "user", "content": "How to handle burst traffic?"},
            ],
        },
        {
            "name": "Very Long (~1500 tok)",
            "max_new_tokens": 80,
            "turns": [
                {"role": "user", "content": "Explain LRU cache eviction."},
                {"role": "assistant", "content": (
                    "LRU uses a doubly linked list and hash map. Access moves "
                    "node to head, eviction removes from tail. O(1) both."
                )},
                {"role": "user", "content": "How does Raft consensus work?"},
                {"role": "assistant", "content": (
                    "Raft elects a leader which replicates log entries. Majority "
                    "ack commits. Failed leader triggers new election."
                )},
                {"role": "user", "content": "Compare PBFT with Raft."},
                {"role": "assistant", "content": (
                    "Raft: crash failures only. PBFT: Byzantine tolerance with "
                    "3f+1 nodes but O(n^2) messages."
                )},
                {"role": "user", "content": "How do transformers use attention?"},
            ],
        },
        {
            "name": "Extended (~2500 tok)",
            "max_new_tokens": 80,
            "turns": [
                {"role": "user", "content": (
                    "Walk me through virtual memory management from process perspective."
                )},
                {"role": "assistant", "content": (
                    "OS creates virtual address space. Pages mapped lazily. "
                    "Page table translates virtual to physical. TLB caches translations."
                )},
                {"role": "user", "content": "What happens on a page fault?"},
                {"role": "assistant", "content": (
                    "CPU raises exception. OS checks validity, allocates frame, "
                    "reads from disk or zeros, updates page table, resumes."
                )},
                {"role": "user", "content": "How does the kernel decide which page to evict?"},
                {"role": "assistant", "content": (
                    "Linux uses active/inactive lists approximating LRU. "
                    "Under pressure, inactive tail pages reclaimed first."
                )},
                {"role": "user", "content": "How does NUMA affect memory allocation?"},
            ],
        },
    ]

    # Old results (without outlier-aware, from Phase 2.5)
    old_compression = {
        "Short (~150 tok)": 1.24,
        "Medium (~400 tok)": 1.27,
        "Long (~800 tok)": 1.32,
        "Very Long (~1500 tok)": 1.27,
        "Extended (~2500 tok)": 1.32,
    }

    print("=" * 80)
    print("Real Model Compression: Before vs After Outlier Fix")
    print("=" * 80)

    # Load model with outlier-aware enabled
    config = FastKVConfig(warmup_steps=50, use_outlier_aware=True)
    print("\nLoading TinyLlama...")
    try:
        hook = FastKVModelHook("TinyLlama/TinyLlama-1.1B-Chat-v1.0", config)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    def _build_prompt(turns, tokenizer):
        try:
            return tokenizer.apply_chat_template(
                turns, tokenize=False, add_generation_prompt=True)
        except Exception:
            parts = [f"{t['role'].capitalize()}: {t['content']}" for t in turns]
            parts.append("Assistant:")
            return "\n\n".join(parts)

    def _word_overlap(a, b):
        wa, wb = set(a.lower().split()), set(b.lower().split())
        return len(wa & wb) / max(len(wa | wb), 1)

    results = []
    for conv in conversations:
        name = conv["name"]
        prompt = _build_prompt(conv["turns"], hook.tokenizer)
        max_new = conv["max_new_tokens"]
        print(f"\n--- {name} ---")

        # FastKV with outlier-aware
        hook.reset()
        gc.collect()
        try:
            fkv_out = hook.generate(prompt, max_new_tokens=max_new)
            stats = hook.fast_kv_cache.tier_managers[0].get_stats()
            new_cr = stats["compression_ratio"]
            outliers = stats.get("total_outliers_stored", 0)
            avg_outliers = stats.get("avg_outliers_per_token", 0)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        # Baseline
        hook.reset()
        gc.collect()
        try:
            bl_out = hook.generate_baseline(prompt, max_new_tokens=max_new)
        except Exception:
            bl_out = ""

        overlap = _word_overlap(fkv_out, bl_out)
        old_cr = old_compression.get(name, 1.0)
        improvement = (new_cr - old_cr) / old_cr * 100

        results.append({
            "name": name,
            "old_cr": old_cr,
            "new_cr": new_cr,
            "improvement": improvement,
            "overlap": overlap,
            "outliers": outliers,
            "avg_outliers": avg_outliers,
        })

        print(f"  Old CR: {old_cr:.2f}x  New CR: {new_cr:.2f}x  "
              f"Improvement: {improvement:+.1f}%  Quality: {overlap:.1%}  "
              f"Outliers: {outliers}")

    # Print summary table
    if results:
        print("\n" + "=" * 80)
        print(f"{'Conversation':<25} | {'Old CR':>8} | {'New CR':>8} | "
              f"{'Change':>8} | {'Quality':>8} | {'Outliers':>8}")
        print("-" * 80)
        for r in results:
            print(f"{r['name']:<25} | {r['old_cr']:>7.2f}x | "
                  f"{r['new_cr']:>7.2f}x | {r['improvement']:>+7.1f}% | "
                  f"{r['overlap']:>7.1%} | {r['outliers']:>8}")
        print("-" * 80)

        avg_new_cr = np.mean([r["new_cr"] for r in results])
        avg_old_cr = np.mean([r["old_cr"] for r in results])
        avg_quality = np.mean([r["overlap"] for r in results])
        all_above = all(r["overlap"] >= 0.85 for r in results)
        print(f"Average old compression:  {avg_old_cr:.2f}x")
        print(f"Average new compression:  {avg_new_cr:.2f}x")
        print(f"Average quality:          {avg_quality:.1%}")
        print(f"All >= 85% quality:       {'YES' if all_above else 'NO'}")
        print("=" * 80)

    hook.cleanup()


def main() -> None:
    """Run all outlier benchmarks."""
    benchmark_quantization_comparison(n_vectors=10000)
    benchmark_real_model()
    print("\nOutlier benchmark complete.")


if __name__ == "__main__":
    main()
