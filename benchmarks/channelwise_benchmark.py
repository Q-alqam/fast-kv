"""Channel-wise quantization benchmark.

Validates that channel-wise quantization fixes the PPL degradation
found in v0.5.1 by preserving per-dimension statistical structure.
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
    compare_quantization_methods,
    dequantize_vector,
    quantize_vector,
    quantize_vector_channelwise,
)


def generate_llm_like_vector(dim: int = 1024) -> np.ndarray:
    """Generate a vector with LLM-like outlier patterns."""
    vec = np.random.randn(dim).astype(np.float32) * 0.3
    n_outliers = max(1, int(dim * 0.02))
    idx = np.random.choice(dim, n_outliers, replace=False)
    vec[idx] = np.random.randn(n_outliers).astype(np.float32) * 50.0
    return vec


def section1_quantization_comparison(n_vectors: int = 10000) -> None:
    """Compare quantization methods on LLM-like vectors."""
    np.random.seed(42)
    vectors = [generate_llm_like_vector() for _ in range(n_vectors)]

    print("=" * 85)
    print(f"Quantization Method Comparison ({n_vectors} LLM-like vectors, dim=1024)")
    print("=" * 85)
    print()

    for bits in [4, 2]:
        print(f"--- {bits}-bit ---")
        agg = {}
        for v in vectors:
            r = compare_quantization_methods(v, bits)
            for method, metrics in r.items():
                if method not in agg:
                    agg[method] = {"mae": [], "cosine": [], "time": []}
                agg[method]["mae"].append(metrics["mae"])
                agg[method]["cosine"].append(metrics["cosine_sim"])
                agg[method]["time"].append(metrics["time"])

        print(f"{'Method':<20} | {'MAE':>10} | {'Cosine Sim':>11} | {'Time (ms)':>10}")
        print("-" * 60)
        for method in ["scalar", "channelwise_128", "channelwise_64", "channelwise_32"]:
            if method in agg:
                mae = np.mean(agg[method]["mae"])
                cos = np.mean(agg[method]["cosine"])
                t = np.sum(agg[method]["time"]) * 1000
                print(f"{method:<20} | {mae:>10.5f} | {cos:>11.8f} | {t:>10.1f}")
        print()


def section2_perplexity_comparison() -> None:
    """Compare PPL: baseline vs scalar vs channel-wise."""
    try:
        import torch
        from fast_kv.model_hook import FastKVModelHook
        from fast_kv.config import FastKVConfig
        from benchmarks.perplexity_benchmark import (
            measure_perplexity_baseline,
            measure_perplexity_fastkv,
        )
    except ImportError as e:
        print(f"Skipping PPL benchmark: {e}")
        return

    texts = [
        (
            "The Turing machine is a mathematical model of computation that "
            "defines an abstract machine which manipulates symbols on a strip "
            "of tape according to a table of rules. Despite the model's "
            "simplicity, given any computer algorithm, a Turing machine capable "
            "of implementing that algorithm's logic can be constructed. The "
            "machine operates on an infinite memory tape divided into discrete "
            "cells, each of which can hold a single symbol drawn from a finite "
            "set of symbols called the alphabet of the machine."
        ),
        (
            "TCP uses a three-way handshake to establish a reliable connection. "
            "The client sends a SYN packet with a random sequence number. The "
            "server responds with SYN-ACK, acknowledging the client's sequence "
            "number and providing its own. The client completes the handshake "
            "with an ACK. After establishment, data flows bidirectionally with "
            "sequence numbers tracking every byte. Retransmission handles "
            "packet loss. Flow control uses a sliding window mechanism."
        ),
        (
            "When designing a distributed system, you must consider the CAP "
            "theorem: you can have at most two of consistency, availability, "
            "and partition tolerance. In practice, network partitions are "
            "inevitable, so the real choice is between consistency and "
            "availability during a partition. Systems like Cassandra choose "
            "availability, while systems like ZooKeeper choose consistency."
        ),
    ]

    print("=" * 75)
    print("Perplexity Comparison — The Core Fix")
    print("=" * 75)
    print()

    configs = [
        ("Baseline (no Fast-KV)", None),
        ("Scalar quantization", FastKVConfig(
            warmup_steps=60, compression_method="scalar")),
        ("Channel-wise g=64", FastKVConfig(
            warmup_steps=60, compression_method="channelwise",
            channelwise_group_size=64)),
        ("Channel-wise g=32", FastKVConfig(
            warmup_steps=60, compression_method="channelwise",
            channelwise_group_size=32)),
    ]

    results = {}
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Load model once
    print(f"Loading {model_name}...")
    base_hook = FastKVModelHook(model_name, FastKVConfig(warmup_steps=60))

    for config_name, config in configs:
        ppls = []
        for text in texts:
            gc.collect()
            if config is None:
                # Baseline
                ppl = measure_perplexity_baseline(
                    base_hook.model, base_hook.tokenizer, text, base_hook.device
                )
            else:
                # Fast-KV with specific config
                hook = FastKVModelHook(model_name, config)
                hook.model = base_hook.model  # share model weights
                hook.tokenizer = base_hook.tokenizer
                ppl = measure_perplexity_fastkv(hook, text)
            ppls.append(ppl)
        results[config_name] = np.mean(ppls)
        print(f"  {config_name}: PPL = {results[config_name]:.2f}")

    # Table
    baseline_ppl = results.get("Baseline (no Fast-KV)", 1.0)
    print()
    print(f"{'Configuration':<28} | {'PPL':>8} | {'vs Baseline':>12}")
    print("-" * 55)
    for name, ppl in results.items():
        if name == "Baseline (no Fast-KV)":
            print(f"{name:<28} | {ppl:>8.2f} | {'---':>12}")
        else:
            inc = (ppl - baseline_ppl) / baseline_ppl * 100
            print(f"{name:<28} | {ppl:>8.2f} | {inc:>+11.1f}%")
    print("=" * 55)

    base_hook.cleanup()


def section3_compression_overhead() -> None:
    """Measure metadata overhead of channel-wise vs scalar."""
    print("=" * 70)
    print("Compression Overhead: Channel-Wise vs Scalar")
    print("=" * 70)
    print()

    dim = 512  # TinyLlama kv_dim * 2
    n_layers = 22
    n_tokens = 1000

    print(f"Model: {n_layers} layers, kv_dim*2={dim}, {n_tokens} tokens")
    print()

    for bits in [4, 2]:
        # Scalar: quantized values + 1 scale + 1 zero_point
        scalar_per_token = dim * bits / 8 + 8  # 8 bytes for scale+zp
        # Channel-wise: quantized values + n_groups * (scale + zp)
        for gs in [128, 64, 32]:
            n_groups = (dim + gs - 1) // gs
            cw_per_token = dim * bits / 8 + n_groups * 8  # 8 bytes per group
            overhead = (cw_per_token - scalar_per_token) / scalar_per_token * 100

            print(f"  {bits}-bit g={gs:>3d}: {cw_per_token:.0f} bytes/token "
                  f"(scalar: {scalar_per_token:.0f}) overhead: +{overhead:.1f}%")
    print()


def main() -> None:
    """Run all channel-wise benchmarks."""
    section1_quantization_comparison(n_vectors=10000)
    section3_compression_overhead()
    section2_perplexity_comparison()
    print("\nChannel-wise benchmark complete.")


if __name__ == "__main__":
    main()
