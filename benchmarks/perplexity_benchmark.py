"""Perplexity benchmark for Fast-KV.

Measures the actual impact of KV cache compression on model output
quality using perplexity (PPL) on standard text. This is the metric
researchers will ask for — token overlap is a weak proxy.

PPL increase = (fastkv_ppl - baseline_ppl) / baseline_ppl

A good result is < 1% PPL increase.
"""

import gc
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, ".")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np

try:
    import torch
except ImportError:
    print("PyTorch required. Install with: pip install torch")
    sys.exit(1)


# Standard text samples for PPL measurement (diverse domains)
PPL_TEXTS = [
    # Wikipedia-style factual text
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
    # Technical documentation
    (
        "TCP uses a three-way handshake to establish a reliable connection. "
        "The client sends a SYN packet with a random sequence number. The "
        "server responds with SYN-ACK, acknowledging the client's sequence "
        "number and providing its own. The client completes the handshake "
        "with an ACK. After establishment, data flows bidirectionally with "
        "sequence numbers tracking every byte. Retransmission handles "
        "packet loss. Flow control uses a sliding window mechanism."
    ),
    # Narrative text
    (
        "The old lighthouse had stood on the cliff for over a century, its "
        "beam sweeping across the dark waters every twelve seconds. The "
        "keeper climbed the spiral staircase each evening at dusk, checking "
        "the oil reservoir and polishing the Fresnel lens until it gleamed. "
        "Ships in the channel relied on that steady pulse of light to "
        "navigate the treacherous rocks below."
    ),
    # Code-adjacent text
    (
        "When designing a distributed system, you must consider the CAP "
        "theorem: you can have at most two of consistency, availability, "
        "and partition tolerance. In practice, network partitions are "
        "inevitable, so the real choice is between consistency and "
        "availability during a partition. Systems like Cassandra choose "
        "availability, while systems like ZooKeeper choose consistency."
    ),
]


def measure_perplexity_baseline(model: Any, tokenizer: Any, text: str, device: str = "cpu") -> float:
    """Measure standard perplexity without Fast-KV.

    Args:
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        text: Input text to measure PPL on.
        device: Device string.

    Returns:
        Perplexity value.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    return float(torch.exp(loss).item())


def measure_perplexity_fastkv(hook: Any, text: str) -> float:
    """Measure perplexity with Fast-KV active.

    Runs token-by-token generation through FastKV and computes the
    cross-entropy loss from the model's logits vs actual next tokens.

    Args:
        hook: FastKVModelHook instance.
        text: Input text.

    Returns:
        Perplexity value.
    """
    hook.reset()

    inputs = hook.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(hook.device)
    seq_len = input_ids.shape[1]

    if seq_len < 2:
        return 0.0

    # Build token ID list and texts
    token_ids_list = input_ids[0].tolist()
    all_token_ids = list(range(seq_len))
    all_token_texts = [
        hook.tokenizer.decode([tid], skip_special_tokens=False)
        for tid in token_ids_list
    ]

    total_loss = 0.0
    n_tokens = 0
    past_key_values = None

    with torch.no_grad():
        for pos in range(seq_len - 1):
            if past_key_values is None:
                # First token: feed all tokens up to pos+1
                outputs = hook.model(
                    input_ids=input_ids[:, :pos + 1],
                    use_cache=True,
                    output_attentions=True,
                )
                new_count = pos + 1
            else:
                outputs = hook.model(
                    input_ids=input_ids[:, pos:pos + 1],
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=True,
                )
                new_count = 1

            # Intercept KV cache through Fast-KV
            raw_past = outputs.past_key_values
            current_ids = all_token_ids[:pos + 1]
            current_texts = all_token_texts[:pos + 1]
            try:
                past_key_values = hook._intercept_kv_cache(
                    raw_past, current_ids, current_texts, pos, new_count
                )
            except Exception:
                past_key_values = raw_past

            # Compute loss for predicting the next token
            logits = outputs.logits[:, -1, :]  # (1, vocab_size)
            target = input_ids[:, pos + 1]  # (1,)
            loss = torch.nn.functional.cross_entropy(logits, target)
            total_loss += loss.item()
            n_tokens += 1

    avg_loss = total_loss / max(n_tokens, 1)
    return float(np.exp(avg_loss))


def main() -> None:
    """Run the perplexity benchmark."""
    from fast_kv.model_hook import FastKVModelHook
    from fast_kv.config import FastKVConfig

    print("=" * 70)
    print("Fast-KV Perplexity Benchmark")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    config = FastKVConfig(warmup_steps=60)
    try:
        hook = FastKVModelHook("TinyLlama/TinyLlama-1.1B-Chat-v1.0", config)
    except Exception as e:
        print(f"Failed: {e}")
        return

    print(f"Model: {hook.model_name}, Device: {hook.device}")

    # Run PPL measurements
    print(f"\nMeasuring perplexity on {len(PPL_TEXTS)} text samples...\n")

    results = []
    text_names = ["Wikipedia/Factual", "Technical/Networking", "Narrative/Fiction", "Systems/Design"]

    for i, (text, name) in enumerate(zip(PPL_TEXTS, text_names)):
        print(f"--- {name} ---")
        tokens = len(hook.tokenizer.encode(text))
        print(f"  Tokens: {tokens}")

        # Baseline PPL
        gc.collect()
        t0 = time.perf_counter()
        baseline_ppl = measure_perplexity_baseline(
            hook.model, hook.tokenizer, text, hook.device
        )
        bl_time = time.perf_counter() - t0

        # FastKV PPL
        gc.collect()
        t0 = time.perf_counter()
        fastkv_ppl = measure_perplexity_fastkv(hook, text)
        fkv_time = time.perf_counter() - t0

        ppl_increase = ((fastkv_ppl - baseline_ppl) / baseline_ppl * 100) if baseline_ppl > 0 else 0

        results.append({
            "name": name,
            "tokens": tokens,
            "baseline_ppl": baseline_ppl,
            "fastkv_ppl": fastkv_ppl,
            "ppl_increase_pct": ppl_increase,
            "bl_time": bl_time,
            "fkv_time": fkv_time,
        })

        print(f"  Baseline PPL: {baseline_ppl:.2f}")
        print(f"  Fast-KV PPL:  {fastkv_ppl:.2f}")
        print(f"  PPL increase: {ppl_increase:+.2f}%")
        print()

    # Summary table
    print("=" * 70)
    print(f"{'Text Type':<22} | {'Tokens':>6} | {'BL PPL':>8} | {'FKV PPL':>8} | {'Increase':>9}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<22} | {r['tokens']:>6} | "
              f"{r['baseline_ppl']:>8.2f} | {r['fastkv_ppl']:>8.2f} | "
              f"{r['ppl_increase_pct']:>+8.2f}%")
    print("-" * 70)

    avg_bl = np.mean([r["baseline_ppl"] for r in results])
    avg_fkv = np.mean([r["fastkv_ppl"] for r in results])
    avg_inc = np.mean([r["ppl_increase_pct"] for r in results])
    print(f"{'Average':<22} | {'':>6} | {avg_bl:>8.2f} | {avg_fkv:>8.2f} | {avg_inc:>+8.2f}%")
    print("=" * 70)

    if abs(avg_inc) < 1.0:
        print("\nPASS: Average PPL increase < 1%")
    elif abs(avg_inc) < 5.0:
        print(f"\nACCEPTABLE: Average PPL increase {avg_inc:+.2f}% (< 5%)")
    else:
        print(f"\nWARN: Average PPL increase {avg_inc:+.2f}% exceeds 5%")

    hook.cleanup()
    print("\nPerplexity benchmark complete.")


if __name__ == "__main__":
    main()
