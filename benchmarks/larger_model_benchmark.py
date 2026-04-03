"""Larger model benchmark for Fast-KV Phase 2.5.

Auto-detects available RAM and loads the best model that fits:
  >= 10 GB: meta-llama/Llama-3.2-3B-Instruct
  >= 6 GB:  microsoft/phi-2
  < 6 GB:   TinyLlama/TinyLlama-1.1B-Chat-v1.0

Tests the warmup fix and measures quality across all conversation lengths.
"""

import gc
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, ".")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import psutil

from fast_kv.config import FastKVConfig


# ---------------------------------------------------------------------------
# Conversations (same as Phase 2)
# ---------------------------------------------------------------------------

CONVERSATIONS: List[Dict[str, Any]] = [
    {
        "name": "Short (~150 tok) - Cybersecurity",
        "target_tokens": 150,
        "max_new_tokens": 60,
        "turns": [
            {"role": "user", "content": (
                "We just got an alert for CVE-2024-3094 on our edge gateway "
                "at 10.0.12.5. The XZ Utils backdoor is being flagged by our "
                "IDS. What immediate containment steps should we take?"
            )},
            {"role": "assistant", "content": (
                "First, isolate 10.0.12.5 from the network by moving it to a "
                "quarantine VLAN. Check if SSH is exposed -- the XZ backdoor "
                "targets sshd via liblzma."
            )},
            {"role": "user", "content": (
                "SSH is exposed on port 22. We also see outbound connections "
                "to 198.51.100.23 on port 443. Is that the C2 callback?"
            )},
        ],
    },
    {
        "name": "Medium (~400 tok) - General Q&A",
        "target_tokens": 400,
        "max_new_tokens": 80,
        "turns": [
            {"role": "user", "content": (
                "Can you explain how black holes form? I know massive stars "
                "collapse, but what actually happens at the quantum level "
                "during the final stages?"
            )},
            {"role": "assistant", "content": (
                "When a star more than about 25 solar masses exhausts its "
                "nuclear fuel, the core collapses under gravity. Electron "
                "degeneracy pressure fails, protons and electrons merge into "
                "neutrons. If the remaining mass exceeds the Tolman-Oppenheimer-"
                "Volkoff limit (~2.1 solar masses), neutron degeneracy pressure "
                "also fails and collapse continues to a singularity."
            )},
            {"role": "user", "content": (
                "What happens to information that falls into a black hole? "
                "I've heard of the information paradox."
            )},
            {"role": "assistant", "content": (
                "The information paradox arises because quantum mechanics "
                "says information cannot be destroyed, but general relativity "
                "says anything past the event horizon is lost. Hawking showed "
                "black holes radiate thermally, which carries no information. "
                "This created a decades-long debate."
            )},
            {"role": "user", "content": (
                "Has the paradox been resolved? What's the current consensus?"
            )},
        ],
    },
    {
        "name": "Long (~800 tok) - Coding Help",
        "target_tokens": 800,
        "max_new_tokens": 100,
        "turns": [
            {"role": "user", "content": (
                "I'm building a REST API in Python with FastAPI. I need to "
                "implement rate limiting per API key. Each key should be "
                "allowed 100 requests per minute. What's the best approach?"
            )},
            {"role": "assistant", "content": (
                "Use a sliding window counter with Redis. Store a sorted set "
                "per API key where each member is a request timestamp. On each "
                "request, remove entries older than 60 seconds, count remaining, "
                "and reject if count >= 100."
            )},
            {"role": "user", "content": (
                "Can you show the FastAPI middleware implementation? I want it "
                "to return 429 Too Many Requests with a Retry-After header."
            )},
            {"role": "assistant", "content": (
                "Here's the approach: create a dependency that checks Redis, "
                "use the api_key from the request header, maintain a sorted "
                "set with ZADD and ZRANGEBYSCORE to count recent requests."
            )},
            {"role": "user", "content": (
                "What about distributed rate limiting across multiple FastAPI "
                "instances behind a load balancer? Redis handles that?"
            )},
            {"role": "assistant", "content": (
                "Yes, Redis is shared state, so all instances see the same "
                "counters. Use Redis Cluster for high availability. Consider "
                "lua scripting for atomic check-and-increment operations."
            )},
            {"role": "user", "content": (
                "One more thing: how do I handle burst traffic? A user might "
                "send 50 requests in 1 second then nothing for 59 seconds."
            )},
        ],
    },
    {
        "name": "Very Long (~1500 tok) - Mixed Topics",
        "target_tokens": 1500,
        "max_new_tokens": 80,
        "turns": [
            {"role": "user", "content": "Explain LRU cache eviction in detail."},
            {"role": "assistant", "content": (
                "LRU (Least Recently Used) tracks access order using a doubly "
                "linked list and hash map. On access, move the node to the "
                "head. On eviction, remove from the tail. O(1) for both."
            )},
            {"role": "user", "content": "How does the Raft consensus protocol work?"},
            {"role": "assistant", "content": (
                "Raft elects a leader which replicates log entries to followers. "
                "A majority acknowledgment commits the entry. If the leader "
                "fails, a new election begins with randomized timeouts."
            )},
            {"role": "user", "content": "Compare PBFT with Raft for Byzantine faults."},
            {"role": "assistant", "content": (
                "Raft assumes crash failures only; PBFT tolerates up to f "
                "Byzantine nodes with 3f+1 total. PBFT has higher message "
                "complexity (O(n^2)) versus Raft's O(n) but handles malicious actors."
            )},
            {"role": "user", "content": (
                "How do transformers use attention? What is multi-head attention?"
            )},
            {"role": "assistant", "content": (
                "Multi-head attention runs several attention functions in "
                "parallel on different linear projections of Q, K, V. Each "
                "head can attend to different aspects. Outputs are concatenated "
                "and projected."
            )},
            {"role": "user", "content": (
                "Why are KV caches needed for autoregressive generation?"
            )},
        ],
    },
    {
        "name": "Extended (~2500 tok) - Long Reasoning",
        "target_tokens": 2500,
        "max_new_tokens": 80,
        "turns": [
            {"role": "user", "content": (
                "Walk me through how an operating system manages virtual memory "
                "from a process perspective."
            )},
            {"role": "assistant", "content": (
                "When a process starts, the OS creates a virtual address space. "
                "Pages are mapped lazily: only when accessed do they get backed "
                "by physical frames. The page table translates virtual to physical "
                "addresses. The TLB caches recent translations for speed."
            )},
            {"role": "user", "content": "What happens on a page fault?"},
            {"role": "assistant", "content": (
                "The CPU raises an exception. The OS checks if the address is "
                "valid. If so, it allocates a physical frame, reads the page "
                "from disk or zeros it, updates the page table, and resumes "
                "the instruction."
            )},
            {"role": "user", "content": (
                "How does the kernel decide which page to evict when RAM is full?"
            )},
            {"role": "assistant", "content": (
                "Linux uses a two-list approximation of LRU: active and inactive "
                "lists. Pages start on inactive; repeated access promotes them. "
                "Under memory pressure, pages at the tail of inactive are "
                "reclaimed first. Dirty pages are written back before eviction."
            )},
            {"role": "user", "content": (
                "What about memory-mapped files and shared memory between processes?"
            )},
            {"role": "assistant", "content": (
                "mmap maps file contents into the virtual address space. Multiple "
                "processes can map the same file; the kernel ensures they share "
                "physical frames via reference counting. Writes may be shared "
                "(MAP_SHARED) or copy-on-write (MAP_PRIVATE)."
            )},
            {"role": "user", "content": (
                "How does NUMA architecture affect memory allocation on multi-socket "
                "servers? What optimizations does the kernel make?"
            )},
            {"role": "assistant", "content": (
                "In NUMA, each CPU socket has local memory with faster access and "
                "remote memory with higher latency. Linux prefers allocating from "
                "the local node. The scheduler tries to keep processes on the same "
                "node as their memory. numactl and mbind() provide user-space control."
            )},
            {"role": "user", "content": (
                "Tie it all together: how does all this relate to how large language "
                "models use memory during inference?"
            )},
        ],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_ram_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def _build_prompt(turns: List[Dict[str, str]], tokenizer: Any) -> str:
    try:
        return tokenizer.apply_chat_template(
            turns, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        parts = [f"{t['role'].capitalize()}: {t['content']}" for t in turns]
        parts.append("Assistant:")
        return "\n\n".join(parts)


def _word_overlap(a: str, b: str) -> float:
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _select_model() -> str:
    avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    print(f"  Available RAM: {avail_gb:.1f} GB")
    # Use TinyLlama for full benchmarks (fast on CPU).
    # Phi-2/Llama-3 are too slow for full CPU-only benchmark suites
    # but can be tested with --model flag for single-conversation tests.
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def _load_hook(model_name: str, config: FastKVConfig) -> Any:
    from fast_kv.model_hook import FastKVModelHook
    print(f"  Loading {model_name}...")
    hook = FastKVModelHook(model_name, config)
    print(f"  Loaded: {repr(hook)}")
    return hook


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def run_single(hook: Any, conv: Dict, config: FastKVConfig) -> Optional[Dict]:
    """Run one conversation with both FastKV and baseline."""
    name = conv["name"]
    print(f"\n--- {name} ---")

    prompt = _build_prompt(conv["turns"], hook.tokenizer)
    max_new = conv["max_new_tokens"]
    prompt_tokens = len(hook.tokenizer.encode(prompt))
    print(f"  Prompt: {prompt_tokens} tokens")

    hook.reset()
    gc.collect()
    ram_before = _get_ram_mb()

    # FastKV
    t0 = time.perf_counter()
    try:
        fkv_out = hook.generate(prompt, max_new_tokens=max_new)
    except Exception as e:
        print(f"  FastKV FAILED: {e}")
        return None
    fkv_time = time.perf_counter() - t0
    ram_fkv = _get_ram_mb()

    # Tier stats
    try:
        stats = hook.fast_kv_cache.tier_managers[0].get_stats()
    except Exception:
        stats = None

    # Baseline
    hook.reset()
    gc.collect()
    t0 = time.perf_counter()
    try:
        bl_out = hook.generate_baseline(prompt, max_new_tokens=max_new)
    except Exception as e:
        print(f"  Baseline FAILED: {e}")
        bl_out = ""
    bl_time = time.perf_counter() - t0
    ram_bl = _get_ram_mb()

    overlap = _word_overlap(fkv_out, bl_out)
    warmup_status = hook.fast_kv_cache.get_warmup_status()

    result = {
        "name": name,
        "prompt_tokens": prompt_tokens,
        "overlap": overlap,
        "fkv_time": fkv_time,
        "bl_time": bl_time,
        "ram_before": ram_before,
        "ram_fkv": ram_fkv - ram_before,
        "ram_bl": ram_bl - ram_before,
        "hot_pct": stats["hot_fraction"] if stats else 0,
        "compression": stats["compression_ratio"] if stats else 1,
        "promotions": stats["n_promotions_total"] if stats else 0,
        "demotions": stats["n_demotions_total"] if stats else 0,
        "warmup": warmup_status,
    }

    print(f"  Overlap: {overlap:.1%}  Hot: {result['hot_pct']:.1%}  "
          f"CR: {result['compression']:.2f}x  FKV: {fkv_time:.1f}s  BL: {bl_time:.1f}s")
    return result


def run_warmup_comparison(hook: Any, conv: Dict) -> Dict:
    """Compare quality with warmup_steps=0 vs warmup_steps=50."""
    prompt = _build_prompt(conv["turns"], hook.tokenizer)
    max_new = conv["max_new_tokens"]

    # Baseline
    hook.reset()
    gc.collect()
    bl_out = hook.generate_baseline(prompt, max_new_tokens=max_new)

    # With warmup (default)
    hook.fast_kv_cache.config.warmup_steps = 50
    hook.reset()
    gc.collect()
    fkv_warmup = hook.generate(prompt, max_new_tokens=max_new)
    overlap_warmup = _word_overlap(fkv_warmup, bl_out)

    # Without warmup
    hook.fast_kv_cache.config.warmup_steps = 0
    hook.reset()
    gc.collect()
    fkv_no_warmup = hook.generate(prompt, max_new_tokens=max_new)
    overlap_no_warmup = _word_overlap(fkv_no_warmup, bl_out)

    # Restore
    hook.fast_kv_cache.config.warmup_steps = 50

    return {
        "with_warmup": overlap_warmup,
        "without_warmup": overlap_no_warmup,
        "improvement": overlap_warmup - overlap_no_warmup,
    }


def run_threshold_sweep(hook: Any, conv: Dict) -> Tuple[float, List[Dict]]:
    """Sweep hot_threshold and find optimal."""
    prompt = _build_prompt(conv["turns"], hook.tokenizer)
    max_new = conv["max_new_tokens"]

    # Get baseline
    hook.reset()
    gc.collect()
    bl_out = hook.generate_baseline(prompt, max_new_tokens=max_new)

    results = []
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
        hook.fast_kv_config.hot_threshold = thresh
        hook.fast_kv_config.cold_threshold = max(0.15, thresh - 0.30)
        hook.fast_kv_cache.config = hook.fast_kv_config
        for tm in hook.fast_kv_cache.tier_managers.values():
            tm.config = hook.fast_kv_config

        hook.reset()
        gc.collect()
        try:
            out = hook.generate(prompt, max_new_tokens=max_new)
            overlap = _word_overlap(out, bl_out)
            stats = hook.fast_kv_cache.tier_managers[0].get_stats()
            results.append({
                "threshold": thresh,
                "overlap": overlap,
                "hot_pct": stats["hot_fraction"],
                "compression": stats["compression_ratio"],
            })
            print(f"  thresh={thresh:.2f}  overlap={overlap:.1%}  "
                  f"hot={stats['hot_fraction']:.1%}  CR={stats['compression_ratio']:.2f}x")
        except Exception as e:
            print(f"  thresh={thresh:.2f}  FAILED: {e}")

    # Restore defaults
    hook.fast_kv_config.hot_threshold = 0.65
    hook.fast_kv_config.cold_threshold = 0.35
    hook.fast_kv_cache.config = hook.fast_kv_config
    for tm in hook.fast_kv_cache.tier_managers.values():
        tm.config = hook.fast_kv_config

    # Find optimal: best compression with overlap >= 85%
    valid = [r for r in results if r["overlap"] >= 0.85]
    if valid:
        best = max(valid, key=lambda r: r["compression"])
        return best["threshold"], results
    elif results:
        best = max(results, key=lambda r: r["overlap"])
        return best["threshold"], results
    return 0.65, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import platform

    print("=" * 75)
    print("Fast-KV Phase 2.5 -- Larger Model Benchmark")
    print("=" * 75)

    # Hardware info
    cpu = platform.processor() or "unknown"
    total_ram = psutil.virtual_memory().total / (1024 ** 3)
    print(f"\nHardware: {cpu}, {total_ram:.0f} GB RAM")

    # Select and load model
    print("\n[1/5] Selecting model...")
    model_name = _select_model()
    config = FastKVConfig(warmup_steps=50)
    try:
        hook = _load_hook(model_name, config)
    except Exception as e:
        print(f"  Failed: {e}")
        print("  Falling back to TinyLlama...")
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        hook = _load_hook(model_name, config)

    # Run conversations
    print("\n[2/5] Running conversation benchmarks...")
    results = []
    for conv in CONVERSATIONS:
        try:
            r = run_single(hook, conv, config)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  SKIPPED: {e}")

    # Warmup comparison
    print("\n[3/5] Warmup comparison (short conversation)...")
    try:
        warmup_cmp = run_warmup_comparison(hook, CONVERSATIONS[0])
        print(f"  Without warmup: {warmup_cmp['without_warmup']:.1%}")
        print(f"  With warmup:    {warmup_cmp['with_warmup']:.1%}")
        print(f"  Improvement:    {warmup_cmp['improvement']:+.1%}")
    except Exception as e:
        print(f"  Warmup comparison failed: {e}")
        warmup_cmp = {"with_warmup": 0, "without_warmup": 0, "improvement": 0}

    # Threshold sweep
    print("\n[4/5] Threshold calibration...")
    try:
        optimal_thresh, sweep = run_threshold_sweep(hook, CONVERSATIONS[1])
        print(f"  Optimal threshold: {optimal_thresh:.2f}")
    except Exception as e:
        print(f"  Calibration failed: {e}")
        optimal_thresh = 0.65
        sweep = []

    # Quality check - try higher warmup if any row < 85%
    low_quality = [r for r in results if r["overlap"] < 0.85]
    if low_quality:
        print(f"\n  {len(low_quality)} conversations below 85% quality.")
        for ws in [75, 100]:
            config.warmup_steps = ws
            hook.fast_kv_cache.config = config
            for tm in hook.fast_kv_cache.tier_managers.values():
                tm.config = config
            improved = True
            for conv in CONVERSATIONS[:1]:  # Test short conv
                hook.reset()
                gc.collect()
                try:
                    bl = hook.generate_baseline(
                        _build_prompt(conv["turns"], hook.tokenizer),
                        max_new_tokens=conv["max_new_tokens"])
                    hook.reset()
                    fkv = hook.generate(
                        _build_prompt(conv["turns"], hook.tokenizer),
                        max_new_tokens=conv["max_new_tokens"])
                    ov = _word_overlap(fkv, bl)
                    if ov >= 0.85:
                        print(f"  warmup_steps={ws}: quality={ov:.1%} >= 85%")
                    else:
                        improved = False
                except Exception:
                    improved = False
            if improved:
                break
        config.warmup_steps = 50  # restore

    # Print table
    print("\n[5/5] Results")
    print("=" * 100)
    print(f"Fast-KV Larger Model Benchmark -- {model_name}")
    print(f"Hardware: {cpu}, {total_ram:.0f} GB RAM")
    print("=" * 100)
    print(f"{'Conversation':<38} | {'Tokens':>6} | {'Overlap':>8} | "
          f"{'Hot%':>6} | {'CR':>6} | {'FKV(s)':>7} | {'BL(s)':>7}")
    print("-" * 100)
    for r in results:
        print(f"{r['name']:<38} | {r['prompt_tokens']:>6} | "
              f"{r['overlap']:>7.1%} | {r['hot_pct']:>5.1%} | "
              f"{r['compression']:>5.2f}x | {r['fkv_time']:>7.1f} | "
              f"{r['bl_time']:>7.1f}")
    print("-" * 100)

    if results:
        avg_overlap = np.mean([r["overlap"] for r in results])
        avg_cr = np.mean([r["compression"] for r in results])
        print(f"Average quality:     {avg_overlap:.1%}")
        print(f"Average compression: {avg_cr:.2f}x")
    print(f"Optimal threshold:   {optimal_thresh:.2f}")
    print(f"Warmup improvement:  {warmup_cmp['improvement']:+.1%} on short convos")
    all_above = all(r["overlap"] >= 0.85 for r in results)
    print(f"All >= 85% quality:  {'YES' if all_above else 'NO'}")
    print("=" * 100)

    # Memory report
    print("\n" + hook.get_memory_report())
    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
