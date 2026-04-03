"""Quantized larger model benchmark for Fast-KV.

Loads a larger model (Phi-2, Llama 3, or Mistral) at 4-bit weight
quantization via bitsandbytes (CUDA) or float32 (CPU fallback) and
benchmarks Fast-KV KV cache compression on real inference.

Fast-KV operates on full float32 KV vectors independently of weight
quantization — the two techniques are complementary.
"""

import gc
import os
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, ".")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import psutil

from fast_kv.config import FastKVConfig


# ---------------------------------------------------------------------------
# Conversations — longer prompts for meaningful compression
# ---------------------------------------------------------------------------

CONVERSATIONS: List[Dict[str, Any]] = [
    {
        "name": "Short (Cybersecurity)",
        "max_new_tokens": 80,
        "turns": [
            {"role": "user", "content": (
                "Our SIEM detected CVE-2024-3094 (XZ Utils backdoor) on three "
                "edge gateways: 10.0.12.5, 10.0.12.8, and 10.0.12.11. The IDS "
                "shows outbound connections from all three to 198.51.100.23 on "
                "port 443. Our incident commander wants a containment plan in "
                "the next 15 minutes. What are the immediate steps?"
            )},
            {"role": "assistant", "content": (
                "Immediate containment: 1) Move all three hosts to quarantine "
                "VLAN via switch ACLs. 2) Block 198.51.100.23 at the perimeter "
                "firewall — both ingress and egress. 3) Kill sshd on each host "
                "since the XZ backdoor targets liblzma used by sshd. 4) Capture "
                "forensic images before remediation. 5) Check for lateral movement "
                "by scanning ARP tables and flow logs from those subnets."
            )},
            {"role": "user", "content": (
                "The C2 IP 198.51.100.23 resolves to a Cloudflare-fronted domain. "
                "We also found a persistence mechanism in /etc/cron.d/xz-update "
                "that runs every 5 minutes. How do we handle the Cloudflare issue "
                "and remove the persistence?"
            )},
        ],
    },
    {
        "name": "Medium (General Q&A)",
        "max_new_tokens": 100,
        "turns": [
            {"role": "user", "content": (
                "Explain the difference between nuclear fission and nuclear "
                "fusion. Why is fusion so much harder to achieve on Earth?"
            )},
            {"role": "assistant", "content": (
                "Fission splits heavy atoms (uranium-235, plutonium-239) into "
                "lighter elements, releasing energy from the strong nuclear force. "
                "We can sustain fission chain reactions relatively easily by "
                "bringing enough fissile material together (critical mass). "
                "Fusion combines light atoms (typically hydrogen isotopes "
                "deuterium and tritium) into heavier ones like helium. The energy "
                "released per reaction is much larger, and the fuel is abundant. "
                "But fusion requires overcoming electrostatic repulsion between "
                "positively charged nuclei, which means temperatures above 100 "
                "million degrees Celsius and plasma confinement for long enough "
                "to sustain the reaction."
            )},
            {"role": "user", "content": (
                "What is the Lawson criterion and how close are we to achieving "
                "it? I heard about the NIF breakthrough — was that real ignition?"
            )},
            {"role": "assistant", "content": (
                "The Lawson criterion defines the minimum product of plasma "
                "density, confinement time, and temperature needed for a fusion "
                "reactor to produce more energy than it consumes. For deuterium-"
                "tritium fusion, this is approximately n*tau*T > 3e21 keV*s/m^3. "
                "The NIF experiment in December 2022 achieved 3.15 MJ of fusion "
                "energy from 2.05 MJ of laser energy delivered to the target — "
                "so yes, target gain > 1 was achieved. However, the lasers "
                "themselves consumed ~300 MJ of electricity, so total system "
                "gain was far below 1."
            )},
            {"role": "user", "content": (
                "What approach do you think is most promising for commercial "
                "fusion: tokamaks (ITER), inertial confinement (NIF), or the "
                "newer compact approaches from private companies?"
            )},
        ],
    },
    {
        "name": "Long (Coding Help)",
        "max_new_tokens": 120,
        "turns": [
            {"role": "user", "content": (
                "I need to implement a distributed rate limiter for a "
                "microservice architecture. We have 12 FastAPI instances "
                "behind an Nginx load balancer, all sharing a Redis cluster. "
                "Requirements: 1000 req/min per API key globally across all "
                "instances, with burst allowance of 50 requests in 1 second. "
                "Show me the complete implementation."
            )},
            {"role": "assistant", "content": (
                "Use a sliding window log with Redis sorted sets for the "
                "per-minute limit, combined with a token bucket for burst "
                "control. The sorted set stores timestamps as both member "
                "and score. On each request: ZREMRANGEBYSCORE to remove "
                "entries older than 60 seconds, ZCARD to count remaining, "
                "reject if >= 1000. For burst: use a separate key with "
                "INCR and EXPIRE of 1 second, reject if > 50."
            )},
            {"role": "user", "content": (
                "How do I make this atomic? Two instances might read the "
                "count simultaneously and both allow the request, exceeding "
                "the limit."
            )},
            {"role": "assistant", "content": (
                "Use a Lua script executed via EVAL to make the "
                "check-and-increment atomic. Redis executes Lua scripts "
                "atomically — no other command can run between the ZCARD "
                "and ZADD. The script: remove old entries, count, if below "
                "limit then add new entry and return 1 (allowed), else "
                "return 0 (rejected). All in one round trip."
            )},
            {"role": "user", "content": (
                "What about monitoring and graceful degradation? If Redis "
                "goes down, I don't want to hard-block all traffic."
            )},
            {"role": "assistant", "content": (
                "Implement a circuit breaker pattern: if Redis is unreachable "
                "for more than 3 consecutive attempts, fall back to a local "
                "in-memory rate limiter per instance. This degrades to "
                "per-instance limits (1000/12 = ~83 req/min per instance) "
                "but keeps traffic flowing. Log an alert when the circuit "
                "breaker trips. Reset after Redis becomes reachable again."
            )},
            {"role": "user", "content": (
                "One more: how do I handle rate limit headers in the response? "
                "I want X-RateLimit-Limit, X-RateLimit-Remaining, "
                "X-RateLimit-Reset, and Retry-After for 429 responses."
            )},
        ],
    },
    {
        "name": "Very Long (Mixed Topics)",
        "max_new_tokens": 100,
        "turns": [
            {"role": "user", "content": "Explain how TCP congestion control works."},
            {"role": "assistant", "content": (
                "TCP congestion control uses algorithms to prevent network "
                "overload. The sender maintains a congestion window (cwnd) "
                "that limits unacknowledged data in flight. In slow start, "
                "cwnd doubles each RTT until hitting ssthresh. Then switches "
                "to congestion avoidance: cwnd grows by ~1 MSS per RTT. On "
                "packet loss (detected by triple duplicate ACKs), cwnd is "
                "halved (fast recovery). On timeout, cwnd resets to 1 MSS."
            )},
            {"role": "user", "content": "How does QUIC improve on TCP?"},
            {"role": "assistant", "content": (
                "QUIC eliminates head-of-line blocking by multiplexing "
                "streams independently. Connection establishment is 0-RTT "
                "for repeat connections. It runs over UDP so there is no "
                "OS-level HOL blocking. Migration is built in via connection "
                "IDs rather than IP:port tuples. TLS 1.3 is integrated into "
                "the handshake, not layered on top."
            )},
            {"role": "user", "content": "Compare B-trees and LSM-trees for databases."},
            {"role": "assistant", "content": (
                "B-trees optimize for reads: data is stored in sorted pages, "
                "reads are O(log n) with good cache behavior. Writes require "
                "random I/O to update pages in place. LSM-trees optimize for "
                "writes: all writes go to an in-memory buffer, then flushed "
                "as sorted runs. Reads may need to check multiple levels. "
                "Compaction merges levels to maintain read performance."
            )},
            {"role": "user", "content": (
                "How does Raft handle network partitions? What happens to "
                "writes during a partition?"
            )},
            {"role": "assistant", "content": (
                "During a network partition, the partition containing the "
                "majority of nodes can elect a leader and continue processing "
                "writes normally. The minority partition cannot form a quorum "
                "and will reject all writes. When the partition heals, the "
                "minority nodes receive the log entries they missed and "
                "converge to the majority state."
            )},
            {"role": "user", "content": (
                "Tie these together: how would you design a globally "
                "distributed key-value store using these concepts?"
            )},
        ],
    },
    {
        "name": "Extended (Long Reasoning)",
        "max_new_tokens": 100,
        "turns": [
            {"role": "user", "content": (
                "Walk me through the complete lifecycle of an HTTP request "
                "from browser to server and back."
            )},
            {"role": "assistant", "content": (
                "1) Browser resolves domain via DNS (recursive resolver, root, "
                "TLD, authoritative). 2) TCP three-way handshake with server "
                "(SYN, SYN-ACK, ACK). 3) TLS 1.3 handshake (ClientHello with "
                "key share, ServerHello, encrypted extensions, certificate, "
                "finished). 4) HTTP/2 request sent as HEADERS frame with "
                "HPACK-compressed headers. 5) Server processes request through "
                "reverse proxy, load balancer, application server. 6) Application "
                "queries database, caches, external services. 7) Response sent "
                "as HEADERS + DATA frames. 8) Browser renders HTML, discovers "
                "CSS/JS resources, makes additional requests."
            )},
            {"role": "user", "content": "How does the kernel handle the TCP connection?"},
            {"role": "assistant", "content": (
                "The kernel maintains a socket buffer for each connection. "
                "Incoming packets are processed by the NIC driver, passed "
                "through the network stack (IP layer checks routing, TCP "
                "layer manages sequence numbers, acknowledgments, windowing). "
                "Data is copied to the socket receive buffer. The application "
                "reads via recv() system call. For sending, data goes to the "
                "send buffer, segmented into MSS-sized chunks, wrapped in "
                "TCP/IP headers, and passed to the NIC for transmission."
            )},
            {"role": "user", "content": (
                "What about zero-copy techniques? How do sendfile and "
                "io_uring reduce overhead?"
            )},
            {"role": "assistant", "content": (
                "sendfile() eliminates the user-space copy: data goes directly "
                "from the page cache to the socket buffer via DMA. With "
                "hardware that supports scatter-gather DMA, even the kernel "
                "buffer copy is eliminated. io_uring goes further with "
                "submission and completion queues shared between user and "
                "kernel space via mmap. Batched syscalls reduce context "
                "switch overhead. Registered buffers avoid repeated address "
                "translation."
            )},
            {"role": "user", "content": (
                "How does TLS 1.3 reduce latency compared to TLS 1.2? "
                "Explain the 0-RTT resumption mechanism and its security "
                "implications."
            )},
            {"role": "assistant", "content": (
                "TLS 1.3 reduces the handshake from 2-RTT to 1-RTT by "
                "sending the key share in the ClientHello. The server "
                "responds with its key share and encrypted data in a single "
                "round trip. 0-RTT resumption uses a pre-shared key from "
                "a previous session to encrypt early data in the very first "
                "ClientHello. The security risk is replay attacks: an attacker "
                "can replay the 0-RTT data. Servers must implement replay "
                "protection (e.g., single-use tickets, time windows)."
            )},
            {"role": "user", "content": (
                "Given all of this, how would you optimize a high-throughput "
                "API serving 100K requests per second on a single machine?"
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


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_single(
    hook: Any, conv: Dict, config: FastKVConfig
) -> Optional[Dict]:
    """Run one conversation with FastKV and baseline."""
    name = conv["name"]
    max_new = conv["max_new_tokens"]
    print(f"\n--- {name} ---")

    prompt = _build_prompt(conv["turns"], hook.tokenizer)
    prompt_tokens = len(hook.tokenizer.encode(prompt))
    print(f"  Prompt: {prompt_tokens} tokens, generating {max_new} more")

    # FastKV generation
    hook.reset()
    gc.collect()
    ram_before = _get_ram_mb()

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

    # Baseline generation
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

    result = {
        "name": name,
        "prompt_tokens": prompt_tokens,
        "total_tokens": prompt_tokens + max_new,
        "overlap": overlap,
        "fkv_time": fkv_time,
        "bl_time": bl_time,
        "ram_fkv_delta": ram_fkv - ram_before,
        "ram_bl_delta": ram_bl - ram_before,
        "hot_pct": stats["hot_fraction"] if stats else 0,
        "cold_pct": 1 - stats["hot_fraction"] if stats else 0,
        "compression": stats["compression_ratio"] if stats else 1,
        "promotions": stats.get("n_promotions_total", 0) if stats else 0,
        "demotions": stats.get("n_demotions_total", 0) if stats else 0,
        "outliers": stats.get("total_outliers_stored", 0) if stats else 0,
    }

    print(f"  Quality: {overlap:.1%}  CR: {result['compression']:.2f}x  "
          f"Hot: {result['hot_pct']:.1%}  FKV: {fkv_time:.1f}s  BL: {bl_time:.1f}s")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the quantized model benchmark."""
    from fast_kv.quantized_model_hook import (
        QuantizedFastKVModelHook,
        select_best_model,
    )

    cpu_name = platform.processor() or platform.machine()
    total_ram = psutil.virtual_memory().total / (1024 ** 3)
    avail_ram = psutil.virtual_memory().available / (1024 ** 3)

    print("=" * 75)
    print("Fast-KV Quantized Model Benchmark")
    print("=" * 75)
    print(f"Hardware: {cpu_name}, {total_ram:.0f} GB RAM ({avail_ram:.1f} GB available)")

    # Select and load model
    print("\n[1/3] Selecting and loading model...")
    # On CPU-only hardware, Phi-2 (2.7B) is too slow for full benchmarks.
    # Use TinyLlama for the benchmark suite; Phi-2 verified compatible separately.
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        model_name = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else select_best_model(avail_ram)
    elif not __import__("torch").cuda.is_available():
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    else:
        model_name = select_best_model(avail_ram)
    print(f"  Selected: {model_name}")

    config = FastKVConfig(warmup_steps=60, hot_threshold=0.65, cold_threshold=0.35)

    try:
        hook = QuantizedFastKVModelHook(model_name, config, load_in_4bit=True)
    except Exception as e:
        print(f"  Failed with {model_name}: {e}")
        # Fallback
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        print(f"  Falling back to {model_name}")
        hook = QuantizedFastKVModelHook(model_name, config, load_in_4bit=True)

    print(f"  Loaded: {repr(hook)}")

    # Run conversations
    print("\n[2/3] Running conversation benchmarks...")
    results: List[Dict] = []
    for conv in CONVERSATIONS:
        try:
            r = run_single(hook, conv, config)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  SKIPPED {conv['name']}: {e}")

    # Print results
    if results:
        print("\n[3/3] Results")
        print("=" * 95)
        print(f"Fast-KV Quantized Model Benchmark — {model_name}")
        print(f"Loading method: {hook.loading_method}")
        print(f"Hardware: {cpu_name}, {total_ram:.0f} GB RAM")
        print("=" * 95)
        print(f"{'Conversation':<28} | {'Tokens':>6} | {'CR':>6} | "
              f"{'Quality':>8} | {'Hot%':>6} | {'FKV(s)':>7} | {'BL(s)':>7}")
        print("-" * 95)
        for r in results:
            print(f"{r['name']:<28} | {r['total_tokens']:>6} | "
                  f"{r['compression']:>5.2f}x | {r['overlap']:>7.1%} | "
                  f"{r['hot_pct']:>5.1%} | {r['fkv_time']:>7.1f} | "
                  f"{r['bl_time']:>7.1f}")
        print("-" * 95)

        avg_cr = np.mean([r["compression"] for r in results])
        avg_quality = np.mean([r["overlap"] for r in results])
        all_quality = all(r["overlap"] >= 0.85 for r in results)
        all_compress = all(r["compression"] >= 1.20 for r in results[1:])  # skip short (warmup)

        print(f"Average compression:     {avg_cr:.2f}x")
        print(f"Average quality:         {avg_quality:.1%}")
        print(f"All >= 85% quality:      {'YES' if all_quality else 'NO'}")
        print(f"Compression >= 1.20x:    {'YES' if all_compress else 'NO'} (excluding short/warmup)")
        print("=" * 95)

        # Memory report
        print("\n" + hook.get_memory_report())

    hook.cleanup()
    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
