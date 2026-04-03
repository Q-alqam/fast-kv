"""Real model benchmark for Fast-KV Phase 2 integration.

Loads a real HuggingFace transformer model and runs multi-turn conversations
of increasing length through the FastKVModelHook. Measures RAM consumption,
generation speed, output quality, and tier statistics for both FastKV-compressed
and baseline (uncompressed) generation paths.
"""

import gc
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, ".")

import numpy as np

from fast_kv.config import FastKVConfig


# ---------------------------------------------------------------------------
# Conversation prompts -- realistic multi-turn exchanges
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
                "neutrons, and if the remnant exceeds the Tolman-Oppenheimer-"
                "Volkoff limit (~2-3 solar masses), nothing stops the collapse."
            )},
            {"role": "user", "content": (
                "What about Hawking radiation? Does the black hole actually "
                "lose mass over time, and how long would it take for a "
                "stellar-mass black hole to evaporate completely?"
            )},
            {"role": "assistant", "content": (
                "Yes, Hawking radiation causes black holes to slowly lose mass. "
                "Virtual particle pairs form near the event horizon; one falls "
                "in while the other escapes as real radiation. For a 10-solar-"
                "mass black hole, evaporation takes roughly 10^67 years -- far "
                "longer than the current age of the universe at 1.38 x 10^10 years."
            )},
            {"role": "user", "content": (
                "That is an incredibly long timescale. Does the information "
                "paradox get resolved when the black hole finally evaporates? "
                "What do physicists currently think about unitarity?"
            )},
        ],
    },
    {
        "name": "Long (~800 tok) - Coding Help",
        "target_tokens": 800,
        "max_new_tokens": 100,
        "turns": [
            {"role": "user", "content": (
                "I am building a Python web scraper using aiohttp and "
                "BeautifulSoup. The scraper needs to handle rate limiting, "
                "retry on 429 responses, rotate user agents, and respect "
                "robots.txt. Can you outline the architecture?"
            )},
            {"role": "assistant", "content": (
                "Here is a solid architecture:\n\n"
                "1. `RateLimiter` class using asyncio.Semaphore with a "
                "token-bucket algorithm for per-domain throttling.\n"
                "2. `RetryMiddleware` that catches 429 and 503 responses, "
                "reads the Retry-After header, and uses exponential backoff "
                "with jitter (base=1s, max=60s, factor=2).\n"
                "3. `UserAgentRotator` cycling through a pool of real browser "
                "user-agent strings.\n"
                "4. `RobotsTxtChecker` that fetches and caches robots.txt per "
                "domain using the `robotsparser` stdlib module.\n"
                "5. `Scraper` orchestrator that creates an `aiohttp.ClientSession`, "
                "injects the middleware stack, and dispatches URLs to worker coroutines."
            )},
            {"role": "user", "content": (
                "Great outline. Now show me the RetryMiddleware implementation. "
                "It should be an aiohttp trace config callback and handle both "
                "429 and 503 status codes with exponential backoff. Include "
                "proper logging and type hints."
            )},
            {"role": "assistant", "content": (
                "```python\n"
                "import asyncio\n"
                "import logging\n"
                "from typing import Optional\n"
                "import aiohttp\n\n"
                "logger = logging.getLogger(__name__)\n\n"
                "class RetryMiddleware:\n"
                "    def __init__(self, max_retries: int = 5, base_delay: float = 1.0):\n"
                "        self.max_retries = max_retries\n"
                "        self.base_delay = base_delay\n\n"
                "    async def send_with_retry(self, session, method, url, **kwargs):\n"
                "        for attempt in range(self.max_retries):\n"
                "            resp = await session.request(method, url, **kwargs)\n"
                "            if resp.status not in (429, 503):\n"
                "                return resp\n"
                "            retry_after = resp.headers.get('Retry-After')\n"
                "            delay = float(retry_after) if retry_after else "
                "self.base_delay * (2 ** attempt)\n"
                "            logger.warning('Retry %d/%d for %s (status %d, wait %.1fs)',\n"
                "                           attempt+1, self.max_retries, url, resp.status, delay)\n"
                "            await asyncio.sleep(delay)\n"
                "        return resp\n"
                "```"
            )},
            {"role": "user", "content": (
                "Can you add jitter to the backoff and also implement a "
                "circuit-breaker pattern? If a domain fails more than 10 times "
                "in 60 seconds, we should stop hitting it entirely for 5 minutes. "
                "Also show unit tests using pytest and aioresponses."
            )},
        ],
    },
    {
        "name": "Very Long (~1500 tok) - Mixed Topics",
        "target_tokens": 1500,
        "max_new_tokens": 120,
        "turns": [
            {"role": "user", "content": (
                "I am preparing for a technical interview at a FAANG company. "
                "Can you walk me through how to solve the LRU Cache problem "
                "using a doubly linked list and a hash map? Explain the time "
                "complexity of each operation."
            )},
            {"role": "assistant", "content": (
                "The LRU Cache combines a hash map (dict) for O(1) key lookup "
                "with a doubly linked list for O(1) insertion and removal. "
                "Operations:\n\n"
                "- get(key): Look up in dict -> O(1). If found, move node to "
                "head of list (most recently used) -> O(1). Return value.\n"
                "- put(key, value): If key exists, update value and move to "
                "head -> O(1). If new, create node at head -> O(1). If over "
                "capacity, remove tail node (least recently used) and delete "
                "from dict -> O(1).\n\n"
                "Total: O(1) for both get and put, O(capacity) space."
            )},
            {"role": "user", "content": (
                "Now let us switch topics. I am also studying distributed systems. "
                "Can you explain the Raft consensus algorithm? How does leader "
                "election work, and what happens during a network partition?"
            )},
            {"role": "assistant", "content": (
                "Raft is a consensus algorithm designed for understandability. "
                "It decomposes consensus into three sub-problems:\n\n"
                "1. Leader Election: Nodes start as followers. If a follower "
                "receives no heartbeat within the election timeout (randomized "
                "150-300ms), it becomes a candidate, increments its term, votes "
                "for itself, and sends RequestVote RPCs. A candidate wins if it "
                "gets votes from a majority. Only one leader per term.\n\n"
                "2. Log Replication: The leader receives client commands, appends "
                "them to its log, and replicates via AppendEntries RPCs. An entry "
                "is committed once replicated to a majority.\n\n"
                "3. Safety: Raft guarantees that committed entries are durable "
                "and will be present in the logs of all future leaders.\n\n"
                "During a network partition, the partition with the majority "
                "of nodes continues operating normally. The minority side cannot "
                "elect a leader (no majority) and stops accepting writes."
            )},
            {"role": "user", "content": (
                "Interesting. What about Byzantine fault tolerance? How does "
                "PBFT differ from Raft, and why is it more expensive? Also, "
                "can you compare it with the consensus mechanism used in "
                "Ethereum post-merge?"
            )},
            {"role": "assistant", "content": (
                "PBFT (Practical Byzantine Fault Tolerance) handles arbitrary "
                "node failures, including malicious behavior. It requires 3f+1 "
                "nodes to tolerate f Byzantine faults. The protocol has three "
                "phases: pre-prepare, prepare, and commit, each requiring "
                "quadratic message complexity O(n^2). This makes it impractical "
                "for large networks.\n\n"
                "Raft only tolerates crash faults (not Byzantine), needs 2f+1 "
                "nodes for f failures, and has linear message complexity.\n\n"
                "Ethereum post-merge uses Gasper, which combines Casper FFG "
                "(finality gadget) with LMD-GHOST (fork choice rule). Validators "
                "stake 32 ETH and attest to blocks each epoch (32 slots of 12s). "
                "Finality comes when 2/3 of stake attests to two consecutive "
                "epochs. It tolerates Byzantine faults up to 1/3 of staked ETH."
            )},
            {"role": "user", "content": (
                "One more topic -- can you explain how transformers handle "
                "long-range dependencies better than RNNs? Specifically, why "
                "does self-attention scale as O(n^2) and what are the main "
                "approaches to reduce this to linear complexity?"
            )},
        ],
    },
    {
        "name": "Extended (~2500 tok) - Long Reasoning",
        "target_tokens": 2500,
        "max_new_tokens": 150,
        "turns": [
            {"role": "user", "content": (
                "I want to understand memory management in modern operating "
                "systems from first principles. Start with virtual memory: why "
                "do we need it, and how does page table translation work on "
                "x86-64?"
            )},
            {"role": "assistant", "content": (
                "Virtual memory serves three purposes: memory isolation between "
                "processes, the illusion of contiguous address space, and "
                "overcommitment (using more memory than physically available).\n\n"
                "On x86-64, virtual addresses are 48 bits (with 5-level paging "
                "extending to 57 bits). Translation uses a 4-level page table:\n"
                "- CR3 register holds the physical address of the PML4 table.\n"
                "- Bits 47:39 of the virtual address index into PML4 (512 entries).\n"
                "- Bits 38:30 index into the Page Directory Pointer Table (PDPT).\n"
                "- Bits 29:21 index into the Page Directory (PD).\n"
                "- Bits 20:12 index into the Page Table (PT).\n"
                "- Bits 11:0 are the page offset (4KB pages).\n\n"
                "The TLB (Translation Lookaside Buffer) caches recent translations "
                "to avoid the 4-level walk on every access. TLB misses cost ~100 "
                "cycles vs ~1 cycle for a TLB hit."
            )},
            {"role": "user", "content": (
                "How does the OS handle page faults? Walk me through the full "
                "lifecycle: from the CPU exception to the process resuming "
                "execution. Include both minor and major page faults."
            )},
            {"role": "assistant", "content": (
                "When the MMU cannot resolve a virtual address:\n\n"
                "1. CPU triggers exception #14 (page fault), saves the faulting "
                "address in CR2 and an error code on the stack.\n\n"
                "2. The kernel's page fault handler runs:\n"
                "   a. Reads CR2 to get the faulting address.\n"
                "   b. Checks the VMA (Virtual Memory Area) to determine if "
                "the access is valid.\n"
                "   c. If invalid (no VMA covers the address, or wrong "
                "permissions), sends SIGSEGV to the process.\n\n"
                "3. For valid faults:\n"
                "   - Minor fault: The page exists in the page cache or is "
                "a zero-fill page. The kernel just updates the page table "
                "entry and flushes the TLB. Cost: ~1-10 microseconds.\n"
                "   - Major fault: The page must be read from disk (swap or "
                "file-backed mapping). The kernel issues a disk I/O, the "
                "process sleeps until it completes (~5-10ms for SSD, ~10ms+ "
                "for HDD), then the PTE is updated.\n\n"
                "4. Copy-on-write faults: If the page is shared and the "
                "access is a write, the kernel allocates a new physical page, "
                "copies the content, and maps the new page as writable.\n\n"
                "5. The handler returns, the CPU retries the faulting "
                "instruction, which now succeeds."
            )},
            {"role": "user", "content": (
                "Now explain how the Linux kernel decides which pages to evict "
                "when memory is under pressure. Cover the page reclamation "
                "algorithm, kswapd, and the OOM killer."
            )},
            {"role": "assistant", "content": (
                "Linux uses a modified two-list LRU algorithm:\n\n"
                "Active list: Pages recently accessed (hot pages).\n"
                "Inactive list: Pages not recently accessed (cold candidates).\n\n"
                "Pages enter the inactive list first. If accessed again, they "
                "get promoted to the active list. The kernel balances the lists "
                "so the inactive list is roughly 50-67%% of total reclaimable.\n\n"
                "kswapd is a per-NUMA-node kernel thread that runs when free "
                "memory drops below the low watermark. It scans the inactive "
                "list and:\n"
                "- Drops clean file-backed pages immediately.\n"
                "- Writes dirty pages to disk (writeback) then drops them.\n"
                "- Swaps out anonymous pages to the swap device.\n\n"
                "If kswapd cannot free enough pages and direct reclaim also "
                "fails, the OOM killer activates. It selects a process to kill "
                "based on a badness score: oom_score = (process RSS / total RAM) "
                "* 1000, adjusted by oom_score_adj (-1000 to +1000). The highest-"
                "scored process receives SIGKILL."
            )},
            {"role": "user", "content": (
                "How do huge pages (2MB and 1GB) improve TLB performance, and "
                "what are the trade-offs? Also explain transparent huge pages "
                "vs explicit hugetlbfs allocation. When should I use each "
                "approach in production?"
            )},
            {"role": "assistant", "content": (
                "Huge pages reduce TLB pressure because each entry covers more "
                "memory:\n"
                "- 4KB pages: 512 TLB entries cover 2MB.\n"
                "- 2MB pages: 512 TLB entries cover 1GB.\n"
                "- 1GB pages: 512 TLB entries cover 512GB.\n\n"
                "This dramatically reduces TLB miss rates for workloads with "
                "large working sets (databases, scientific computing, VMs).\n\n"
                "Trade-offs:\n"
                "- Internal fragmentation: a 5KB allocation wastes 2043KB with "
                "2MB pages.\n"
                "- External fragmentation: the kernel needs contiguous 2MB "
                "physical regions, which become scarce over time.\n"
                "- Swap complexity: huge pages must be split before swapping.\n\n"
                "Transparent Huge Pages (THP): The kernel automatically promotes "
                "4KB pages to 2MB pages in the background (khugepaged). Pros: no "
                "application changes. Cons: latency spikes from compaction and "
                "splitting; memory bloat if only part of the huge page is used.\n\n"
                "hugetlbfs: Explicit reservation at boot (hugepages=N) or runtime. "
                "Pros: guaranteed availability, no compaction stalls. Cons: "
                "requires application modification (mmap with MAP_HUGETLB or "
                "mounting hugetlbfs).\n\n"
                "Recommendation: Use hugetlbfs for databases (PostgreSQL, Oracle) "
                "and VMs (QEMU/KVM). Disable THP for latency-sensitive apps "
                "(Redis, Java) that suffer from compaction pauses."
            )},
            {"role": "user", "content": (
                "Finally, tie this all together: in a KV cache system like "
                "Fast-KV, how would you design the memory allocator to minimize "
                "page faults and TLB misses? Consider that hot tier tokens are "
                "accessed frequently while cold tier tokens are accessed rarely. "
                "What OS-level techniques would you use?"
            )},
        ],
    },
]


def _get_process_ram_mb() -> float:
    """Return the current process RSS in megabytes using psutil.

    Returns:
        Process RSS in MB, or 0.0 if psutil is unavailable.
    """
    try:
        import psutil
        proc = psutil.Process()
        return proc.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def _build_prompt(turns: List[Dict[str, str]], tokenizer: Any) -> str:
    """Build a prompt string from multi-turn conversation history.

    Attempts to use the tokenizer's chat template if available, otherwise
    falls back to a simple newline-separated format.

    Args:
        turns: List of dicts with 'role' and 'content' keys.
        tokenizer: A HuggingFace tokenizer instance.

    Returns:
        Formatted prompt string.
    """
    try:
        prompt = tokenizer.apply_chat_template(
            turns, tokenize=False, add_generation_prompt=True
        )
        return prompt
    except Exception:
        # Fallback: simple format
        parts: List[str] = []
        for turn in turns:
            role = turn["role"].capitalize()
            parts.append(f"{role}: {turn['content']}")
        parts.append("Assistant:")
        return "\n\n".join(parts)


def _word_overlap(text_a: str, text_b: str) -> float:
    """Compute word-level Jaccard overlap between two texts.

    Args:
        text_a: First text.
        text_b: Second text.

    Returns:
        Overlap ratio in [0.0, 1.0].
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a and not words_b:
        return 1.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union) if union else 1.0


def _load_model_hook(config: FastKVConfig) -> Any:
    """Attempt to load a FastKVModelHook with TinyLlama, falling back to gpt2.

    Args:
        config: FastKVConfig for the hook.

    Returns:
        A FastKVModelHook instance.

    Raises:
        RuntimeError: If no model could be loaded.
    """
    from fast_kv.model_hook import FastKVModelHook

    models_to_try = [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "gpt2",
    ]

    for model_name in models_to_try:
        try:
            print(f"  Attempting to load {model_name}...")
            hook = FastKVModelHook(model_name, config)
            print(f"  Successfully loaded {model_name}")
            return hook
        except Exception as exc:
            print(f"  Failed to load {model_name}: {exc}")

    raise RuntimeError("Could not load any model. Install transformers and torch.")


def run_conversation_benchmark(
    hook: Any,
    conversation: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Run a single conversation through both FastKV and baseline generation.

    Measures RAM, timing, output quality, and tier statistics.

    Args:
        hook: FastKVModelHook instance.
        conversation: A dict with 'name', 'turns', 'max_new_tokens', etc.

    Returns:
        Dictionary with benchmark results, or None on failure.
    """
    name: str = conversation["name"]
    turns: List[Dict[str, str]] = conversation["turns"]
    max_new_tokens: int = conversation["max_new_tokens"]

    print(f"\n--- {name} ---")

    prompt = _build_prompt(turns, hook.tokenizer)
    prompt_tokens = len(hook.tokenizer.encode(prompt))
    print(f"  Prompt tokens: {prompt_tokens}")

    hook.reset()
    gc.collect()

    # --- RAM before ---
    ram_before = _get_process_ram_mb()

    # --- FastKV generation ---
    t0 = time.perf_counter()
    try:
        fastkv_output = hook.generate(prompt, max_new_tokens=max_new_tokens)
    except Exception as exc:
        print(f"  FastKV generation failed: {exc}")
        return None
    fastkv_time = time.perf_counter() - t0
    ram_after_fastkv = _get_process_ram_mb()

    # --- Tier stats ---
    tier_stats: Optional[Dict] = None
    try:
        cache = hook.fast_kv_cache
        tm0 = cache.tier_managers[0]
        tier_stats = tm0.get_stats()
    except Exception:
        pass

    # --- Baseline generation ---
    hook.reset()
    gc.collect()

    t0 = time.perf_counter()
    try:
        baseline_output = hook.generate_baseline(prompt, max_new_tokens=max_new_tokens)
    except Exception as exc:
        print(f"  Baseline generation failed: {exc}")
        baseline_output = ""
    baseline_time = time.perf_counter() - t0
    ram_after_baseline = _get_process_ram_mb()

    # --- Output quality ---
    overlap = _word_overlap(fastkv_output, baseline_output)

    # --- Memory report ---
    try:
        mem_report = hook.get_memory_report()
    except Exception:
        mem_report = "(unavailable)"

    result = {
        "name": name,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": max_new_tokens,
        "fastkv_output_len": len(fastkv_output.split()),
        "baseline_output_len": len(baseline_output.split()),
        "fastkv_time_s": fastkv_time,
        "baseline_time_s": baseline_time,
        "speedup": baseline_time / fastkv_time if fastkv_time > 0 else 0.0,
        "word_overlap": overlap,
        "ram_before_mb": ram_before,
        "ram_after_fastkv_mb": ram_after_fastkv,
        "ram_after_baseline_mb": ram_after_baseline,
        "ram_delta_fastkv_mb": ram_after_fastkv - ram_before,
        "ram_delta_baseline_mb": ram_after_baseline - ram_before,
        "tier_stats": tier_stats,
        "memory_report": mem_report,
    }

    # Print summary
    print(f"  FastKV time:     {fastkv_time:.2f}s")
    print(f"  Baseline time:   {baseline_time:.2f}s")
    print(f"  Speedup:         {result['speedup']:.2f}x")
    print(f"  Word overlap:    {overlap:.2%}")
    print(f"  RAM delta (FKV): {result['ram_delta_fastkv_mb']:+.1f} MB")
    print(f"  RAM delta (BL):  {result['ram_delta_baseline_mb']:+.1f} MB")
    if tier_stats:
        print(f"  Hot tier:        {tier_stats['n_hot']} tokens "
              f"({tier_stats['hot_fraction']:.1%})")
        print(f"  Cold tier:       {tier_stats['n_cold']} tokens")
        print(f"  Promotions:      {tier_stats['n_promotions_total']}")
        print(f"  Demotions:       {tier_stats['n_demotions_total']}")
        print(f"  Compression:     {tier_stats['compression_ratio']:.2f}x")

    return result


def run_threshold_calibration(
    hook: Any,
    conversation: Dict[str, Any],
    thresholds: List[float],
) -> List[Dict[str, Any]]:
    """Sweep hot_threshold values and measure their effect on tier balance.

    Uses the same conversation for each threshold to provide a controlled
    comparison.

    Args:
        hook: FastKVModelHook instance.
        conversation: The conversation dict to use.
        thresholds: List of hot_threshold values to try.

    Returns:
        List of result dicts, one per threshold.
    """
    results: List[Dict[str, Any]] = []
    prompt = _build_prompt(conversation["turns"], hook.tokenizer)
    max_new_tokens = conversation["max_new_tokens"]

    for thresh in thresholds:
        hook.fast_kv_cache.config.hot_threshold = thresh
        # Adjust cold threshold to maintain a reasonable hysteresis gap
        hook.fast_kv_cache.config.cold_threshold = max(0.10, thresh - 0.30)
        hook.reset()
        gc.collect()

        try:
            t0 = time.perf_counter()
            output = hook.generate(prompt, max_new_tokens=max_new_tokens)
            gen_time = time.perf_counter() - t0
        except Exception as exc:
            print(f"  threshold={thresh:.2f} FAILED: {exc}")
            continue

        try:
            cache = hook.fast_kv_cache
            stats = cache.tier_managers[0].get_stats()
        except Exception:
            stats = {}

        entry = {
            "hot_threshold": thresh,
            "cold_threshold": hook.fast_kv_cache.config.cold_threshold,
            "gen_time_s": gen_time,
            "output_words": len(output.split()),
            "n_hot": stats.get("n_hot", 0),
            "n_cold": stats.get("n_cold", 0),
            "hot_fraction": stats.get("hot_fraction", 0.0),
            "compression_ratio": stats.get("compression_ratio", 0.0),
            "promotions": stats.get("n_promotions_total", 0),
            "demotions": stats.get("n_demotions_total", 0),
        }
        results.append(entry)
        print(f"  thresh={thresh:.2f}  hot={entry['n_hot']:>4}  "
              f"cold={entry['n_cold']:>4}  "
              f"hot%={entry['hot_fraction']:.1%}  "
              f"CR={entry['compression_ratio']:.2f}x  "
              f"time={gen_time:.2f}s")

    return results


def print_comparison_table(results: List[Dict[str, Any]]) -> None:
    """Print a formatted comparison table of all conversation benchmarks.

    Args:
        results: List of result dicts from run_conversation_benchmark.
    """
    header = (
        f"{'Conversation':<36} | {'Prompt':>6} | {'FKV(s)':>7} | "
        f"{'BL(s)':>7} | {'Speed':>6} | {'Overlap':>8} | "
        f"{'RAM FKV':>8} | {'RAM BL':>8} | {'Hot%':>6} | {'CR':>5}"
    )
    sep = "-" * len(header)

    print("\n" + sep)
    print("COMPARISON TABLE")
    print(sep)
    print(header)
    print(sep)

    for r in results:
        ts = r.get("tier_stats") or {}
        hot_frac = ts.get("hot_fraction", 0.0)
        cr = ts.get("compression_ratio", 0.0)
        print(
            f"{r['name']:<36} | "
            f"{r['prompt_tokens']:>6} | "
            f"{r['fastkv_time_s']:>7.2f} | "
            f"{r['baseline_time_s']:>7.2f} | "
            f"{r['speedup']:>5.2f}x | "
            f"{r['word_overlap']:>7.1%} | "
            f"{r['ram_delta_fastkv_mb']:>+7.1f} | "
            f"{r['ram_delta_baseline_mb']:>+7.1f} | "
            f"{hot_frac:>5.1%} | "
            f"{cr:>5.2f}"
        )

    print(sep)


def main() -> None:
    """Run the full real model benchmark suite."""
    print("=" * 80)
    print("Fast-KV Phase 2 -- Real Model Benchmark")
    print("=" * 80)

    config = FastKVConfig()

    # --- Load model ---
    print("\n[1/3] Loading model...")
    try:
        hook = _load_model_hook(config)
    except RuntimeError as exc:
        print(f"FATAL: {exc}")
        sys.exit(1)

    # --- Run conversations ---
    print("\n[2/3] Running conversation benchmarks...")
    results: List[Dict[str, Any]] = []

    for conversation in CONVERSATIONS:
        try:
            result = run_conversation_benchmark(hook, conversation)
            if result is not None:
                results.append(result)
        except Exception as exc:
            print(f"  SKIPPED {conversation['name']}: {exc}")

    if results:
        print_comparison_table(results)
    else:
        print("\nNo conversations completed successfully.")

    # --- Threshold calibration ---
    print("\n[3/3] Threshold calibration sweep...")
    # Use the medium conversation for calibration
    medium_conv = CONVERSATIONS[1]  # Medium (~400 tok) - General Q&A
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]

    print(f"  Using: {medium_conv['name']}")
    print(f"  Sweeping hot_threshold: {thresholds}")

    try:
        cal_results = run_threshold_calibration(hook, medium_conv, thresholds)
        if cal_results:
            print("\n  Calibration Summary:")
            print(f"  {'Threshold':>10} | {'Hot%':>7} | {'CR':>6} | "
                  f"{'Promos':>7} | {'Demos':>7} | {'Time(s)':>8}")
            print("  " + "-" * 62)
            for cr in cal_results:
                print(f"  {cr['hot_threshold']:>10.2f} | "
                      f"{cr['hot_fraction']:>6.1%} | "
                      f"{cr['compression_ratio']:>6.2f} | "
                      f"{cr['promotions']:>7} | "
                      f"{cr['demotions']:>7} | "
                      f"{cr['gen_time_s']:>8.2f}")
    except Exception as exc:
        print(f"  Calibration failed: {exc}")

    # --- Final memory report ---
    print("\n" + "=" * 80)
    print("Final Memory Report")
    print("=" * 80)
    try:
        print(hook.get_memory_report())
    except Exception:
        print("(Memory report unavailable)")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
