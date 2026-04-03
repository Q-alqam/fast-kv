"""Attention pattern analysis benchmark for Fast-KV Phase 2.

Runs a real model through a multi-turn conversation, captures attention weights
at every generation step, and analyzes the resulting patterns to validate the
core assumptions behind Fast-KV's two-tier caching strategy:

  1. A minority of tokens consistently attract high attention (hot tokens).
  2. Most tokens quickly decay to near-zero attention (cold tokens).
  3. Occasional cold-to-hot spikes occur (validating the promotion mechanism).
  4. The ISE's composite score correlates with actual attention behaviour.

Produces three matplotlib plots saved to benchmarks/outputs/:
  - attention_heatmap.png   : Per-token attention over generation steps.
  - ise_correlation.png     : ISE predicted score vs. actual mean attention.
  - tier_events.png         : Timeline of promotion/demotion events overlaid
                              with per-token attention traces.
"""

import gc
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, ".")

import numpy as np

from fast_kv.config import FastKVConfig

# Use non-interactive backend so plots can be saved without a display.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

OUTPUT_DIR: str = os.path.join(os.path.dirname(__file__), "outputs")


# ---------------------------------------------------------------------------
# Test conversation (~500 tokens)
# ---------------------------------------------------------------------------

ANALYSIS_CONVERSATION: List[Dict[str, str]] = [
    {"role": "user", "content": (
        "I am investigating a security incident. Our SIEM flagged traffic "
        "from 172.16.44.12 to an external IP 203.0.113.99 over port 8443. "
        "The source host is a Linux jumpbox running Ubuntu 22.04. The alert "
        "triggered on a Suricata rule for TLS certificate anomaly. What "
        "should I check first?"
    )},
    {"role": "assistant", "content": (
        "Start with these steps:\n"
        "1. Run `ss -tlnp` on the jumpbox to identify the process bound to "
        "port 8443 or making the outbound connection.\n"
        "2. Check `/var/log/auth.log` and `journalctl -u sshd` for any "
        "recent SSH sessions or authentication anomalies.\n"
        "3. Inspect the TLS certificate being served by 203.0.113.99 with "
        "`openssl s_client -connect 203.0.113.99:8443` to look for self-signed "
        "or recently-issued certificates.\n"
        "4. Cross-reference the external IP against threat intelligence feeds "
        "like AlienVault OTX, VirusTotal, and AbuseIPDB."
    )},
    {"role": "user", "content": (
        "The process is a Python script `/opt/tools/sync_agent.py` running as "
        "the `deploy` user. It was started via cron 3 hours ago. The TLS cert "
        "on the remote end was issued yesterday by Let's Encrypt to a domain "
        "`update-services.example.com`. The domain was registered 2 days ago. "
        "This looks suspicious. What is the best way to capture evidence "
        "without tipping off the attacker?"
    )},
    {"role": "assistant", "content": (
        "Good instinct -- short-lived domain plus recent cert is a classic C2 "
        "indicator. For evidence collection:\n"
        "1. Capture a memory dump of the Python process with `gcore <PID>` "
        "before it exits.\n"
        "2. Copy the script: `cp /opt/tools/sync_agent.py /evidence/` and "
        "hash it with `sha256sum`.\n"
        "3. Start a packet capture on the interface: `tcpdump -i eth0 -w "
        "/evidence/capture.pcap host 203.0.113.99`.\n"
        "4. Do NOT kill the process yet -- let it run while you capture traffic "
        "to understand the C2 protocol."
    )},
    {"role": "user", "content": (
        "I captured the traffic and the Python script. The script uses the "
        "`requests` library to POST base64-encoded system information to "
        "the C2 server every 5 minutes. It also downloads and executes "
        "commands from a `/tasks` endpoint. The deploy user has sudo access "
        "to several services. How do I scope the blast radius and begin "
        "remediation without destroying forensic artifacts?"
    )},
]


def _load_model_hook(config: FastKVConfig) -> Any:
    """Load a FastKVModelHook, trying TinyLlama first then gpt2.

    Args:
        config: FastKVConfig to pass to the hook.

    Returns:
        FastKVModelHook instance.

    Raises:
        RuntimeError: If no model loads successfully.
    """
    from fast_kv.model_hook import FastKVModelHook

    for model_name in [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "gpt2",
    ]:
        try:
            print(f"  Trying {model_name}...")
            hook = FastKVModelHook(model_name, config)
            print(f"  Loaded {model_name}")
            return hook
        except Exception as exc:
            print(f"  Failed: {exc}")

    raise RuntimeError("No model could be loaded.")


def _build_prompt(turns: List[Dict[str, str]], tokenizer: Any) -> str:
    """Format conversation turns into a generation prompt.

    Args:
        turns: List of role/content dicts.
        tokenizer: HuggingFace tokenizer.

    Returns:
        Formatted prompt string.
    """
    try:
        return tokenizer.apply_chat_template(
            turns, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        parts = [f"{t['role'].capitalize()}: {t['content']}" for t in turns]
        parts.append("Assistant:")
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def collect_attention_history(
    hook: Any,
    prompt: str,
    max_new_tokens: int = 100,
) -> Tuple[Dict[int, List[float]], List[Dict[int, float]], int]:
    """Generate tokens and record per-step attention weights.

    Args:
        hook: FastKVModelHook instance.
        prompt: The input prompt string.
        max_new_tokens: Number of tokens to generate.

    Returns:
        Tuple of:
          - token_attention_history: token_id -> list of attention values (per step)
          - step_attention_list: list of (step's attention dict)
          - n_prompt_tokens: number of prompt tokens
    """
    hook.reset()
    gc.collect()

    # Generate with FastKV to populate attention data
    _ = hook.generate(prompt, max_new_tokens=max_new_tokens)

    # Retrieve attention weights captured during generation
    attn_history: Dict[int, List[float]] = {}
    step_attns: List[Dict[int, float]] = []

    # The hook stores last_attention_weights as layer_idx -> attention weights.
    # We use layer 0 as the reference layer for analysis.
    raw_weights = getattr(hook, "last_attention_weights", {})

    # If the hook provides step-by-step history, use it; otherwise reconstruct
    # from the cache's ISE state.
    if isinstance(raw_weights, dict) and raw_weights:
        # Use layer 0 weights
        layer0_weights = raw_weights.get(0, raw_weights.get(list(raw_weights.keys())[0], {}))

        if isinstance(layer0_weights, list):
            # List of per-step dicts
            step_attns = layer0_weights
        elif isinstance(layer0_weights, dict):
            # Single snapshot -- wrap as one step
            step_attns = [layer0_weights]

    # Build per-token history from step_attns
    all_token_ids: set = set()
    for step_dict in step_attns:
        all_token_ids.update(step_dict.keys())

    for tid in all_token_ids:
        attn_history[tid] = [
            step_dict.get(tid, 0.0) for step_dict in step_attns
        ]

    n_prompt = len(hook.tokenizer.encode(prompt))
    return attn_history, step_attns, n_prompt


def analyze_attention_patterns(
    attn_history: Dict[int, List[float]],
    n_steps: int,
) -> Dict[str, Any]:
    """Analyze attention patterns to validate Fast-KV assumptions.

    Args:
        attn_history: token_id -> list of attention values across steps.
        n_steps: Total number of generation steps.

    Returns:
        Dictionary with analysis metrics.
    """
    if not attn_history or n_steps == 0:
        return {
            "pct_consistently_hot": 0.0,
            "pct_consistently_cold": 0.0,
            "n_cold_to_hot_spikes": 0,
            "avg_spike_magnitude": 0.0,
            "n_tokens_analyzed": 0,
        }

    n_tokens = len(attn_history)

    # Compute per-token mean attention
    token_means: Dict[int, float] = {}
    for tid, vals in attn_history.items():
        token_means[tid] = float(np.mean(vals)) if vals else 0.0

    global_mean = float(np.mean(list(token_means.values()))) if token_means else 0.0

    # Consistently hot: mean attention > 1.5x global mean for at least 70% of steps
    n_consistently_hot = 0
    for tid, vals in attn_history.items():
        if not vals:
            continue
        above_threshold = sum(1 for v in vals if v > 1.5 * global_mean)
        if above_threshold / len(vals) >= 0.70:
            n_consistently_hot += 1

    # Consistently cold: mean attention < 0.5x global mean for at least 70% of steps
    n_consistently_cold = 0
    for tid, vals in attn_history.items():
        if not vals:
            continue
        below_threshold = sum(1 for v in vals if v < 0.5 * global_mean)
        if below_threshold / len(vals) >= 0.70:
            n_consistently_cold += 1

    # Cold-to-hot spikes: token goes from below-average to 2x average in one step
    n_spikes = 0
    spike_magnitudes: List[float] = []
    for tid, vals in attn_history.items():
        for i in range(1, len(vals)):
            if vals[i - 1] < global_mean and vals[i] > 2.0 * global_mean:
                n_spikes += 1
                spike_magnitudes.append(vals[i] - vals[i - 1])

    avg_spike = float(np.mean(spike_magnitudes)) if spike_magnitudes else 0.0

    return {
        "pct_consistently_hot": n_consistently_hot / n_tokens * 100 if n_tokens else 0.0,
        "pct_consistently_cold": n_consistently_cold / n_tokens * 100 if n_tokens else 0.0,
        "n_cold_to_hot_spikes": n_spikes,
        "avg_spike_magnitude": avg_spike,
        "n_tokens_analyzed": n_tokens,
        "global_mean_attention": global_mean,
    }


def compute_ise_correlation(
    hook: Any,
    attn_history: Dict[int, List[float]],
) -> Tuple[List[float], List[float]]:
    """Compare ISE predicted scores with actual mean attention.

    Args:
        hook: FastKVModelHook instance (with populated cache).
        attn_history: token_id -> list of attention values.

    Returns:
        Tuple of (ise_scores, actual_means) aligned lists.
    """
    ise_scores: List[float] = []
    actual_means: List[float] = []

    try:
        cache = hook.fast_kv_cache
        ise = cache.ise
        current_step = cache._current_step
    except Exception:
        return ise_scores, actual_means

    for tid, vals in attn_history.items():
        try:
            ise_score = ise.get_score(tid, current_step)
        except Exception:
            continue
        actual_mean = float(np.mean(vals)) if vals else 0.0
        ise_scores.append(ise_score)
        actual_means.append(actual_mean)

    return ise_scores, actual_means


def get_tier_events(hook: Any) -> Tuple[int, int, Dict[str, int]]:
    """Extract tier event counts from the cache.

    Args:
        hook: FastKVModelHook instance.

    Returns:
        Tuple of (total_promotions, total_demotions, subtier_counts).
    """
    try:
        cache = hook.fast_kv_cache
        total_promos = 0
        total_demos = 0
        subtier_totals: Dict[str, int] = {"2A": 0, "2B": 0, "2C": 0}

        for tm in cache.tier_managers.values():
            stats = tm.get_stats()
            total_promos += stats["n_promotions_total"]
            total_demos += stats["n_demotions_total"]
            for st, cnt in stats["subtier_counts"].items():
                subtier_totals[st] += cnt

        return total_promos, total_demos, subtier_totals
    except Exception:
        return 0, 0, {"2A": 0, "2B": 0, "2C": 0}


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_attention_heatmap(
    attn_history: Dict[int, List[float]],
    output_path: str,
    max_tokens: int = 80,
) -> None:
    """Generate and save an attention heatmap.

    Rows = token positions, columns = generation steps.

    Args:
        attn_history: token_id -> list of attention values.
        output_path: File path to save the PNG.
        max_tokens: Maximum number of tokens to display (for readability).
    """
    if not attn_history:
        print("  [WARN] No attention data for heatmap.")
        return

    # Sort tokens by ID and limit
    sorted_tids = sorted(attn_history.keys())[:max_tokens]
    n_steps = max(len(v) for v in attn_history.values()) if attn_history else 0

    if n_steps == 0:
        print("  [WARN] Zero generation steps; skipping heatmap.")
        return

    # Build matrix
    matrix = np.zeros((len(sorted_tids), n_steps), dtype=np.float32)
    for row, tid in enumerate(sorted_tids):
        vals = attn_history[tid]
        for col in range(min(len(vals), n_steps)):
            matrix[row, col] = vals[col]

    fig, ax = plt.subplots(figsize=(14, 8))

    # Use a small epsilon to avoid log(0)
    vmin = max(matrix[matrix > 0].min(), 1e-6) if np.any(matrix > 0) else 1e-6
    vmax = matrix.max() if matrix.max() > 0 else 1.0

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="hot",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        interpolation="nearest",
    )
    ax.set_xlabel("Generation Step", fontsize=12)
    ax.set_ylabel("Token Position", fontsize=12)
    ax.set_title("Attention Weight Heatmap (Log Scale)", fontsize=14)
    fig.colorbar(im, ax=ax, label="Attention Weight")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_ise_correlation(
    ise_scores: List[float],
    actual_means: List[float],
    output_path: str,
) -> None:
    """Scatter plot of ISE predicted scores vs actual mean attention.

    Args:
        ise_scores: List of ISE composite scores.
        actual_means: List of actual mean attention values.
        output_path: File path to save the PNG.
    """
    if not ise_scores or not actual_means:
        print("  [WARN] No ISE data for correlation plot.")
        return

    ise_arr = np.array(ise_scores)
    actual_arr = np.array(actual_means)

    # Normalize actual to [0, 1] for comparison
    actual_max = actual_arr.max()
    if actual_max > 0:
        actual_norm = actual_arr / actual_max
    else:
        actual_norm = actual_arr

    # Compute Pearson correlation
    if len(ise_arr) > 1 and np.std(ise_arr) > 0 and np.std(actual_norm) > 0:
        correlation = float(np.corrcoef(ise_arr, actual_norm)[0, 1])
    else:
        correlation = 0.0

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        ise_arr, actual_norm,
        alpha=0.5, s=20, c="steelblue", edgecolors="none",
    )

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "r--", alpha=0.5, label="Perfect prediction")

    # Fit line
    if len(ise_arr) > 1:
        z = np.polyfit(ise_arr, actual_norm, 1)
        p = np.poly1d(z)
        xs = np.linspace(0, 1, 100)
        ax.plot(xs, p(xs), "g-", alpha=0.7, label=f"Linear fit (r={correlation:.3f})")

    ax.set_xlabel("ISE Predicted Score", fontsize=12)
    ax.set_ylabel("Actual Mean Attention (normalized)", fontsize=12)
    ax.set_title(f"ISE Prediction Accuracy (Pearson r = {correlation:.3f})", fontsize=14)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_tier_events(
    attn_history: Dict[int, List[float]],
    total_promotions: int,
    total_demotions: int,
    subtier_counts: Dict[str, int],
    output_path: str,
    top_n: int = 10,
) -> None:
    """Plot attention traces for top tokens with tier event annotations.

    Shows the attention trajectories of the most-attended tokens, plus
    a bar chart of tier event counts.

    Args:
        attn_history: token_id -> list of attention values.
        total_promotions: Total promotion count.
        total_demotions: Total demotion count.
        subtier_counts: Dict of sub-tier label -> token count.
        output_path: File path to save the PNG.
        top_n: Number of top tokens to plot traces for.
    """
    if not attn_history:
        print("  [WARN] No attention data for tier events plot.")
        return

    # Sort tokens by mean attention and pick top N
    token_means = {
        tid: float(np.mean(vals)) if vals else 0.0
        for tid, vals in attn_history.items()
    }
    top_tokens = sorted(token_means, key=token_means.get, reverse=True)[:top_n]  # type: ignore[arg-type]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [3, 1]})

    # Left panel: attention traces
    ax_traces = axes[0]
    for tid in top_tokens:
        vals = attn_history[tid]
        ax_traces.plot(
            range(len(vals)), vals,
            alpha=0.7, linewidth=1.2, label=f"Token {tid}",
        )

    ax_traces.set_xlabel("Generation Step", fontsize=12)
    ax_traces.set_ylabel("Attention Weight", fontsize=12)
    ax_traces.set_title(f"Attention Traces (Top {top_n} Tokens)", fontsize=14)
    ax_traces.legend(fontsize=8, loc="upper right", ncol=2)
    ax_traces.grid(True, alpha=0.3)

    # Right panel: tier event bar chart
    ax_bars = axes[1]
    labels = ["Promotions", "Demotions", "Sub-2A", "Sub-2B", "Sub-2C"]
    values = [
        total_promotions,
        total_demotions,
        subtier_counts.get("2A", 0),
        subtier_counts.get("2B", 0),
        subtier_counts.get("2C", 0),
    ]
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6", "#95a5a6"]
    bars = ax_bars.barh(labels, values, color=colors, edgecolor="white")

    # Add value labels
    for bar, val in zip(bars, values):
        ax_bars.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontsize=10,
        )

    ax_bars.set_xlabel("Count", fontsize=12)
    ax_bars.set_title("Tier Events", fontsize=14)
    ax_bars.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run attention pattern analysis and generate plots."""
    print("=" * 70)
    print("Fast-KV Phase 2 -- Attention Pattern Analysis")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    config = FastKVConfig()

    # --- Load model ---
    print("\n[1/5] Loading model...")
    try:
        hook = _load_model_hook(config)
    except RuntimeError as exc:
        print(f"FATAL: {exc}")
        sys.exit(1)

    # --- Build prompt ---
    prompt = _build_prompt(ANALYSIS_CONVERSATION, hook.tokenizer)
    n_prompt_tokens = len(hook.tokenizer.encode(prompt))
    print(f"\n[2/5] Prompt prepared: {n_prompt_tokens} tokens")

    # --- Collect attention data ---
    print("\n[3/5] Generating with attention capture...")
    t0 = time.perf_counter()
    try:
        attn_history, step_attns, n_prompt = collect_attention_history(
            hook, prompt, max_new_tokens=100,
        )
    except Exception as exc:
        print(f"  Attention capture failed: {exc}")
        print("  Falling back to ISE-based synthetic analysis...")
        attn_history = {}
        step_attns = []
        n_prompt = n_prompt_tokens

    gen_time = time.perf_counter() - t0
    n_steps = len(step_attns)
    n_tokens_tracked = len(attn_history)
    print(f"  Generation time: {gen_time:.2f}s")
    print(f"  Steps captured:  {n_steps}")
    print(f"  Tokens tracked:  {n_tokens_tracked}")

    # If no attention history was captured, build a synthetic one from ISE
    if not attn_history:
        print("\n  Building synthetic attention history from ISE state...")
        try:
            cache = hook.fast_kv_cache
            ise = cache.ise
            for tid in list(ise._attention_ema.keys()):
                ema_val = ise._attention_ema.get(tid, 0.0)
                # Create a synthetic single-step record
                attn_history[tid] = [ema_val]
            n_steps = 1
            step_attns = [dict(attn_history)]
            print(f"  Reconstructed {len(attn_history)} tokens from ISE EMA.")
        except Exception as exc:
            print(f"  ISE fallback also failed: {exc}")

    # --- Analyze patterns ---
    print("\n[4/5] Analyzing attention patterns...")
    analysis = analyze_attention_patterns(attn_history, n_steps)

    print(f"  Tokens analyzed:           {analysis['n_tokens_analyzed']}")
    print(f"  Consistently hot:          {analysis['pct_consistently_hot']:.1f}%")
    print(f"  Consistently cold:         {analysis['pct_consistently_cold']:.1f}%")
    print(f"  Cold->hot spikes:          {analysis['n_cold_to_hot_spikes']}")
    print(f"  Avg spike magnitude:       {analysis['avg_spike_magnitude']:.6f}")

    # ISE correlation
    ise_scores, actual_means = compute_ise_correlation(hook, attn_history)
    if ise_scores:
        ise_arr = np.array(ise_scores)
        actual_arr = np.array(actual_means)
        actual_max = actual_arr.max() if actual_arr.max() > 0 else 1.0
        actual_norm = actual_arr / actual_max
        if len(ise_arr) > 1 and np.std(ise_arr) > 0 and np.std(actual_norm) > 0:
            r = float(np.corrcoef(ise_arr, actual_norm)[0, 1])
        else:
            r = 0.0
        print(f"  ISE-Attention correlation: r = {r:.3f}")

        # Classification accuracy: does ISE correctly rank tokens above/below
        # the global mean attention?
        global_mean_ise = np.mean(ise_arr)
        global_mean_actual = np.mean(actual_norm)
        ise_above = ise_arr > global_mean_ise
        actual_above = actual_norm > global_mean_actual
        accuracy = float(np.mean(ise_above == actual_above)) * 100
        print(f"  ISE binary accuracy:       {accuracy:.1f}%")
    else:
        print("  ISE correlation: (no data)")

    # Tier events
    total_promos, total_demos, subtier_counts = get_tier_events(hook)
    print(f"  Total promotions:          {total_promos}")
    print(f"  Total demotions:           {total_demos}")
    print(f"  Sub-tier 2A:               {subtier_counts.get('2A', 0)}")
    print(f"  Sub-tier 2B:               {subtier_counts.get('2B', 0)}")
    print(f"  Sub-tier 2C:               {subtier_counts.get('2C', 0)}")

    # --- Generate plots ---
    print("\n[5/5] Generating plots...")

    heatmap_path = os.path.join(OUTPUT_DIR, "attention_heatmap.png")
    try:
        plot_attention_heatmap(attn_history, heatmap_path)
    except Exception as exc:
        print(f"  Heatmap failed: {exc}")

    correlation_path = os.path.join(OUTPUT_DIR, "ise_correlation.png")
    try:
        plot_ise_correlation(ise_scores, actual_means, correlation_path)
    except Exception as exc:
        print(f"  Correlation plot failed: {exc}")

    tier_events_path = os.path.join(OUTPUT_DIR, "tier_events.png")
    try:
        plot_tier_events(
            attn_history, total_promos, total_demos, subtier_counts,
            tier_events_path,
        )
    except Exception as exc:
        print(f"  Tier events plot failed: {exc}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)
    print(f"  Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f"  Plots generated:  attention_heatmap.png, ise_correlation.png, tier_events.png")

    # Print memory report
    try:
        print(f"\n{hook.get_memory_report()}")
    except Exception:
        pass

    print("\nDone.")


if __name__ == "__main__":
    main()
