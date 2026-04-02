"""End-to-end demonstration of Fast-KV in action.

Simulates a 500-token cybersecurity conversation with realistic token text,
feeds each token through the Fast-KV cache, and produces memory reports
and visualizations.
"""

import sys
import os

import numpy as np
from scipy.special import softmax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fast_kv.config import FastKVConfig
from fast_kv.fast_kv_cache import FastKVCache


# Cybersecurity conversation token vocabulary
CYBER_STOPWORDS = [
    "the", "a", "an", "in", "on", "at", "to", "of", "for", "by", "with",
    "and", "or", "but", "is", "are", "was", "were", ".", ",", ";", ":",
    "from", "that", "this", "has", "had", "have", "been", "be",
]
CYBER_NOUNS = [
    "attacker", "vulnerability", "exploit", "payload", "shellcode",
    "privilege", "escalation", "buffer", "overflow", "injection",
    "malware", "ransomware", "backdoor", "rootkit", "trojan",
    "firewall", "endpoint", "daemon", "socket", "kernel",
    "credential", "hash", "certificate", "token", "session",
]
CYBER_VERBS = [
    "detected", "exploited", "escalated", "injected", "encrypted",
    "exfiltrated", "scanned", "patched", "blocked", "terminated",
    "compromised", "intercepted", "decoded", "obfuscated", "executed",
]
CYBER_ENTITIES = [
    "CVE-2024-1234", "CVE-2024-5678", "CVE-2023-9999", "CVE-2024-0001",
    "192.168.1.1", "10.0.0.1", "172.16.0.50", "8.8.8.8",
    "Linux", "Windows", "Apache", "Nginx", "OpenSSH",
    "MITRE", "NIST", "CIS", "OWASP",
    "/etc/passwd", "/var/log/auth.log", "/tmp/.hidden",
    "sshd", "httpd", "systemd", "cron",
]
QUESTION_TOKENS = ["who", "what", "where", "when", "why", "how"]
FIRST_PERSON = ["I", "my", "we", "our"]

ALL_TOKENS = (
    CYBER_STOPWORDS + CYBER_NOUNS + CYBER_VERBS + CYBER_ENTITIES
    + QUESTION_TOKENS + FIRST_PERSON
)


def generate_cyber_token(step: int) -> str:
    """Generate a realistic cybersecurity conversation token.

    Distribution: ~35% stopwords, ~25% nouns, ~15% verbs, ~20% entities, ~5% questions.
    """
    rng = np.random.RandomState(step * 7 + 13)
    r = rng.random()
    if r < 0.35:
        return rng.choice(CYBER_STOPWORDS)
    elif r < 0.60:
        return rng.choice(CYBER_NOUNS)
    elif r < 0.75:
        return rng.choice(CYBER_VERBS)
    elif r < 0.95:
        return rng.choice(CYBER_ENTITIES)
    else:
        return rng.choice(QUESTION_TOKENS + FIRST_PERSON)


def main() -> None:
    """Run the end-to-end Fast-KV demo."""
    np.random.seed(42)

    # Configuration
    config = FastKVConfig()
    model_config = {
        "n_layers": 32,
        "kv_dim": 1024,
        "dtype": "float32",
    }
    n_tokens = 500

    print("=" * 70)
    print("  Fast-KV End-to-End Demo")
    print("  Simulating a 500-token cybersecurity conversation")
    print(f"  Model: Llama 3 8B ({model_config['n_layers']} layers, kv_dim={model_config['kv_dim']})")
    print("=" * 70)
    print()

    cache = FastKVCache(config, model_config)

    # Track data for visualization
    hot_counts = []
    cold_counts = []
    ram_baseline = []
    ram_fastkv = []
    all_scores = []
    token_labels = []  # 'hot' or 'cold' at end

    bytes_per_token = model_config["kv_dim"] * 2 * 4 * model_config["n_layers"]

    for step in range(n_tokens):
        token_text = generate_cyber_token(step)
        key_vec = np.random.randn(model_config["kv_dim"]).astype(np.float32)
        val_vec = np.random.randn(model_config["kv_dim"]).astype(np.float32)

        # Generate realistic attention weights — important tokens get more attention
        n_existing = step + 1
        raw_attn = np.random.randn(n_existing) * 0.5
        # Boost attention for entity/proper noun tokens
        for i in range(n_existing):
            t = generate_cyber_token(i)
            if t in CYBER_ENTITIES or t in QUESTION_TOKENS or t in FIRST_PERSON:
                raw_attn[i] += 3.0  # Strong boost for important tokens
            elif t in CYBER_NOUNS or t in CYBER_VERBS:
                raw_attn[i] += 1.0  # Moderate boost for content words
            # Stopwords get no boost (remain low)
        attn_probs = softmax(raw_attn)
        attention_weights = {
            i: float(attn_probs[i]) for i in range(n_existing)
        }

        # Update all layers
        for layer_id in range(model_config["n_layers"]):
            kv = np.random.randn(model_config["kv_dim"]).astype(np.float32)
            vv = np.random.randn(model_config["kv_dim"]).astype(np.float32)
            cache.update(layer_id, step, token_text, kv, vv, attention_weights, step)

        # Track stats
        stats = cache.tier_managers[0].get_stats()
        hot_counts.append(stats["n_hot"])
        cold_counts.append(stats["n_cold"])
        ram_baseline.append((step + 1) * bytes_per_token / (1024 ** 2))

        total_ram = sum(
            tm.get_stats()["estimated_ram_total_mb"]
            for tm in cache.tier_managers.values()
        )
        ram_fastkv.append(total_ram)

        # Print progress every 50 tokens
        if (step + 1) % 50 == 0:
            print(f"\n--- After {step + 1} tokens ---")
            print(cache.get_memory_report())

    # Final report
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(cache.get_memory_report())

    # Collect scores for visualization
    tm0 = cache.tier_managers[0]
    for tid in range(n_tokens):
        score = cache.ise.get_score(tid, n_tokens)
        all_scores.append(score)
        token_labels.append(tm0.token_tiers.get(tid, "unknown"))

    hot_scores = [s for s, l in zip(all_scores, token_labels) if l == "hot"]
    cold_scores = [s for s, l in zip(all_scores, token_labels) if l == "cold"]

    print(f"\nScore distribution:")
    print(f"  Hot tokens:  n={len(hot_scores)}, "
          f"mean={np.mean(hot_scores):.3f}, min={np.min(hot_scores):.3f}" if hot_scores else "  Hot tokens:  n=0")
    print(f"  Cold tokens: n={len(cold_scores)}, "
          f"mean={np.mean(cold_scores):.3f}, max={np.max(cold_scores):.3f}" if cold_scores else "  Cold tokens: n=0")

    # Generate visualizations
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Importance score distribution
        ax1 = axes[0]
        if hot_scores:
            ax1.hist(hot_scores, bins=30, alpha=0.7, label="Hot (Tier 1)", color="red")
        if cold_scores:
            ax1.hist(cold_scores, bins=30, alpha=0.7, label="Cold (Tier 2)", color="blue")
        ax1.axvline(x=config.hot_threshold, color="red", linestyle="--", label=f"Hot threshold ({config.hot_threshold})")
        ax1.axvline(x=config.cold_threshold, color="blue", linestyle="--", label=f"Cold threshold ({config.cold_threshold})")
        ax1.set_xlabel("Importance Score")
        ax1.set_ylabel("Count")
        ax1.set_title("Token Importance Score Distribution")
        ax1.legend()

        # Plot 2: RAM usage over time
        ax2 = axes[1]
        steps = list(range(1, n_tokens + 1))
        ax2.plot(steps, ram_baseline, label="Baseline (no compression)", color="red", linewidth=2)
        ax2.plot(steps, ram_fastkv, label="Fast-KV", color="green", linewidth=2)
        ax2.fill_between(steps, ram_fastkv, ram_baseline, alpha=0.2, color="green", label="RAM saved")
        ax2.set_xlabel("Tokens Processed")
        ax2.set_ylabel("RAM Usage (MB)")
        ax2.set_title("RAM Usage: Baseline vs Fast-KV")
        ax2.legend()

        plt.tight_layout()
        plt.savefig("fast_kv_demo.png", dpi=150)
        print(f"\nVisualization saved to: fast_kv_demo.png")

    except ImportError:
        print("\nMatplotlib not available — skipping visualization.")


if __name__ == "__main__":
    main()
