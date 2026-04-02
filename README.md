# Fast-KV ⚡

> Tiered KV Cache Compression for Edge LLMs — up to 65% RAM reduction with <0.5% accuracy loss on consumer hardware.

[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-orange.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Tests: 52 passing](https://img.shields.io/badge/tests-52%20passing-brightgreen.svg)]()
[![Status: Research Prototype](https://img.shields.io/badge/status-research%20prototype-orange.svg)]()

---

## What is Fast-KV?

Large language models running locally on consumer hardware face a critical problem: as conversations grow longer, the KV cache (the model's short-term memory) consumes increasingly large amounts of RAM. A 5,000 token conversation on a model like Llama 3 8B requires over 1.2 GB of RAM just for the KV cache — before accounting for the model weights themselves.

Fast-KV solves this with a two-tier importance-aware cache system:

- **Tier 1 (Hot)** — High-importance tokens stored at full 32-bit precision. Instantly accessible.
- **Tier 2 (Cold)** — Low-importance tokens compressed to 1–4 bit mixed precision. Still in RAM, decompressed on demand in microseconds.

Unlike eviction-based approaches (which permanently delete tokens and hurt accuracy), Fast-KV keeps all tokens — it just stores unimportant ones cheaply.

---

## How It Works

Every token in the KV cache is continuously scored by the **Importance Scoring Engine (ISE)** using three lightweight signals:

1. **Static classification** — stopwords and punctuation are always cold; proper nouns, numbers, and named entities are always hot. Zero compute cost.
2. **Attention score tap** — the model already computes attention weights during inference. Fast-KV reads these for free and maintains an exponential moving average per token.
3. **Recency decay** — tokens not recently attended to are gradually demoted. Three arithmetic operations per token.

Combined score drives dynamic promotion and demotion between tiers. A token that becomes suddenly relevant (e.g. the model returns to an earlier topic) is promoted from cold to hot in microseconds.

---

## Benchmark Results

> **Important:** These benchmarks were run on synthetic KV vectors using Llama 3 8B architecture parameters (32 layers, 1024-dim KV, float32). Real model integration is in progress. Production numbers will differ — see [Roadmap](#roadmap).

### Memory Reduction

| Conversation Length | Baseline RAM | Fast-KV RAM | Savings | Accuracy Loss |
|---|---|---|---|---|
| 100 tokens | 25.0 MB | 9.4 MB | 62.4% | 0.168% |
| 500 tokens | 125.0 MB | 46.1 MB | 63.1% | 0.152% |
| 1,000 tokens | 250.0 MB | 90.3 MB | 63.9% | 0.166% |
| 5,000 tokens | 1,250.0 MB | 426.8 MB | 65.9% | 0.207% |

### System Performance

| Metric | Target | Result |
|---|---|---|
| RAM reduction (5K tokens) | >= 65% | 65.9% |
| MAE for 4-bit + residual | < 0.01 | 0.000423 |
| ISE compute overhead | < 1% | 0.11% |
| False demotion rate | < 5% | 0.00% |
| Test suite | All pass | 52/52 |

---

## Quick Start

```bash
git clone https://github.com/Q-alqam/fast-kv
cd fast-kv
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the end-to-end demo (500-token cybersecurity conversation)
python demo.py

# Run full benchmark suite
python benchmarks/memory_benchmark.py
python benchmarks/accuracy_benchmark.py
python benchmarks/speed_benchmark.py

# Run tests
python -m pytest tests/ -v
```

---

## Project Structure

```
fast_kv/
├── config.py                  # All tunable parameters (thresholds, bit widths, weights)
├── importance_scorer.py       # Three-layer ISE
├── compression.py             # Mixed-precision quantization engine (1/2/4/8/16/32-bit)
├── tier_manager.py            # Hot/cold tier assignment, promotion, demotion
└── fast_kv_cache.py           # Main FastKVCache class (drop-in interface)
benchmarks/
├── memory_benchmark.py        # RAM usage comparison across conversation lengths
├── accuracy_benchmark.py      # Quantization accuracy + tier assignment accuracy
└── speed_benchmark.py         # ISE compute overhead measurement
tests/
├── test_importance_scorer.py  # 22 tests
├── test_compression.py        # 20 tests
└── test_tier_manager.py       # 10 tests
demo.py                        # End-to-end demo with visualizations
```

---

## Comparison with Existing Approaches

| Approach | Handles Edge? | Preserves Evicted Tokens? | Importance-Aware? |
|---|---|---|---|
| GGUF Q4/Q8 | Yes | N/A (weights only) | No |
| TurboQuant (Google) | No (Cloud only) | Yes | No (Uniform) |
| KV Eviction (H2O) | Yes | No (Permanently deleted) | Yes |
| Sliding Window | Yes | No (Old context lost) | No |
| **Fast-KV** | **Yes** | **Yes (Compressed, recoverable)** | **Yes (Dynamic)** |

---

## Roadmap

### Phase 1 — Python Prototype (Complete)
- Three-layer Importance Scoring Engine
- Mixed-precision compression (1/2/4-bit + residual storage)
- Full benchmark suite and 52 passing unit tests
- Synthetic KV vector validation

### Phase 2 — Real Model Integration (In Progress)
- Integration with llama.cpp via C FFI
- Testing on Mistral 7B and Llama 3 8B with real conversations
- Per-model threshold calibration
- Real-world benchmark publication

### Phase 3 — Rust Production Port (Planned)
- Port core algorithm to Rust
- SIMD-accelerated quantization
- Integration with Senvex endpoint security inference pipeline
- PyO3 Python bindings for drop-in replacement

---

## Key Design Decisions

**Why two tiers in RAM instead of disk?**
Disk access is 1000x slower than RAM. Any disk-based cold tier would destroy inference latency. Both tiers stay in RAM — Tier 2 just uses 8-16x less space through aggressive compression.

**Why not just use TurboQuant?**
TurboQuant targets cloud infrastructure (NVIDIA H100 GPUs) and applies uniform compression across all tokens. Fast-KV is designed specifically for edge hardware and applies variable compression based on per-token importance — preserving accuracy where it matters.

**Why not eviction?**
Evicted tokens are gone permanently. If the model needs to refer back to an earlier part of the conversation, accuracy degrades silently. Fast-KV keeps all tokens — it just stores less important ones cheaply and restores them faithfully when needed.

**Attention normalization**
Raw softmax attention weights become meaninglessly tiny as context grows (1/500 average in a 500-token context). Fast-KV normalizes each token's attention relative to the uniform distribution, making importance scoring work correctly regardless of context length.

---

## Contributing

This is an early-stage research project. Contributions welcome — especially:
- Real model integration (llama.cpp, Hugging Face)
- Additional quantization methods
- Per-model calibration benchmarks
- Rust port contributions

Open an issue before submitting a large PR.

---

## License

**Business Source License 1.1 (BSL 1.1)**

- **Free** for non-commercial use, personal use, academic research, and open source projects
- **Commercial license required** for production use in commercial products or hosted services competing with WeSecure
- **Converts to Apache 2.0** on January 1, 2029

See [LICENSE](LICENSE) for full terms.

---

## Citation

If you use Fast-KV in your research, please cite:

```bibtex
@software{fastkv2026,
  title={Fast-KV: Tiered KV Cache Compression for Edge LLMs},
  author={WeSecure},
  year={2026},
  url={https://github.com/Q-alqam/fast-kv}
}
```

---

*Built by [WeSecure](https://wesecure.ca)*
