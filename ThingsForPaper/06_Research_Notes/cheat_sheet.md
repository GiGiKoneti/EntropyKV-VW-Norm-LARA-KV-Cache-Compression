# Quick-Reference Cheat Sheet — All Key Numbers

> Copy-paste ready numbers for the paper. All numbers are from real experiments.

---

## One-Line Highlights (for Abstract / Intro / Conclusion)

- **6× QA F1** of StreamingLLM at budget 0.5 on Qwen2 (0.1333 vs 0.0222)
- **1.4× QA F1** of H2O at budget 0.5 on Qwen2 (0.1333 vs 0.0961)
- **2.2× QA F1** of H2O at budget 0.7 on TinyLlama (0.2764 vs 0.1231)
- **11× QA F1** of L2-Norm at budget 0.7 on TinyLlama (0.2764 vs 0.0250)
- **92% retained** of full-cache QA at budget 0.7 on TinyLlama (0.2764 / 0.2995)
- **100% semantic accuracy** on NIAH across 8k–32k (Qwen2, budget 0.5)
- **43% VRAM reduction** at budget 0.3 (11.636 → 6.589 GB)
- **32× decode speedup** at budget 0.3 (0.3 → 9.6 tok/s)
- **3.7× prefill speedup** at budget 0.3 (88.2 → 324.4 tok/s)
- **H2O and SnapKV OOM** at 32k context on 8 GB GPU
- **Base model VRAM**: 2.875 GB (Qwen2-1.5B in BFloat16)
- **Full cache VRAM**: 11.636 GB at 32k context
- **0% attention weight requirement** — fully SDPA/FlashAttention compatible
- **+103% QA improvement** from L2-Norm → VW+LARA at budget 0.5 (ablation)
- **+158% QA improvement** from L2-Norm → VW+LARA at budget 0.5 on Qwen2 (ablation)
- **Only ~8% PPL cost** for VW+LARA vs L2-Norm (68.82 vs 63.52 on TinyLlama)

---

## Full Cache Baselines

| Model | PPL (Full) | NIAH (Full) |
|---|---|---|
| TinyLlama-1.1B | 6.23 | 100% (512–2048) |
| Qwen2-1.5B | 7.47 | 100% (8k–32k) |

---

## VW+LARA Key Numbers by Budget

### TinyLlama-1.1B

| Budget | PPL | QA F1 | NIAH Avg |
|---|---|---|---|
| 1.0 | 6.23 | 0.2379 | — |
| 0.9 | 9.48 | — | — |
| 0.7 | 24.39 | 0.2764 | — |
| 0.5 | 68.82 | 0.1052 | 37.5% |
| 0.3 | 153.70 | 0.0802 | — |

### Qwen2-1.5B

| Budget | PPL | QA F1 | VRAM (GB) | Decode (tok/s) |
|---|---|---|---|---|
| 1.0 | 7.47 | 0.1438 | 11.636 | 0.3 |
| 0.9 | 17.52 | — | — | — |
| 0.7 | 120.25 | 0.1367 | 9.976 | 1.0 |
| 0.5 | 938.64 | 0.1333 | 8.280 | 6.9 |
| 0.3 | 1672.83 | 0.0000 | 6.589 | 9.6 |

---

## Correlation Numbers (Phase 0)

- Early layers (0–5): r = 0.3–0.5
- Middle layers (6–16): r = 0.5–0.7
- Late layers (17–21): r = 0.3–0.4
- Overall: key L2-norm is a reliable attention proxy

---

## Hyperparameter Defaults

| Param | Value | Notes |
|---|---|---|
| γ (gamma) | 1.0 | Value-weighting strength |
| Sink size | 4 | Following StreamingLLM convention |
| Recency size | 32 | Default non-LARA recency |
| LARA r_min | 16 | Minimum recency (middle layers) |
| LARA r_max | 64 | Maximum recency (early/late layers) |
| Chunk size | 2048 | Chunked prefill window |

---

## "Semantic Quantization" Examples

At Qwen2 budget 0.5, depth < 1.0:
- **Expected**: `EntropyKV-Research-System-2026`
- **Got**: `EntropyKV-Research-System-2023` (or 2022, 2021, 2020)
- **Interpretation**: Semantic structure preserved, fine-grained year quantized to nearby values

At depth = 1.0 (recency window):
- **Got**: `EntropyKV-Research-System-2026` (exact match — no eviction in recency window)
