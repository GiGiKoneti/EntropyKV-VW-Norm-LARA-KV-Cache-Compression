# Figure Captions — Ready for LaTeX

> Pre-written captions for every figure. Copy-paste into \caption{} in LaTeX.

---

## TinyLlama Figures

### ppl_vs_budget.png
**Caption**: Sliding-window perplexity on WikiText-2 as a function of KV cache budget ratio for TinyLlama-1.1B-Chat-v1.0. SnapKV (prefill-phase selection) achieves the lowest perplexity. Among online eviction methods, EntropyKV (VW+LARA) outperforms StreamingLLM and Random at all budgets.

### qa_performance_vs_budget.png
**Caption**: Downstream QA F1 score (LongBench, 5 samples) vs. KV cache budget ratio on TinyLlama-1.1B. EntropyKV (VW+LARA) demonstrates the most graceful degradation, retaining 92\% of full-cache performance at budget 0.7 and maintaining non-zero F1 at budget 0.3 where L2-Norm collapses to zero.

### niah_heatmap_entropykv_vw_norm_budget_0.5.png
**Caption**: Needle-in-a-Haystack retrieval accuracy for EntropyKV (VW-Norm + LARA) at budget 0.5 on TinyLlama-1.1B. The method uniquely preserves retrieval at early context depths (0.0 and 0.2), demonstrating LARA's ability to protect structurally important early-layer caches.

### niah_heatmap_streaming_budget_0.5.png
**Caption**: NIAH retrieval for StreamingLLM at budget 0.5. Retrieval succeeds only in the last 40\% of the document (depths $\geq$ 0.6), illustrating the fundamental limitation of pure recency-based eviction.

### niah_heatmap_h2o_budget_0.5.png
**Caption**: NIAH retrieval for H2O at budget 0.5. Complete retrieval failure (0\% accuracy) across all context lengths and depths, demonstrating that uniform attention-based eviction destroys critical retrieval keys under aggressive compression.

### niah_heatmap_entropykv_l2_norm_budget_0.5.png
**Caption**: NIAH retrieval for EntropyKV (L2-Norm, no LARA) at budget 0.5. Complete failure, identical to H2O, confirming that layer-adaptive allocation (LARA) is the critical component enabling retrieval preservation.

---

## Qwen2 Figures

### qwen2_ppl_vs_budget.png
**Caption**: Sliding-window perplexity vs. KV cache budget ratio on Qwen2-1.5B-Instruct at 32k context. Log scale. All methods show steep degradation below budget 0.7, with SnapKV maintaining the best perplexity. StreamingLLM shows the worst perplexity at budget 0.3 (2323.17).

### qwen2_qa_vs_budget.png
**Caption**: Downstream QA F1 score vs. budget on Qwen2-1.5B at 32k context. EntropyKV (VW+LARA) shows the flattest degradation curve, retaining 0.1333 F1 at budget 0.5 — 6$\times$ StreamingLLM (0.0222) and 2.6$\times$ L2-Norm (0.0517). StreamingLLM collapses entirely at budget 0.3.

### qwen2_vram_profiling.png
**Caption**: Peak VRAM usage for Qwen2-1.5B at 32k context across methods and budgets. The dashed red line marks the 8\,GB consumer GPU limit. At budget 0.3, all methods fit within 6.6\,GB. H2O and SnapKV are excluded (OOM due to eager attention).

### qwen2_throughput_profiling.png
**Caption**: Prefill and decode throughput (tokens/second) for Qwen2-1.5B at 32k context. Left: prefill throughput scales from 88 tok/s (full cache) to 324 tok/s at budget 0.3. Right: decode throughput improves 32$\times$ (0.3 to 9.6 tok/s) at budget 0.3 for EntropyKV (VW+LARA).

### qwen2_efficiency_tradeoff.png
**Caption**: Efficiency tradeoff: peak VRAM vs. decode throughput for Qwen2-1.5B at 32k context. Each point is labeled with its budget ratio. The red dashed line marks the 8\,GB GPU limit. Lower budgets achieve both lower VRAM and higher throughput, with EntropyKV variants matching StreamingLLM's efficiency profile.

### niah_heatmap_qwen2_*_entropykv_vw_norm_budget_0.5.png
**Caption**: NIAH retrieval heatmap for Qwen2-1.5B with EntropyKV (VW+LARA) at budget 0.5 across 8k–32k contexts. Despite only 16.7\% exact-match accuracy, the model achieves 100\% semantic accuracy — consistently retrieving the needle's structure while quantizing fine-grained numeric details (``semantic quantization'').

### niah_heatmap_qwen2_*_full_budget_1.0.png
**Caption**: Full-cache baseline NIAH retrieval for Qwen2-1.5B-Instruct across 8k–32k context lengths. Perfect 100\% retrieval accuracy at all context lengths and depths, confirming the model's native long-context capability.

---

## Correlation Analysis Figures

### correlation_heatmap.png
**Caption**: Pearson correlation between key vector L2 norms and per-position attention scores across all 22 layers and attention heads of TinyLlama-1.1B. Middle layers (6–16) show the strongest positive correlation ($r = 0.5$–$0.7$), validating key norms as attention proxies.

### correlation_by_layer.png
**Caption**: Layer-wise mean Pearson correlation between key L2 norms and attention scores. The U-shaped pattern of \emph{weaker} correlation at early and late layers directly motivates LARA's U-shaped recency allocation — layers where key norms are less predictive receive larger recency windows.

### scatter_score_vs_attention.png
**Caption**: Scatter plot of key vector L2 norm vs. accumulated attention score for representative positions. The positive correlation confirms that key vector magnitudes encode meaningful information about token importance, enabling attention-free eviction.
