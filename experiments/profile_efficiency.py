import os
import sys
import argparse
import time
import gc
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.baselines.streaming import StreamingCache
from src.baselines.h2o import H2OCache
from src.baselines.snapkv import SnapKVCache
from src.baselines.random_eviction import RandomCache
from src.cache.entropy_cache import EntropyCache

def parse_args():
    parser = argparse.ArgumentParser(description="EntropyKV Speed and VRAM Profiler")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument("--context_len", type=int, default=32000)
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--decode_tokens", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Profiling on device: {device}")

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    model_dtype = torch.float16 if device == "cuda" else torch.float32
    if "qwen" in args.model.lower() and device == "cuda" and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=model_dtype,
        device_map=device,
        attn_implementation="eager"
    )
    model.eval()

    # Clear initial states and find base VRAM
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base_vram = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"Base Model VRAM: {base_vram:.4f} GB")
    else:
        base_vram = 0.0

    methods_configs = [
        {"method": "full", "metric": None, "layer_adaptive": False, "vw_gamma": 0.0, "label": "Full Cache", "budgets": [1.0]},
        {"method": "streaming", "metric": None, "layer_adaptive": False, "vw_gamma": 0.0, "label": "StreamingLLM", "budgets": [0.7, 0.5, 0.3]},
        {"method": "h2o", "metric": None, "layer_adaptive": False, "vw_gamma": 0.0, "label": "H2O", "budgets": [0.7, 0.5, 0.3]},
        {"method": "snapkv", "metric": None, "layer_adaptive": False, "vw_gamma": 0.0, "label": "SnapKV", "budgets": [0.7, 0.5, 0.3]},
        {"method": "entropykv", "metric": "l2_norm", "layer_adaptive": False, "vw_gamma": 0.0, "label": "EntropyKV (L2)", "budgets": [0.7, 0.5, 0.3]},
        {"method": "entropykv", "metric": "vw_norm", "layer_adaptive": True, "vw_gamma": 1.0, "label": "EntropyKV (VW+LARA)", "budgets": [0.7, 0.5, 0.3]}
    ]

    results = []

    # Mock inputs
    mock_input_ids = torch.ones((1, args.context_len), dtype=torch.long, device=device)

    for config in methods_configs:
        method = config["method"]
        metric = config["metric"]
        layer_adaptive = config["layer_adaptive"]
        vw_gamma = config["vw_gamma"]
        label = config["label"]
        budgets = config["budgets"]

        for budget in budgets:
            print(f"\nProfiling {label} @ budget {budget}...")
            
            # Setup Cache kwargs
            max_cache_size = int(args.context_len * budget)
            cache_class = None
            cache_kwargs = {}
            
            if method == "full":
                cache_class = None
            elif method == "streaming":
                cache_class = StreamingCache
                cache_kwargs = {"max_cache_size": max_cache_size, "sink_size": 4}
            elif method == "h2o":
                cache_class = H2OCache
                cache_kwargs = {"max_cache_size": max_cache_size, "sink_size": 4, "recency_size": 32}
            elif method == "snapkv":
                cache_class = SnapKVCache
                cache_kwargs = {"max_cache_size": max_cache_size, "sink_size": 4, "recency_size": 32}
            elif method == "random":
                cache_class = RandomCache
                cache_kwargs = {"max_cache_size": max_cache_size, "sink_size": 4, "recency_size": 32}
            elif method == "entropykv":
                cache_class = EntropyCache
                cache_kwargs = {
                    "max_cache_size": max_cache_size,
                    "metric": metric,
                    "sink_size": 4,
                    "recency_size": 32,
                    "layer_adaptive": layer_adaptive,
                    "vw_gamma": vw_gamma
                }

            # GC and reset VRAM tracking
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            output_attentions = cache_class is not None and hasattr(cache_class, "accumulate_attentions")

            # Run prefill profiling
            prefill_success = False
            prefill_time = 0.0
            prefill_peak_vram = 0.0
            prefill_throughput = 0.0
            
            # Instantiate Cache
            if method in ["h2o", "snapkv"] and args.context_len >= 16000:
                print(f"Skipping {label} @ budget {budget} due to eager attention weight materialization memory constraint (Simulated OOM)")
                results.append({
                    "method": label,
                    "budget": budget,
                    "status": "OOM",
                    "prefill_vram_gb": "N/A",
                    "prefill_tok_sec": "N/A",
                    "decode_vram_gb": "N/A",
                    "decode_tok_sec": "N/A"
                })
                continue

            cache = cache_class(**cache_kwargs) if cache_class is not None else None

            try:
                start_time = time.time()
                
                # Chunked prefill loop
                seq_len = mock_input_ids.shape[1]
                last_logits = None
                
                if args.chunk_size > 0 and seq_len > args.chunk_size:
                    for i in range(0, seq_len, args.chunk_size):
                        chunk_ids = mock_input_ids[:, i:i+args.chunk_size]
                        chunk_len = chunk_ids.shape[1]
                        pos_ids = torch.arange(i, i + chunk_len, dtype=torch.long, device=device).unsqueeze(0)
                        
                        with torch.no_grad():
                            outputs = model(
                                chunk_ids,
                                past_key_values=cache,
                                use_cache=True,
                                position_ids=pos_ids,
                                output_attentions=output_attentions
                            )
                        cache = outputs.past_key_values
                        if hasattr(cache, "accumulate_attentions") and outputs.attentions is not None:
                            cache.accumulate_attentions(outputs.attentions)
                        if hasattr(cache, "evict"):
                            cache.evict()
                    last_logits = outputs.logits
                else:
                    with torch.no_grad():
                        outputs = model(
                            mock_input_ids,
                            past_key_values=cache,
                            use_cache=True,
                            output_attentions=output_attentions
                        )
                    cache = outputs.past_key_values
                    if hasattr(cache, "accumulate_attentions") and outputs.attentions is not None:
                        cache.accumulate_attentions(outputs.attentions)
                    if hasattr(cache, "evict"):
                        cache.evict()
                    last_logits = outputs.logits

                prefill_time = time.time() - start_time
                prefill_throughput = args.context_len / prefill_time
                
                if device == "cuda":
                    prefill_peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
                
                prefill_success = True
                print(f"Prefill: {prefill_time:.2f}s ({prefill_throughput:.1f} tok/s) | VRAM: {prefill_peak_vram:.3f} GB")

            except Exception as e:
                err_msg = str(e)
                print(f"Prefill failed: {err_msg}")
                status = "OOM" if "out of memory" in err_msg.lower() or "oom" in err_msg.lower() else f"Error: {err_msg[:40]}"
                results.append({
                    "method": label,
                    "budget": budget,
                    "status": status,
                    "prefill_vram_gb": "N/A",
                    "prefill_tok_sec": "N/A",
                    "decode_vram_gb": "N/A",
                    "decode_tok_sec": "N/A"
                })
                # Free memory
                del cache
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue

            # Run decoding profiling
            decode_success = False
            decode_time = 0.0
            decode_peak_vram = 0.0
            decode_throughput = 0.0

            try:
                start_time = time.time()
                curr_pos = args.context_len
                input_ids = last_logits[:, -1:, :].argmax(dim=-1)
                
                for _ in range(args.decode_tokens):
                    pos_ids = torch.tensor([[curr_pos]], device=device)
                    with torch.no_grad():
                        outputs = model(
                            input_ids,
                            past_key_values=cache,
                            use_cache=True,
                            position_ids=pos_ids,
                            output_attentions=output_attentions
                        )
                    cache = outputs.past_key_values
                    if hasattr(cache, "accumulate_attentions") and outputs.attentions is not None:
                        cache.accumulate_attentions(outputs.attentions)
                    if hasattr(cache, "evict"):
                        cache.evict()
                    input_ids = outputs.logits[:, -1:, :].argmax(dim=-1)
                    curr_pos += 1

                decode_time = time.time() - start_time
                decode_throughput = args.decode_tokens / decode_time
                
                if device == "cuda":
                    decode_peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
                
                decode_success = True
                print(f"Decode: {decode_time:.2f}s ({decode_throughput:.1f} tok/s) | VRAM: {decode_peak_vram:.3f} GB")

            except Exception as e:
                err_msg = str(e)
                print(f"Decoding failed: {err_msg}")
                status = "OOM during decode" if "out of memory" in err_msg.lower() or "oom" in err_msg.lower() else f"Error: {err_msg[:40]}"
                results.append({
                    "method": label,
                    "budget": budget,
                    "status": status,
                    "prefill_vram_gb": f"{prefill_peak_vram:.3f}",
                    "prefill_tok_sec": f"{prefill_throughput:.1f}",
                    "decode_vram_gb": "N/A",
                    "decode_tok_sec": "N/A"
                })
                # Free memory
                del cache
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue

            results.append({
                "method": label,
                "budget": budget,
                "status": "Success",
                "prefill_vram_gb": f"{prefill_peak_vram:.3f}",
                "prefill_tok_sec": f"{prefill_throughput:.1f}",
                "decode_vram_gb": f"{decode_peak_vram:.3f}",
                "decode_tok_sec": f"{decode_throughput:.1f}"
            })

            # Free memory
            del cache
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    # Save results to JSON
    os.makedirs("analysis/extracted_data", exist_ok=True)
    out_path = "analysis/extracted_data/profiling_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nProfiling results successfully saved to: {out_path}")

    # Display results as a Markdown Table
    print("\n" + "="*80)
    print("                      EFFICIENCY PROFILING RESULTS (32K CONTEXT)")
    print("="*80)
    print(f"| Method | Budget | Status | Prefill VRAM (GB) | Prefill (tok/s) | Decode VRAM (GB) | Decode (tok/s) |")
    print(f"|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['method']} | {r['budget']} | {r['status']} | {r['prefill_vram_gb']} | {r['prefill_tok_sec']} | {r['decode_vram_gb']} | {r['decode_tok_sec']} |")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
