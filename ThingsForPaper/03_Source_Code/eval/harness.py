import os
import argparse
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from src.eval.perplexity import compute_perplexity
from src.baselines.streaming import StreamingCache
from src.baselines.h2o import H2OCache
from src.baselines.snapkv import SnapKVCache
from src.baselines.random_eviction import RandomCache
from src.cache.entropy_cache import EntropyCache

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="EntropyKV Evaluation Harness")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--budget_ratio", type=float, default=0.5, help="KV cache retention ratio (0.1 to 1.0)")
    parser.add_argument("--method", type=str, default="entropykv", choices=["full", "streaming", "h2o", "snapkv", "random", "entropykv"])
    parser.add_argument("--metric", type=str, default="l2_norm", choices=["l2_norm", "variance", "shannon", "vw_norm"])
    parser.add_argument("--layer_adaptive", action="store_true", help="Use layer-adaptive budgets (U-shape curve)")
    parser.add_argument("--vw_gamma", type=float, default=0.0, help="Exponent for Value vector in Value-Weighted Key Norm")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sliding window sequence length")
    parser.add_argument("--stride", type=int, default=512, help="Sliding window stride size")
    parser.add_argument("--sink_size", type=int, default=4, help="Attention sink size")
    parser.add_argument("--recency_size", type=int, default=32, help="Recency window size")
    parser.add_argument("--mode", type=str, default="chunk", choices=["chunk", "token"], help="Evaluation mode")
    parser.add_argument("--num_eval_tokens", type=int, default=2048, help="Number of tokens from validation set to evaluate on")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    
    # 1. Determine Device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")
    
    # 2. Load Tokenizer and Model
    print(f"Loading tokenizer and model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    model_dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32
    if "qwen" in args.model.lower() and device == "cuda" and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=model_dtype,
        device_map=device,
        attn_implementation="eager" # Eager is required for output_attentions=True
    )
    model.eval()
    
    # 3. Load and Prepare WikiText-2 Dataset
    print("Loading WikiText-2 validation split...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    
    # Concatenate non-empty text items
    print(f"Preparing dataset and tokenizing up to {args.num_eval_tokens} tokens...")
    full_text = "\n\n".join([item["text"] for item in dataset if item["text"].strip() != ""])
    tokens = tokenizer(full_text, return_tensors="pt").input_ids
    
    # Prune tokens to target size
    input_ids = tokens[:, :args.num_eval_tokens]
    print(f"Tokens prepared. Shape: {input_ids.shape}")
    
    # 4. Configure Cache Class
    max_cache_size = int(args.seq_len * args.budget_ratio)
    
    cache_class = None
    cache_kwargs = {}
    
    if args.method == "full":
        print(f"Method: Full Cache (No Eviction)")
    elif args.method == "streaming":
        cache_class = StreamingCache
        cache_kwargs = {
            "max_cache_size": max_cache_size,
            "sink_size": args.sink_size
        }
        print(f"Method: StreamingLLM (Budget={max_cache_size}, Sinks={args.sink_size})")
    elif args.method == "h2o":
        cache_class = H2OCache
        cache_kwargs = {
            "max_cache_size": max_cache_size,
            "sink_size": args.sink_size,
            "recency_size": args.recency_size
        }
        print(f"Method: H2O (Budget={max_cache_size}, Sinks={args.sink_size}, Recency={args.recency_size})")
    elif args.method == "snapkv":
        cache_class = SnapKVCache
        cache_kwargs = {
            "max_cache_size": max_cache_size,
            "sink_size": args.sink_size,
            "recency_size": args.recency_size
        }
        print(f"Method: SnapKV (Budget={max_cache_size}, Sinks={args.sink_size}, Recency={args.recency_size})")
    elif args.method == "random":
        cache_class = RandomCache
        cache_kwargs = {
            "max_cache_size": max_cache_size,
            "sink_size": args.sink_size,
            "recency_size": args.recency_size
        }
        print(f"Method: Random (Budget={max_cache_size}, Sinks={args.sink_size}, Recency={args.recency_size})")
    elif args.method == "entropykv":
        cache_class = EntropyCache
        cache_kwargs = {
            "max_cache_size": max_cache_size,
            "metric": args.metric,
            "sink_size": args.sink_size,
            "recency_size": args.recency_size,
            "layer_adaptive": args.layer_adaptive,
            "vw_gamma": args.vw_gamma
        }
        print(f"Method: EntropyKV (Budget={max_cache_size}, Metric={args.metric}, Sinks={args.sink_size}, Recency={args.recency_size}, LayerAdaptive={args.layer_adaptive}, vw_gamma={args.vw_gamma})")
        
    # Reset peak memory stats before evaluation
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    start_time = time.time()
    
    # 5. Execute Perplexity Computation
    try:
        ppl = compute_perplexity(
            model=model,
            input_ids=input_ids,
            seq_len=args.seq_len,
            stride=args.stride,
            device=device,
            cache_class=cache_class,
            cache_kwargs=cache_kwargs,
            mode=args.mode
        )
    except Exception as e:
        print(f"Error during perplexity evaluation: {e}")
        raise e
        
    elapsed_time = time.time() - start_time
    
    # 6. Retrieve Memory usage
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) # in GB
    else:
        peak_vram = 0.0 # No CUDA memory tracking
        
    # Calculate average processing speed (tokens / second)
    total_tokens_processed = input_ids.shape[1]
    tokens_per_sec = total_tokens_processed / elapsed_time if elapsed_time > 0 else 0
    
    # 7. Print Master Evaluation Metrics
    print("\n" + "="*50)
    print("                 EVALUATION RESULTS")
    print("="*50)
    print(f"Model:           {args.model}")
    print(f"Method:          {args.method.upper()}")
    if args.method == "entropykv":
        print(f"Scoring Metric:  {args.metric}")
    print(f"Budget Ratio:    {args.budget_ratio:.2f} ({max_cache_size} tokens)")
    print(f"Evaluation Mode: {args.mode}")
    print(f"Tokens Checked:  {total_tokens_processed}")
    print("-"*50)
    print(f"Perplexity:      {ppl:.4f}")
    print(f"Peak VRAM (GB):  {peak_vram:.4f} GB")
    print(f"Elapsed Time:    {elapsed_time:.2f} seconds")
    print(f"Throughput:      {tokens_per_sec:.2f} tokens/second")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
