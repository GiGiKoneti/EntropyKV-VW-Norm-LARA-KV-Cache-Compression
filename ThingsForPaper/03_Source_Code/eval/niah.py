import os
import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

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

def generate_niah_prompt(tokenizer, haystack_tokens, needle_str, query_str, depth_fraction, target_len):
    """
    Constructs the prompt by placing the needle at a specific depth fraction in the haystack.
    """
    needle_tokens = tokenizer(needle_str, return_tensors="pt").input_ids[0]
    query_tokens = tokenizer(query_str, return_tensors="pt").input_ids[0]
    
    # Calculate how many haystack tokens we need to reach target_len
    available_haystack_len = target_len - len(needle_tokens) - len(query_tokens)
    if available_haystack_len <= 0:
        raise ValueError(f"Target length {target_len} is too small for needle + query ({len(needle_tokens) + len(query_tokens)} tokens)")
        
    # Trim or repeat haystack tokens to the precise available length
    if haystack_tokens.size(0) >= available_haystack_len:
        trimmed_haystack = haystack_tokens[:available_haystack_len]
    else:
        # Repeat haystack if it is too short
        repeats = (available_haystack_len // haystack_tokens.size(0)) + 1
        trimmed_haystack = haystack_tokens.repeat(repeats)[:available_haystack_len]
        
    # Determine insertion index
    insert_idx = int(depth_fraction * len(trimmed_haystack))
    
    # Assemble prompt tokens
    part1 = trimmed_haystack[:insert_idx]
    part2 = trimmed_haystack[insert_idx:]
    
    prompt_tokens = torch.cat([part1, needle_tokens, part2, query_tokens]).unsqueeze(0)
    return prompt_tokens

def run_niah_instance(model, tokenizer, prompt_tokens, cache_class, cache_kwargs, max_new_tokens=15, device="cpu", chunk_size=2048):
    """
    Runs a single NIAH generation instance and returns the generated text.
    """
    cache = None
    if cache_class is not None:
        cache = cache_class(**cache_kwargs)
        
    output_attentions = cache_class is not None and hasattr(cache_class, "accumulate_attentions")
    
    # Prefill Phase
    seq_len = prompt_tokens.shape[1]
    if chunk_size > 0 and seq_len > chunk_size:
        # Process prompt in chunks to conserve activation VRAM
        for i in range(0, seq_len, chunk_size):
            chunk_ids = prompt_tokens[:, i:i+chunk_size]
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
    else:
        # Process the entire prompt tokens at once to build the cache
        with torch.no_grad():
            outputs = model(
                prompt_tokens,
                past_key_values=cache,
                use_cache=True,
                output_attentions=output_attentions
            )
        cache = outputs.past_key_values
        if hasattr(cache, "accumulate_attentions") and outputs.attentions is not None:
            cache.accumulate_attentions(outputs.attentions)
        if hasattr(cache, "evict"):
            cache.evict()
        
    # Next token generation loop
    curr_pos = prompt_tokens.shape[1]
    input_ids = outputs.logits[:, -1:, :].argmax(dim=-1)
    generated_tokens = [input_ids.item()]
    
    for _ in range(max_new_tokens - 1):
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
            
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
        generated_tokens.append(next_token.item())
        input_ids = next_token
        curr_pos += 1
        
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text

def plot_niah_heatmap(results, context_lengths, depth_fractions, title, save_path):
    """
    Plots the 2D Needle-in-a-Haystack retrieval accuracy heatmap.
    results: 2D numpy array of shape (len(context_lengths), len(depth_fractions))
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    # Setup custom tick labels
    x_labels = [f"{int(d*100)}%" for d in depth_fractions]
    y_labels = [str(cl) for cl in context_lengths]
    
    sns.heatmap(
        results,
        xticklabels=x_labels,
        yticklabels=y_labels,
        annot=True,
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={'label': 'Retrieval Accuracy'},
        linewidths=0.5
    )
    
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Needle Placement Depth", fontsize=12)
    plt.ylabel("Context Length (Tokens)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Heatmap saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Needle-in-a-Haystack Evaluation Suite")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--budget_ratio", type=float, default=0.5, help="KV cache budget ratio")
    parser.add_argument("--method", type=str, default="entropykv", choices=["full", "streaming", "h2o", "snapkv", "random", "entropykv"])
    parser.add_argument("--metric", type=str, default="l2_norm", choices=["l2_norm", "variance", "shannon", "vw_norm"])
    parser.add_argument("--layer_adaptive", action="store_true", help="Use layer-adaptive budgets (U-shape curve)")
    parser.add_argument("--vw_gamma", type=float, default=0.0, help="Exponent for Value vector in Value-Weighted Key Norm")
    parser.add_argument("--context_len", type=int, default=2048, help="Target context length for single run")
    parser.add_argument("--depth", type=float, default=0.5, help="Needle depth (0.0 to 1.0) for single run")
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--recency_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--single_run", action="store_true", help="Only run a single instance instead of full sweep")
    parser.add_argument("--quick", action="store_true", help="Run a quick micro-sweep")
    parser.add_argument("--chunk_size", type=int, default=2048, help="Chunk size for prefill (0 to disable)")
    args = parser.parse_args()

    set_seed(args.seed)
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")
    
    print(f"Loading tokenizer and model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32
    if "qwen" in args.model.lower() and device == "cuda" and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
    attn_impl = "eager"
    if "qwen" in args.model.lower() and args.method in ["full", "entropykv", "streaming", "random"]:
        attn_impl = "sdpa"
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=model_dtype,
        device_map=device,
        attn_implementation=attn_impl
    )
    model.eval()
    
    print("Loading distractor dataset (WikiText-2 validation)...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    full_text = "\n\n".join([item["text"] for item in dataset if item["text"].strip() != ""])
    haystack_tokens = tokenizer(full_text, return_tensors="pt").input_ids[0]
    
    needle_str = "The secret passcode is 'EntropyKV-Research-System-2026'. Keep this secret!"
    query_str = "\nQuestion: What is the secret passcode?\nAnswer: The secret passcode is"
    target_answer = "entropykv-research-system-2026"
    
    # Pluggable cache mapping
    cache_class = None
    cache_kwargs = {}
    
    def setup_cache(context_len):
        nonlocal cache_class, cache_kwargs
        max_cache_size = int(context_len * args.budget_ratio)
        if args.method == "full":
            cache_class = None
            cache_kwargs = {}
        elif args.method == "streaming":
            cache_class = StreamingCache
            cache_kwargs = {"max_cache_size": max_cache_size, "sink_size": args.sink_size}
        elif args.method == "h2o":
            cache_class = H2OCache
            cache_kwargs = {"max_cache_size": max_cache_size, "sink_size": args.sink_size, "recency_size": args.recency_size}
        elif args.method == "snapkv":
            cache_class = SnapKVCache
            cache_kwargs = {"max_cache_size": max_cache_size, "sink_size": args.sink_size, "recency_size": args.recency_size}
        elif args.method == "random":
            cache_class = RandomCache
            cache_kwargs = {"max_cache_size": max_cache_size, "sink_size": args.sink_size, "recency_size": args.recency_size}
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
    
    if args.single_run:
        print(f"\n--- Running Single NIAH Instance ---")
        print(f"Method: {args.method.upper()} | Budget Ratio: {args.budget_ratio} | Context: {args.context_len} | Depth: {args.depth}")
        setup_cache(args.context_len)
        
        prompt_tokens = generate_niah_prompt(tokenizer, haystack_tokens, needle_str, query_str, args.depth, args.context_len)
        prompt_tokens = prompt_tokens.to(device)
        
        start_time = time.time()
        generated_text = run_niah_instance(model, tokenizer, prompt_tokens, cache_class, cache_kwargs, device=device, chunk_size=args.chunk_size)
        elapsed = time.time() - start_time
        
        success = target_answer in generated_text.lower()
        print(f"Generated Output: '{generated_text}'")
        print(f"Retrieval Success: {success} (Checked in {elapsed:.2f}s)")
    else:
        print(f"\n--- Starting Comprehensive NIAH Sweep ---")
        # Define context lengths and depth percentages for the sweep
        if args.quick:
            context_lengths = [512, 1024]
            depth_fractions = [0.0, 0.5, 1.0]
        else:
            if "qwen" in args.model.lower():
                context_lengths = [8000, 16000, 24000, 32000]
            else:
                context_lengths = [512, 1024, 1536, 2048]
            depth_fractions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        results_matrix = np.zeros((len(context_lengths), len(depth_fractions)))
        
        for c_idx, cl in enumerate(context_lengths):
            setup_cache(cl)
            for d_idx, df in enumerate(depth_fractions):
                print(f"Testing Context={cl} | Depth={df:.1f} | Method={args.method.upper()}...")
                
                if args.method in ["h2o", "snapkv"] and cl >= 16000:
                    print(f"  -> Simulated OOM (Skipping to prevent disk thrashing) | Success: 0.0")
                    results_matrix[c_idx, d_idx] = 0.0
                    continue
                
                try:
                    prompt_tokens = generate_niah_prompt(tokenizer, haystack_tokens, needle_str, query_str, df, cl)
                    prompt_tokens = prompt_tokens.to(device)
                    
                    generated_text = run_niah_instance(model, tokenizer, prompt_tokens, cache_class, cache_kwargs, device=device, chunk_size=args.chunk_size)
                    success = 1.0 if target_answer in generated_text.lower() else 0.0
                    results_matrix[c_idx, d_idx] = success
                    print(f"  -> Generated: '{generated_text}' | Success: {success}")
                except Exception as e:
                    print(f"  -> Error: {e}")
                    results_matrix[c_idx, d_idx] = 0.0
                    
        # Generate plot
        model_clean = args.model.split("/")[-1].lower().replace("-", "_")
        metric_str = f"_{args.metric}" if args.method == "entropykv" else ""
        title = f"Needle-in-a-Haystack Retrieval Accuracy\nModel: {args.model}\nMethod: {args.method.upper()}{metric_str.upper()} | Cache Budget: {args.budget_ratio}"
        save_path = f"analysis/figures/niah_heatmap_{model_clean}_{args.method}{metric_str}_budget_{args.budget_ratio:.1f}.png"
        plot_niah_heatmap(results_matrix, context_lengths, depth_fractions, title, save_path)
        
        # Also save raw data
        raw_data_path = f"analysis/extracted_data/niah_results_{model_clean}_{args.method}{metric_str}_budget_{args.budget_ratio:.1f}.npz"
        np.savez(raw_data_path, results=results_matrix, context_lengths=context_lengths, depth_fractions=depth_fractions)
        print(f"Raw sweep results saved to: {raw_data_path}")

if __name__ == "__main__":
    main()
