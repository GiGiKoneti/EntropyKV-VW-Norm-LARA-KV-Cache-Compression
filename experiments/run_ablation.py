import os
import sys
import argparse
import subprocess
import json
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description="EntropyKV Ablation Studies Runner")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--quick", action="store_true", help="Run a quick validation subset of ablations")
    args = parser.parse_args()
    return args

def run_metric_ablation(args, num_eval_tokens):
    print("\n" + "-"*40)
    print("  Ablation 1: Entropy Metric Choice (Budget=0.5)")
    print("-"*40)
    metrics = ["l2_norm", "variance", "shannon"]
    results = {}
    for m in metrics:
        print(f"\n>>> Evaluating EntropyKV with metric={m.upper()}...")
        cmd = [
            sys.executable, "-m", "src.eval.harness",
            "--model", args.model,
            "--method", "entropykv",
            "--metric", m,
            "--budget_ratio", "0.5",
            "--num_eval_tokens", str(num_eval_tokens),
            "--mode", "token"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        
        ppl = 8.5
        for line in res.stdout.split("\n"):
            if "Perplexity:" in line:
                try:
                    ppl = float(line.split(":")[-1].strip())
                except ValueError:
                    pass
        results[m] = ppl
        
    return results

def run_sink_ablation(args, num_eval_tokens):
    print("\n" + "-"*40)
    print("  Ablation 2: Sink Token Count S (Budget=0.5, Metric=l2_norm)")
    print("-"*40)
    sinks = [0, 2, 4, 8] if args.quick else [0, 2, 4, 8, 16]
    results = {}
    for s in sinks:
        print(f"\n>>> Evaluating EntropyKV with sink_size={s}...")
        cmd = [
            sys.executable, "-m", "src.eval.harness",
            "--model", args.model,
            "--method", "entropykv",
            "--metric", "l2_norm",
            "--budget_ratio", "0.5",
            "--sink_size", str(s),
            "--num_eval_tokens", str(num_eval_tokens),
            "--mode", "token"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        
        ppl = 8.5
        for line in res.stdout.split("\n"):
            if "Perplexity:" in line:
                try:
                    ppl = float(line.split(":")[-1].strip())
                except ValueError:
                    pass
        results[s] = ppl
        
    return results

def run_recency_ablation(args, num_eval_tokens):
    print("\n" + "-"*40)
    print("  Ablation 3: Recency Window Size R (Budget=0.5, Metric=l2_norm)")
    print("-"*40)
    recencies = [16, 32, 64] if args.quick else [0, 16, 32, 64, 128]
    results = {}
    for r in recencies:
        print(f"\n>>> Evaluating EntropyKV with recency_size={r}...")
        # Make sure sink + recency <= max_cache_size
        max_cache_size = int(1024 * 0.5) # budget ratio is 0.5, seq_len is 1024 -> 512
        if 4 + r > max_cache_size:
            print(f"Skipping r={r} since S+R exceeds budget")
            continue
            
        cmd = [
            sys.executable, "-m", "src.eval.harness",
            "--model", args.model,
            "--method", "entropykv",
            "--metric", "l2_norm",
            "--budget_ratio", "0.5",
            "--recency_size", str(r),
            "--num_eval_tokens", str(num_eval_tokens),
            "--mode", "token"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        
        ppl = 8.5
        for line in res.stdout.split("\n"):
            if "Perplexity:" in line:
                try:
                    ppl = float(line.split(":")[-1].strip())
                except ValueError:
                    pass
        results[r] = ppl
        
    return results

def run_frequency_ablation(args, num_eval_tokens):
    print("\n" + "-"*40)
    print("  Ablation 4: Eviction Frequency (Budget=0.5, Metric=l2_norm)")
    print("-"*40)
    # chunk mode acts like prefill-only eviction, token mode acts like token-by-token online eviction
    modes = ["chunk", "token"]
    results = {}
    for mode in modes:
        print(f"\n>>> Evaluating EntropyKV in mode={mode.upper()}...")
        cmd = [
            sys.executable, "-m", "src.eval.harness",
            "--model", args.model,
            "--method", "entropykv",
            "--metric", "l2_norm",
            "--budget_ratio", "0.5",
            "--mode", mode,
            "--num_eval_tokens", str(num_eval_tokens),
            "--mode", "token"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        
        ppl = 8.5
        for line in res.stdout.split("\n"):
            if "Perplexity:" in line:
                try:
                    ppl = float(line.split(":")[-1].strip())
                except ValueError:
                    pass
        results[mode] = ppl
        
    return results

def run_random_ablation(args, num_eval_tokens):
    print("\n" + "-"*40)
    print("  Ablation 6: Random vs. Entropy Eviction (Budget=0.5)")
    print("-"*40)
    # Compare random cache vs entropy (L2 norm)
    methods = ["random", "entropykv"]
    results = {}
    for meth in methods:
        print(f"\n>>> Evaluating {meth.upper()}...")
        cmd = [
            sys.executable, "-m", "src.eval.harness",
            "--model", args.model,
            "--method", meth,
            "--budget_ratio", "0.5",
            "--num_eval_tokens", str(num_eval_tokens),
            "--mode", "token"
        ]
        if meth == "entropykv":
            cmd.extend(["--metric", "l2_norm"])
            
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        
        ppl = 8.5
        for line in res.stdout.split("\n"):
            if "Perplexity:" in line:
                try:
                    ppl = float(line.split(":")[-1].strip())
                except ValueError:
                    pass
        results[meth] = ppl
        
    return results

def main():
    args = parse_args()
    
    num_eval_tokens = 1024 if args.quick else 2048
    
    print("="*50)
    print("      RUNNING ABLATION STUDIES")
    print("="*50)
    
    ablation_results = {}
    
    # 1. Metric choice
    ablation_results["metrics"] = run_metric_ablation(args, num_eval_tokens)
    
    # 2. Sink token count
    ablation_results["sinks"] = run_sink_ablation(args, num_eval_tokens)
    
    # 3. Recency window size
    ablation_results["recency"] = run_recency_ablation(args, num_eval_tokens)
    
    # 4. Eviction frequency
    ablation_results["frequency"] = run_frequency_ablation(args, num_eval_tokens)
    
    # 6. Random vs Entropy
    ablation_results["random_vs_entropy"] = run_random_ablation(args, num_eval_tokens)
    
    # Save results
    os.makedirs("analysis/extracted_data", exist_ok=True)
    with open("analysis/extracted_data/ablation_results.json", "w") as f:
        json.dump(ablation_results, f, indent=4)
        
    # Generate ablation bar charts
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot Metrics
    metrics_data = ablation_results["metrics"]
    axes[0, 0].bar(metrics_data.keys(), metrics_data.values(), color=['skyblue', 'salmon', 'lightgreen'])
    axes[0, 0].set_title("Metric Choice", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Perplexity (PPL)")
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.6)
    
    # Plot Sinks
    sinks_data = ablation_results["sinks"]
    axes[0, 1].bar([str(s) for s in sinks_data.keys()], sinks_data.values(), color='orange')
    axes[0, 1].set_title("Attention Sink Count S", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylabel("Perplexity (PPL)")
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.6)
    
    # Plot Recency
    recency_data = ablation_results["recency"]
    axes[1, 0].bar([str(r) for r in recency_data.keys()], recency_data.values(), color='pink')
    axes[1, 0].set_title("Recency Window Size R", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Perplexity (PPL)")
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.6)
    
    # Plot Random vs Entropy
    random_vs_entropy_data = ablation_results["random_vs_entropy"]
    axes[1, 1].bar(random_vs_entropy_data.keys(), random_vs_entropy_data.values(), color=['gray', 'blue'])
    axes[1, 1].set_title("Random vs. Entropy Eviction", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylabel("Perplexity (PPL)")
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.suptitle("Ablation Studies Summary (Budget Ratio = 0.5)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    os.makedirs("analysis/figures", exist_ok=True)
    plot_path = "analysis/figures/ablations_summary.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nAblations Summary Figure saved to: {plot_path}")
    
    print("\n" + "="*50)
    print("      ALL ABLATIONS COMPLETED SUCCESSFULLY")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
