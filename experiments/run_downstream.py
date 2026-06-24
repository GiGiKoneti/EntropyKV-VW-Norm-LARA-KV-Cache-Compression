import os
import sys
import argparse
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description="EntropyKV Downstream Task Sweeps Runner")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--task", type=str, default="all", choices=["niah", "qa", "all"])
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples for downstream QA")
    parser.add_argument("--quick", action="store_true", help="Run a quick micro-sweep to verify plotting")
    args = parser.parse_args()
    return args

def run_qa_sweeps(args):
    print("\n" + "="*50)
    print("      RUNNING DOWNSTREAM QA SWEEPS")
    print("="*50)
    
    budgets = [1.0, 0.7, 0.5, 0.3]
    if args.quick:
        budgets = [1.0, 0.5]
        
    methods_configs = [
        {"method": "streaming", "metric": None, "layer_adaptive": False, "vw_gamma": 0.0, "label": "STREAMING"},
        {"method": "h2o", "metric": None, "layer_adaptive": False, "vw_gamma": 0.0, "label": "H2O"},
        {"method": "entropykv", "metric": "l2_norm", "layer_adaptive": False, "vw_gamma": 0.0, "label": "ENTROPYKV (L2_NORM)"},
        {"method": "entropykv", "metric": "vw_norm", "layer_adaptive": True, "vw_gamma": 1.0, "label": "ENTROPYKV (VW-NORM + LARA)"}
    ]
    
    # Store results for plotting: {method_label: {budget: f1_score}}
    plot_data = {}
    
    for config in methods_configs:
        method = config["method"]
        metric = config["metric"]
        layer_adaptive = config.get("layer_adaptive", False)
        vw_gamma = config.get("vw_gamma", 0.0)
        label = config["label"]
            
        plot_data[label] = {}
        
        for budget in budgets:
            print(f"\n>>> Executing Downstream QA: {label} at budget {budget}...")
            cmd = [
                sys.executable, "-m", "src.eval.longbench",
                "--model", args.model,
                "--method", method,
                "--budget_ratio", str(budget),
                "--num_samples", str(args.num_samples)
            ]
            if metric:
                cmd.extend(["--metric", metric])
            if layer_adaptive:
                cmd.append("--layer_adaptive")
            if vw_gamma > 0.0:
                cmd.extend(["--vw_gamma", str(vw_gamma)])
                
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stdout
            print(output)
            
            # Parse F1 from stdout
            f1_match = [line for line in output.split("\n") if "Average F1:" in line]
            if f1_match:
                try:
                    f1_val = float(f1_match[0].split(":")[-1].strip())
                    plot_data[label][budget] = f1_val
                except ValueError:
                    plot_data[label][budget] = 0.0
            else:
                plot_data[label][budget] = 0.0
                
    # Save raw JSON results
    os.makedirs("analysis/extracted_data", exist_ok=True)
    with open("analysis/extracted_data/qa_sweeps_results.json", "w") as f:
        json.dump(plot_data, f, indent=4)
        
    # Generate downstream QA plot
    plt.figure(figsize=(8, 5))
    for label, budget_scores in plot_data.items():
        sorted_budgets = sorted(budget_scores.keys(), reverse=True)
        scores = [budget_scores[b] for b in sorted_budgets]
        plt.plot(sorted_budgets, scores, marker='o', label=label, linewidth=2)
        
    plt.title("Downstream QA Performance vs. KV Cache Budget", fontsize=12, fontweight="bold")
    plt.xlabel("KV Cache Budget Ratio", fontsize=10)
    plt.ylabel("Token-Level F1 Score", fontsize=10)
    plt.xlim(1.05, 0.25) # reverse x-axis to show budget reduction left-to-right
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    os.makedirs("analysis/figures", exist_ok=True)
    plot_path = "analysis/figures/qa_performance_vs_budget.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nDownstream QA Comparison Plot saved to: {plot_path}")

def run_niah_sweeps(args):
    print("\n" + "="*50)
    print("      RUNNING NEEDLE-IN-A-HAYSTACK SWEEPS")
    print("="*50)
    
    methods_configs = [
        {"method": "streaming", "metric": None, "budget": 0.5, "layer_adaptive": False, "vw_gamma": 0.0, "label": "STREAMING"},
        {"method": "h2o", "metric": None, "budget": 0.5, "layer_adaptive": False, "vw_gamma": 0.0, "label": "H2O"},
        {"method": "entropykv", "metric": "l2_norm", "budget": 0.5, "layer_adaptive": False, "vw_gamma": 0.0, "label": "ENTROPYKV (L2_NORM)"},
        {"method": "entropykv", "metric": "vw_norm", "budget": 0.5, "layer_adaptive": True, "vw_gamma": 1.0, "label": "ENTROPYKV (VW-NORM + LARA)"}
    ]
    
    if args.quick:
        # Micro-sweep parameters to run quickly
        print("Quick mode active: executing limited NIAH runs...")
        
    for config in methods_configs:
        method = config["method"]
        metric = config["metric"]
        budget = config["budget"]
        layer_adaptive = config.get("layer_adaptive", False)
        vw_gamma = config.get("vw_gamma", 0.0)
        label = config["label"]
            
        print(f"\n>>> Executing NIAH Sweep: {label} at budget {budget}...")
        cmd = [
            sys.executable, "-m", "src.eval.niah",
            "--model", args.model,
            "--method", method,
            "--budget_ratio", str(budget)
        ]
        if metric:
            cmd.extend(["--metric", metric])
        if layer_adaptive:
            cmd.append("--layer_adaptive")
        if vw_gamma > 0.0:
            cmd.extend(["--vw_gamma", str(vw_gamma)])
        if args.quick:
            cmd.append("--quick")
            
        # Run sweep
        subprocess.run(cmd)

def main():
    args = parse_args()
    
    if args.task in ["qa", "all"]:
        run_qa_sweeps(args)
        
    if args.task in ["niah", "all"]:
        run_niah_sweeps(args)
        
    print("\n" + "="*50)
    print("      ALL SWEEPS COMPLETED SUCCESSFULLY")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
