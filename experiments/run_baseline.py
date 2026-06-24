import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Run Baseline Sweeps")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--method", type=str, default="streaming", choices=["full", "streaming", "h2o"])
    parser.add_argument("--mode", type=str, default="token", choices=["chunk", "token"])
    parser.add_argument("--num_eval_tokens", type=int, default=2048)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    args = parser.parse_args()
    
    if args.method == "full":
        budgets = [1.0]
    else:
        # Standard budget sweep ratios
        budgets = [1.0, 0.9, 0.7, 0.5, 0.3]
        
    print(f"==================================================")
    print(f"    STARTING BASELINE SWEEP: {args.method.upper()}")
    print(f"==================================================")
    
    for budget in budgets:
        cmd = [
            sys.executable, "-m", "src.eval.harness",
            "--model", args.model,
            "--method", args.method,
            "--budget_ratio", str(budget),
            "--mode", args.mode,
            "--num_eval_tokens", str(args.num_eval_tokens),
            "--seq_len", str(args.seq_len),
            "--stride", str(args.stride)
        ]
        print(f"\n>>> Running {args.method.upper()} at budget {budget}...")
        subprocess.run(cmd)
        
    print(f"\n==================================================")
    print(f"    SWEEP COMPLETED SUCCESSFULLY")
    print(f"==================================================")

if __name__ == "__main__":
    main()
