import os
import glob
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
INPUT_DIR = "analysis/extracted_data"
FIGURES_DIR = "analysis/figures"

def compute_entropy_proxies(keys):
    """
    Computes different entropy/variance proxies for keys.
    keys: shape (num_layers, num_kv_heads, seq_len, head_dim)
    Returns a dict of proxies, each of shape (num_layers, num_kv_heads, seq_len)
    """
    num_layers, num_kv_heads, seq_len, head_dim = keys.shape
    
    # 1. Option A: Shannon entropy of softmax-normalized key vector
    # Avoid underflow/overflow by shifting key components
    shifted_keys = keys - np.max(keys, axis=-1, keepdims=True)
    exp_keys = np.exp(shifted_keys)
    prob_a = exp_keys / (np.sum(exp_keys, axis=-1, keepdims=True) + 1e-9)
    entropy_a = -np.sum(prob_a * np.log(prob_a + 1e-9), axis=-1)
    
    # 2. Option B: Variance of key vector components
    variance_b = np.var(keys, axis=-1)
    
    # 3. Option C: L2 norm of key vector
    l2_norm_c = np.linalg.norm(keys, axis=-1)
    
    # 4. Option D: Shannon entropy of absolute-value normalized key vector
    abs_keys = np.abs(keys)
    prob_d = abs_keys / (np.sum(abs_keys, axis=-1, keepdims=True) + 1e-9)
    entropy_d = -np.sum(prob_d * np.log(prob_d + 1e-9), axis=-1)
    
    return {
        "Shannon Entropy (Softmax)": entropy_a,
        "Variance": variance_b,
        "L2 Norm": l2_norm_c,
        "Abs Shannon Entropy": entropy_d
    }

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Load all sample npz files
    npz_files = sorted(glob.glob(os.path.join(INPUT_DIR, "sample_*.npz")))
    if not npz_files:
        print(f"Error: No extracted data files found in {INPUT_DIR}. Run instrument_forward_pass.py first.")
        return
        
    print(f"Found {len(npz_files)} sample files. Performing correlation analysis...")
    
    # We will accumulate correlation results for each proxy option
    # Structure: {option: [samples of shape (num_layers, num_kv_heads)]}
    results = {}
    
    for file_path in npz_files:
        data = np.load(file_path)
        keys = data["keys"]          # Shape: (num_layers, num_kv_heads, seq_len, head_dim)
        cum_att = data["cum_att"]    # Shape: (num_layers, num_kv_heads, seq_len)
        
        num_layers, num_kv_heads, seq_len, _ = keys.shape
        proxies = compute_entropy_proxies(keys)
        
        for name, proxy_val in proxies.items():
            if name not in results:
                results[name] = []
            
            # Compute Spearman correlation for each layer and head
            corrs = np.zeros((num_layers, num_kv_heads))
            for l in range(num_layers):
                for h in range(num_kv_heads):
                    score = proxy_val[l, h, :]
                    att = cum_att[l, h, :]
                    
                    # Spearman rank correlation
                    r, p = stats.spearmanr(score, att)
                    if np.isnan(r):
                        r = 0.0
                    corrs[l, h] = r
            
            results[name].append(corrs)
            
    # Calculate average correlations across samples
    mean_correlations = {}
    for name in results:
        # stack to shape (num_samples, num_layers, num_kv_heads) and average
        stacked = np.stack(results[name], axis=0)
        mean_correlations[name] = np.mean(stacked, axis=0)
        
        # Overall absolute mean correlation
        overall_mean = np.mean(mean_correlations[name])
        overall_abs_mean = np.mean(np.abs(mean_correlations[name]))
        print(f"Option: {name}")
        print(f"  Overall mean r: {overall_mean:.4f}")
        print(f"  Overall mean |r|: {overall_abs_mean:.4f}")
        
    # Let's find the best performing proxy (based on average absolute correlation |r|)
    best_proxy_name = max(mean_correlations.keys(), key=lambda k: np.mean(np.abs(mean_correlations[k])))
    best_corrs = mean_correlations[best_proxy_name]
    print(f"\nBest performing proxy: {best_proxy_name} with mean |r| = {np.mean(np.abs(best_corrs)):.4f}")
    
    # 5. Plotting Results
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Line Plot of mean correlation per layer depth (H2 check)
    plt.figure(figsize=(8, 5))
    for name, corrs in mean_correlations.items():
        layer_means = np.mean(corrs, axis=1) # average over heads
        # We plot absolute value of correlation to compare strength
        plt.plot(layer_means, marker='o', label=name)
        
    plt.title("Mean Spearman Correlation per Layer Depth")
    plt.xlabel("Layer Index")
    plt.ylabel("Mean Spearman Correlation (r)")
    plt.legend()
    plt.tight_layout()
    plot1_path = os.path.join(FIGURES_DIR, "correlation_by_layer.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved Plot 1 to {plot1_path}")
    
    # Plot 2: Heatmap of mean correlation for the best proxy (Layers x Heads)
    plt.figure(figsize=(10, 8))
    # We display actual r (could be positive or negative)
    sns.heatmap(best_corrs, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar_kws={'label': 'Spearman Correlation (r)'})
    plt.title(f"Mean Correlation Heatmap for {best_proxy_name}")
    plt.xlabel("Key-Value Head Index")
    plt.ylabel("Layer Index")
    plt.tight_layout()
    plot2_path = os.path.join(FIGURES_DIR, "correlation_heatmap.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved Plot 2 to {plot2_path}")
    
    # Plot 3: Scatter plot of proxy score vs cumulative attention for a representative layer and head
    # Let's pick a middle/late layer (e.g. layer 15 of 22) and head 0
    rep_layer = min(15, num_layers - 1)
    rep_head = 0
    
    # Gather data from the first sample
    first_sample_data = np.load(npz_files[0])
    first_keys = first_sample_data["keys"]
    first_cum_att = first_sample_data["cum_att"]
    
    first_proxies = compute_entropy_proxies(first_keys)
    rep_score = first_proxies[best_proxy_name][rep_layer, rep_head, :]
    rep_att = first_cum_att[rep_layer, rep_head, :]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(rep_score, rep_att, alpha=0.5, color='darkblue', edgecolors='none')
    plt.title(f"Score vs Cumulative Attention (Layer {rep_layer}, KV Head {rep_head})\nProxy: {best_proxy_name}")
    plt.xlabel(f"Entropy Proxy Score ({best_proxy_name})")
    plt.ylabel("Cumulative Future Attention Weight")
    plt.tight_layout()
    plot3_path = os.path.join(FIGURES_DIR, "scatter_score_vs_attention.png")
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"Saved Plot 3 to {plot3_path}")
    
    print("\nPhase 0 Correlation Analysis completed successfully.")

if __name__ == "__main__":
    main()
