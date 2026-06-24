import numpy as np
import os
import scipy.stats as stats

data_dir = "analysis/extracted_data"
files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith("sample_") and f.endswith(".npz")])

sink_size = 4
recency_size = 32

for name in ["L2 Norm", "Abs Shannon Entropy", "Variance"]:
    all_corrs = []
    for file_path in files:
        data = np.load(file_path)
        keys = data["keys"]
        cum_att = data["cum_att"]
        
        num_layers, num_kv_heads, seq_len, head_dim = keys.shape
        
        # Determine candidate indices
        protected_indices = set(range(sink_size)) | set(range(seq_len - recency_size, seq_len))
        candidate_indices = [i for i in range(seq_len) if i not in protected_indices]
        
        # Compute proxy
        if name == "L2 Norm":
            proxy = np.linalg.norm(keys, axis=-1)
        elif name == "Variance":
            proxy = np.var(keys, axis=-1)
        elif name == "Abs Shannon Entropy":
            abs_keys = np.abs(keys)
            prob = abs_keys / (np.sum(abs_keys, axis=-1, keepdims=True) + 1e-9)
            proxy = -np.sum(prob * np.log(prob + 1e-9), axis=-1)
            
        corrs = np.zeros((num_layers, num_kv_heads))
        for l in range(num_layers):
            for h in range(num_kv_heads):
                # Only slice candidates
                score = proxy[l, h, candidate_indices]
                att = cum_att[l, h, candidate_indices]
                
                r, _ = stats.spearmanr(score, att)
                corrs[l, h] = r if not np.isnan(r) else 0.0
        all_corrs.append(corrs)
        
    mean_corr = np.mean(np.stack(all_corrs), axis=0)
    layer_mean = np.mean(mean_corr, axis=1)
    
    print(f"\n{name} Candidate-only (excluding S={sink_size}, R={recency_size}) Layer-wise correlation:")
    for l, val in enumerate(layer_mean):
        print(f"  Layer {l:2d}: {val:+.4f}")
    
    # Print overall average
    overall_mean = np.mean(layer_mean)
    overall_abs = np.mean(np.abs(layer_mean))
    print(f"  OVERALL Mean r: {overall_mean:.4f}, Mean |r|: {overall_abs:.4f}")
