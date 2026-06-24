import numpy as np
import os

data_dir = "analysis/extracted_data"
files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith("sample_") and f.endswith(".npz")])

if not files:
    print("No data files found.")
    exit()

# We will calculate L2 Norm and Softmax Shannon Entropy layer-wise correlations
import scipy.stats as stats

for name in ["L2 Norm", "Shannon Entropy (Softmax)", "Abs Shannon Entropy", "Variance"]:
    all_corrs = []
    for file_path in files:
        data = np.load(file_path)
        keys = data["keys"]
        cum_att = data["cum_att"]
        
        num_layers, num_kv_heads, seq_len, head_dim = keys.shape
        
        # Compute proxy
        if name == "L2 Norm":
            proxy = np.linalg.norm(keys, axis=-1)
        elif name == "Variance":
            proxy = np.var(keys, axis=-1)
        elif name == "Shannon Entropy (Softmax)":
            shifted_keys = keys - np.max(keys, axis=-1, keepdims=True)
            exp_keys = np.exp(shifted_keys)
            prob = exp_keys / (np.sum(exp_keys, axis=-1, keepdims=True) + 1e-9)
            proxy = -np.sum(prob * np.log(prob + 1e-9), axis=-1)
        elif name == "Abs Shannon Entropy":
            abs_keys = np.abs(keys)
            prob = abs_keys / (np.sum(abs_keys, axis=-1, keepdims=True) + 1e-9)
            proxy = -np.sum(prob * np.log(prob + 1e-9), axis=-1)
            
        corrs = np.zeros((num_layers, num_kv_heads))
        for l in range(num_layers):
            for h in range(num_kv_heads):
                r, _ = stats.spearmanr(proxy[l, h, :], cum_att[l, h, :])
                corrs[l, h] = r if not np.isnan(r) else 0.0
        all_corrs.append(corrs)
        
    mean_corr = np.mean(np.stack(all_corrs), axis=0)
    layer_mean = np.mean(mean_corr, axis=1)
    
    print(f"\n{name} Layer-wise correlation:")
    for l, val in enumerate(layer_mean):
        print(f"  Layer {l:2d}: {val:+.4f}")
