import numpy as np

methods = ["streaming", "h2o", "entropykv_l2_norm", "entropykv_vw_norm"]
for name in methods:
    path = f"analysis/extracted_data/niah_results_{name}_budget_0.5.npz"
    try:
        data = np.load(path)
        print(f"=== {name.upper()} ===")
        print("Context lengths:", data["context_lengths"])
        print("Depth fractions:", data["depth_fractions"])
        print("Results Matrix (Context x Depth):")
        print(data["results"])
        print(f"Average Accuracy: {np.mean(data['results']) * 100:.2f}%\n")
    except Exception as e:
        print(f"Error loading {path}: {e}")
