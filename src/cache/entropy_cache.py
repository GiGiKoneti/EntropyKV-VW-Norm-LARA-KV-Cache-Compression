import torch
from transformers.cache_utils import DynamicCache
from src.cache.utils import compute_l2_norm, compute_variance, compute_shannon_entropy, compute_value_weighted_norm

class EntropyCache(DynamicCache):
    def __init__(
        self, 
        max_cache_size: int, 
        metric: str = "l2_norm", 
        sink_size: int = 4, 
        recency_size: int = 32,
        layer_adaptive: bool = False,
        vw_gamma: float = 0.0
    ):
        """
        EntropyKV online KV cache eviction.
        
        Args:
            max_cache_size: maximum number of tokens to keep in cache.
            metric: proxy metric to use for scoring ('l2_norm', 'variance', 'shannon', 'vw_norm').
            sink_size: number of initial attention sink tokens to protect (S).
            recency_size: number of recent tokens to protect (R).
            layer_adaptive: whether to distribute layer budgets dynamically using a U-shaped profile.
            vw_gamma: value-weighted key norm exponent (gamma).
        """
        super().__init__()
        self.max_cache_size = max_cache_size
        self.metric = metric.lower()
        self.sink_size = sink_size
        self.recency_size = recency_size
        self.layer_adaptive = layer_adaptive
        self.vw_gamma = vw_gamma
        
        if self.metric not in ["l2_norm", "variance", "shannon", "vw_norm"]:
            raise ValueError(f"Unknown metric: {metric}. Must be 'l2_norm', 'variance', 'shannon', or 'vw_norm'.")
            
        if self.sink_size + self.recency_size > self.max_cache_size:
            raise ValueError(
                f"Combined sink_size ({sink_size}) and recency_size ({recency_size}) "
                f"exceeds max_cache_size ({max_cache_size})."
            )
            
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None
    ):
        return super().update(key_states, value_states, layer_idx, cache_kwargs)
        
    def evict(self):
        """
        Prunes the KV cache to max_cache_size by keeping attention sinks, recency window, 
        and the top heavy-hitters based on key-vector entropy scoring metrics.
        Supports layer-adaptive static recency allocation and value-weighted key norm eviction.
        """
        num_layers = len(self.layers) if hasattr(self, "layers") and len(self.layers) > 0 else len(self.key_cache)
        
        if self.layer_adaptive and not hasattr(self, "layer_recency_sizes"):
            import numpy as np
            L = num_layers
            x = np.arange(L) - (L - 1) / 2.0
            x2 = x ** 2
            mean_x2 = np.mean(x2) if L > 1 else 1.0
            b = 0.5
            a = (1.0 - b) / mean_x2 if L > 1 else 0.0
            r = a * x2 + b
            
            self.layer_recency_sizes = []
            for idx in range(L):
                l_recency = int(round(self.recency_size * r[idx]))
                max_recency = self.max_cache_size - self.sink_size - 1
                if l_recency < 8:
                    l_recency = 8
                if l_recency > max_recency:
                    l_recency = max_recency
                self.layer_recency_sizes.append(l_recency)
                
        for l in range(num_layers):
            if hasattr(self, "layers") and len(self.layers) > 0:
                keys = self.layers[l].keys
                values = self.layers[l].values
            else:
                keys = self.key_cache[l]
                values = self.value_cache[l]
                
            seq_len = keys.shape[2]
            recency_size = self.layer_recency_sizes[l] if self.layer_adaptive else self.recency_size
            
            # Perform online eviction if the sequence length exceeds the budget
            if seq_len > self.max_cache_size:
                # Protected ranges:
                # - Attention sinks: range(0, S)
                # - Recency window: range(seq_len - R, seq_len)
                protected_indices = set(range(self.sink_size)) | set(range(seq_len - recency_size, seq_len))
                candidate_indices = [i for i in range(seq_len) if i not in protected_indices]
                
                num_keep = self.max_cache_size - self.sink_size - recency_size
                
                if num_keep <= 0 or not candidate_indices:
                    # If budget is extremely small, act like StreamingLLM
                    keep_indices = sorted(list(protected_indices))
                    pruned_keys = keys[:, :, keep_indices, :]
                    pruned_values = values[:, :, keep_indices, :]
                    
                    if hasattr(self, "layers") and len(self.layers) > 0:
                        self.layers[l].keys = pruned_keys
                        self.layers[l].values = pruned_values
                    else:
                        self.key_cache[l] = pruned_keys
                        self.value_cache[l] = pruned_values
                    continue
                    
                # 3. Compute token importance scores across the head dimension
                # key shape: (batch_size, num_kv_heads, seq_len, head_dim)
                if self.metric == "l2_norm":
                    scores = compute_l2_norm(keys) # (batch, num_heads, seq_len)
                    largest = False
                elif self.metric == "variance":
                    scores = compute_variance(keys)
                    largest = False
                elif self.metric == "shannon":
                    scores = compute_shannon_entropy(keys)
                    largest = True
                elif self.metric == "vw_norm":
                    scores = compute_value_weighted_norm(keys, values, gamma=self.vw_gamma)
                    largest = False
                    
                # Advanced Vectorized Slicing using torch.gather for maximum native GPU speed
                batch_size, num_heads, _ = scores.shape
                device = keys.device
                
                # Get candidate scores: shape (batch, num_heads, len(candidate_indices))
                candidate_scores = scores[:, :, candidate_indices]
                
                # Top-k indices within candidates: shape (batch, num_heads, num_keep)
                top_k_indices = torch.topk(candidate_scores, num_keep, largest=largest).indices
                
                # Map to real indices in the sequence
                candidate_tensor = torch.tensor(candidate_indices, device=device).view(1, 1, -1).expand(batch_size, num_heads, -1)
                top_real_indices = torch.gather(candidate_tensor, dim=2, index=top_k_indices)
                
                # Protected indices tensors
                sink_tensor = torch.arange(self.sink_size, device=device).view(1, 1, -1).expand(batch_size, num_heads, -1)
                recency_tensor = torch.arange(seq_len - recency_size, seq_len, device=device).view(1, 1, -1).expand(batch_size, num_heads, -1)
                
                # Combine keep indices: shape (batch, num_heads, max_cache_size)
                keep_indices = torch.cat([sink_tensor, top_real_indices, recency_tensor], dim=-1)
                keep_indices = torch.sort(keep_indices, dim=-1).values
                
                # Gather key-values in parallel across heads
                head_dim = keys.shape[-1]
                gather_index = keep_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                pruned_keys = torch.gather(keys, dim=2, index=gather_index)
                pruned_values = torch.gather(values, dim=2, index=gather_index)
                
                # Overwrite cache tensors for this layer (supporting both transformers v4 and v5)
                if hasattr(self, "layers") and len(self.layers) > 0:
                    self.layers[l].keys = pruned_keys
                    self.layers[l].values = pruned_values
                else:
                    self.key_cache[l] = pruned_keys
                    self.value_cache[l] = pruned_values
