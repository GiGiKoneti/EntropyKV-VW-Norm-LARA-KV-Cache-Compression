import torch
import random
from transformers.cache_utils import DynamicCache

class RandomCache(DynamicCache):
    def __init__(self, max_cache_size: int, sink_size: int = 4, recency_size: int = 32):
        """
        Random eviction KV cache baseline.
        
        Args:
            max_cache_size: maximum number of tokens to keep in cache.
            sink_size: number of initial attention sink tokens to protect (S).
            recency_size: number of recent tokens to protect (R).
        """
        super().__init__()
        self.max_cache_size = max_cache_size
        self.sink_size = sink_size
        self.recency_size = recency_size
        
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
        Prunes the KV cache to max_cache_size by keeping attention sinks, the recency window, 
        and randomly selected tokens in the middle.
        """
        num_layers = len(self.layers) if hasattr(self, "layers") and len(self.layers) > 0 else len(self.key_cache)
        for l in range(num_layers):
            if hasattr(self, "layers") and len(self.layers) > 0:
                keys = self.layers[l].keys
                values = self.layers[l].values
            else:
                keys = self.key_cache[l]
                values = self.value_cache[l]
                
            seq_len = keys.shape[2]
            
            if seq_len > self.max_cache_size:
                protected_indices = set(range(self.sink_size)) | set(range(seq_len - self.recency_size, seq_len))
                candidate_indices = [i for i in range(seq_len) if i not in protected_indices]
                
                num_keep = self.max_cache_size - self.sink_size - self.recency_size
                
                if num_keep <= 0 or not candidate_indices:
                    keep_indices = sorted(list(protected_indices))
                else:
                    # Randomly sample middle indices to keep
                    random_keep = random.sample(candidate_indices, num_keep)
                    keep_indices = sorted(list(protected_indices) + random_keep)
                    
                device = keys.device
                keep_tensor = torch.tensor(keep_indices, device=device)
                
                pruned_keys = keys[:, :, keep_tensor, :]
                pruned_values = values[:, :, keep_tensor, :]
                
                if hasattr(self, "layers") and len(self.layers) > 0:
                    self.layers[l].keys = pruned_keys
                    self.layers[l].values = pruned_values
                else:
                    self.key_cache[l] = pruned_keys
                    self.value_cache[l] = pruned_values
