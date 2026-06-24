import torch
from transformers.cache_utils import DynamicCache

class StreamingCache(DynamicCache):
    def __init__(self, max_cache_size: int, sink_size: int = 4):
        """
        StreamingLLM online KV eviction cache.
        
        Args:
            max_cache_size: maximum number of tokens to keep in cache (S + W).
            sink_size: number of initial tokens to keep as attention sinks (S).
        """
        super().__init__()
        self.max_cache_size = max_cache_size
        self.sink_size = sink_size
        self.recency_size = max_cache_size - sink_size
        
        if self.recency_size <= 0:
            raise ValueError(f"max_cache_size ({max_cache_size}) must be larger than sink_size ({sink_size}).")
            
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
        Prunes the KV cache to max_cache_size by keeping attention sinks and the recency window.
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
                sink_keys = keys[:, :, :self.sink_size, :]
                recency_keys = keys[:, :, -self.recency_size:, :]
                
                sink_values = values[:, :, :self.sink_size, :]
                recency_values = values[:, :, -self.recency_size:, :]
                
                # Concatenate along sequence dimension (dim=2)
                pruned_keys = torch.cat([sink_keys, recency_keys], dim=2)
                pruned_values = torch.cat([sink_values, recency_values], dim=2)
                
                # Overwrite the cache tensors for this layer (supporting both transformers v4 and v5)
                if hasattr(self, "layers") and len(self.layers) > 0:
                    self.layers[l].keys = pruned_keys
                    self.layers[l].values = pruned_values
                else:
                    self.key_cache[l] = pruned_keys
                    self.value_cache[l] = pruned_values
