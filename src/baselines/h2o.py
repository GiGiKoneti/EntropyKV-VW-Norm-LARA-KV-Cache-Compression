import torch
from transformers.cache_utils import DynamicCache

class H2OCache(DynamicCache):
    def __init__(self, max_cache_size: int, sink_size: int = 4, recency_size: int = 32):
        """
        H2O (Heavy Hitter Oracle) KV cache eviction.
        
        Args:
            max_cache_size: maximum number of tokens to keep in cache.
            sink_size: number of initial attention sink tokens to protect (S).
            recency_size: number of recent tokens to protect (R).
        """
        super().__init__()
        self.max_cache_size = max_cache_size
        self.sink_size = sink_size
        self.recency_size = recency_size
        
        # List to store rolling sum of attention weights per layer
        # Each element is a tensor of shape (batch_size, num_heads, seq_len)
        self.attention_accumulators = []
        
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None
    ):
        # H2O updates cache normally. Eviction is triggered manually after attention weights are observed.
        return super().update(key_states, value_states, layer_idx, cache_kwargs)
        
    def accumulate_attentions(self, attentions):
        """
        Accumulates attention weights for each layer.
        attentions: tuple of tensors, one per layer, each of shape (batch, num_heads, 1, seq_len_k)
        """
        num_layers = len(attentions)
        
        if len(self.attention_accumulators) == 0:
            for l in range(num_layers):
                # Sum along the query length dimension (which is 1 during decoding, but >1 during prefill)
                att_step = attentions[l].sum(dim=2) # shape: (batch, num_query_heads, seq_len_k)
                
                batch_size, num_query_heads, seq_len_k = att_step.shape
                if hasattr(self, "layers") and len(self.layers) > l:
                    num_kv_heads = self.layers[l].keys.shape[1]
                elif len(self.key_cache) > l:
                    num_kv_heads = self.key_cache[l].shape[1]
                else:
                    num_kv_heads = num_query_heads
                
                if num_query_heads > num_kv_heads:
                    group_size = num_query_heads // num_kv_heads
                    att_step = att_step.view(batch_size, num_kv_heads, group_size, seq_len_k).mean(dim=2)
                
                self.attention_accumulators.append(att_step)
        else:
            for l in range(num_layers):
                att_step = attentions[l].sum(dim=2) # shape: (batch, num_query_heads, seq_len_k)
                
                batch_size, num_query_heads, seq_len_k = att_step.shape
                if hasattr(self, "layers") and len(self.layers) > l:
                    num_kv_heads = self.layers[l].keys.shape[1]
                elif len(self.key_cache) > l:
                    num_kv_heads = self.key_cache[l].shape[1]
                else:
                    num_kv_heads = num_query_heads
                
                if num_query_heads > num_kv_heads:
                    group_size = num_query_heads // num_kv_heads
                    att_step = att_step.view(batch_size, num_kv_heads, group_size, seq_len_k).mean(dim=2)
                
                prev_accum = self.attention_accumulators[l]
                
                # Check for matching shapes
                prev_len = prev_accum.shape[-1]
                updated_accum = prev_accum + att_step[:, :, :prev_len]
                new_token_score = att_step[:, :, prev_len:]
                
                self.attention_accumulators[l] = torch.cat([updated_accum, new_token_score], dim=-1)
                
    def evict(self):
        """
        Prunes the KV cache to max_cache_size by keeping attention sinks, recency window, 
        and the top heavy hitters based on accumulated attention scores.
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
                # Identify protected and candidate ranges
                # Sinks are index [0 : sink_size]
                # Recency window is [seq_len - recency_size : seq_len]
                protected_indices = set(range(self.sink_size)) | set(range(seq_len - self.recency_size, seq_len))
                candidate_indices = [i for i in range(seq_len) if i not in protected_indices]
                
                num_heavy_hitters = self.max_cache_size - self.sink_size - self.recency_size
                
                if num_heavy_hitters <= 0 or not candidate_indices:
                    # Fallback to StreamingLLM if heavy hitter budget is zero or negative
                    keep_indices = sorted(list(protected_indices))
                    pruned_keys = keys[:, :, keep_indices, :]
                    pruned_values = values[:, :, keep_indices, :]
                    
                    if hasattr(self, "layers") and len(self.layers) > 0:
                        self.layers[l].keys = pruned_keys
                        self.layers[l].values = pruned_values
                    else:
                        self.key_cache[l] = pruned_keys
                        self.value_cache[l] = pruned_values
                        
                    if len(self.attention_accumulators) > l:
                        self.attention_accumulators[l] = self.attention_accumulators[l][:, :, keep_indices]
                else:
                    accum = self.attention_accumulators[l]
                    batch_size, num_heads, _ = accum.shape
                    device = keys.device
                    
                    # candidate scores: shape (batch, num_heads, len(candidate_indices))
                    candidate_scores = accum[:, :, candidate_indices]
                    
                    # Top heavy hitters indices: shape (batch, num_heads, num_heavy_hitters)
                    top_k_indices = torch.topk(candidate_scores, num_heavy_hitters).indices
                    
                    # Map to real indices
                    candidate_tensor = torch.tensor(candidate_indices, device=device).view(1, 1, -1).expand(batch_size, num_heads, -1)
                    top_real_indices = torch.gather(candidate_tensor, dim=2, index=top_k_indices)
                    
                    # Protected indices tensors
                    sink_tensor = torch.arange(self.sink_size, device=device).view(1, 1, -1).expand(batch_size, num_heads, -1)
                    recency_tensor = torch.arange(seq_len - self.recency_size, seq_len, device=device).view(1, 1, -1).expand(batch_size, num_heads, -1)
                    
                    # Combine keep indices: shape (batch, num_heads, max_cache_size)
                    keep_indices = torch.cat([sink_tensor, top_real_indices, recency_tensor], dim=-1)
                    keep_indices = torch.sort(keep_indices, dim=-1).values
                    
                    # Gather key-values and accumulators in parallel across heads
                    head_dim = keys.shape[-1]
                    gather_index = keep_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    pruned_keys = torch.gather(keys, dim=2, index=gather_index)
                    pruned_values = torch.gather(values, dim=2, index=gather_index)
                    
                    # Gather rolling accumulators
                    new_accum = torch.gather(accum, dim=2, index=keep_indices)
                    
                    if hasattr(self, "layers") and len(self.layers) > 0:
                        self.layers[l].keys = pruned_keys
                        self.layers[l].values = pruned_values
                    else:
                        self.key_cache[l] = pruned_keys
                        self.value_cache[l] = pruned_values
                        
                    self.attention_accumulators[l] = new_accum
