import torch
from transformers.cache_utils import DynamicCache

class SnapKVCache(DynamicCache):
    def __init__(
        self, 
        max_cache_size: int, 
        sink_size: int = 4, 
        recency_size: int = 32,
        obs_size: int = 32
    ):
        """
        SnapKV online cache eviction baseline.
        
        Args:
            max_cache_size: maximum number of tokens to keep in cache.
            sink_size: number of initial attention sink tokens to protect (S).
            recency_size: number of recent tokens to protect (R).
            obs_size: observation window size (L_obs) to average attentions.
        """
        super().__init__()
        self.max_cache_size = max_cache_size
        self.sink_size = sink_size
        self.recency_size = recency_size
        self.obs_size = obs_size
        
        self.prefill_attentions = None
        self.is_prefill = True
        
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None
    ):
        return super().update(key_states, value_states, layer_idx, cache_kwargs)
        
    def accumulate_attentions(self, attentions):
        """
        Accumulates attention weights during the prefill phase.
        attentions: tuple of tensors, one per layer, each of shape (batch, num_heads, q_len, k_len)
        """
        if self.is_prefill:
            self.prefill_attentions = []
            for l in range(len(attentions)):
                att = attentions[l]
                obs_len = min(self.obs_size, att.shape[2])
                # Average over the last obs_len query tokens to save memory
                scores = att[:, :, -obs_len:, :].mean(dim=-2) # shape: (batch, num_heads, k_len)
                self.prefill_attentions.append(scores)
            
    def evict(self):
        """
        Prunes the KV cache based on the SnapKV strategy.
        At prefill, selects key tokens with the highest average attention weights from 
        the observation window queries.
        At decoding, performs FIFO sliding window eviction on the recency window.
        """
        num_layers = len(self.layers) if hasattr(self, "layers") and len(self.layers) > 0 else len(self.key_cache)
        
        if self.is_prefill:
            self.is_prefill = False
            
            if self.prefill_attentions is None:
                # If no attention scores accumulated, fallback to streaming/fifo eviction
                self._fallback_eviction()
                return
                
            for l in range(num_layers):
                if hasattr(self, "layers") and len(self.layers) > 0:
                    keys = self.layers[l].keys
                    values = self.layers[l].values
                else:
                    keys = self.key_cache[l]
                    values = self.value_cache[l]
                    
                seq_len = keys.shape[2]
                
                if seq_len > self.max_cache_size:
                    # Retrieve the pre-computed scores for layer l: shape (batch, num_heads, k_len)
                    scores = self.prefill_attentions[l]
                    
                    batch_size, num_query_heads, k_len = scores.shape
                    if hasattr(self, "layers") and len(self.layers) > l:
                        num_kv_heads = self.layers[l].keys.shape[1]
                    elif len(self.key_cache) > l:
                        num_kv_heads = self.key_cache[l].shape[1]
                    else:
                        num_kv_heads = num_query_heads
                    
                    if num_query_heads > num_kv_heads:
                        group_size = num_query_heads // num_kv_heads
                        scores = scores.view(batch_size, num_kv_heads, group_size, k_len).mean(dim=2)
                    
                    # Protected indices:
                    # - Sinks: 0 to sink_size
                    # - Recency window / Observation window: seq_len - recency_size to seq_len
                    protected_indices = set(range(self.sink_size)) | set(range(seq_len - self.recency_size, seq_len))
                    candidate_indices = [i for i in range(seq_len) if i not in protected_indices]
                    
                    num_keep = self.max_cache_size - self.sink_size - self.recency_size
                    
                    if num_keep <= 0 or not candidate_indices:
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
                    else:
                        device = keys.device
                        batch_size, num_heads, _ = scores.shape
                        
                        # Candidate scores: shape (batch, num_heads, len(candidate_indices))
                        candidate_scores = scores[:, :, candidate_indices]
                        
                        # Top-k indices
                        top_k_indices = torch.topk(candidate_scores, num_keep, largest=True).indices
                        
                        # Map to real sequence indices
                        candidate_tensor = torch.tensor(candidate_indices, device=device).view(1, 1, -1).expand(batch_size, num_heads, -1)
                        top_real_indices = torch.gather(candidate_tensor, dim=2, index=top_k_indices)
                        
                        # Protected indices tensors
                        sink_tensor = torch.arange(self.sink_size, device=device).view(1, 1, -1).expand(batch_size, num_heads, -1)
                        recency_tensor = torch.arange(seq_len - self.recency_size, seq_len, device=device).view(1, 1, -1).expand(batch_size, num_heads, -1)
                        
                        # Combine keep indices: shape (batch, num_heads, max_cache_size)
                        keep_indices = torch.cat([sink_tensor, top_real_indices, recency_tensor], dim=-1)
                        keep_indices = torch.sort(keep_indices, dim=-1).values
                        
                    # Slice key-values
                    head_dim = keys.shape[-1]
                    gather_index = keep_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    pruned_keys = torch.gather(keys, dim=2, index=gather_index)
                    pruned_values = torch.gather(values, dim=2, index=gather_index)
                    
                    if hasattr(self, "layers") and len(self.layers) > 0:
                        self.layers[l].keys = pruned_keys
                        self.layers[l].values = pruned_values
                    else:
                        self.key_cache[l] = pruned_keys
                        self.value_cache[l] = pruned_values
            # Free memory by clearing prefill attentions
            self.prefill_attentions = None
        else:
            # During decoding, new tokens are appended.
            # To keep the cache within max_cache_size, slide the recency window by evicting 
            # the oldest non-sink token in the recency window.
            for l in range(num_layers):
                if hasattr(self, "layers") and len(self.layers) > 0:
                    keys = self.layers[l].keys
                    values = self.layers[l].values
                else:
                    keys = self.key_cache[l]
                    values = self.value_cache[l]
                    
                seq_len = keys.shape[2]
                if seq_len > self.max_cache_size:
                    # Index of oldest token in the recency window to evict is: max_cache_size - recency_size.
                    # We retain range(0, max_cache_size - recency_size) and range(max_cache_size - recency_size + 1, seq_len)
                    keep_idx = list(range(0, self.max_cache_size - self.recency_size)) + list(range(self.max_cache_size - self.recency_size + 1, seq_len))
                    
                    pruned_keys = keys[:, :, keep_idx, :]
                    pruned_values = values[:, :, keep_idx, :]
                    
                    if hasattr(self, "layers") and len(self.layers) > 0:
                        self.layers[l].keys = pruned_keys
                        self.layers[l].values = pruned_values
                    else:
                        self.key_cache[l] = pruned_keys
                        self.value_cache[l] = pruned_values
                        
    def _fallback_eviction(self):
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
                recency_keys = keys[:, :, -(self.max_cache_size - self.sink_size):, :]
                
                sink_values = values[:, :, :self.sink_size, :]
                recency_values = values[:, :, -(self.max_cache_size - self.sink_size):, :]
                
                pruned_keys = torch.cat([sink_keys, recency_keys], dim=2)
                pruned_values = torch.cat([sink_values, recency_values], dim=2)
                
                if hasattr(self, "layers") and len(self.layers) > 0:
                    self.layers[l].keys = pruned_keys
                    self.layers[l].values = pruned_values
                else:
                    self.key_cache[l] = pruned_keys
                    self.value_cache[l] = pruned_values
