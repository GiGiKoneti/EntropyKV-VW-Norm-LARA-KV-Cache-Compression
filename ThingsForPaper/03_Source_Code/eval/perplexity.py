import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

def compute_perplexity(
    model, 
    input_ids, 
    seq_len=1024, 
    stride=512, 
    device="cpu", 
    cache_class=None, 
    cache_kwargs=None,
    mode="chunk"
):
    """
    Computes sliding-window perplexity with KV cache eviction.
    
    Standard evaluation protocol matching the H2O / SnapKV / PyramidKV literature:
    
    For each overlapping window of `seq_len` tokens:
    - 'chunk' mode: Feed the entire window in one forward pass. After the forward pass,
      evict the cache down to max_cache_size. The eviction has no effect on the logits
      within this window (they were already computed), but this mode is useful for
      measuring SnapKV-style prefill-only eviction and as a fast sanity check.
      
    - 'token' mode: Feed the window token-by-token through a persistent cache. Eviction
      fires after each token. Each subsequent token's prediction is conditioned ONLY on
      what the cache retains. This is the gold-standard evaluation for online eviction
      methods like EntropyKV and H2O — the eviction directly affects prediction quality.
    
    The cache is RESET between windows to keep evaluation independent of window ordering,
    matching the standard sliding-window PPL protocol.
    
    Args:
        model: HuggingFace AutoModelForCausalLM.
        input_ids: torch.LongTensor of shape (1, total_tokens).
        seq_len: Sliding window size in tokens.
        stride: Step size between windows.
        device: Device to run on ('cpu', 'cuda', 'mps').
        cache_class: Class of the custom cache (e.g. EntropyCache, H2OCache).
                     If None, standard full-cache forward pass is used.
        cache_kwargs: Dict of init arguments for cache_class.
        mode: 'chunk' (fast, prefill-style) or 'token' (exact, autoregressive).
    """
    model.eval()
    seq_len_total = input_ids.size(1)
    
    nlls = []
    prev_end_loc = 0
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    
    # Determine if the cache needs attention scores (e.g., for H2O, SnapKV)
    output_attentions = cache_class is not None and hasattr(cache_class, "accumulate_attentions")
    
    # Iterate over overlapping windows
    pbar = tqdm(range(0, seq_len_total, stride), desc=f"Evaluating PPL ({mode} mode)")
    for begin_loc in pbar:
        end_loc = min(begin_loc + seq_len, seq_len_total)
        trg_len = end_loc - prev_end_loc  # number of new (non-overlapping) tokens
        
        # Fresh cache per window
        cache = None
        if cache_class is not None:
            kwargs = cache_kwargs or {}
            cache = cache_class(**kwargs)
            
        chunk_ids = input_ids[:, begin_loc:end_loc].to(device)
        
        if mode == "token":
            # =================================================================
            # TOKEN MODE: autoregressive decoding within the window.
            # Each token is fed one at a time. The cache persists within the
            # window and eviction fires after each token insertion. Subsequent
            # token predictions are conditioned only on what the cache retains.
            # =================================================================
            for t in range(chunk_ids.size(1) - 1):
                input_token = chunk_ids[:, t:t+1]
                target_token = chunk_ids[:, t+1]
                
                with torch.no_grad():
                    outputs = model(
                        input_token, 
                        past_key_values=cache, 
                        use_cache=True, 
                        output_attentions=output_attentions
                    )
                cache = outputs.past_key_values
                
                # Accumulate attention weights if supported (e.g. for H2O)
                if hasattr(cache, "accumulate_attentions") and outputs.attentions is not None:
                    cache.accumulate_attentions(outputs.attentions)
                
                # Online cache eviction
                if hasattr(cache, "evict"):
                    cache.evict()
                    
                # Only score tokens in the non-overlapping region
                global_pos = begin_loc + t + 1
                if global_pos >= prev_end_loc:
                    logits = outputs.logits.view(-1, outputs.logits.size(-1))
                    loss = loss_fct(logits, target_token.view(-1))
                    nlls.extend(loss.tolist())
        
        elif mode == "chunk":
            # =================================================================
            # CHUNK MODE: process the full window in one forward pass.
            # Eviction runs post-forward-pass. This doesn't affect this
            # window's logits but exercises prefill-only eviction logic.
            # For methods that don't use online eviction (like SnapKV's prefill
            # selection), this is the natural evaluation mode.
            # =================================================================
            with torch.no_grad():
                outputs = model(
                    chunk_ids, 
                    past_key_values=cache, 
                    use_cache=True, 
                    output_attentions=output_attentions
                )
                
            # Accumulate attention weights if supported (e.g. for H2O, SnapKV)
            if hasattr(cache, "accumulate_attentions") and outputs.attentions is not None:
                cache.accumulate_attentions(outputs.attentions)
                
            # Post-forward eviction (prefill-only style)
            if hasattr(cache, "evict"):
                cache.evict()
                
            # Logits: predict token[i+1] from position i
            logits = outputs.logits[0, :-1, :]
            targets = chunk_ids[0, 1:]
            losses = loss_fct(logits, targets)
            
            # Only score tokens in the non-overlapping region
            for idx in range(len(losses)):
                global_pos = begin_loc + idx + 1
                if global_pos >= prev_end_loc:
                    nlls.append(losses[idx].item())
        
        else:
            raise ValueError(f"Unknown perplexity mode: {mode}. Use 'chunk' or 'token'.")
            
        prev_end_loc = end_loc
        if end_loc == seq_len_total:
            break
            
    if not nlls:
        return float("nan")
        
    mean_nll = np.mean(nlls)
    ppl = np.exp(mean_nll)
    return ppl
