import torch


def compute_l2_norm(keys: torch.Tensor) -> torch.Tensor:
    """
    Computes the L2 norm across the head dimension.
    keys: shape (batch_size, num_heads, seq_len, head_dim)
    Returns: shape (batch_size, num_heads, seq_len)
    """
    return torch.linalg.norm(keys, dim=-1)


def compute_variance(keys: torch.Tensor) -> torch.Tensor:
    """
    Computes variance of the components across the head dimension.
    keys: shape (batch_size, num_heads, seq_len, head_dim)
    Returns: shape (batch_size, num_heads, seq_len)
    """
    return torch.var(keys, dim=-1)


def compute_shannon_entropy(keys: torch.Tensor) -> torch.Tensor:
    """
    Computes the Shannon entropy across the head dimension of the softmax-normalized key vector.
    keys: shape (batch_size, num_heads, seq_len, head_dim)
    Returns: shape (batch_size, num_heads, seq_len)
    """
    # Cast to float32 to prevent FP16 numerical underflow and NaN propagation
    keys_f32 = keys.float()
    probs = torch.softmax(keys_f32, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    return entropy.to(keys.dtype)


def compute_abs_entropy(keys: torch.Tensor) -> torch.Tensor:
    """
    Computes the Shannon entropy of the absolute-value-normalized key vector.
    Takes absolute values, normalizes to sum to 1, then computes H = -Σ p log p.
    This avoids sign ambiguity in softmax normalization.
    keys: shape (batch_size, num_heads, seq_len, head_dim)
    Returns: shape (batch_size, num_heads, seq_len)
    """
    # Cast to float32 to prevent FP16 numerical underflow and NaN propagation
    keys_f32 = keys.float()
    abs_keys = torch.abs(keys_f32) + 1e-9  # avoid division by zero
    probs = abs_keys / abs_keys.sum(dim=-1, keepdim=True)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    return entropy.to(keys.dtype)


def compute_value_weighted_norm(keys: torch.Tensor, values: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """
    Computes value-weighted key norm: Score = ||k||_2 / (||v||_2^gamma + 1e-9).
    We select tokens with the smallest scores to preserve both small-norm keys
    (which correlate with high attention) and large-norm values (outliers).
    keys: shape (batch_size, num_heads, seq_len, head_dim)
    values: shape (batch_size, num_heads, seq_len, head_dim)
    Returns: shape (batch_size, num_heads, seq_len)
    """
    k_norm = torch.linalg.norm(keys.float(), dim=-1)
    v_norm = torch.linalg.norm(values.float(), dim=-1)
    score = k_norm / (torch.pow(v_norm, gamma) + 1e-9)
    return score.to(keys.dtype)


