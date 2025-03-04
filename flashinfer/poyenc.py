import torch
from einops import rearrange, repeat

def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
    key_leftpad=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )

def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool = False,
    window_left: int = -1,
    sm_scale: float = 0.0,
    logits_soft_cap: float = 0.0
) -> torch.Tensor:
    if causal:
        window_size = (window_left, 0)
    else:
        window_size = (-1, -1)

    head_dim = query.shape[2]
    seqlen_q = query.shape[0]
    seqlen_k = key.shape[0]

    attn_weights = sm_scale * torch.einsum("qhd,khd->hqk", query.float(), key.float())
    if 0 < logits_soft_cap:
        attn_weights = logits_soft_cap * torch.tanh(attn_weights / logits_soft_cap)
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            device=query.device,
        )
        attn_weights.masked_fill_(local_mask, float("-inf"))
    attn_weights = torch.softmax(attn_weights, dim=-1)
    if window_size[0] >= 0 or window_size[1] >= 0:
        attn_weights = attn_weights.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value.float())
    return out.to(query)

def find_mismatches(A, B, atol=1e-5, rtol=1e-5):
    # Compute the absolute difference
    diff = torch.abs(A - B)

    # Compute the relative difference (avoid division by zero)
    rel_diff = diff / torch.abs(B).clamp(min=1e-8)

    # Identify mismatches (where both atol and rtol conditions fail)
    mismatches = (diff > atol) & (rel_diff > rtol)

    # Get mismatch positions
    mismatch_positions = torch.nonzero(mismatches, as_tuple=False)

    return mismatch_positions

def validate_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    causal: bool = False,
    logits_soft_cap: float = 0.0,
    sm_scale: float = 1.0,
    window_left: int = -1,
    kv_layout: str = "NHD"
):
    batch_size = qo_indptr.size(0) - 1
    head_dim = query.size(-1)
    dtype = query.dtype
    num_kv_heads = key.size(1) if kv_layout == "HND" else key.size(2)

    for i in range(batch_size):
            perm_dims = [0, 2, 1, 3] if kv_layout == "HND" else [0, 1, 2, 3]
            perm_dims_last = [1, 0, 2] if kv_layout == "HND" else [0, 1, 2]
            qi = query[qo_indptr[i] : qo_indptr[i + 1]]
            used_kv_indices = kv_indices[kv_indptr[i] : kv_indptr[i + 1]]
            ki = torch.cat(
            [
                key[used_kv_indices[:-1]]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    key[
                        used_kv_indices[-1], :, : kv_last_page_len[i]
                    ]
                    if kv_layout == "HND"
                    else key[
                        used_kv_indices[-1], : kv_last_page_len[i], :
                    ]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
            )
            vi = torch.cat(
                [
                    value[used_kv_indices[:-1]]
                    .permute(*perm_dims)
                    .reshape(-1, num_kv_heads, head_dim),
                    (
                        value[
                            used_kv_indices[-1], :, : kv_last_page_len[i]
                        ]
                        if kv_layout == "HND"
                        else value[
                            used_kv_indices[-1], : kv_last_page_len[i], :
                        ]
                    )
                    .permute(*perm_dims_last)
                    .reshape(-1, num_kv_heads, head_dim),
                ],
                dim=0,
            )

            rtol, atol = (1e-3, 1e-3) if dtype == torch.float16 else (1e-3, 1e-3)
    
            o_ref_i = ref_masked_attention(qi, ki, vi, 
                                        causal=causal, 
                                        logits_soft_cap=logits_soft_cap,
                                        sm_scale=sm_scale)
            o_i = out[qo_indptr[i] : qo_indptr[i + 1]]

            mismatch = find_mismatches(o_i, o_ref_i, atol, rtol)
            if 0 < mismatch.size(0):
                return False

    return True
