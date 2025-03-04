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
    logits_soft_cap: float = 0.0
) -> torch.Tensor:
    if causal:
        window_size = (window_left, 0)
    else:
        window_size = (-1, -1)

    head_dim = query.shape[2]
    seqlen_q = query.shape[0]
    seqlen_k = key.shape[0]
    scale = 1.0 / math.sqrt(head_dim)

    attn_weights = scale * torch.einsum("qhd,khd->hqk", query.float(), key.float())
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

def validate_attention()
    for i in range(batch_size):
            perm_dims = [0, 2, 1, 3] if kv_layout == "HND" else [0, 1, 2, 3]
            perm_dims_last = [1, 0, 2] if kv_layout == "HND" else [0, 1, 2]
            qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
            used_kv_indices = kv_indices_cpu[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1]]
            ki = torch.cat(
                [
                    kv_data_fp32[used_kv_indices[:-1], 0]
                    .permute(*perm_dims)
                    .reshape(-1, num_kv_heads, head_dim),
                    (
                        kv_data_fp32[
                            used_kv_indices[-1], 0, :, : kv_last_page_len_cpu[i]
                        ]
                        if kv_layout == "HND"
                        else kv_data_fp32[
                            used_kv_indices[-1], 0, : kv_last_page_len_cpu[i], :
                        ]
                    )
                    .permute(*perm_dims_last)
                    .reshape(-1, num_kv_heads, head_dim),
                ],
                dim=0,
            ).to(dtype)
            vi = torch.cat(
                [
                    kv_data_fp32[used_kv_indices[:-1], 1]
                    .permute(*perm_dims)
                    .reshape(-1, num_kv_heads, head_dim),
                    (
                        kv_data_fp32[
                            used_kv_indices[-1], 1, :, : kv_last_page_len_cpu[i]
                        ]
                        if kv_layout == "HND"
                        else kv_data_fp32[
                            used_kv_indices[-1], 1, : kv_last_page_len_cpu[i], :
                        ]
                    )
                    .permute(*perm_dims_last)
                    .reshape(-1, num_kv_heads, head_dim),
                ],
                dim=0,
            ).to(dtype)
            # FIXME: add back single_prefill_with_kv_cache() call after finish implementation
            """
            o_ref_i = flashinfer.prefill.single_prefill_with_kv_cache(
                qi,
                ki,
                vi,
                causal=causal,
                pos_encoding_mode=pos_encoding_mode,
                logits_soft_cap=logits_soft_cap,
            )
            """
            # NOTICE: for now, we use ref_masked_attention() as reference function
            assert pos_encoding_mode == 'NONE'
            assert not return_lse
    
            rtol, atol = (1e-3, 1e-3) if dtype == torch.float16 else (1e-2, 1e-2)
    
            o_ref_i = ref_masked_attention(qi, ki, vi, causal=causal, logits_soft_cap=logits_soft_cap)
            o_i = o[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
            torch.testing.assert_close(o_i, o_ref_i, rtol=rtol, atol=atol)
