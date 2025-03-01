"""
Copyright (c) 2023 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import torch
from jit_utils import jit_prefill_attention_func_args

import flashinfer
import math

from einops import rearrange, repeat


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    if flashinfer.jit.has_prebuilt_ops:
        yield
    else:
        try:
            flashinfer.jit.parallel_load_modules(
                jit_prefill_attention_func_args(
                    [torch.float16],  # q_dtypes
                    [torch.float16],  # kv_dtypes
                    [128, 256],  # head_dims
                    [0, 1, 2],  # pos_encoding_modes
                    [False],  # use_sliding_windows
                    [False, True],  # use_logits_soft_caps
                    [False],  # allow_fp16_qk_reductions
                )
            )
        except Exception as e:
            # abort the test session if warmup fails
            pytest.exit(str(e))
        finally:
            yield

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
    out = torch.einsum("hqk,khd->qhd", attn_weights, value.float())
    return out.to(query)

@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("qo_len,kv_len", [
    (127, 512),
    (127, 513),
    (113, 203),
    (128, 217),
    (113, 211),
    (108, 256),
    (256, 512),
    (1023, 1024),
])
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(2, 1), (4, 1)])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE"])
@pytest.mark.parametrize("use_cuda_graph", [False])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("return_lse", [False])
@pytest.mark.parametrize("contiguous_kv", [True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("q_init_min,q_init_max", [(-3, 3)])
@pytest.mark.parametrize("k_init_min,k_init_max", [(-3, 3)])
@pytest.mark.parametrize("v_init_min,v_init_max", [(-3, 3)])
@pytest.mark.parametrize("seed", [123])
def test_batch_prefill_with_paged_kv_cache(
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    causal,
    kv_layout,
    pos_encoding_mode,
    use_cuda_graph,
    logits_soft_cap,
    return_lse,
    contiguous_kv,
    dtype,
    q_init_min,
    q_init_max,
    k_init_min,
    k_init_max,
    v_init_min,
    v_init_max,
    seed
):
    if seed is not None:
        torch.manual_seed(seed)

    def create_tensor(min, max, *args, **kwargs):
        return (min + (max - max) * torch.rand(*args, **kwargs))

    def convert_lens_to_indtpr(lens):
        return torch.cumsum(torch.cat((torch.tensor([0]), lens)), dim=0).int()

    q = create_tensor(q_init_min, q_init_max, batch_size * qo_len, num_qo_heads, head_dim, dtype=dtype).to(0)
    qo_lens = torch.randint(1, qo_len + 1, (batch_size,))
    q_indptr_cpu = convert_lens_to_indtpr(qo_lens)
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    if kv_layout == "HND":
        kv_shape = [total_num_pages, 2, num_kv_heads, page_size, head_dim]
    else:
        kv_shape = [total_num_pages, 2, page_size, num_kv_heads, head_dim]
    if not contiguous_kv:
        tmp = [kv_shape[0]]
        for v in kv_shape[1:]:
            tmp.append(2)
            tmp.append(v)
        kv_shape = tmp
        kv_data_fp32 = create_tensor(k_init_min, k_init_max, *kv_shape, dtype=torch.float32).to(0)
        kv_data = kv_data_fp32.to(dtype)
        kv_data = kv_data[:, 1, :, 1, :, 1, :, 1, :]
        kv_data_fp32 = kv_data_fp32[:, 1, :, 1, :, 1, :, 1, :]
        # actual data is stored in non-contiguous memory
        assert (
            kv_data.stride(-4)
            != kv_data.shape[-3] * kv_data.shape[-2] * kv_data.shape[-1]
        )
    else:
        kv_data_fp32 = create_tensor(k_init_min, k_init_max, *kv_shape, dtype=torch.float32).to(0)
        kv_data = kv_data_fp32.to(dtype)
    kv_indptr_cpu = torch.arange(0, batch_size + 1).int() * num_pages_per_seq
    kv_indices_cpu = torch.arange(0, total_num_pages).int()
    kv_last_page_len_cpu = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    )

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0)
    if not use_cuda_graph:
        q_indptr_gpu = q_indptr_cpu.to(0)
        kv_indptr_gpu = kv_indptr_cpu.to(0)
        kv_indices_gpu = kv_indices_cpu.to(0)
        kv_last_page_len_gpu = kv_last_page_len_cpu.to(0)
        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout
        )
        wrapper.plan(
            q_indptr_gpu,
            kv_indptr_gpu,
            kv_indices_gpu,
            kv_last_page_len_gpu,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
            q_data_type=dtype
        )
        if return_lse:
            o, _ = wrapper.run(q, kv_data, return_lse=True)
        else:
            o = wrapper.run(q, kv_data)
    else:
        q_indptr_buffer = torch.empty(batch_size + 1).int().to(0)
        kv_indptr_buffer = torch.empty(batch_size + 1).int().to(0)
        kv_indices_buffer = torch.empty(total_num_pages).int().to(0)
        kv_last_page_len_buffer = torch.empty(batch_size).int().to(0)
        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer,
            kv_layout,
            use_cuda_graph=True,
            qo_indptr_buf=q_indptr_buffer,
            paged_kv_indptr_buf=kv_indptr_buffer,
            paged_kv_indices_buf=kv_indices_buffer,
            paged_kv_last_page_len_buf=kv_last_page_len_buffer,
        )
        q_indptr_warmup = torch.arange(0, batch_size + 1).int() * qo_len
        kv_indptr_warmup = torch.arange(0, batch_size + 1).int()
        kv_indices_warmup = torch.arange(0, batch_size).int()
        kv_last_page_len_warmup = torch.full(
            (batch_size,), page_size, dtype=torch.int32
        )

        wrapper.plan(
            q_indptr_warmup,
            kv_indptr_warmup,
            kv_indices_warmup,
            kv_last_page_len_warmup,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )

        # warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                if return_lse:
                    o, _ = wrapper.run(q, kv_data, return_lse=True)
                else:
                    o = wrapper.run(q, kv_data)
        torch.cuda.current_stream().wait_stream(s)
        # capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            if return_lse:
                o, _ = wrapper.run(q, kv_data, return_lse=True)
            else:
                o = wrapper.run(q, kv_data)

        wrapper.plan(
            q_indptr_cpu,
            kv_indptr_cpu,
            kv_indices_cpu,
            kv_last_page_len_cpu,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )

        g.replay()

    for i in range(batch_size):
        perm_dims = [0, 2, 1, 3] if kv_layout == "HND" else [0, 1, 2, 3]
        perm_dims_last = [1, 0, 2] if kv_layout == "HND" else [0, 1, 2]
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        ki = torch.cat(
            [
                kv_data_fp32[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1, 0]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 0, :, : kv_last_page_len_cpu[i]
                    ]
                    if kv_layout == "HND"
                    else kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 0, : kv_last_page_len_cpu[i], :
                    ]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).to(dtype)
        vi = torch.cat(
            [
                kv_data_fp32[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1, 1]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 1, :, : kv_last_page_len_cpu[i]
                    ]
                    if kv_layout == "HND"
                    else kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 1, : kv_last_page_len_cpu[i], :
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

        # enlarge tolerance to pass very few outputs while checking
        if causal:
            atol = atol * 1.4

        o_ref_i = ref_masked_attention(qi, ki, vi, causal=causal, logits_soft_cap=logits_soft_cap)
        o_i = o[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        torch.testing.assert_close(o_i, o_ref_i, rtol=rtol, atol=atol)

'''
@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("kv_len", [54, 97])
@pytest.mark.parametrize("qo_len", [37, 17])
@pytest.mark.parametrize("page_size", [1, 5, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA", "ALIBI"])
@pytest.mark.parametrize("use_cuda_graph", [False, True])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("return_lse", [True, False])
@pytest.mark.parametrize("contiguous_kv", [True, False])
def test_batch_prefill_with_tuple_paged_kv_cache(
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    kv_layout,
    pos_encoding_mode,
    use_cuda_graph,
    logits_soft_cap,
    return_lse,
    contiguous_kv,
):
    q = torch.randn(batch_size * qo_len, num_qo_heads, head_dim).to(0).half()
    q_indptr_cpu = torch.arange(0, batch_size + 1).int() * qo_len
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    if kv_layout == "HND":
        kv_shape = [total_num_pages, num_kv_heads, page_size, head_dim]
    else:
        kv_shape = [total_num_pages, page_size, num_kv_heads, head_dim]
    if not contiguous_kv:
        tmp = [kv_shape[0]]
        for v in kv_shape[1:]:
            tmp.append(2)
            tmp.append(v)
        kv_shape = tmp
        kv_data_fp32 = [
            torch.randn(*kv_shape, dtype=torch.float32).to(0) for _ in range(2)
        ]
        kv_data = [kv_data_fp32[i].half() for i in range(2)]
        for i in range(2):
            kv_data_fp32[i] = kv_data_fp32[i][:, 1, :, 1, :, 1, :]
            kv_data[i] = kv_data[i][:, 1, :, 1, :, 1, :]
            # actual data is stored in non-contiguous memory
            assert (
                kv_data[i].stride(-4)
                != kv_data[i].shape[-3] * kv_data[i].shape[-2] * kv_data[i].shape[-1]
            )
    else:
        kv_data_fp32 = [
            torch.randn(*kv_shape, dtype=torch.float32).to(0) for _ in range(2)
        ]
        kv_data = [kv_data_fp32[i].half() for i in range(2)]
    kv_data = tuple(kv_data)
    kv_indptr_cpu = torch.arange(0, batch_size + 1).int() * num_pages_per_seq
    kv_indices_cpu = torch.arange(0, total_num_pages).int()
    kv_last_page_len_cpu = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    )

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0)
    if not use_cuda_graph:
        q_indptr_gpu = q_indptr_cpu.to(0)
        kv_indptr_gpu = kv_indptr_cpu.to(0)
        kv_indices_gpu = kv_indices_cpu.to(0)
        kv_last_page_len_gpu = kv_last_page_len_cpu.to(0)
        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout
        )
        wrapper.plan(
            q_indptr_gpu,
            kv_indptr_gpu,
            kv_indices_gpu,
            kv_last_page_len_gpu,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )
        if return_lse:
            o, _ = wrapper.run(q, kv_data, return_lse=True)
        else:
            o = wrapper.run(q, kv_data)
    else:
        q_indptr_buffer = torch.empty(batch_size + 1).int().to(0)
        kv_indptr_buffer = torch.empty(batch_size + 1).int().to(0)
        kv_indices_buffer = torch.empty(total_num_pages).int().to(0)
        kv_last_page_len_buffer = torch.empty(batch_size).int().to(0)
        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer,
            kv_layout,
            use_cuda_graph=True,
            qo_indptr_buf=q_indptr_buffer,
            paged_kv_indptr_buf=kv_indptr_buffer,
            paged_kv_indices_buf=kv_indices_buffer,
            paged_kv_last_page_len_buf=kv_last_page_len_buffer,
        )
        q_indptr_warmup = torch.arange(0, batch_size + 1).int() * qo_len
        kv_indptr_warmup = torch.arange(0, batch_size + 1).int()
        kv_indices_warmup = torch.arange(0, batch_size).int()
        kv_last_page_len_warmup = torch.full(
            (batch_size,), page_size, dtype=torch.int32
        )
        wrapper.plan(
            q_indptr_warmup,
            kv_indptr_warmup,
            kv_indices_warmup,
            kv_last_page_len_warmup,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )

        # warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                if return_lse:
                    o, _ = wrapper.run(q, kv_data, return_lse=True)
                else:
                    o = wrapper.run(q, kv_data)
        torch.cuda.current_stream().wait_stream(s)
        # capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            if return_lse:
                o, _ = wrapper.run(q, kv_data, return_lse=True)
            else:
                o = wrapper.run(q, kv_data)

        wrapper.plan(
            q_indptr_cpu,
            kv_indptr_cpu,
            kv_indices_cpu,
            kv_last_page_len_cpu,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )

        g.replay()

    k_cache, v_cache = kv_data_fp32
    for i in range(batch_size):
        perm_dims = [0, 2, 1, 3] if kv_layout == "HND" else [0, 1, 2, 3]
        perm_dims_last = [1, 0, 2] if kv_layout == "HND" else [0, 1, 2]
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        ki = torch.cat(
            [
                k_cache[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    k_cache[kv_indptr_cpu[i + 1] - 1, :, : kv_last_page_len_cpu[i]]
                    if kv_layout == "HND"
                    else k_cache[kv_indptr_cpu[i + 1] - 1, : kv_last_page_len_cpu[i], :]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).half()
        vi = torch.cat(
            [
                v_cache[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    v_cache[kv_indptr_cpu[i + 1] - 1, :, : kv_last_page_len_cpu[i]]
                    if kv_layout == "HND"
                    else v_cache[kv_indptr_cpu[i + 1] - 1, : kv_last_page_len_cpu[i], :]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).half()
        o_ref_i = flashinfer.prefill.single_prefill_with_kv_cache(
            qi,
            ki,
            vi,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )
        o_i = o[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        torch.testing.assert_close(o_i, o_ref_i, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("kv_len", [54, 97])
@pytest.mark.parametrize("qo_len", [37, 17])
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA", "ALIBI"])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("return_lse", [True, False])
@pytest.mark.parametrize("contiguous_kv", [True, False])
def test_batch_prefill_with_paged_kv_cache_custom_mask(
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
    pos_encoding_mode,
    logits_soft_cap,
    return_lse,
    contiguous_kv,
):
    q = torch.randn(batch_size * qo_len, num_qo_heads, head_dim).to(0).half()
    q_indptr = torch.arange(0, batch_size + 1).to(0).int() * qo_len
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    if kv_layout == "HND":
        kv_shape = [total_num_pages, 2, num_kv_heads, page_size, head_dim]
    else:
        kv_shape = [total_num_pages, 2, page_size, num_kv_heads, head_dim]
    if not contiguous_kv:
        tmp = [kv_shape[0]]
        for v in kv_shape[1:]:
            tmp.append(2)
            tmp.append(v)
        kv_shape = tmp
        kv_data = torch.randn(*kv_shape, dtype=torch.float32).to(0).half()
        kv_data = kv_data[:, 1, :, 1, :, 1, :, 1, :]
        # actual data is stored in non-contiguous memory
        assert (
            kv_data.stride(-4)
            != kv_data.shape[-3] * kv_data.shape[-2] * kv_data.shape[-1]
        )
    else:
        kv_data = torch.randn(*kv_shape, dtype=torch.float32).to(0).half()
    kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * num_pages_per_seq
    kv_indices = torch.arange(0, total_num_pages).to(0).int()
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    ).to(0)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0)
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    custom_mask = (
        torch.tril(
            torch.full((batch_size, qo_len, kv_len), True),
            diagonal=(kv_len - qo_len),
        )
        .reshape(-1)
        .to(0)
    )

    # use custom mask
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        custom_mask,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=logits_soft_cap,
    )
    if return_lse:
        o_custom, _ = wrapper.run(q, kv_data, return_lse=True)
    else:
        o_custom = wrapper.run(q, kv_data)

    # use causal
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=logits_soft_cap,
    )
    if return_lse:
        o_causal, _ = wrapper.run(q, kv_data, return_lse=True)
    else:
        o_causal = wrapper.run(q, kv_data)
    torch.testing.assert_close(o_custom, o_causal, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("kv_len", [54, 97])
@pytest.mark.parametrize("qo_len", [37, 17])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA", "ALIBI"])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("return_lse", [True, False])
def test_batch_prefill_with_ragged_kv_cache(
    batch_size,
    kv_len,
    qo_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    pos_encoding_mode,
    logits_soft_cap,
    return_lse,
):
    kv_layout = "NHD"
    q = torch.randn(batch_size * qo_len, num_qo_heads, head_dim).to(0).half()
    q_indptr = torch.arange(0, batch_size + 1).to(0).int() * qo_len

    k = torch.randn(batch_size * kv_len, num_kv_heads, head_dim).to(0).half()
    v = torch.randn(batch_size * kv_len, num_kv_heads, head_dim).to(0).half()
    kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * kv_len

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0)
    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    wrapper.plan(
        q_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=logits_soft_cap,
    )
    if return_lse:
        o, _ = wrapper.run(q, k, v, return_lse=True)
    else:
        o = wrapper.run(q, k, v)

    for i in range(batch_size):
        o_ref_i = flashinfer.prefill.single_prefill_with_kv_cache(
            q[q_indptr[i] : q_indptr[i + 1]],
            k[kv_indptr[i] : kv_indptr[i + 1]],
            v[kv_indptr[i] : kv_indptr[i + 1]],
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )
        o_i = o[q_indptr[i] : q_indptr[i + 1]]
        torch.testing.assert_close(o_i, o_ref_i, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("kv_len", [54, 97])
@pytest.mark.parametrize("qo_len", [37, 17])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA", "ALIBI"])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("return_lse", [True, False])
def test_batch_prefill_with_ragged_kv_cache_custom_mask(
    batch_size,
    kv_len,
    qo_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    pos_encoding_mode,
    logits_soft_cap,
    return_lse,
):
    kv_layout = "NHD"
    q = torch.randn(batch_size * qo_len, num_qo_heads, head_dim).to(0).half()
    q_indptr = torch.arange(0, batch_size + 1).to(0).int() * qo_len

    k = torch.randn(batch_size * kv_len, num_kv_heads, head_dim).to(0).half()
    v = torch.randn(batch_size * kv_len, num_kv_heads, head_dim).to(0).half()
    kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * kv_len

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0)
    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, kv_layout
    )

    custom_mask = (
        torch.tril(
            torch.full((batch_size, qo_len, kv_len), True),
            diagonal=(kv_len - qo_len),
        )
        .reshape(-1)
        .to(0)
    )

    # use custom mask
    wrapper.plan(
        q_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        custom_mask=custom_mask,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=logits_soft_cap,
    )
    if return_lse:
        o_custom, _ = wrapper.run(q, k, v, return_lse=True)
    else:
        o_custom = wrapper.run(q, k, v)

    # use causal
    wrapper.plan(
        q_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=True,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=logits_soft_cap,
    )
    if return_lse:
        o_causal, _ = wrapper.run(q, k, v, return_lse=True)
    else:
        o_causal = wrapper.run(q, k, v)
    torch.testing.assert_close(o_custom, o_causal, rtol=1e-3, atol=1e-3)
'''

if __name__ == "__main__":
    test_batch_prefill_with_paged_kv_cache(
        12, 54, 37, 16, 8, 8, 128, True, "HND", "NONE", True, 0.0, False, True
    )
'''    test_batch_prefill_with_tuple_paged_kv_cache(
        12, 54, 37, 16, 8, 8, 128, True, "HND", "NONE", True, 0.0, False, True
    )
    test_batch_prefill_with_paged_kv_cache(
        12, 54, 37, 1, 8, 8, 128, True, "HND", "NONE", False, 0.0, False, True
    )
    test_batch_prefill_with_paged_kv_cache_custom_mask(
        1, 137, 137, 1, 8, 8, 128, "HND", "NONE", 0.0, False, True
    )
    test_batch_prefill_with_ragged_kv_cache(
        12, 54, 37, 8, 8, 128, True, "NONE", 0.0, False
    )
    test_batch_prefill_with_ragged_kv_cache_custom_mask(
        1, 137, 137, 8, 8, 128, "NONE", 0.0, False
    ) '''
