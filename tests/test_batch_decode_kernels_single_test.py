"""
run decoder with dummy data for a single batch input without checking results
This is for profiling purpose only
"""

import torch
import flashinfer
import argparse
def test_batch_decode_with_paged_kv_cache(
    batch_size,
    kv_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
    pos_encoding_mode,
    logits_soft_cap,
    q_dtype,
    kv_dtype,
    contiguous_kv,
    warmup
):
    q = torch.randn(batch_size, num_qo_heads, head_dim).to(0).to(q_dtype)
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
        kv_data_fp32 = torch.randn(*kv_shape, dtype=torch.float32).to(0)
        kv_data = kv_data_fp32.to(kv_dtype)
        kv_data = kv_data[:, 1, :, 1, :, 1, :, 1, :]
        kv_data_fp32 = kv_data_fp32[:, 1, :, 1, :, 1, :, 1, :]
        # actual data is stored in non-contiguous memory
        assert (
            kv_data.stride(-4)
            != kv_data.shape[-3] * kv_data.shape[-2] * kv_data.shape[-1]
        )
    else:
        kv_data_fp32 = torch.randn(*kv_shape, dtype=torch.float32).to(0)
        kv_data = kv_data_fp32.to(kv_dtype)
    kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * num_pages_per_seq
    kv_indices = torch.arange(0, total_num_pages).to(0).int()
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    ).to(0)

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(0)
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        logits_soft_cap=logits_soft_cap,
        pos_encoding_mode=pos_encoding_mode,
        data_type=kv_dtype,
        q_data_type=q_dtype,
    )
    
    for _ in range(warmup):
        print("=======warmp run()=======")
        o = wrapper.run(q, kv_data)
        
    o = wrapper.run(q, kv_data)



if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Run test_batch_decode_with_paged_kv_cache with arguments")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (default: 256)")
    parser.add_argument("--kv_len", type=int, default=128, help="KV length (default: 128)")
    parser.add_argument("--page_size", type=int, default=8, help="Page size (default: 8)")
    parser.add_argument("--num_kv_heads", type=int, default=1, help="Number of KV heads (default: 8)")
    parser.add_argument("--num_qo_heads", type=int, default=8, help="Number of QO heads (default: 8)")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension (default: 256)")
    parser.add_argument("--warmup", type=int, default=10, help="warmup (default: 2)")
    args = parser.parse_args()

    test_batch_decode_with_paged_kv_cache(
        args.batch_size,
        args.kv_len,
        args.page_size,
        args.num_kv_heads,
        args.num_qo_heads,
        args.head_dim,
        "NHD",
        "NONE",
        0.0,
        torch.float16,
        torch.float16,
        True,
        args.warmup
    )
    print(" BatchDecodeWithPagedKVCacheDispatched "+" ".join(f"{key}={value}" for key, value in vars(args).items()))
