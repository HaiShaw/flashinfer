import pytest
import torch
import flashinfer


# @pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("batch_size", [256])
# @pytest.mark.parametrize("kv_len", [54, 97, 512])
@pytest.mark.parametrize("kv_len", [54])
# @pytest.mark.parametrize("page_size", [1, 8, 16])
@pytest.mark.parametrize("page_size", [8])
# @pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_kv_heads", [8])
# @pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("num_qo_heads", [8])
# @pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("head_dim", [128])
# @pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("kv_layout", ["NHD"])
# @pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA", "ALIBI"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE"])
# @pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("logits_soft_cap", [0.0])
@pytest.mark.parametrize("q_dtype", [torch.float16])
@pytest.mark.parametrize(
    # "kv_dtype", [torch.float16, torch.float8_e4m3fn, torch.float8_e5m2]
    "kv_dtype", [torch.float16]
)
# @pytest.mark.parametrize("contiguous_kv", [True, False])
@pytest.mark.parametrize("contiguous_kv", [True])
def test_aiter_batch_decode_with_paged_kv_cache(
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
):
    q = torch.randn(batch_size, num_qo_heads, head_dim, device="cuda:0", dtype=q_dtype)
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
        kv_data_fp32 = torch.randn(*kv_shape, dtype=torch.float32, device=q.device)
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
    wrapper = flashinfer.decode.AiterDecodeWithPagedKVCacheWrapper(
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
        kv_data_type=kv_dtype,
        q_data_type=q_dtype,
    )

    o = wrapper.run(q, kv_data)


def run_example(
    *,
    workspace_size=128 * 1024 * 1024,
    max_num_pages=128,
    kv_last_page_lens=[1, 7, 14, 4, 3, 1, 16],
    kv_page_indptrs=[0, 17, 29, 44, 48, 66, 100, 128],
    num_kv_heads=8,
    num_qo_heads=64,
    head_dim=128,
    page_size=16,
    batch_size=7,
    num_layers=32,
    q_dtype=torch.float16,
    kv_dtype=torch.float16,
):
    workspace_buffer = torch.empty(workspace_size, dtype=torch.uint8, device="cuda:0")
    decode_wrapper = flashinfer.decode.AiterDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD")

    kv_page_indices = torch.arange(max_num_pages).int().to("cuda:0")
    kv_last_page_len = torch.tensor(kv_last_page_lens, dtype=torch.int32, device="cuda:0")

    kv_page_indptr = torch.tensor(kv_page_indptrs, dtype=torch.int32, device="cuda:0")

    decode_wrapper.plan(kv_page_indptr,
                        kv_page_indices,
                        kv_last_page_len,
                        num_qo_heads,
                        num_kv_heads,
                        head_dim,
                        page_size)

    kv_cache_at_layer = [
        torch.randn(
            max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=kv_dtype, device="cuda:0"
        ) for _ in range(num_layers)
    ]

    outputs = []
    for i in range(num_layers):
        q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=q_dtype, device="cuda:0")
        kv_cache = kv_cache_at_layer[i]
        o = decode_wrapper.run(q, kv_cache)
        outputs.append(o)


if __name__ == "__main__":
    # test_aiter_batch_decode_with_paged_kv_cache(
    #     batch_size=256,
    #     kv_len=54,
    #     page_size=8,
    #     num_kv_heads=8,
    #     num_qo_heads=8,
    #     head_dim=128,
    #     kv_layout="NHD",
    #     pos_encoding_mode="NONE",
    #     logits_soft_cap=0.0,
    #     q_dtype=torch.float16,
    #     kv_dtype=torch.float16,
    #     contiguous_kv=True,
    # )
    run_example()
