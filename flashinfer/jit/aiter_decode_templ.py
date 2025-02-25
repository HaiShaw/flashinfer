# TODO: license header
suffix_source_dict = {
    '.hip': r"""// generated with aiter_decode_templ.py
#include "pytorch_extension_utils.h"

void {{kernel_name}}(
    torch::Tensor& out, // [num_seqs, num_heads, head_size]
    torch::Tensor& workspace_buffer,
    torch::Tensor& query,       // [num_seqs, num_heads, head_size]
    torch::Tensor& key_cache,   // [num_blocks, num_heads, block_size, head_size] or
                                // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor& value_cache, // [num_blocks, num_heads, block_size, head_size] or
                                // [num_blocks, block_size, num_heads, head_size]
    double scale,
    torch::Tensor& kv_indptr,         // [num_seqs + 1]
    torch::Tensor& kv_page_indices,   // [max_num_blocks]
    std::optional<torch::Tensor>& kv_last_page_lens, // [num_seqs]
    int64_t block_size,
    int64_t max_num_partitions,
    const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype,
    const std::string& kv_cache_layout,
    float logits_soft_cap,
    torch::Tensor& k_scale,
    torch::Tensor& v_scale,
    const c10::optional<torch::Tensor>& fp8_out_scale,
    int64_t partition_size) {

    paged_attention_custom_launcher<
        {{query_dtype}},
        {{key_value_dtype}},
        {{kvcache_dtype}},
        {{block_size}},
        {{head_size}},
        {{out_dtype}},
        {{partition_size_old}},
        {{alibi_enabled}},
        {{logits_soft_cap_enabled}}>(
        out,
        workspace_buffer,
        query,
        key_cache,
        value_cache,
        scale,
        kv_indptr,
        kv_page_indices,
        kv_last_page_lens,
        max_num_partitions,
        alibi_slopes,
        kv_cache_layout,
        logits_soft_cap,
        k_scale,
        v_scale,
        fp8_out_scale);
}

""",
    '_pybind.cc': r"""// generated with aiter_decode_templ.py
#include "pytorch_extension_utils.h"

void {{kernel_name}}(
    torch::Tensor& out, // [num_seqs, num_heads, head_size]
    torch::Tensor& workspace_buffer,
    torch::Tensor& query,       // [num_seqs, num_heads, head_size]
    torch::Tensor& key_cache,   // [num_blocks, num_heads, block_size, head_size] or
                                // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor& value_cache, // [num_blocks, num_heads, block_size, head_size] or
                                // [num_blocks, block_size, num_heads, head_size]
    double scale,
    torch::Tensor& kv_indptr,         // [num_seqs + 1]
    torch::Tensor& kv_page_indices,   // [max_num_blocks]
    std::optional<torch::Tensor>& kv_last_page_lens, // [num_seqs]
    int64_t block_size,
    int64_t max_num_partitions,
    const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype,
    const std::string& kv_cache_layout,
    float logits_soft_cap,
    torch::Tensor& k_scale,
    torch::Tensor& v_scale,
    const c10::optional<torch::Tensor>& fp8_out_scale,
    int64_t partition_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &{{kernel_name}},
        "AITER paged attention");
""",
}
