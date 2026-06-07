from flashinfer.jit.attention.utils import generate_additional_params


def test_generate_additional_params_emits_nvfp4_kv_sf_strides():
    decl, func_params, setter = generate_additional_params(
        ["maybe_k_cache_sf", "maybe_v_cache_sf"],
        ["uint8_t", "uint8_t"],
        [],
        [],
    )

    assert "uint32_t maybe_k_cache_sf_stride_page;" in decl
    assert "uint32_t maybe_k_cache_sf_stride_h;" in decl
    assert "uint32_t maybe_k_cache_sf_stride_n;" in decl
    assert "uint32_t maybe_v_cache_sf_stride_page;" in decl
    assert "uint32_t maybe_v_cache_sf_stride_h;" in decl
    assert "uint32_t maybe_v_cache_sf_stride_n;" in decl
    assert "Optional<ffi::Tensor> maybe_k_cache_sf" in func_params
    assert "Optional<ffi::Tensor> maybe_v_cache_sf" in func_params
    assert "params.maybe_k_cache_sf_stride_page" in setter
    assert "params.maybe_v_cache_sf_stride_page" in setter
    assert "kv_layout == QKVLayout::kNHD" in setter


def test_generate_additional_params_does_not_emit_unrelated_strides():
    decl, _, setter = generate_additional_params(
        ["maybe_alibi_slopes"],
        ["float"],
        ["sm_scale"],
        ["double"],
    )

    assert "stride_page" not in decl
    assert "stride_page" not in setter
