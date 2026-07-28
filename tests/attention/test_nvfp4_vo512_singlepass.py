"""Single-pass NVFP4 qk512/vo512 VO-split equivalence with the two-pass vo256 path.

Gemma-4 / DiffusionGemma full-attention layers use a symmetric head_dim of 512.
FlashInfer's FA2 NVFP4 paged prefill runs these through the in-kernel VO-split
path (``USE_VO_SPLIT`` is active whenever ``NUM_MMA_D_VO > 16``, i.e. head_dim_vo
== 512): QK is computed once, P is staged through ``p_smem``, and the four KV
warps each accumulate a 128-wide slice of the full-512 PV, with the FP4 V
dequant (packed e2m1 + per-16 UE4M3 scale factors) applied in
``vosplit_compute_pv``. This retires the two-pass orchestration some callers use
today -- running the asymmetric qk512/vo256 kernel twice over byte-sliced V
halves and concatenating -- which recomputes the entire QK side per half.

This test pins the single-pass output against that two-pass reference. The V
halves are exact BYTE SLICES of the same quantized V (the per-16 SF blocks align
at the 256 boundary), so both paths consume identical FP4 V bytes; the only
difference is softmax re-association and the P re-quantization normalizer, so
bit-identity is not expected but the outputs must agree tightly. Both paths are
additionally bounded against a float32 reference attention on the dequantized
K/V (a dequantization oracle, not a re-quantized approximation).
"""

import pytest
import torch
import torch.nn.functional as F

import flashinfer
from flashinfer.utils import get_compute_capability
from tests.test_helpers.utils_fp4 import create_nvfp4_kv, nvfp4_to_float

HEAD_DIM = 512
HALF = HEAD_DIM // 2  # VO split boundary (256)


def _head_dim_512_supported() -> bool:
    major, _ = get_compute_capability(torch.device("cuda:0"))
    return major >= 8


def _plan_run(q, k_packed, v_packed, k_sf, v_sf, k_gs, v_gs, q_indptr, kv_indptr,
              kv_indices, kv_last_page_len, num_qo_heads, num_kv_heads,
              head_dim_vo, page_size, causal, q_dtype, workspace):
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        HEAD_DIM,  # head_dim_qk
        page_size,
        head_dim_vo=head_dim_vo,
        causal=causal,
        pos_encoding_mode="NONE",
        logits_soft_cap=0.0,
        kv_data_type=torch.uint8,
        q_data_type=q_dtype,
    )
    return wrapper.run(
        q,
        (k_packed, v_packed),
        k_scale=k_gs.item(),
        v_scale=v_gs.item(),
        kv_cache_sf=(k_sf, v_sf),
    )


def _exact_ref(q, k_dq, v_dq, kv_indptr, kv_last_page_len, q_indptr, batch_size,
               num_kv_heads, page_size, causal):
    """float32 attention on dequantized paged K/V, per batch item."""
    outs = []
    for i in range(batch_size):
        qi = q[q_indptr[i] : q_indptr[i + 1]]
        full_k = k_dq[kv_indptr[i] : kv_indptr[i + 1] - 1]
        last_k = k_dq[kv_indptr[i + 1] - 1, : kv_last_page_len[i]]
        ki = torch.cat(
            [full_k.reshape(-1, num_kv_heads, k_dq.shape[-1]),
             last_k.reshape(-1, num_kv_heads, k_dq.shape[-1])], dim=0)
        full_v = v_dq[kv_indptr[i] : kv_indptr[i + 1] - 1]
        last_v = v_dq[kv_indptr[i + 1] - 1, : kv_last_page_len[i]]
        vi = torch.cat(
            [full_v.reshape(-1, num_kv_heads, v_dq.shape[-1]),
             last_v.reshape(-1, num_kv_heads, v_dq.shape[-1])], dim=0)
        outs.append(
            flashinfer.prefill.single_prefill_with_kv_cache(
                qi, ki, vi, causal=causal, pos_encoding_mode="NONE",
                logits_soft_cap=0.0,
            )
        )
    return outs


@pytest.mark.parametrize("q_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "qo_len",
    [1, 64],  # 1 -> CTA16 (decode); 64 -> CTA32 (long-q prefill) at head_dim>=512
)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("num_kv_heads", [1, 2])
def test_nvfp4_vo512_singlepass_matches_twopass(q_dtype, qo_len, causal, num_kv_heads):
    if not _head_dim_512_supported():
        pytest.skip("16-bit FA2 head_dim > 256 requires SM80+")
    if qo_len == 1 and causal:
        pytest.skip("single-token causal decode is trivial; covered by qo_len>1")

    torch.manual_seed(19)
    batch_size = 2
    kv_len = 128
    page_size = 16
    num_qo_heads = 2 * num_kv_heads

    q = torch.randn(batch_size * qo_len, num_qo_heads, HEAD_DIM,
                    device="cuda:0", dtype=q_dtype)
    q_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32) * qo_len

    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32) * num_pages_per_seq
    kv_indices = torch.arange(0, total_num_pages, dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32)

    # Full-512 NVFP4 K and V.
    kv_shape = (total_num_pages, page_size, num_kv_heads, HEAD_DIM // 2)
    k_packed, k_sf, k_gs = create_nvfp4_kv(kv_shape, "cuda:0")
    v_packed, v_sf, v_gs = create_nvfp4_kv(kv_shape, "cuda:0")

    q_g = q_indptr.to("cuda:0")
    kvi_g = kv_indptr.to("cuda:0")
    kvidx_g = kv_indices.to("cuda:0")
    kvlp_g = kv_last_page_len.to("cuda:0")
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")

    # --- single-pass qk512 / vo512 (the VO-split PV path under test) ---
    o_single = _plan_run(
        q, k_packed, v_packed, k_sf, v_sf, k_gs, v_gs, q_g, kvi_g, kvidx_g,
        kvlp_g, num_qo_heads, num_kv_heads, HEAD_DIM, page_size, causal,
        q_dtype, workspace)
    assert o_single.shape == (batch_size * qo_len, num_qo_heads, HEAD_DIM)
    assert torch.isfinite(o_single).all()

    # --- two-pass reference: qk512 / vo256 twice over byte-sliced V halves ---
    # packed byte boundary 128 == VO dim 256; SF boundary 16 == VO dim 256.
    halves = []
    for lo_b, lo_sf in ((0, 0), (HALF // 2, HALF // 16)):
        v_h = v_packed[..., lo_b : lo_b + HALF // 2].contiguous()
        vsf_h = v_sf[..., lo_sf : lo_sf + HALF // 16].contiguous()
        o_h = _plan_run(
            q, k_packed, v_h, k_sf, vsf_h, k_gs, v_gs, q_g, kvi_g, kvidx_g,
            kvlp_g, num_qo_heads, num_kv_heads, HALF, page_size, causal,
            q_dtype, workspace)
        halves.append(o_h)
    o_two = torch.cat(halves, dim=-1)

    # single-pass ~= two-pass (re-association only).
    cos = F.cosine_similarity(
        o_single.float().reshape(1, -1), o_two.float().reshape(1, -1)).item()
    mad = (o_single.float() - o_two.float()).abs().max().item()
    assert cos >= 0.9999, f"single vs two-pass cos={cos:.7f} mad={mad:.3e}"

    # both paths bounded against the float32 dequant oracle.
    k_dq = nvfp4_to_float(k_packed, k_sf, k_gs).to(q_dtype)
    v_dq = nvfp4_to_float(v_packed, v_sf, v_gs).to(q_dtype)
    refs = _exact_ref(q, k_dq, v_dq, kv_indptr, kv_last_page_len, q_indptr,
                      batch_size, num_kv_heads, page_size, causal)
    for i in range(batch_size):
        sl = slice(q_indptr[i], q_indptr[i + 1])
        torch.testing.assert_close(o_single[sl], refs[i], rtol=1e-1, atol=1e-1)
        torch.testing.assert_close(o_two[sl], refs[i], rtol=1e-1, atol=1e-1)


if __name__ == "__main__":
    import sys

    rc = 0
    for qd in (torch.bfloat16, torch.float16):
        for ql in (1, 64):
            for ca in (False, True):
                for h in (1, 2):
                    if ql == 1 and ca:
                        continue
                    try:
                        test_nvfp4_vo512_singlepass_matches_twopass(qd, ql, ca, h)
                        print(f"PASS q={qd} qo={ql} causal={ca} kvh={h}", flush=True)
                    except Exception as e:  # noqa
                        rc = 1
                        print(f"FAIL q={qd} qo={ql} causal={ca} kvh={h}: "
                              f"{type(e).__name__}: {e}", flush=True)
    print("VO512_SINGLEPASS_EQUIV " + ("PASS" if rc == 0 else "FAIL"), flush=True)
    sys.exit(rc)
