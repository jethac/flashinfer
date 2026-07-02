"""A4Q ladder step 2: batch paged nvf4-QK vs current fp4 path on dequantized Q."""
import math
import torch
import torch.nn.functional as F
from flashinfer import nvfp4_quantize_paged_kv_cache, BatchPrefillWithPagedKVCacheWrapper
from flashinfer.prefill import nvfp4_quantize_q

torch.manual_seed(7)
DEV = "cuda"
NQO, NKV, HD, PAGE = 32, 16, 128, 16
workspace = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=DEV)


def plan_args(batch, kv_len, qo_len):
    qo_indptr = torch.arange(0, batch + 1, device=DEV, dtype=torch.int32) * qo_len
    pages_per_seq = (kv_len + PAGE - 1) // PAGE
    kv_indptr = torch.arange(0, batch + 1, device=DEV, dtype=torch.int32) * pages_per_seq
    kv_indices = torch.arange(0, batch * pages_per_seq, device=DEV, dtype=torch.int32)
    last_len = kv_len - (pages_per_seq - 1) * PAGE
    kv_last = torch.full((batch,), last_len, device=DEV, dtype=torch.int32)
    return qo_indptr, kv_indptr, kv_indices, kv_last


def run_case(batch, kv_len, qo_len, causal):
    num_pages = batch * ((kv_len + PAGE - 1) // PAGE)
    kbf = torch.randn(num_pages, PAGE, NKV, HD, device=DEV, dtype=torch.bfloat16) * 0.5
    vbf = torch.randn(num_pages, PAGE, NKV, HD, device=DEV, dtype=torch.bfloat16) * 0.5
    (k4, v4), (ksf, vsf), kg, vg = nvfp4_quantize_paged_kv_cache(kbf, vbf, kv_layout="NHD")
    kv_cache = torch.stack([k4, v4], dim=1)
    kv_sf = torch.stack([ksf, vsf], dim=1)
    kg = float(kg) if not torch.is_tensor(kg) else kg.item()
    vg = float(vg) if not torch.is_tensor(vg) else vg.item()

    q = torch.randn(batch * qo_len, NQO, HD, device=DEV, dtype=torch.bfloat16) * 0.5
    qp, qsf, qdq = nvfp4_quantize_q(q)
    sm = 1.0 / math.sqrt(HD)

    qi, ki, kidx, kl = plan_args(batch, kv_len, qo_len)

    ref_w = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    ref_w.plan(qi, ki, kidx, kl, NQO, NKV, HD, PAGE, causal=causal, sm_scale=sm,
               q_data_type=torch.bfloat16, kv_data_type=torch.uint8,
               o_data_type=torch.bfloat16)
    ref = ref_w.run(qdq.to(torch.bfloat16), kv_cache, k_scale=kg, v_scale=vg,
                    kv_cache_sf=kv_sf)

    a4q_w = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    a4q_w.plan(qi, ki, kidx, kl, NQO, NKV, HD, PAGE, causal=causal, sm_scale=sm,
               q_data_type=torch.bfloat16, kv_data_type=torch.uint8,
               o_data_type=torch.bfloat16, use_nvf4_qk=True)
    out = a4q_w.run(qp, kv_cache, k_scale=kg, v_scale=vg, kv_cache_sf=kv_sf, q_sf=qsf)

    cos = F.cosine_similarity(
        out.float().reshape(1, -1), ref.float().reshape(1, -1)).item()
    mad = (out.float() - ref.float()).abs().max().item()
    ok = cos >= 0.9999
    print(f"[paged] batch={batch} kv={kv_len} qo={qo_len} causal={causal}: "
          f"cos={cos:.7f} max_abs_diff={mad:.4e} {'PASS' if ok else 'FAIL'}", flush=True)
    del kbf, vbf, kv_cache, kv_sf, q
    torch.cuda.empty_cache()
    return ok


allok = True
allok &= run_case(2, 8192, 8192, True)
allok &= run_case(2, 8192, 8192, False)
allok &= run_case(2, 8192, 1, False)   # decode-like (CTA_TILE_Q=16)
allok &= run_case(3, 4096, 333, True)  # append-style, uneven lengths
print("A4Q_PAGED " + ("PASS" if allok else "FAIL"), flush=True)
