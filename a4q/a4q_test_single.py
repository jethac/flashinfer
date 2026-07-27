"""A4Q ladder step 1: single-prefill nvf4-QK vs current fp4 path on dequantized Q."""
import math
import torch
import torch.nn.functional as F
from flashinfer import single_prefill_with_kv_cache
from flashinfer.prefill import nvfp4_quantize_q

torch.manual_seed(3)
DEV = "cuda"
NQO, NKV, HD = 32, 16, 128


def run_case(qo_len, kv_len, causal):
    q = torch.randn(qo_len, NQO, HD, device=DEV, dtype=torch.bfloat16) * 0.7
    k = torch.randn(kv_len, NKV, HD, device=DEV, dtype=torch.bfloat16) * 0.7
    v = torch.randn(kv_len, NKV, HD, device=DEV, dtype=torch.bfloat16) * 0.7
    kp, ksf, _ = nvfp4_quantize_q(k)
    vp, vsf, _ = nvfp4_quantize_q(v)
    qp, qsf, qdq = nvfp4_quantize_q(q)
    sm = 1.0 / math.sqrt(HD)
    kv_sf = (ksf.view(torch.float8_e4m3fn), vsf.view(torch.float8_e4m3fn))
    ref = single_prefill_with_kv_cache(
        qdq.to(torch.bfloat16), kp, vp, causal=causal, kv_cache_sf=kv_sf,
        o_dtype=torch.bfloat16, sm_scale=sm, backend="fa2")
    out = single_prefill_with_kv_cache(
        qp, kp, vp, causal=causal, kv_cache_sf=kv_sf, q_sf=qsf,
        use_nvf4_qk=True, o_dtype=torch.bfloat16, sm_scale=sm, backend="fa2")
    cos = F.cosine_similarity(
        out.float().reshape(1, -1), ref.float().reshape(1, -1)).item()
    mad = (out.float() - ref.float()).abs().max().item()
    ok = cos >= 0.9999
    print(f"[single] qo={qo_len} kv={kv_len} causal={causal}: cos={cos:.7f} "
          f"max_abs_diff={mad:.4e} {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


allok = True
for causal in (False, True):
    allok &= run_case(1024, 1024, causal)
# small qo exercises CTA_TILE_Q=16 path
allok &= run_case(17, 1024, True)
allok &= run_case(64, 1024, False)
print("A4Q_SINGLE " + ("PASS" if allok else "FAIL"), flush=True)
