"""A4Q ladder step 3: real Gemma-27B capture layers (L41, L47, L5) paged A4Q vs
current-fp4-path-on-dequantized-Q reference. Gate: cos >= 0.999 per layer."""
import math
import torch
import torch.nn.functional as F
from flashinfer import nvfp4_quantize_paged_kv_cache, BatchPrefillWithPagedKVCacheWrapper
from flashinfer.prefill import nvfp4_quantize_q

DEV = "cuda"
PAGE = 16
d = torch.load("/root/nano-kv/gemma_kv.pt", map_location="cpu")
cap, nkv, hd, nh = d["cap"], d["nkv"], d["hd"], d["nh"]
print("capture keys sample:", list(cap.keys())[:6], "nkv", nkv, "hd", hd, "nh", nh,
      flush=True)
workspace = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=DEV)

allok = True
for L in ["L41", "L47", "L5"]:
    q = cap[f"{L}.q"].to(DEV).to(torch.bfloat16)
    k = cap[f"{L}.k"].to(DEV).to(torch.bfloat16)
    v = cap[f"{L}.v"].to(DEV).to(torch.bfloat16)
    T = q.shape[0]
    pages = (T + PAGE - 1) // PAGE
    kpad = torch.zeros(pages * PAGE, nkv, hd, device=DEV, dtype=torch.bfloat16)
    vpad = torch.zeros_like(kpad)
    kpad[:T], vpad[:T] = k, v
    kpad = kpad.reshape(pages, PAGE, nkv, hd)
    vpad = vpad.reshape(pages, PAGE, nkv, hd)
    (k4, v4), (ksf, vsf), kg, vg = nvfp4_quantize_paged_kv_cache(kpad, vpad, kv_layout="NHD")
    kv_cache = torch.stack([k4, v4], dim=1)
    kv_sf = torch.stack([ksf, vsf], dim=1)
    kg = float(kg) if not torch.is_tensor(kg) else kg.item()
    vg = float(vg) if not torch.is_tensor(vg) else vg.item()
    qp, qsf, qdq = nvfp4_quantize_q(q)
    sm = 1.0 / math.sqrt(hd)

    qi = torch.tensor([0, T], device=DEV, dtype=torch.int32)
    ki = torch.tensor([0, pages], device=DEV, dtype=torch.int32)
    kidx = torch.arange(pages, device=DEV, dtype=torch.int32)
    kl = torch.tensor([T - (pages - 1) * PAGE], device=DEV, dtype=torch.int32)

    ref_w = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    ref_w.plan(qi, ki, kidx, kl, nh, nkv, hd, PAGE, causal=True, sm_scale=sm,
               q_data_type=torch.bfloat16, kv_data_type=torch.uint8,
               o_data_type=torch.bfloat16)
    ref = ref_w.run(qdq.to(torch.bfloat16), kv_cache, k_scale=kg, v_scale=vg,
                    kv_cache_sf=kv_sf)

    a4q_w = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    a4q_w.plan(qi, ki, kidx, kl, nh, nkv, hd, PAGE, causal=True, sm_scale=sm,
               q_data_type=torch.bfloat16, kv_data_type=torch.uint8,
               o_data_type=torch.bfloat16, use_nvf4_qk=True)
    out = a4q_w.run(qp, kv_cache, k_scale=kg, v_scale=vg, kv_cache_sf=kv_sf, q_sf=qsf)

    cos = F.cosine_similarity(out.float().reshape(1, -1), ref.float().reshape(1, -1)).item()
    mad = (out.float() - ref.float()).abs().max().item()
    # also report vs the unquantized-Q fp4 baseline for context (not the gate)
    ref_bf16q = ref_w.run(q, kv_cache, k_scale=kg, v_scale=vg, kv_cache_sf=kv_sf)
    cos_q = F.cosine_similarity(out.float().reshape(1, -1),
                                ref_bf16q.float().reshape(1, -1)).item()
    ok = cos >= 0.999
    print(f"[real {L}] T={T}: cos_vs_ref={cos:.7f} max_abs_diff={mad:.4e} "
          f"cos_vs_bf16q_fp4kv={cos_q:.6f} {'PASS' if ok else 'FAIL'}", flush=True)
    allok &= ok

print("A4Q_REAL " + ("PASS" if allok else "FAIL"), flush=True)
