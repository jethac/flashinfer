"""A4Q step 3 rerun, end-to-end through nvfp4_quantize_q_cuda: real Gemma layers,
A4Q run gets RAW bf16 q (auto-quantized by the CUDA op inside run()); reference is
the current fp4 path on the CUDA-op-dequantized Q. Gate: cos >= 0.999 per layer."""
import math
import torch
import torch.nn.functional as F
from flashinfer import (nvfp4_quantize_paged_kv_cache, nvfp4_quantize_q_cuda,
                        BatchPrefillWithPagedKVCacheWrapper)

DEV = "cuda"
PAGE = 16
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
                    device=DEV)


def dequant_q(packed, sf):
    # packed [..., D/2] u8, sf [..., D/16] u8 -> float32 [..., D]
    shape = packed.shape[:-1]
    D = packed.shape[-1] * 2
    codes = torch.empty(*shape, D, dtype=torch.uint8, device=packed.device)
    codes[..., 0::2] = packed & 0x0F
    codes[..., 1::2] = (packed >> 4) & 0x0F
    vals = E2M1[codes.long()]
    sf_f = sf.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    return vals * sf_f


d = torch.load("/root/nano-kv/gemma_kv.pt", map_location="cpu")
cap, nkv, hd, nh = d["cap"], d["nkv"], d["hd"], d["nh"]
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
    qp, qsf = nvfp4_quantize_q_cuda(q)
    qdq = dequant_q(qp, qsf)
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
    # RAW bf16 q -> exercises the auto-quant (CUDA op) path inside run()
    out = a4q_w.run(q, kv_cache, k_scale=kg, v_scale=vg, kv_cache_sf=kv_sf)

    cos = F.cosine_similarity(out.float().reshape(1, -1), ref.float().reshape(1, -1)).item()
    mad = (out.float() - ref.float()).abs().max().item()
    ok = cos >= 0.999
    print(f"[real-e2e {L}] T={T}: cos_vs_ref={cos:.7f} max_abs_diff={mad:.4e} "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    allok &= ok

print("A4Q_REAL_E2E " + ("PASS" if allok else "FAIL"), flush=True)
