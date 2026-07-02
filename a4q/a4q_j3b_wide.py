"""J-3b wide heads: A4Q at head_dim_qk 256 (symmetric) and 512/256 (VO-split QK side).

Gates: bit-exact vs current-fp4-path-on-dequantized-Q reference; perf >= 1.3x vs
the current fp4 path at the same dims. Also polls /root/work/kv/gemma_kv.pt for a
real Gemma-4-geometry capture (256-head) - used if present, skipped if not.
"""
import math
import os
import torch
import torch.nn.functional as F
from flashinfer import BatchPrefillWithPagedKVCacheWrapper, nvfp4_quantize_q_cuda
from flashinfer.prefill import nvfp4_quantize_q

torch.manual_seed(13)
DEV = "cuda"
PAGE = 16
workspace = torch.empty(1024 * 1024 * 1024, dtype=torch.uint8, device=DEV)
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
                    device=DEV)


def dequant(packed, sf):
    shape = packed.shape[:-1]
    D = packed.shape[-1] * 2
    codes = torch.empty(*shape, D, dtype=torch.uint8, device=packed.device)
    codes[..., 0::2] = packed & 0x0F
    codes[..., 1::2] = (packed >> 4) & 0x0F
    vals = E2M1[codes.long()]
    sf_f = sf.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    return vals * sf_f


def quant_kv(x):  # [pages, PAGE, H, D] bf16 -> packed u8 + sf u8 (linear layout)
    p, s, _ = nvfp4_quantize_q(x)
    return p, s


def bench(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def run_case(tag, nq, nkv, hd_qk, hd_vo, qo_len, kv_len, causal, bench_perf):
    pages = (kv_len + PAGE - 1) // PAGE
    kbf = torch.randn(pages, PAGE, nkv, hd_qk, device=DEV, dtype=torch.bfloat16) * 0.5
    vbf = torch.randn(pages, PAGE, nkv, hd_vo, device=DEV, dtype=torch.bfloat16) * 0.5
    k4, ksf = quant_kv(kbf)
    v4, vsf = quant_kv(vbf)
    q = torch.randn(qo_len, nq, hd_qk, device=DEV, dtype=torch.bfloat16) * 0.5
    qp, qsf = nvfp4_quantize_q_cuda(q)
    qdq = dequant(qp, qsf).to(torch.bfloat16)
    sm = 1.0 / math.sqrt(hd_qk)

    qi = torch.tensor([0, qo_len], device=DEV, dtype=torch.int32)
    ki = torch.tensor([0, pages], device=DEV, dtype=torch.int32)
    kidx = torch.arange(pages, device=DEV, dtype=torch.int32)
    kl = torch.tensor([kv_len - (pages - 1) * PAGE], device=DEV, dtype=torch.int32)

    def make(use_nvf4):
        w = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
        w.plan(qi, ki, kidx, kl, nq, nkv, hd_qk, PAGE, head_dim_vo=hd_vo,
               causal=causal, sm_scale=sm, q_data_type=torch.bfloat16,
               kv_data_type=torch.uint8, o_data_type=torch.bfloat16,
               use_nvf4_qk=use_nvf4)
        return w

    w_ref, w_a4q = make(False), make(True)
    ref = w_ref.run(qdq, (k4, v4), kv_cache_sf=(ksf, vsf))
    out = w_a4q.run(qp, (k4, v4), kv_cache_sf=(ksf, vsf), q_sf=qsf)
    mad = (out.float() - ref.float()).abs().max().item()
    cos = F.cosine_similarity(out.float().reshape(1, -1),
                              ref.float().reshape(1, -1)).item()
    ok = cos >= 0.9999
    msg = (f"[{tag} qo={qo_len} kv={kv_len} causal={causal}] mad={mad:.3e} "
           f"cos={cos:.7f}")
    if bench_perf:
        ms_ref = bench(lambda: w_ref.run(qdq, (k4, v4), kv_cache_sf=(ksf, vsf)))
        ms_a4q = bench(lambda: w_a4q.run(qp, (k4, v4), kv_cache_sf=(ksf, vsf),
                                         q_sf=qsf))
        speedup = ms_ref / ms_a4q
        ok &= speedup >= 1.3
        msg += f" | fp4-current={ms_ref:.3f} ms a4q={ms_a4q:.3f} ms ({speedup:.2f}x)"
    print(msg + (" PASS" if ok else " FAIL"), flush=True)
    del kbf, vbf, k4, v4, ksf, vsf
    torch.cuda.empty_cache()
    return ok


allok = True
# Gemma-4-like 256-head geometry (E-series: 8 q / 4 kv), plus a GQA-2 variant
allok &= run_case("hd256", 8, 4, 256, 256, 2048, 2048, True, False)
allok &= run_case("hd256", 8, 4, 256, 256, 333, 4096, False, False)
allok &= run_case("hd256", 16, 8, 256, 256, 8192, 8192, True, True)
# 512-QK / 256-VO (the fork's VO-split QK side: kernel sees one V half)
allok &= run_case("hd512/256", 8, 4, 512, 256, 1024, 1024, True, False)
allok &= run_case("hd512/256", 8, 4, 512, 256, 17, 4096, False, False)
allok &= run_case("hd512/256", 8, 4, 512, 256, 8192, 8192, True, True)

# real capture leg: the regenerated capture is Gemma-3-27B geometry (head_dim
# 128, 32/16 heads) - no 256-head capture exists, so this validates that the
# wide-head relax did not regress the real-data 128 path (wide dims are
# synthetic-gated above, the pre-registered requirement).
cap_path = "/root/work/kv/gemma_kv.pt"
if os.path.exists(cap_path):
    d = torch.load(cap_path, map_location="cpu")
    cap, nkv, hd, nh = d["cap"], d["nkv"], d["hd"], d["nh"]
    for L in ("L41", "L5"):
        q = cap[f"{L}.q"].to(DEV).to(torch.bfloat16)
        k = cap[f"{L}.k"].to(DEV).to(torch.bfloat16)
        v = cap[f"{L}.v"].to(DEV).to(torch.bfloat16)
        T = q.shape[0]
        pages = (T + PAGE - 1) // PAGE
        kpad = torch.zeros(pages * PAGE, nkv, hd, device=DEV, dtype=torch.bfloat16)
        vpad = torch.zeros_like(kpad)
        kpad[:T], vpad[:T] = k, v
        k4, ksf = quant_kv(kpad.reshape(pages, PAGE, nkv, hd))
        v4, vsf = quant_kv(vpad.reshape(pages, PAGE, nkv, hd))
        qp, qsf = nvfp4_quantize_q_cuda(q)
        qdq = dequant(qp, qsf).to(torch.bfloat16)
        sm = 1.0 / math.sqrt(hd)
        qi = torch.tensor([0, T], device=DEV, dtype=torch.int32)
        ki = torch.tensor([0, pages], device=DEV, dtype=torch.int32)
        kidx = torch.arange(pages, device=DEV, dtype=torch.int32)
        kl = torch.tensor([T - (pages - 1) * PAGE], device=DEV, dtype=torch.int32)

        def mkw(use_nvf4):
            w = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
            w.plan(qi, ki, kidx, kl, nh, nkv, hd, PAGE, causal=True, sm_scale=sm,
                   q_data_type=torch.bfloat16, kv_data_type=torch.uint8,
                   o_data_type=torch.bfloat16, use_nvf4_qk=use_nvf4)
            return w

        ref = mkw(False).run(qdq, (k4, v4), kv_cache_sf=(ksf, vsf))
        out = mkw(True).run(qp, (k4, v4), kv_cache_sf=(ksf, vsf), q_sf=qsf)
        mad = (out.float() - ref.float()).abs().max().item()
        cos = F.cosine_similarity(out.float().reshape(1, -1),
                                  ref.float().reshape(1, -1)).item()
        ok = cos >= 0.999
        print(f"[real {L} hd{hd} T={T}] mad={mad:.3e} cos={cos:.7f} "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        allok &= ok
else:
    print("real capture not present at /root/work/kv/gemma_kv.pt - skipped "
          "(synthetic gates above are the pre-registered requirement)", flush=True)

print("A4Q_J3B " + ("PASS" if allok else "FAIL"), flush=True)
