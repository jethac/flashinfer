"""A4Q K-5 table (v2, CUDA Q-quantize): batch-prefill paged perf, Gemma-27B shapes.
batch 1, causal, kv_len = qo_len in {8192, 32768}. CUDA events, warmup 3, >=5 iters.
Reports: bf16-FA2, fp4-FA2, A4Q kernel, CUDA q-quant, TOTAL A4Q (quant+kernel).
Gate: TOTAL >= 1.5x faster than fp4-FA2 at 8k+.
"""
import math
import torch
import torch.nn.functional as F
from flashinfer import (nvfp4_quantize_paged_kv_cache, nvfp4_quantize_q_cuda,
                        BatchPrefillWithPagedKVCacheWrapper)

torch.manual_seed(7)
DEV = "cuda"
NQO, NKV, HD, PAGE = 32, 16, 128, 16
workspace = torch.empty(768 * 1024 * 1024, dtype=torch.uint8, device=DEV)


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


def plan_args(kv_len, qo_len):
    qi = torch.tensor([0, qo_len], device=DEV, dtype=torch.int32)
    pages = (kv_len + PAGE - 1) // PAGE
    ki = torch.tensor([0, pages], device=DEV, dtype=torch.int32)
    kidx = torch.arange(pages, device=DEV, dtype=torch.int32)
    kl = torch.tensor([kv_len - (pages - 1) * PAGE], device=DEV, dtype=torch.int32)
    return qi, ki, kidx, kl


for ctx in (8192, 32768):
    pages = (ctx + PAGE - 1) // PAGE
    kbf = torch.randn(pages, PAGE, NKV, HD, device=DEV, dtype=torch.bfloat16) * 0.5
    vbf = torch.randn(pages, PAGE, NKV, HD, device=DEV, dtype=torch.bfloat16) * 0.5
    (k4, v4), (ksf, vsf), kg, vg = nvfp4_quantize_paged_kv_cache(kbf, vbf, kv_layout="NHD")
    kv4_cache = torch.stack([k4, v4], dim=1)
    kv_sf = torch.stack([ksf, vsf], dim=1)
    kg = float(kg) if not torch.is_tensor(kg) else kg.item()
    vg = float(vg) if not torch.is_tensor(vg) else vg.item()
    q = torch.randn(ctx, NQO, HD, device=DEV, dtype=torch.bfloat16) * 0.5
    sm = 1.0 / math.sqrt(HD)
    qi, ki, kidx, kl = plan_args(ctx, ctx)

    quant_ms = bench(lambda: nvfp4_quantize_q_cuda(q))
    qp, qsf = nvfp4_quantize_q_cuda(q)

    # bf16 baseline
    w_bf = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    w_bf.plan(qi, ki, kidx, kl, NQO, NKV, HD, PAGE, causal=True, sm_scale=sm,
              q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
              o_data_type=torch.bfloat16)
    kv_bf = torch.stack([kbf, vbf], dim=1)
    ms_bf = bench(lambda: w_bf.run(q, kv_bf))

    # fp4-FA2 shipped path
    w_fp4 = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    w_fp4.plan(qi, ki, kidx, kl, NQO, NKV, HD, PAGE, causal=True, sm_scale=sm,
               q_data_type=torch.bfloat16, kv_data_type=torch.uint8,
               o_data_type=torch.bfloat16)
    ms_fp4 = bench(lambda: w_fp4.run(q, kv4_cache, k_scale=kg, v_scale=vg,
                                     kv_cache_sf=kv_sf))

    # A4Q nvf4-QK path
    w_a4q = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    w_a4q.plan(qi, ki, kidx, kl, NQO, NKV, HD, PAGE, causal=True, sm_scale=sm,
               q_data_type=torch.bfloat16, kv_data_type=torch.uint8,
               o_data_type=torch.bfloat16, use_nvf4_qk=True)
    out_a4q = w_a4q.run(q, kv4_cache, k_scale=kg, v_scale=vg, kv_cache_sf=kv_sf)
    out_fp4 = w_fp4.run(q, kv4_cache, k_scale=kg, v_scale=vg, kv_cache_sf=kv_sf)
    cos = F.cosine_similarity(out_a4q.float().reshape(1, -1),
                              out_fp4.float().reshape(1, -1)).item()
    # kernel-only (pre-quantized q)
    ms_a4q = bench(lambda: w_a4q.run(qp, kv4_cache, k_scale=kg, v_scale=vg,
                                     kv_cache_sf=kv_sf, q_sf=qsf))
    # TOTAL: raw bf16 q -> auto CUDA quant + kernel
    ms_total = bench(lambda: w_a4q.run(q, kv4_cache, k_scale=kg, v_scale=vg,
                                       kv_cache_sf=kv_sf))

    gate = ms_fp4 / ms_total
    print(f"[perf ctx={ctx}] bf16-FA2={ms_bf:.3f} ms  fp4-FA2={ms_fp4:.3f} ms  "
          f"A4Q-kernel={ms_a4q:.3f} ms  q-quant-cuda={quant_ms:.3f} ms  "
          f"A4Q-TOTAL={ms_total:.3f} ms  fp4/TOTAL={gate:.2f}x "
          f"({'PASS' if gate >= 1.5 else 'FAIL'} gate>=1.5x)  "
          f"TOTAL/bf16={ms_total / ms_bf:.3f}x  cos_vs_fp4_bf16q={cos:.6f}",
          flush=True)
    del kbf, vbf, kv_bf, kv4_cache, kv_sf, q, qp, qsf
    torch.cuda.empty_cache()

print("A4Q_BENCH2_DONE", flush=True)
