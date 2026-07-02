"""J-3 A4Q-decode ladder: bit-exact decode shapes + split-KV validation + perf.

- Correctness: BatchDecodeWithPagedKVCacheWrapper (tensor cores, fa2),
  batch {1,8,32} x kv {8k,32k,100k}. Reference = current fp4 path (dequantized
  Q, split-KV disabled, the shipped configuration). A4Q no-split must be
  bit-exact; A4Q split-KV must match within fp32 merge rounding.
- Split-KV regression shape: prefill wrapper, qo<<kv (the original corruption
  repro geometry), a4q+split vs ref.
- Perf: batch-1 decode, kv in {8k, 32k, 100k}; baseline = K-1 methodology
  (current fp4 path; K-1 raw @32k: fp4-FA2 0.610 ms, bf16 0.211 ms).
  Gate: a4q(split) >= 2x faster than current fp4 at 32k.

KV codes are synthesized directly as random packed e2m1 bytes + bounded random
ue4m3 SF bytes (no giant bf16 staging; stresses arbitrary code patterns).
"""
import math
import torch
import torch.nn.functional as F
from flashinfer import (BatchDecodeWithPagedKVCacheWrapper,
                        BatchPrefillWithPagedKVCacheWrapper, nvfp4_quantize_q_cuda)

torch.manual_seed(11)
DEV = "cuda"
NQO, NKV, HD, PAGE = 32, 16, 128, 16
workspace = torch.empty(1024 * 1024 * 1024, dtype=torch.uint8, device=DEV)
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
                    device=DEV)


def dequant_q(packed, sf):
    shape = packed.shape[:-1]
    D = packed.shape[-1] * 2
    codes = torch.empty(*shape, D, dtype=torch.uint8, device=packed.device)
    codes[..., 0::2] = packed & 0x0F
    codes[..., 1::2] = (packed >> 4) & 0x0F
    vals = E2M1[codes.long()]
    sf_f = sf.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    return vals * sf_f


def make_fp4_kv(num_pages):
    k4 = torch.randint(0, 256, (num_pages, PAGE, NKV, HD // 2), device=DEV,
                       dtype=torch.uint8)
    v4 = torch.randint(0, 256, (num_pages, PAGE, NKV, HD // 2), device=DEV,
                       dtype=torch.uint8)
    # sane ue4m3 SF range (~0.01 .. ~2.0): exponent fields 0x38..0x60
    ksf = torch.randint(0x30, 0x58, (num_pages, PAGE, NKV, HD // 16), device=DEV,
                        dtype=torch.uint8)
    vsf = torch.randint(0x30, 0x58, (num_pages, PAGE, NKV, HD // 16), device=DEV,
                        dtype=torch.uint8)
    return (torch.stack([k4, v4], dim=1), torch.stack([ksf, vsf], dim=1))


def decode_plan_args(batch, kv_len):
    pages = (kv_len + PAGE - 1) // PAGE
    indptr = torch.arange(0, batch + 1, device=DEV, dtype=torch.int32) * pages
    indices = torch.arange(0, batch * pages, device=DEV, dtype=torch.int32)
    last = torch.full((batch,), kv_len - (pages - 1) * PAGE, device=DEV,
                      dtype=torch.int32)
    return indptr, indices, last


def bench(fn, iters=20, warmup=5):
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


def make_decode_wrapper(use_nvf4, disable_split, batch, kv_len):
    w = BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD",
                                           use_tensor_cores=True, backend="fa2")
    indptr, indices, last = decode_plan_args(batch, kv_len)
    w.plan(indptr, indices, last, NQO, NKV, HD, PAGE, sm_scale=1.0 / math.sqrt(HD),
           q_data_type=torch.bfloat16, kv_data_type=torch.uint8,
           o_data_type=torch.bfloat16, disable_split_kv=disable_split,
           use_nvf4_qk=use_nvf4)
    return w


allok = True


def bf16_matched_control(batch, kv_len, kv_cache, kv_sf, qdq):
    """Distribution-matched merge control: the SAME dequantized KV data in bf16
    through the same flash-decoding split/merge path (no fp4, no a4q). Returns
    the split-vs-nosplit lse deviation of the shared merge math on identical
    logit statistics."""
    kbf = dequant_q(kv_cache[:, 0], kv_sf[:, 0]).to(torch.bfloat16)
    vbf = dequant_q(kv_cache[:, 1], kv_sf[:, 1]).to(torch.bfloat16)
    kvbf = torch.stack([kbf, vbf], dim=1)
    del kbf, vbf

    def bf16_run(disable_split):
        w = BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD",
                                               use_tensor_cores=True, backend="fa2")
        indptr, indices, last = decode_plan_args(batch, kv_len)
        w.plan(indptr, indices, last, NQO, NKV, HD, PAGE,
               sm_scale=1.0 / math.sqrt(HD), q_data_type=torch.bfloat16,
               kv_data_type=torch.bfloat16, o_data_type=torch.bfloat16,
               disable_split_kv=disable_split)
        return w.run(qdq, kvbf, return_lse=True)

    _, l_ns = bf16_run(True)
    _, l_sp = bf16_run(False)
    lse_mad = (l_sp - l_ns).abs().max().item()
    del kvbf
    torch.cuda.empty_cache()
    return max(lse_mad, 1e-5)


print("=== J-3 correctness: decode shapes ===", flush=True)
for batch in (1, 8, 32):
    for kv_len in (8192, 32768, 100352):
        pages = (kv_len + PAGE - 1) // PAGE
        kv_cache, kv_sf = make_fp4_kv(batch * pages)
        q = torch.randn(batch, NQO, HD, device=DEV, dtype=torch.bfloat16) * 0.5
        qp, qsf = nvfp4_quantize_q_cuda(q)
        qdq = dequant_q(qp, qsf).to(torch.bfloat16)

        ref = make_decode_wrapper(False, True, batch, kv_len).run(
            qdq, kv_cache, kv_cache_sf=kv_sf)
        out_ns, lse_ns = make_decode_wrapper(True, True, batch, kv_len).run(
            qp, kv_cache, kv_cache_sf=kv_sf, q_sf=qsf, return_lse=True)
        out_sp, lse_sp = make_decode_wrapper(True, False, batch, kv_len).run(
            qp, kv_cache, kv_cache_sf=kv_sf, q_sf=qsf, return_lse=True)

        mad_ns = (out_ns.float() - ref.float()).abs().max().item()
        cos_ns = F.cosine_similarity(out_ns.float().reshape(1, -1),
                                     ref.float().reshape(1, -1)).item()
        mad_sp = (out_sp.float() - ref.float()).abs().max().item()
        cos_sp = F.cosine_similarity(out_sp.float().reshape(1, -1),
                                     ref.float().reshape(1, -1)).item()
        # split merges partial softmax states in fp32; bf16 outputs may differ
        # from the single-pass result by last-ulp rounding. Gate on ulp-relative
        # output error (2 bf16 ulps of the output range) AND on the fp32 LSE
        # (merge corruption would show up there far above 1e-4).
        ulp2 = ref.float().abs().max().item() * (2.0 ** -7)
        lse_mad = (lse_sp - lse_ns).abs().max().item()
        if lse_mad > 0 and batch <= 8:
            # split actually engaged: yardstick = same data through the bf16
            # merge path (identical logit statistics, shared merge math)
            lse_gate = 3.0 * bf16_matched_control(batch, kv_len, kv_cache, kv_sf,
                                                  qdq)
        else:
            lse_gate = 1e-4  # split didn't engage (or huge batch): near-zero
        ok = (mad_ns == 0.0) and cos_sp >= 0.99999 and mad_sp <= ulp2 \
            and lse_mad <= lse_gate
        print(f"[decode b={batch} kv={kv_len}] nosplit: mad={mad_ns:.3e} "
              f"cos={cos_ns:.7f} | split: mad={mad_sp:.3e} (2ulp={ulp2:.3e}) "
              f"cos={cos_sp:.7f} lse_mad={lse_mad:.3e} (gate={lse_gate:.3e}) "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        allok &= ok
        del kv_cache, kv_sf
        torch.cuda.empty_cache()

print("=== J-3 split-KV regression geometry (prefill wrapper, qo<<kv) ===", flush=True)
for qo_len, kv_len in ((17, 32768), (1, 100352), (64, 65536)):
    pages = (kv_len + PAGE - 1) // PAGE
    kv_cache, kv_sf = make_fp4_kv(pages)
    q = torch.randn(qo_len, NQO, HD, device=DEV, dtype=torch.bfloat16) * 0.5
    qp, qsf = nvfp4_quantize_q_cuda(q)
    qdq = dequant_q(qp, qsf).to(torch.bfloat16)
    qi = torch.tensor([0, qo_len], device=DEV, dtype=torch.int32)
    ki = torch.tensor([0, pages], device=DEV, dtype=torch.int32)
    kidx = torch.arange(pages, device=DEV, dtype=torch.int32)
    kl = torch.tensor([kv_len - (pages - 1) * PAGE], device=DEV, dtype=torch.int32)

    def prefill_out(use_nvf4, qq, qsf_):
        w = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
        w.plan(qi, ki, kidx, kl, NQO, NKV, HD, PAGE, causal=True,
               sm_scale=1.0 / math.sqrt(HD), q_data_type=torch.bfloat16,
               kv_data_type=torch.uint8, o_data_type=torch.bfloat16,
               use_nvf4_qk=use_nvf4)
        return w.run(qq, kv_cache, kv_cache_sf=kv_sf, q_sf=qsf_)

    ref = prefill_out(False, qdq, None)          # current path: split auto-disabled
    out = prefill_out(True, qp, qsf)             # a4q: split-KV re-enabled
    mad = (out.float() - ref.float()).abs().max().item()
    cos = F.cosine_similarity(out.float().reshape(1, -1),
                              ref.float().reshape(1, -1)).item()
    ulp2 = ref.float().abs().max().item() * (2.0 ** -7)
    ok = cos >= 0.99999 and mad <= ulp2
    print(f"[extend qo={qo_len} kv={kv_len}] a4q+split vs ref: mad={mad:.3e} "
          f"(2ulp={ulp2:.3e}) cos={cos:.7f} {'PASS' if ok else 'FAIL'}", flush=True)
    allok &= ok
    del kv_cache, kv_sf
    torch.cuda.empty_cache()

print("=== J-3 perf: batch-1 decode attention ===", flush=True)
for kv_len in (8192, 32768, 100352):
    pages = (kv_len + PAGE - 1) // PAGE
    kv_cache, kv_sf = make_fp4_kv(pages)
    q = torch.randn(1, NQO, HD, device=DEV, dtype=torch.bfloat16) * 0.5
    qp, qsf = nvfp4_quantize_q_cuda(q)
    qdq = dequant_q(qp, qsf).to(torch.bfloat16)
    kbf = torch.randn(pages, PAGE, NKV, HD, device=DEV, dtype=torch.bfloat16) * 0.5
    vbf = torch.randn_like(kbf)

    # K-1 methodology baseline: current fp4 path (split auto-disabled)
    qi = torch.tensor([0, 1], device=DEV, dtype=torch.int32)
    ki = torch.tensor([0, pages], device=DEV, dtype=torch.int32)
    kidx = torch.arange(pages, device=DEV, dtype=torch.int32)
    kl = torch.tensor([kv_len - (pages - 1) * PAGE], device=DEV, dtype=torch.int32)
    w_fp4 = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    w_fp4.plan(qi, ki, kidx, kl, NQO, NKV, HD, PAGE, causal=True,
               sm_scale=1.0 / math.sqrt(HD), q_data_type=torch.bfloat16,
               kv_data_type=torch.uint8, o_data_type=torch.bfloat16)
    ms_fp4 = bench(lambda: w_fp4.run(qdq, kv_cache, kv_cache_sf=kv_sf))

    # bf16 decode baseline (split on, decode wrapper)
    w_bf = BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD",
                                              use_tensor_cores=True, backend="fa2")
    indptr, indices, last = decode_plan_args(1, kv_len)
    w_bf.plan(indptr, indices, last, NQO, NKV, HD, PAGE,
              sm_scale=1.0 / math.sqrt(HD), q_data_type=torch.bfloat16,
              kv_data_type=torch.bfloat16, o_data_type=torch.bfloat16)
    kvbf = torch.stack([kbf, vbf], dim=1)
    ms_bf = bench(lambda: w_bf.run(qdq, kvbf))

    w_ns = make_decode_wrapper(True, True, 1, kv_len)
    ms_ns = bench(lambda: w_ns.run(qp, kv_cache, kv_cache_sf=kv_sf, q_sf=qsf))
    w_sp = make_decode_wrapper(True, False, 1, kv_len)
    ms_sp = bench(lambda: w_sp.run(qp, kv_cache, kv_cache_sf=kv_sf, q_sf=qsf))

    gate = ms_fp4 / ms_sp
    tag = "PASS" if (kv_len != 32768 or gate >= 2.0) else "FAIL"
    print(f"[perf b=1 kv={kv_len}] fp4-current={ms_fp4:.3f} ms  bf16={ms_bf:.3f} ms  "
          f"a4q-nosplit={ms_ns:.3f} ms  a4q-split={ms_sp:.3f} ms  "
          f"fp4/a4q-split={gate:.2f}x {tag if kv_len == 32768 else ''}", flush=True)
    if kv_len == 32768:
        allok &= gate >= 2.0
    del kv_cache, kv_sf, kbf, vbf, kvbf
    torch.cuda.empty_cache()

print("A4Q_J3 " + ("PASS" if allok else "FAIL"), flush=True)
