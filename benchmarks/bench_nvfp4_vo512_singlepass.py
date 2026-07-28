"""Bench: single-pass NVFP4 qk512/vo512 VO-split vs the two-pass qk512/vo256 path.

Measures wrapper.run() wall time (CUDA events, warmup + mean-of-N) for the
single-pass VO-split kernel against the two-pass reference (two runs of the
asymmetric qk512/vo256 kernel over byte-sliced V halves, both under one timer --
the same V bytes as the single-pass full-512 tensors). Reports the per-shape
saving. Gemma-4 / DiffusionGemma full-attention geometry (symmetric head_dim
512). Kernel-level, so run on a dedicated GPU for stable numbers.

Usage:
    python benchmarks/bench_nvfp4_vo512_singlepass.py
"""

import argparse

import torch

import flashinfer
from tests.test_helpers.utils_fp4 import create_nvfp4_kv

HEAD_DIM = 512
HALF = HEAD_DIM // 2


def _bench(fn, iters, warmup=5):
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


def _plan(q_indptr, kv_indptr, kv_indices, kv_last, nq, nkv, head_dim_vo,
          page_size, causal, q_dtype, workspace):
    w = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    w.plan(q_indptr, kv_indptr, kv_indices, kv_last, nq, nkv, HEAD_DIM,
           page_size, head_dim_vo=head_dim_vo, causal=causal,
           pos_encoding_mode="NONE", logits_soft_cap=0.0,
           kv_data_type=torch.uint8, q_data_type=q_dtype)
    return w


def run_cell(tag, batch_size, qo_len, kv_len, num_kv_heads, causal, q_dtype,
             iters):
    dev = "cuda:0"
    page_size = 16
    num_qo_heads = 2 * num_kv_heads
    q = torch.randn(batch_size * qo_len, num_qo_heads, HEAD_DIM, device=dev,
                    dtype=q_dtype) * 0.5
    q_indptr = (torch.arange(0, batch_size + 1, dtype=torch.int32) * qo_len).to(dev)
    npages = (kv_len + page_size - 1) // page_size
    total = npages * batch_size
    kv_indptr = (torch.arange(0, batch_size + 1, dtype=torch.int32) * npages).to(dev)
    kv_indices = torch.arange(0, total, dtype=torch.int32, device=dev)
    kv_last = torch.full((batch_size,), (kv_len - 1) % page_size + 1,
                         dtype=torch.int32, device=dev)

    kv_shape = (total, page_size, num_kv_heads, HEAD_DIM // 2)
    k_packed, k_sf, k_gs = create_nvfp4_kv(kv_shape, dev)
    v_packed, v_sf, v_gs = create_nvfp4_kv(kv_shape, dev)
    ks, vs = k_gs.item(), v_gs.item()

    w1 = _plan(q_indptr, kv_indptr, kv_indices, kv_last, num_qo_heads,
               num_kv_heads, HEAD_DIM, page_size, causal, q_dtype, _WS)

    def single():
        w1.run(q, (k_packed, v_packed), k_scale=ks, v_scale=vs,
               kv_cache_sf=(k_sf, v_sf))

    vh, vsfh = [], []
    for lo_b, lo_sf in ((0, 0), (HALF // 2, HALF // 16)):
        vh.append(v_packed[..., lo_b:lo_b + HALF // 2].contiguous())
        vsfh.append(v_sf[..., lo_sf:lo_sf + HALF // 16].contiguous())
    w2 = _plan(q_indptr, kv_indptr, kv_indices, kv_last, num_qo_heads,
               num_kv_heads, HALF, page_size, causal, q_dtype, _WS)

    def two():
        w2.run(q, (k_packed, vh[0]), k_scale=ks, v_scale=vs,
               kv_cache_sf=(k_sf, vsfh[0]))
        w2.run(q, (k_packed, vh[1]), k_scale=ks, v_scale=vs,
               kv_cache_sf=(k_sf, vsfh[1]))

    t1 = _bench(single, iters)
    t2 = _bench(two, iters)
    save = 100.0 * (1.0 - t1 / t2)
    print(f"[{tag}] q={q_dtype} single={t1:.4f}ms two-pass={t2:.4f}ms "
          f"ratio={t1 / t2:.3f} saves={save:.1f}%", flush=True)
    del k_packed, v_packed, k_sf, v_sf, vh, vsfh, q
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    args = ap.parse_args()
    qd = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    _WS = torch.empty(512 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    # prefill cells (long-q, CTA32): batch 1, qo == kv, causal
    run_cell("prefill-2k", 1, 2048, 2048, 8, True, qd, 20)
    run_cell("prefill-4k", 1, 4096, 4096, 8, True, qd, 20)
    run_cell("prefill-8k", 1, 8192, 8192, 8, True, qd, 10)
    # decode cells (short-q, CTA16): batch 8, qo == 1
    run_cell("decode-b8-4k", 8, 1, 4096, 8, False, qd, 50)
    run_cell("decode-b8-8k", 8, 1, 8192, 8, False, qd, 50)
    print("BENCH_DONE", flush=True)
