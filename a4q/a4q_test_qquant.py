"""A4Q: acceptance test for nvfp4_quantize_q_cuda vs the torch reference.

Gate: bit-exact packed codes + sf bytes on random (bf16/fp16, several scales),
constructed tie/edge cases, and the real Gemma capture Q tensors.
"""
import torch
from flashinfer import nvfp4_quantize_q_cuda
from flashinfer.prefill import nvfp4_quantize_q

torch.manual_seed(5)
DEV = "cuda"
allok = True


def check(name, x):
    global allok
    p_ref, s_ref, _ = nvfp4_quantize_q(x)
    p_cu, s_cu = nvfp4_quantize_q_cuda(x)
    pm = (p_ref != p_cu).sum().item()
    sm = (s_ref != s_cu).sum().item()
    ok = pm == 0 and sm == 0
    total_p, total_s = p_ref.numel(), s_ref.numel()
    msg = f"[qquant {name}] packed_mismatch={pm}/{total_p} sf_mismatch={sm}/{total_s}"
    if not ok:
        # locate a few examples for diagnosis
        bad = (p_ref != p_cu).reshape(-1).nonzero()[:5].flatten().tolist()
        msg += f" first_bad_packed_idx={bad}"
        # dequant-level check: codes may differ only as -0 vs +0 etc.
        import torch.nn.functional as F
    print(msg + (" PASS" if ok else " FAIL"), flush=True)
    allok &= ok
    return ok


# random bf16 / fp16 at several scales (incl. tiny values to hit sf underflow)
for dt in (torch.bfloat16, torch.float16):
    for scale in (0.5, 4.0, 64.0, 1e-3, 1e-5):
        x = torch.randn(4096, 32, 128, device=DEV, dtype=dt) * scale
        check(f"randn dt={dt} scale={scale}", x)

# constructed tie / edge cases: amax=6 -> sf = e4m3(1.0) = 1.0 exactly, so
# y == x and midpoints are exact.
tie_vals = [6.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.75, -1.75, -3.5,
            -0.25, 0.0, -0.0, 0.1, -0.1]
x = torch.tensor(tie_vals, device=DEV, dtype=torch.bfloat16).repeat(64, 32, 8)
check("ties(sf=1)", x)
# same but scaled by an exact power of two (sf = 2^k exact in e4m3)
check("ties(sf=4)", x * 4.0)
# zero rows / tiny rows
z = torch.zeros(128, 32, 128, device=DEV, dtype=torch.bfloat16)
check("zeros", z)
check("subnormal-ish", torch.full((128, 32, 128), 1e-7, device=DEV,
                                  dtype=torch.bfloat16))

# real capture Q tensors
d = torch.load("/root/nano-kv/gemma_kv.pt", map_location="cpu")
for L in ("L41", "L47", "L5"):
    q = d["cap"][f"{L}.q"].to(DEV).to(torch.bfloat16)
    check(f"real {L}", q)

print("A4Q_QQUANT " + ("PASS" if allok else "FAIL"), flush=True)
