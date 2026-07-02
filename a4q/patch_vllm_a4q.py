#!/usr/bin/env python3
"""A4Q-WIRE: env-gated nvf4 block-scaled QK MMA for the fa2-nvfp4 KV PREFILL path
(VLLM_NVFP4_A4Q=1). Decode stays on the current path (A4Q v1 scope decision).

Patches the installed vllm/v1/attention/backends/flashinfer.py (idempotent;
marker A4Q-WIRE). Usage: python patch_vllm_a4q.py <path-to-flashinfer.py>

Wiring summary:
- _fa2_nvfp4_prefill_jit_args grows use_nvf4_qk: appends the maybe_q_sf
  additional tensor, sets jit_kwargs["use_nvf4_qk"], and suffixes the module
  uri with _a4q1 so a4q modules never collide with current-path modules.
- Builder gains self.a4q_prefill = env && use_fa2_nvfp4_kv && head_dim==128
  && vo_split==1 && !dcp (A4Q v1 kernel scope). Outside that scope the flag
  self-disables and everything runs the current path.
- The PLAIN prefill wrapper is constructed with the a4q jit module and its
  plan() calls pass use_nvf4_qk; at run() the fork wrapper auto-quantizes the
  bf16 prefill query slice with flashinfer.nvfp4_quantize_q_cuda and passes
  the packed q (bf16 view) + maybe_q_sf — so no forward-site changes needed.
- Fallbacks kept on the CURRENT path: mm-prefix custom-mask wrapper (built
  with use_a4q=False), DCP (already rejected for fa2-nvfp4), cascade wrapper,
  decode wrapper, and the env-gated spark fresh-wrapper replay tracer.
- Precedence over K0B-QF16: when VLLM_NVFP4_A4Q is set, q_data_type is forced
  back to model dtype (q travels as packed uint8 viewed as bf16, not fp16).

Requires the fi-a4q FlashInfer fork (branch a4q-prototype) with
use_nvf4_qk / maybe_q_sf / nvfp4_quantize_q_cuda support installed.
"""
import ast
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
if "A4Q-WIRE" in src:
    print("ALREADY PATCHED")
    sys.exit(0)
assert "K0B-QF16" in src, "expected the K0B-QF16 patch to be applied first"


def rep(old, new, n=1):
    global src
    cnt = src.count(old)
    assert cnt == n, f"anchor x{n} expected, found x{cnt}: {old[:80]!r}"
    src = src.replace(old, new)


# --- 1. env helper + jit-args builder ---------------------------------------
rep(
    "def _fa2_nvfp4_prefill_jit_args(\n",
    "def _vllm_nvfp4_a4q_enabled() -> bool:  # A4Q-WIRE\n"
    '    return os.environ.get("VLLM_NVFP4_A4Q", "0") not in ("", "0")\n'
    "\n"
    "\n"
    "def _fa2_nvfp4_prefill_jit_args(\n",
)
rep(
    "    use_fp16_qk_reduction: bool = False,\n"
    ") -> tuple[list[Any], dict[str, Any]]:",
    "    use_fp16_qk_reduction: bool = False,\n"
    "    use_nvf4_qk: bool = False,  # A4Q-WIRE\n"
    ") -> tuple[list[Any], dict[str, Any]]:",
)
rep(
    '        f"fp16_qk_{int(use_fp16_qk_reduction)}"\n    )',
    '        f"fp16_qk_{int(use_fp16_qk_reduction)}"\n'
    '        + ("_a4q1" if use_nvf4_qk else "")  # A4Q-WIRE\n'
    "    )",
)
rep(
    '            "maybe_k_cache_sf",\n'
    '            "maybe_v_cache_sf",\n'
    "        ],\n"
    "        [\n"
    '            "uint8_t",\n'
    '            "int32_t",\n'
    '            "float",\n'
    '            "uint32_t",\n'
    '            "uint16_t",\n'
    '            "uint16_t",\n'
    '            "uint8_t",\n'
    '            "uint8_t",\n'
    "        ],",
    '            "maybe_k_cache_sf",\n'
    '            "maybe_v_cache_sf",\n'
    "        ]\n"
    '        + (["maybe_q_sf"] if use_nvf4_qk else []),  # A4Q-WIRE\n'
    "        [\n"
    '            "uint8_t",\n'
    '            "int32_t",\n'
    '            "float",\n'
    '            "uint32_t",\n'
    '            "uint16_t",\n'
    '            "uint16_t",\n'
    '            "uint8_t",\n'
    '            "uint8_t",\n'
    "        ]\n"
    '        + (["uint8_t"] if use_nvf4_qk else []),  # A4Q-WIRE',
)
rep(
    '        "use_fp16_qk_reduction": use_fp16_qk_reduction,\n'
    '        "fp8_enabled": False,\n'
    "    }",
    '        "use_fp16_qk_reduction": use_fp16_qk_reduction,\n'
    '        "fp8_enabled": False,\n'
    '        "use_nvf4_qk": use_nvf4_qk,  # A4Q-WIRE\n'
    "    }",
)

# --- 2. builder flag (after vo_split is known) -------------------------------
rep(
    "        self.vo_split = _vo_split_factor(\n"
    "            self.head_dim, self.use_fa2_nvfp4_kv\n"
    "        )\n",
    "        self.vo_split = _vo_split_factor(\n"
    "            self.head_dim, self.use_fa2_nvfp4_kv\n"
    "        )\n"
    "        # A4Q-WIRE: nvf4 block-scaled QK MMA for prefill (env-gated;\n"
    "        # v1 kernel scope: head_dim 128, no VO split, no DCP). Outside\n"
    "        # scope the flag self-disables and the current path runs.\n"
    "        self.a4q_prefill = (\n"
    "            _vllm_nvfp4_a4q_enabled()\n"
    "            and self.use_fa2_nvfp4_kv\n"
    "            and self.head_dim == 128\n"
    "            and self.vo_split == 1\n"
    "            and not self.use_dcp\n"
    "        )\n"
    "        if self.a4q_prefill:\n"
    "            logger.info_once(\n"
    '                "A4Q: nvf4 block-scaled QK MMA enabled for FA2 NVFP4 '
    'prefill."\n'
    "            )\n"
    "        elif _vllm_nvfp4_a4q_enabled() and self.use_fa2_nvfp4_kv:\n"
    "            logger.info_once(\n"
    '                "A4Q requested but out of v1 scope (head_dim=%d, '
    'vo_split=%d, dcp=%s); using the current fp4 prefill path.",\n'
    "                self.head_dim, self.vo_split, self.use_dcp,\n"
    "            )\n",
)

# --- 3. A4Q precedence over K0B-QF16 (both q_data_type sites) ----------------
rep(
    '            if _os.environ.get("VLLM_NVFP4_KV_QF16") and getattr(\n'
    '                    self, "use_fa2_nvfp4_kv", False):\n'
    "                self.q_data_type = torch.float16\n",
    '            if _os.environ.get("VLLM_NVFP4_KV_QF16") and getattr(\n'
    '                    self, "use_fa2_nvfp4_kv", False):\n'
    "                self.q_data_type = torch.float16\n"
    "            if _vllm_nvfp4_a4q_enabled() and getattr(\n"
    '                    self, "use_fa2_nvfp4_kv", False):\n'
    "                # A4Q-WIRE: precedence over QF16 - q travels as packed\n"
    "                # uint8 viewed as the model dtype, not fp16.\n"
    "                self.q_data_type = self.model_config.dtype\n",
    n=2,
)

# --- 4. wrapper construction --------------------------------------------------
rep(
    "    def _make_paged_prefill_wrapper(self) -> "
    "BatchPrefillWithPagedKVCacheWrapper:\n"
    "        if self.use_fa2_nvfp4_kv:\n",
    "    def _make_paged_prefill_wrapper(\n"
    "        self, use_a4q: bool | None = None\n"
    "    ) -> BatchPrefillWithPagedKVCacheWrapper:\n"
    "        if use_a4q is None:  # A4Q-WIRE\n"
    "            use_a4q = self.a4q_prefill\n"
    "        if self.use_fa2_nvfp4_kv:\n",
)
rep(
    "                use_sliding_window=self.window_left >= 0,\n"
    "                use_logits_soft_cap=(self.logits_soft_cap or 0.0) > 0,\n"
    "            )",
    "                use_sliding_window=self.window_left >= 0,\n"
    "                use_logits_soft_cap=(self.logits_soft_cap or 0.0) > 0,\n"
    "                use_nvf4_qk=use_a4q,  # A4Q-WIRE\n"
    "            )",
)
rep(
    "            self._mm_prefill_wrapper = self._make_paged_prefill_wrapper()\n",
    "            # A4Q-WIRE: mm-prefix custom-mask path stays on the current\n"
    "            # (non-a4q) prefill kernels.\n"
    "            self._mm_prefill_wrapper = self._make_paged_prefill_wrapper(\n"
    "                use_a4q=False\n"
    "            )\n",
)

# --- 5. plan sites ------------------------------------------------------------
# 5a. shared mm-group plan: the plain-causal group reuses the (possibly a4q)
#     main prefill wrapper; mm custom-mask groups use the non-a4q mm wrapper.
rep(
    "                q_data_type=self.q_data_type,\n"
    "                kv_data_type=self.kv_cache_dtype,\n"
    "                o_data_type=o_dtype,\n"
    "                fixed_split_size=self.prefill_fixed_split_size,\n"
    "                disable_split_kv=self.disable_split_kv,\n"
    "            )\n"
    "            wrapper.vllm_prefill_fixed_split_size = "
    "self.prefill_fixed_split_size",
    "                q_data_type=self.q_data_type,\n"
    "                kv_data_type=self.kv_cache_dtype,\n"
    "                o_data_type=o_dtype,\n"
    "                fixed_split_size=self.prefill_fixed_split_size,\n"
    "                disable_split_kv=self.disable_split_kv,\n"
    "                use_nvf4_qk=(not group_mm) and self.a4q_prefill,"
    "  # A4Q-WIRE\n"
    "            )\n"
    "            wrapper.vllm_prefill_fixed_split_size = "
    "self.prefill_fixed_split_size",
)
# 5b. plain prefill plan.
rep(
    "                            o_data_type=o_dtype,\n"
    "                            fixed_split_size=self.prefill_fixed_split_size,\n"
    "                            disable_split_kv=self.disable_split_kv,\n"
    "                        )",
    "                            o_data_type=o_dtype,\n"
    "                            fixed_split_size=self.prefill_fixed_split_size,\n"
    "                            disable_split_kv=self.disable_split_kv,\n"
    "                            use_nvf4_qk=self.a4q_prefill,  # A4Q-WIRE\n"
    "                        )",
)

ast.parse(src)
open(path, "w", encoding="utf-8").write(src)
print(
    "PATCHED A4Q-WIRE (jit args + builder flag + QF16 precedence x2 + "
    "wrapper ctor + 2 plan sites; forward untouched - the fork wrapper "
    "auto-quantizes q via nvfp4_quantize_q_cuda)"
)
