/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/pos_enc.cuh>

#include <cstdlib>
#include <cstdio>
#include <atomic>
#include <cstdint>
#include <type_traits>

#include "batch_prefill_config.inc"
#include "tvm/ffi/container/array.h"
#include "tvm_ffi_utils.h"

namespace flashinfer {

template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          PosEncodingMode POS_ENCODING_MODE, bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                   float* tmp_s, bool enable_pdl,
                                                   cudaStream_t stream);

template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          PosEncodingMode POS_ENCODING_MODE, bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithRaggedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                    float* tmp_s, bool enable_pdl,
                                                    cudaStream_t stream);

}  // namespace flashinfer

using namespace flashinfer;

using tvm::ffi::Array;
using tvm::ffi::Optional;

namespace {

bool SparkPrefillDebugEnabled() {
  const char* value = std::getenv("FLASHINFER_PREFILL_DEBUG_ONCE");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

uint64_t SparkNextPrefillDebugCallId() {
  static std::atomic<uint64_t> next_call_id{1};
  return next_call_id.fetch_add(1, std::memory_order_relaxed);
}

void SparkPrintDType(const char* name, DLDataType dtype) {
  std::fprintf(stderr, "%s_dtype={code=%u,bits=%u,lanes=%u} ", name, dtype.code, dtype.bits,
               dtype.lanes);
}

void SparkPrintTensorView(const char* name, const TensorView& tensor) {
  std::fprintf(stderr, "%s={ptr=%p,device=%d,ndim=%d,shape=[", name, tensor.data_ptr(),
               tensor.device().device_id, tensor.ndim());
  for (int i = 0; i < tensor.ndim(); ++i) {
    std::fprintf(stderr, "%s%lld", i == 0 ? "" : ",",
                 static_cast<long long>(tensor.size(i)));
  }
  std::fprintf(stderr, "],stride=[");
  for (int i = 0; i < tensor.ndim(); ++i) {
    std::fprintf(stderr, "%s%lld", i == 0 ? "" : ",",
                 static_cast<long long>(tensor.stride(i)));
  }
  std::fprintf(stderr, "],");
  SparkPrintDType("", tensor.dtype());
  std::fprintf(stderr, "} ");
}

void SparkPrintOptionalTensorView(const char* name, const Optional<TensorView>& maybe_tensor) {
  if (maybe_tensor.has_value()) {
    SparkPrintTensorView(name, maybe_tensor.value());
  } else {
    std::fprintf(stderr, "%s=null ", name);
  }
}

template <typename T>
constexpr bool SparkIsFp4x2KvType() {
#if defined(FLASHINFER_ENABLE_FP4_E2M1) && \
    (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120800)
  return std::is_same_v<T, __nv_fp4x2_e2m1>;
#else
  return false;
#endif
}

template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_>
void SparkPrintPrefillJitIdentity(uint64_t debug_call_id, const char* path,
                                  const PrefillPlanInfo& plan_info,
                                  int64_t layout, int64_t window_left, int64_t batch_size,
                                  int64_t num_qo_heads, int64_t num_kv_heads,
                                  int64_t page_size) {
  std::fprintf(stderr,
               "[flashinfer][prefill-debug] call_id=%llu module_uri=%s module_key=%s path=%s "
               "compiled={dtype_q=%s,dtype_kv=%s,"
               "dtype_o=%s,idtype=%s,head_dim_qk=%d,head_dim_vo=%d,require_fp4_kv=%d,"
               "use_swa=%d,use_logits_cap=%d,posenc=%d,use_fp16_qk_reduction=%d,"
               "sizeof_q=%zu,sizeof_kv=%zu,sizeof_o=%zu,sizeof_id=%zu,"
               "is_kv_fp4x2=%d,additional_tensors=%s,additional_tensor_dtypes=%s,"
               "additional_scalars=%s,additional_scalar_dtypes=%s} "
               "runtime={layout=%lld,window_left=%lld,batch_size=%lld,num_qo_heads=%lld,"
               "num_kv_heads=%lld,page_size=%lld,split_kv=%d,cta_tile_q=%d,"
               "enable_cuda_graph=%d,padded_batch_size=%u,total_num_rows=%u}\n",
               static_cast<unsigned long long>(debug_call_id), JIT_MODULE_URI, JIT_MODULE_KEY,
               path, JIT_DTYPE_Q_NAME, JIT_DTYPE_KV_NAME, JIT_DTYPE_O_NAME, JIT_IDTYPE_NAME,
               HEAD_DIM_QK, HEAD_DIM_VO, static_cast<int>(REQUIRE_FP4_KV_CACHE),
               static_cast<int>(USE_SLIDING_WINDOW), static_cast<int>(USE_LOGITS_SOFT_CAP),
               static_cast<int>(POS_ENCODING_MODE), static_cast<int>(USE_FP16_QK_REDUCTION),
               sizeof(DTypeQ_), sizeof(DTypeKV_), sizeof(DTypeO_), sizeof(IdType_),
               static_cast<int>(SparkIsFp4x2KvType<DTypeKV_>()),
               JIT_ADDITIONAL_TENSOR_NAMES, JIT_ADDITIONAL_TENSOR_DTYPES,
               JIT_ADDITIONAL_SCALAR_NAMES, JIT_ADDITIONAL_SCALAR_DTYPES,
               static_cast<long long>(layout), static_cast<long long>(window_left),
               static_cast<long long>(batch_size), static_cast<long long>(num_qo_heads),
               static_cast<long long>(num_kv_heads), static_cast<long long>(page_size),
               static_cast<int>(plan_info.split_kv), plan_info.cta_tile_q,
               static_cast<int>(plan_info.enable_cuda_graph), plan_info.padded_batch_size,
               plan_info.total_num_rows);
}

}  // namespace

Array<int64_t> BatchPrefillWithKVCachePlan(
    TensorView float_workspace_buffer, TensorView int_workspace_buffer,
    TensorView page_locked_int_workspace_buffer, TensorView qo_indptr, TensorView kv_indptr,
    TensorView kv_len_arr, int64_t total_num_rows, int64_t batch_size, int64_t num_qo_heads,
    int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph, int64_t head_dim_qk,
    int64_t head_dim_vo, bool causal, int64_t window_left, int64_t fixed_split_size,
    bool disable_split_kv, int64_t num_colocated_ctas = 0) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * get_element_size(float_workspace_buffer);
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * get_element_size(int_workspace_buffer);

  PrefillPlanInfo plan_info;

  ffi::CUDADeviceGuard device_guard(float_workspace_buffer.device().device_id);
  const cudaStream_t stream = get_stream(float_workspace_buffer.device());
  cudaError_t status = PrefillPlan<IdType>(
      float_workspace_buffer.data_ptr(), float_workspace_size_in_bytes,
      int_workspace_buffer.data_ptr(), page_locked_int_workspace_buffer.data_ptr(),
      int_workspace_size_in_bytes, plan_info, static_cast<IdType*>(qo_indptr.data_ptr()),
      static_cast<IdType*>(kv_indptr.data_ptr()), total_num_rows, batch_size, num_qo_heads,
      num_kv_heads, head_dim_qk, head_dim_vo, page_size, enable_cuda_graph,
      /*sizeof_dtype_o=*/2, window_left, fixed_split_size, disable_split_kv, num_colocated_ctas,
      stream);

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "Failed to plan prefill with error: " << cudaGetErrorString(status);

  return Array(plan_info.ToVector());
}

void BatchPrefillWithRaggedKVCacheRun(TensorView float_workspace_buffer,
                                      TensorView int_workspace_buffer, Array<int64_t> plan_info_vec,
                                      TensorView q, TensorView k, TensorView v,
                                      TensorView qo_indptr, TensorView kv_indptr, TensorView o,
                                      Optional<TensorView> maybe_lse, int64_t mask_mode_code,
                                      int64_t layout, int64_t window_left,
                                      bool enable_pdl ADDITIONAL_FUNC_PARAMS) {
  PrefillPlanInfo plan_info;
  plan_info.FromVector(std::vector<int64_t>(plan_info_vec.begin(), plan_info_vec.end()));
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);

  int64_t num_qo_heads = q.size(1);
  int64_t head_dim_qk = q.size(2);
  int64_t num_kv_heads = (kv_layout == QKVLayout::kNHD) ? k.size(1) : k.size(0);
  uint32_t q_stride_n = q.stride(0), q_stride_h = q.stride(1), k_stride_n, k_stride_h, v_stride_n,
           v_stride_h;
  if (kv_layout == QKVLayout::kNHD) {
    k_stride_n = k.stride(0);
    k_stride_h = k.stride(1);
    v_stride_n = v.stride(0);
    v_stride_h = v.stride(1);
  } else {
    k_stride_h = k.stride(0);
    k_stride_n = k.stride(1);
    v_stride_h = v.stride(0);
    v_stride_n = v.stride(1);
  }

  if (maybe_lse.has_value()) {
    const auto& lse = *maybe_lse;
    TVM_FFI_ICHECK_EQ(lse.size(0), q.size(0));
    TVM_FFI_ICHECK_EQ(lse.size(1), q.size(1));
  }

  void* float_buffer_ptr = float_workspace_buffer.data_ptr();
  void* int_buffer_ptr = int_workspace_buffer.data_ptr();

  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);

  ffi::CUDADeviceGuard device_guard(float_workspace_buffer.device().device_id);
  const cudaStream_t stream = get_stream(float_workspace_buffer.device());

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, USE_FP16_QK_REDUCTION, AttentionVariant,
      RaggedParams, PagedParams, [&] {
        RaggedParams params;
        static bool debug_printed = false;

        params.q = static_cast<DTypeQ*>(q.data_ptr());
        params.k = static_cast<DTypeKV*>(k.data_ptr());
        params.v = static_cast<DTypeKV*>(v.data_ptr());
        params.o = static_cast<DTypeO*>(o.data_ptr());
        params.lse =
            maybe_lse.has_value() ? static_cast<float*>(maybe_lse.value().data_ptr()) : nullptr;
        params.q_indptr = static_cast<IdType*>(qo_indptr.data_ptr());
        params.kv_indptr = static_cast<IdType*>(kv_indptr.data_ptr());
        params.num_qo_heads = num_qo_heads;
        params.num_kv_heads = num_kv_heads;
        params.group_size = uint_fastdiv(num_qo_heads / num_kv_heads);
        params.q_stride_n = q_stride_n;
        params.q_stride_h = q_stride_h;
        params.k_stride_n = k_stride_n;
        params.k_stride_h = k_stride_h;
        params.v_stride_n = v_stride_n;
        params.v_stride_h = v_stride_h;
        params.window_left = window_left;

        params.request_indices = nullptr;
        params.qo_tile_indices = nullptr;
        params.kv_tile_indices = nullptr;
        params.merge_indptr = nullptr;
        params.o_indptr = nullptr;
        params.kv_chunk_size_ptr = nullptr;
        params.block_valid_mask = nullptr;
        params.total_num_rows = nullptr;
        params.max_total_num_rows = 0;
        params.padded_batch_size = 0;
        params.partition_kv = false;

        ADDITIONAL_PARAMS_SETTER

        if (SparkPrefillDebugEnabled() && !debug_printed) {
          debug_printed = true;
          const uint64_t debug_call_id = SparkNextPrefillDebugCallId();
          SparkPrintPrefillJitIdentity<DTypeQ, DTypeKV, DTypeO, IdType>(
              debug_call_id, "ragged", plan_info, layout, window_left,
              /*batch_size=*/kv_indptr.size(0) - 1, num_qo_heads, num_kv_heads,
              /*page_size=*/0);
          std::fprintf(stderr,
                       "[flashinfer][prefill-debug] tensors call_id=%llu module_uri=%s "
                       "module_key=%s path=ragged ",
                       static_cast<unsigned long long>(debug_call_id), JIT_MODULE_URI,
                       JIT_MODULE_KEY);
          SparkPrintTensorView("float_workspace", float_workspace_buffer);
          SparkPrintTensorView("int_workspace", int_workspace_buffer);
          SparkPrintTensorView("q", q);
          SparkPrintTensorView("k", k);
          SparkPrintTensorView("v", v);
          SparkPrintTensorView("qo_indptr", qo_indptr);
          SparkPrintTensorView("kv_indptr", kv_indptr);
          SparkPrintTensorView("o", o);
          SparkPrintOptionalTensorView("maybe_lse", maybe_lse);
          std::fprintf(stderr, "\n");
        }

        DTypeO* tmp_v = nullptr;
        float* tmp_s = nullptr;

        params.request_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.request_indices_offset);
        params.qo_tile_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
        params.kv_tile_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_tile_indices_offset);
        params.o_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.o_indptr_offset);
        params.kv_chunk_size_ptr =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_chunk_size_ptr_offset);
        if (plan_info.split_kv) {
          params.merge_indptr =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.merge_indptr_offset);
          tmp_v = GetPtrFromBaseOffset<DTypeO>(float_buffer_ptr, plan_info.v_offset);
          tmp_s = GetPtrFromBaseOffset<float>(float_buffer_ptr, plan_info.s_offset);
          if (plan_info.enable_cuda_graph) {
            params.block_valid_mask =
                GetPtrFromBaseOffset<bool>(int_buffer_ptr, plan_info.block_valid_mask_offset);
          }
        }
        params.padded_batch_size = plan_info.padded_batch_size;
        params.max_total_num_rows = plan_info.total_num_rows;
        if (plan_info.enable_cuda_graph) {
          params.total_num_rows =
              GetPtrFromBaseOffset<uint32_t>(int_buffer_ptr, plan_info.total_num_rows_offset);
        }

        cudaError_t status = cudaSuccess;

        DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
          status = flashinfer::BatchPrefillWithRaggedKVCacheDispatched<
              CTA_TILE_Q, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
              /*use_fp16_qk_reduction=*/USE_FP16_QK_REDUCTION, MASK_MODE, AttentionVariant,
              RaggedParams>(params, tmp_v, tmp_s, enable_pdl, stream);
        });

        TVM_FFI_ICHECK(status == cudaSuccess)
            << "BatchPrefillWithRaggedKVCache failed with error " << cudaGetErrorString(status);
        return true;
      });
}

void BatchPrefillWithPagedKVCacheRun(TensorView float_workspace_buffer,
                                     TensorView int_workspace_buffer, Array<int64_t> plan_info_vec,
                                     TensorView q, TensorView paged_k_cache,
                                     TensorView paged_v_cache, TensorView qo_indptr,
                                     TensorView paged_kv_indptr, TensorView paged_kv_indices,
                                     TensorView paged_kv_last_page_len, TensorView o,
                                     Optional<TensorView> maybe_lse, int64_t mask_mode_code,
                                     int64_t layout, int64_t window_left,
                                     bool enable_pdl ADDITIONAL_FUNC_PARAMS) {
  PrefillPlanInfo plan_info;
  plan_info.FromVector(std::vector<int64_t>(plan_info_vec.begin(), plan_info_vec.end()));
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  int64_t batch_size = paged_kv_indptr.size(0) - 1;
  int64_t num_qo_heads = q.size(1);
  int64_t num_kv_heads, page_size;
  uint32_t head_dim_qk = q.size(2);
  if (kv_layout == QKVLayout::kHND) {
    num_kv_heads = paged_k_cache.size(1);
    page_size = paged_k_cache.size(2);
  } else {
    page_size = paged_k_cache.size(1);
    num_kv_heads = paged_k_cache.size(2);
  }

  if (maybe_lse) {
    const auto& lse = *maybe_lse;
    TVM_FFI_ICHECK_EQ(lse.size(0), q.size(0));
    TVM_FFI_ICHECK_EQ(lse.size(1), q.size(1));
  }

  void* float_buffer_ptr = static_cast<void*>(float_workspace_buffer.data_ptr());
  void* int_buffer_ptr = static_cast<void*>(int_workspace_buffer.data_ptr());

  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);

  // get q_stride_n and q_stride_h
  const auto q_stride_n = q.stride(0);
  const auto q_stride_h = q.stride(1);

  // get kv-cache strides
  auto k_cache_strides = paged_k_cache.strides();
  auto v_cache_strides = paged_v_cache.strides();
  TVM_FFI_ICHECK_EQ(paged_k_cache.ndim(), paged_v_cache.ndim());

  ffi::CUDADeviceGuard device_guard(float_workspace_buffer.device().device_id);
  const cudaStream_t stream = get_stream(float_workspace_buffer.device());

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, USE_FP16_QK_REDUCTION, AttentionVariant,
      RaggedParams, PagedParams, [&] {
        PagedParams params;
        static bool debug_printed = false;

        params.q = static_cast<DTypeQ*>(q.data_ptr());
        paged_kv_t<DTypeKV, IdType> paged_kv(
            num_kv_heads, page_size, HEAD_DIM_VO, batch_size, kv_layout,
            static_cast<DTypeKV*>(paged_k_cache.data_ptr()),
            static_cast<DTypeKV*>(paged_v_cache.data_ptr()), k_cache_strides.data(),
            v_cache_strides.data(),
            static_cast<IdType*>(paged_kv_indices.data_ptr()),
            static_cast<IdType*>(paged_kv_indptr.data_ptr()),
            static_cast<IdType*>(paged_kv_last_page_len.data_ptr()));
        params.paged_kv = paged_kv;
        params.q_indptr = static_cast<IdType*>(qo_indptr.data_ptr());
        params.o = static_cast<DTypeO*>(o.data_ptr());

        params.lse = maybe_lse ? static_cast<float*>(maybe_lse.value().data_ptr()) : nullptr;
        params.num_qo_heads = num_qo_heads;
        params.group_size = uint_fastdiv(num_qo_heads / paged_kv.num_heads);
        params.q_stride_n = q_stride_n;
        params.q_stride_h = q_stride_h;
        params.window_left = window_left;

        params.request_indices = nullptr;
        params.qo_tile_indices = nullptr;
        params.kv_tile_indices = nullptr;
        params.merge_indptr = nullptr;
        params.o_indptr = nullptr;
        params.kv_chunk_size_ptr = nullptr;
        params.block_valid_mask = nullptr;
        params.total_num_rows = nullptr;
        params.max_total_num_rows = 0;
        params.padded_batch_size = 0;
        params.partition_kv = false;

        ADDITIONAL_PARAMS_SETTER

        if (SparkPrefillDebugEnabled() && !debug_printed) {
          debug_printed = true;
          const uint64_t debug_call_id = SparkNextPrefillDebugCallId();
          SparkPrintPrefillJitIdentity<DTypeQ, DTypeKV, DTypeO, IdType>(
              debug_call_id, "paged", plan_info, layout, window_left, batch_size, num_qo_heads,
              num_kv_heads, page_size);
          std::fprintf(stderr,
                       "[flashinfer][prefill-debug] tensors call_id=%llu module_uri=%s "
                       "module_key=%s path=paged ",
                       static_cast<unsigned long long>(debug_call_id), JIT_MODULE_URI,
                       JIT_MODULE_KEY);
          SparkPrintTensorView("float_workspace", float_workspace_buffer);
          SparkPrintTensorView("int_workspace", int_workspace_buffer);
          SparkPrintTensorView("q", q);
          SparkPrintTensorView("paged_k_cache", paged_k_cache);
          SparkPrintTensorView("paged_v_cache", paged_v_cache);
          SparkPrintTensorView("qo_indptr", qo_indptr);
          SparkPrintTensorView("paged_kv_indptr", paged_kv_indptr);
          SparkPrintTensorView("paged_kv_indices", paged_kv_indices);
          SparkPrintTensorView("paged_kv_last_page_len", paged_kv_last_page_len);
          SparkPrintTensorView("o", o);
          SparkPrintOptionalTensorView("maybe_lse", maybe_lse);
          std::fprintf(stderr, "\n");
        }

        DTypeO* tmp_v = nullptr;
        float* tmp_s = nullptr;

        params.request_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.request_indices_offset);
        params.qo_tile_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
        params.kv_tile_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_tile_indices_offset);
        params.o_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.o_indptr_offset);
        params.kv_chunk_size_ptr =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_chunk_size_ptr_offset);
        if (plan_info.split_kv) {
          params.merge_indptr =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.merge_indptr_offset);
          tmp_v = GetPtrFromBaseOffset<DTypeO>(float_buffer_ptr, plan_info.v_offset);
          tmp_s = GetPtrFromBaseOffset<float>(float_buffer_ptr, plan_info.s_offset);
          if (plan_info.enable_cuda_graph) {
            params.block_valid_mask =
                GetPtrFromBaseOffset<bool>(int_buffer_ptr, plan_info.block_valid_mask_offset);
          }
        }
        params.padded_batch_size = plan_info.padded_batch_size;
        params.max_total_num_rows = plan_info.total_num_rows;
        if (plan_info.enable_cuda_graph) {
          params.total_num_rows =
              GetPtrFromBaseOffset<uint32_t>(int_buffer_ptr, plan_info.total_num_rows_offset);
        }

        cudaError_t status = cudaSuccess;

        DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
          status = flashinfer::BatchPrefillWithPagedKVCacheDispatched<
              CTA_TILE_Q, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
              /*use_fp16_qk_reduction=*/USE_FP16_QK_REDUCTION, MASK_MODE, AttentionVariant,
              PagedParams>(params, tmp_v, tmp_s, enable_pdl, stream);
        });

        TVM_FFI_ICHECK(status == cudaSuccess)
            << "BatchPrefillWithPagedKVCache failed with error " << cudaGetErrorString(status);
        return true;
      });
}
