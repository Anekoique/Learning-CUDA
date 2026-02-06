#include <vector>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../tester/utils.h"

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
extern "C" cudaError_t cudaGetDeviceProperties_v2(cudaDeviceProp* prop, int device) {
  return cudaGetDeviceProperties(prop, device);
}
#endif

template <typename T>
__device__ __forceinline__ float to_float(T v) {
  return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float to_float<half>(half v) {
  return __half2float(v);
}

template <typename T>
__device__ __forceinline__ T from_float(float v) {
  return static_cast<T>(v);
}

template <>
__device__ __forceinline__ half from_float<half>(float v) {
  return __float2half_rn(v);
}

__device__ __forceinline__ double warp_reduce_sum(double v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

__device__ __forceinline__ double kahan_add(double sum, double& c, double x) {
  const double y = x - c;
  const double t = sum + y;
  c = (t - sum) - y;
  return t;
}

__global__ void flash_attention_float_kernel(const float* q, const float* k, const float* v, float* o,
                                             int batch_size, int target_seq_len, int src_seq_len,
                                             int query_heads, int kv_heads, int head_dim,
                                             bool is_causal, float scale) {
  const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int total = batch_size * target_seq_len * query_heads;
  if (idx >= total) return;

  const int tq = target_seq_len * query_heads;
  const int b = idx / tq;
  const int rem = idx - b * tq;
  const int t = rem / query_heads;
  const int qh = rem - t * query_heads;

  const int group = query_heads / kv_heads;
  const int kvh = qh / group;

  const size_t q_base = (static_cast<size_t>(b) * target_seq_len + t) * query_heads * head_dim
                      + static_cast<size_t>(qh) * head_dim;

  float row_max = -INFINITY;
  for (int s = 0; s < src_seq_len; ++s) {
    if (is_causal && s > t) continue;
    const size_t kv_base = (static_cast<size_t>(b) * src_seq_len + s) * kv_heads * head_dim
                         + static_cast<size_t>(kvh) * head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      dot = fmaf(q[q_base + d], k[kv_base + d], dot);
    }
    const float score = dot * scale;
    row_max = fmaxf(row_max, score);
  }

  float denom = 0.0f;
  for (int s = 0; s < src_seq_len; ++s) {
    if (is_causal && s > t) continue;
    const size_t kv_base = (static_cast<size_t>(b) * src_seq_len + s) * kv_heads * head_dim
                         + static_cast<size_t>(kvh) * head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      dot = fmaf(q[q_base + d], k[kv_base + d], dot);
    }
    denom += __expf(dot * scale - row_max);
  }

  const float inv_denom = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
  for (int d = 0; d < head_dim; ++d) {
    float out = 0.0f;
    for (int s = 0; s < src_seq_len; ++s) {
      if (is_causal && s > t) continue;
      const size_t kv_base = (static_cast<size_t>(b) * src_seq_len + s) * kv_heads * head_dim
                           + static_cast<size_t>(kvh) * head_dim;
      float dot = 0.0f;
      for (int dd = 0; dd < head_dim; ++dd) {
        dot = fmaf(q[q_base + dd], k[kv_base + dd], dot);
      }
      const float w = __expf(dot * scale - row_max);
      out = fmaf(w, v[kv_base + d], out);
    }
    o[q_base + d] = out * inv_denom;
  }
}

template <typename T>
__global__ void trace_kernel(const T* input, size_t rows, size_t cols, T* out) {
  const size_t diag = rows < cols ? rows : cols;
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= diag) return;
  const T val = input[idx * cols + idx];
  atomicAdd(out, val);
}

template <typename T, int BLOCK_M, int BLOCK_N>
__global__ void flash_attention_tiled_kernel(const T* q, const T* k, const T* v, T* o,
                                             int batch_size, int target_seq_len, int src_seq_len,
                                             int query_heads, int kv_heads, int head_dim,
                                             bool is_causal, double scale) {
  using OutAccum = typename std::conditional<std::is_same<T, float>::value, double, float>::type;

  extern __shared__ unsigned char smem_raw[];
  float* s_k = reinterpret_cast<float*>(smem_raw);
  float* s_v = s_k + static_cast<size_t>(BLOCK_N) * head_dim;
  float* s_q = s_v + static_cast<size_t>(BLOCK_N) * head_dim;
  uintptr_t s_o_addr = reinterpret_cast<uintptr_t>(s_q + static_cast<size_t>(BLOCK_M) * head_dim);
  const uintptr_t align_mask = static_cast<uintptr_t>(alignof(OutAccum) - 1);
  s_o_addr = (s_o_addr + align_mask) & ~align_mask;
  OutAccum* s_o = reinterpret_cast<OutAccum*>(s_o_addr);

  const int b = blockIdx.z;
  const int qh = blockIdx.y;
  const int tile_m = blockIdx.x;

  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  const int t = tile_m * BLOCK_M + warp_id;
  const bool valid = (t < target_seq_len);

  const int kvh = static_cast<int>((static_cast<long long>(qh) * kv_heads) / query_heads);
  const size_t q_base = (static_cast<size_t>(b) * target_seq_len + t) * query_heads * head_dim
                      + static_cast<size_t>(qh) * head_dim;

  for (int d = threadIdx.x; d < BLOCK_M * head_dim; d += blockDim.x) {
    const int warp = d / head_dim;
    const int dd = d - warp * head_dim;
    const int tt = tile_m * BLOCK_M + warp;
    if (tt < target_seq_len) {
      const size_t q_idx = (static_cast<size_t>(b) * target_seq_len + tt) * query_heads * head_dim
                         + static_cast<size_t>(qh) * head_dim + dd;
      s_q[d] = to_float(q[q_idx]);
      s_o[d] = static_cast<OutAccum>(0);
    } else {
      s_q[d] = 0.0f;
      s_o[d] = static_cast<OutAccum>(0);
    }
  }
  __syncthreads();

  double m = -INFINITY;

  for (int tile_n = 0; tile_n < src_seq_len; tile_n += BLOCK_N) {
    for (int idx = threadIdx.x; idx < BLOCK_N * head_dim; idx += blockDim.x) {
      const int j = idx / head_dim;
      const int d = idx - j * head_dim;
      const int s = tile_n + j;
      if (s < src_seq_len) {
        const size_t kv_base = (static_cast<size_t>(b) * src_seq_len + s) * kv_heads * head_dim
                             + static_cast<size_t>(kvh) * head_dim + d;
        s_k[idx] = to_float(k[kv_base]);
      } else {
        s_k[idx] = 0.0f;
      }
    }
    __syncthreads();

    for (int j = 0; j < BLOCK_N; ++j) {
      const int s = tile_n + j;
      if (s >= src_seq_len) break;

      double partial = 0.0;
      for (int d = lane; d < head_dim; d += 32) {
        partial += static_cast<double>(s_q[warp_id * head_dim + d]) *
                   static_cast<double>(s_k[j * head_dim + d]);
      }
      double score = warp_reduce_sum(partial);
      score = __shfl_sync(0xffffffff, score, 0) * scale;
      if (!valid || (is_causal && s > t)) {
        score = -INFINITY;
      }
      if (lane == 0) {
        m = fmax(m, score);
      }
    }
    __syncthreads();
  }

  double l = 0.0;
  for (int tile_n = 0; tile_n < src_seq_len; tile_n += BLOCK_N) {
    for (int idx = threadIdx.x; idx < BLOCK_N * head_dim; idx += blockDim.x) {
      const int j = idx / head_dim;
      const int d = idx - j * head_dim;
      const int s = tile_n + j;
      if (s < src_seq_len) {
        const size_t kv_base = (static_cast<size_t>(b) * src_seq_len + s) * kv_heads * head_dim
                             + static_cast<size_t>(kvh) * head_dim + d;
        s_k[idx] = to_float(k[kv_base]);
        s_v[idx] = to_float(v[kv_base]);
      } else {
        s_k[idx] = 0.0f;
        s_v[idx] = 0.0f;
      }
    }
    __syncthreads();

    for (int j = 0; j < BLOCK_N; ++j) {
      const int s = tile_n + j;
      if (s >= src_seq_len) break;
      if (!valid || (is_causal && s > t)) continue;

      double partial = 0.0;
      for (int d = lane; d < head_dim; d += 32) {
        partial += static_cast<double>(s_q[warp_id * head_dim + d]) *
                   static_cast<double>(s_k[j * head_dim + d]);
      }
      double score = warp_reduce_sum(partial);
      score = __shfl_sync(0xffffffff, score, 0) * scale;

      double w = 0.0;
      if (lane == 0) {
        w = exp(score - m);
        l += w;
      }
      w = __shfl_sync(0xffffffff, w, 0);

      for (int d = lane; d < head_dim; d += 32) {
        s_o[warp_id * head_dim + d] =
            static_cast<OutAccum>(static_cast<double>(s_o[warp_id * head_dim + d]) +
                                  w * static_cast<double>(s_v[j * head_dim + d]));
      }
    }
    __syncthreads();
  }

  if (valid) {
    const double l_b = __shfl_sync(0xffffffff, l, 0);
    if (l_b > 0.0) {
      const double inv_l = 1.0 / l_b;
      for (int d = lane; d < head_dim; d += 32) {
        const float out = static_cast<float>(static_cast<double>(s_o[warp_id * head_dim + d]) * inv_l);
        const size_t o_idx = q_base + d;
        o[o_idx] = from_float<T>(out);
      }
    } else {
      for (int d = lane; d < head_dim; d += 32) {
        const size_t o_idx = q_base + d;
        o[o_idx] = from_float<T>(0.0f);
      }
    }
  }
}

template <typename T>
__global__ void flash_attention_fallback_kernel(const T* q, const T* k, const T* v, T* o,
                                                int batch_size, int target_seq_len, int src_seq_len,
                                                int query_heads, int kv_heads, int head_dim,
                                                bool is_causal, double scale) {
  const int t = blockIdx.x;
  const int qh = blockIdx.y;
  const int b = blockIdx.z;
  if (t >= target_seq_len) return;

  const int kvh = static_cast<int>((static_cast<long long>(qh) * kv_heads) / query_heads);
  const size_t q_base = (static_cast<size_t>(b) * target_seq_len + t) * query_heads * head_dim
                      + static_cast<size_t>(qh) * head_dim;

  __shared__ double m_shared;
  __shared__ double l_shared;
  if (threadIdx.x == 0) {
    double m = -INFINITY;
    for (int s = 0; s < src_seq_len; ++s) {
      if (is_causal && s > t) break;
      double dot = 0.0;
      double dot_c = 0.0;
      const size_t kv_base = (static_cast<size_t>(b) * src_seq_len + s) * kv_heads * head_dim
                           + static_cast<size_t>(kvh) * head_dim;
      for (int d = 0; d < head_dim; ++d) {
        const double prod = static_cast<double>(to_float(q[q_base + d])) *
                            static_cast<double>(to_float(k[kv_base + d]));
        dot = kahan_add(dot, dot_c, prod);
      }
      dot *= scale;
      if (dot > m) m = dot;
    }
    m_shared = m;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    double l = 0.0;
    double l_c = 0.0;
    for (int s = 0; s < src_seq_len; ++s) {
      if (is_causal && s > t) break;
      double dot = 0.0;
      double dot_c = 0.0;
      const size_t kv_base = (static_cast<size_t>(b) * src_seq_len + s) * kv_heads * head_dim
                           + static_cast<size_t>(kvh) * head_dim;
      for (int d = 0; d < head_dim; ++d) {
        const double prod = static_cast<double>(to_float(q[q_base + d])) *
                            static_cast<double>(to_float(k[kv_base + d]));
        dot = kahan_add(dot, dot_c, prod);
      }
      l = kahan_add(l, l_c, exp(dot * scale - m_shared));
    }
    l_shared = l;
  }
  __syncthreads();

  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    double out = 0.0;
    double out_c = 0.0;
    for (int s = 0; s < src_seq_len; ++s) {
      if (is_causal && s > t) break;
      const size_t kv_base = (static_cast<size_t>(b) * src_seq_len + s) * kv_heads * head_dim
                           + static_cast<size_t>(kvh) * head_dim;
      double dot = 0.0;
      double dot_c = 0.0;
      for (int dd = 0; dd < head_dim; ++dd) {
        const double prod = static_cast<double>(to_float(q[q_base + dd])) *
                            static_cast<double>(to_float(k[kv_base + dd]));
        dot = kahan_add(dot, dot_c, prod);
      }
      dot = exp(dot * scale - m_shared);
      out = kahan_add(out, out_c, dot * static_cast<double>(to_float(v[kv_base + d])));
    }
    const float out_val = (l_shared > 0.0) ? static_cast<float>(out / l_shared) : 0.0f;
    o[q_base + d] = from_float<T>(out_val);
  }
}

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  if (rows == 0 || cols == 0) {
    return T(0);
  }

  const size_t diag = rows < cols ? rows : cols;
  if (diag == 0) {
    return T(0);
  }

  T* d_input = nullptr;
  T* d_out = nullptr;
  RUNTIME_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_out, sizeof(T)));
  RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemset(d_out, 0, sizeof(T)));

  const int threads = 256;
  const int blocks = static_cast<int>((diag + threads - 1) / threads);
  trace_kernel<<<blocks, threads>>>(d_input, rows, cols, d_out);
  RUNTIME_CHECK(cudaGetLastError());

  T h_out = T(0);
  RUNTIME_CHECK(cudaMemcpy(&h_out, d_out, sizeof(T), cudaMemcpyDeviceToHost));

  RUNTIME_CHECK(cudaFree(d_input));
  RUNTIME_CHECK(cudaFree(d_out));
  return h_out;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  if (batch_size <= 0 || target_seq_len <= 0 || src_seq_len <= 0 ||
      query_heads <= 0 || kv_heads <= 0 || head_dim <= 0) {
    return;
  }

  const size_t q_size = static_cast<size_t>(batch_size) * target_seq_len * query_heads * head_dim;
  const size_t k_size = static_cast<size_t>(batch_size) * src_seq_len * kv_heads * head_dim;
  const size_t v_size = k_size;
  h_o.resize(q_size);

  T* d_q = nullptr;
  T* d_k = nullptr;
  T* d_v = nullptr;
  T* d_o = nullptr;

  RUNTIME_CHECK(cudaMalloc(&d_q, q_size * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_k, k_size * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_v, v_size * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_o, q_size * sizeof(T)));

  RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), k_size * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), v_size * sizeof(T), cudaMemcpyHostToDevice));

  const double scale = 1.0 / sqrt(static_cast<double>(head_dim));

  if constexpr (std::is_same<T, float>::value) {
    const int total = batch_size * target_seq_len * query_heads;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    flash_attention_float_kernel<<<blocks, threads>>>(
        reinterpret_cast<const float*>(d_q),
        reinterpret_cast<const float*>(d_k),
        reinterpret_cast<const float*>(d_v),
        reinterpret_cast<float*>(d_o),
        batch_size, target_seq_len, src_seq_len, query_heads, kv_heads, head_dim, is_causal,
        static_cast<float>(scale));
    RUNTIME_CHECK(cudaGetLastError());
  } else {
    int max_smem = 0;
    RUNTIME_CHECK(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlock, 0));

    constexpr int BLOCK_M = 4;
    const int threads = BLOCK_M * 32;
    const dim3 grid((target_seq_len + BLOCK_M - 1) / BLOCK_M, query_heads, batch_size);

    using OutAccum = typename std::conditional<std::is_same<T, float>::value, double, float>::type;
    const auto smem_for_block_n = [&](int block_n) -> size_t {
      const size_t base_smem =
          static_cast<size_t>((2 * block_n + BLOCK_M) * head_dim) * sizeof(float);
      return base_smem + (alignof(OutAccum) - 1) +
             static_cast<size_t>(BLOCK_M * head_dim) * sizeof(OutAccum);
    };

    bool launched = false;
    const size_t smem_32 = smem_for_block_n(32);
    const size_t smem_16 = smem_for_block_n(16);
    const size_t smem_8 = smem_for_block_n(8);
    const size_t smem_4 = smem_for_block_n(4);

    if (smem_32 <= static_cast<size_t>(max_smem)) {
      flash_attention_tiled_kernel<T, BLOCK_M, 32><<<grid, threads, smem_32>>>(
          d_q, d_k, d_v, d_o, batch_size, target_seq_len, src_seq_len,
          query_heads, kv_heads, head_dim, is_causal, scale);
      launched = true;
    } else if (smem_16 <= static_cast<size_t>(max_smem)) {
      flash_attention_tiled_kernel<T, BLOCK_M, 16><<<grid, threads, smem_16>>>(
          d_q, d_k, d_v, d_o, batch_size, target_seq_len, src_seq_len,
          query_heads, kv_heads, head_dim, is_causal, scale);
      launched = true;
    } else if (smem_8 <= static_cast<size_t>(max_smem)) {
      flash_attention_tiled_kernel<T, BLOCK_M, 8><<<grid, threads, smem_8>>>(
          d_q, d_k, d_v, d_o, batch_size, target_seq_len, src_seq_len,
          query_heads, kv_heads, head_dim, is_causal, scale);
      launched = true;
    } else if (smem_4 <= static_cast<size_t>(max_smem)) {
      flash_attention_tiled_kernel<T, BLOCK_M, 4><<<grid, threads, smem_4>>>(
          d_q, d_k, d_v, d_o, batch_size, target_seq_len, src_seq_len,
          query_heads, kv_heads, head_dim, is_causal, scale);
      launched = true;
    }

    if (launched) {
      RUNTIME_CHECK(cudaGetLastError());
    } else {
      flash_attention_fallback_kernel<T><<<dim3(target_seq_len, query_heads, batch_size), 256>>>(
          d_q, d_k, d_v, d_o,
          batch_size, target_seq_len, src_seq_len,
          query_heads, kv_heads, head_dim, is_causal,
          scale);
      RUNTIME_CHECK(cudaGetLastError());
    }
  }

  RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, q_size * sizeof(T), cudaMemcpyDeviceToHost));

  RUNTIME_CHECK(cudaFree(d_q));
  RUNTIME_CHECK(cudaFree(d_k));
  RUNTIME_CHECK(cudaFree(d_v));
  RUNTIME_CHECK(cudaFree(d_o));
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
