#include "cuda_runtime.h"
#include "float.h"


constexpr int MAX_TOPK = 256;

__device__ __forceinline__ void insert(float& value1, float& value2, int& idx1, int& idx2, float candidate, int candidate_idx) {
  bool greater = false;
  float old_value = 0.0f;
  int old_idx = 0;

  // 手动展开
  greater = (candidate > value1);
  old_value = value1;
  old_idx = idx1;
  value1 = (greater) ? candidate : old_value;
  idx1 = (greater) ? candidate_idx : old_idx;
  candidate = (greater) ? old_value : candidate;
  candidate_idx = (greater) ? old_idx : candidate_idx;

  greater = (candidate > value2);
  old_value = value2;
  old_idx = idx2;
  value2 = (greater) ? candidate : old_value;
  idx2 = (greater) ? candidate_idx : old_idx;
  // candidate = (greater) ? old_value : candidate;
  // candidate_idx = (greater) ? old_idx : candidate_idx;
}

#define FULL_MASK 0xffffffff

template<unsigned int stride>
__device__ __forceinline__ void reduce(float& value1, float& value2, int& idx1, int& idx2) {
  float candidate1 = __shfl_down_sync(FULL_MASK, value1, stride);
  int candidate_idx1 = __shfl_down_sync(FULL_MASK, idx1, stride);
  float candidate2 = __shfl_down_sync(FULL_MASK, value2, stride);
  int candidate_idx2 = __shfl_down_sync(FULL_MASK, idx2, stride);
  insert(value1, value2, idx1, idx2, candidate1, candidate_idx1);
  insert(value1, value2, idx1, idx2, candidate2, candidate_idx2);
}

__global__ void kernel(const float* logits, float* weights, int* indices, int M, int E, int k) {
  const int tc = threadIdx.x;
  const int tr = blockDim.y * blockIdx.x + threadIdx.y;

  if (tr >= M) {
    return;
  }
  float value1 = -FLT_MAX;
  float value2 = -FLT_MAX;
  int idx1 = -1;
  int idx2 = -1;
  
  for (int j = tc; j < E; j += warpSize) {
    const int input_idx = tr * E + j;
    if (j < E) {
      float candidate = logits[input_idx];
      insert(value1, value2, idx1, idx2, candidate, j);
    }
  }

  reduce<16>(value1, value2, idx1, idx2);
  reduce<8>(value1, value2, idx1, idx2);
  reduce<4>(value1, value2, idx1, idx2);
  reduce<2>(value1, value2, idx1, idx2);
  reduce<1>(value1, value2, idx1, idx2);
  if (tc == 0) {
    float max_v = value1;
    value1 = __expf(value1 - max_v);
    value2 = __expf(value2 - max_v);
    float sum = value1 + value2;
    value1 = value1 / sum;
    value2 = value2 / sum;
    const int output_idx = tr * k;
    weights[output_idx] = value1;
    weights[output_idx + 1] = value2;
    indices[output_idx] = idx1;
    indices[output_idx + 1] = idx2;
  }
}

// Correctness-first fallback for arbitrary 1 <= k <= 256. Each thread owns
// one row and keeps its sorted Top-K list in thread-local memory.
__global__ void generic_kernel(const float* logits, float* weights, int* indices,
                               int M, int E, int k) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M) {
    return;
  }

  float top_values[MAX_TOPK];
  int top_indices[MAX_TOPK];

  for (int i = 0; i < k; ++i) {
    top_values[i] = -FLT_MAX;
    top_indices[i] = -1;
  }

  const int input_base = row * E;
  for (int expert = 0; expert < E; ++expert) {
    const float candidate = logits[input_base + expert];

    // top_values is kept in descending order. Values that cannot enter the
    // current Top-K can be rejected without shifting the array.
    if (candidate <= top_values[k - 1]) {
      continue;
    }

    int pos = k - 1;
    while (pos > 0 && candidate > top_values[pos - 1]) {
      top_values[pos] = top_values[pos - 1];
      top_indices[pos] = top_indices[pos - 1];
      --pos;
    }
    top_values[pos] = candidate;
    top_indices[pos] = expert;
  }

  const float max_value = top_values[0];
  float sum = 0.0f;
  for (int i = 0; i < k; ++i) {
    top_values[i] = __expf(top_values[i] - max_value);
    sum += top_values[i];
  }

  const float inv_sum = 1.0f / sum;
  const int output_base = row * k;
  for (int i = 0; i < k; ++i) {
    weights[output_base + i] = top_values[i] * inv_sum;
    indices[output_base + i] = top_indices[i];
  }
}

extern "C" void solve(const float* logits, float* weights, int* indices, int M, int E, int k) {
  if (M <= 0 || E <= 0 || k <= 0 || k > E || k > MAX_TOPK) {
    return;
  }

  if (k == 2) {
    const int tile_c_size = 32;
    const int tile_r_size = 8;
    const int block_nums = (M + tile_r_size - 1) / tile_r_size;

    dim3 grid(block_nums, 1, 1);
    dim3 block(tile_c_size, tile_r_size, 1);
    kernel<<<grid, block>>>(logits, weights, indices, M, E, k);
    return;
  }

  const int thread_count = 64;
  const int block_count = (M + thread_count - 1) / thread_count;
  generic_kernel<<<block_count, thread_count>>>(
      logits, weights, indices, M, E, k);
}
