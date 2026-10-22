#include "cuda_runtime.h"


#define FULL_MASK 0xffffffff
__device__ void warp32(float& sum) {
  sum += __shfl_down_sync(FULL_MASK, sum, 16);
  sum += __shfl_down_sync(FULL_MASK, sum, 8);
  sum += __shfl_down_sync(FULL_MASK, sum, 4);
  sum += __shfl_down_sync(FULL_MASK, sum, 2);
  sum += __shfl_down_sync(FULL_MASK, sum, 1);
}

template<unsigned int TILE_SIZE, unsigned int ITEMS_PER_THREAD>
__global__ void kernel(const float* a, const float* b, float* res, int N) {
  const int tx = threadIdx.x;
  constexpr int items_per_block = TILE_SIZE * ITEMS_PER_THREAD;
  const int block_st = blockIdx.x * items_per_block;

  float sum = 0.0f;
  __shared__ float s_sum[TILE_SIZE];
  for (int idx = tx; idx < items_per_block; idx += TILE_SIZE) {
    const int input_idx = block_st + idx;
    if (input_idx < N) {
      sum += a[input_idx] * b[input_idx];
    }
  }
  s_sum[tx] = sum;
  __syncthreads();

  for (int ceil = TILE_SIZE / 2; ceil >= 32; ceil >>= 1) {
    if (tx < ceil) {
      sum = sum + s_sum[tx + ceil];
      s_sum[tx] = sum;
    }
    __syncthreads();
  }

  warp32(sum);

  if (tx == 0) {
    atomicAdd(res, sum);
  }
}

__global__ void naive_kernel(const float* a, const float* b, float* res, int N) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= N) return;

  float contrib = a[i] * b[i];

  atomicAdd(res, contrib);
}


extern "C" void solve(const float* a, const float* b, float* res, int N) {
  const int tile_size = 256;
  const int items_per_thread = 64;
  const int items_per_block = tile_size * items_per_thread;
  const int block_nums = (N + items_per_block - 1) / items_per_block;
  // kernel<tile_size, items_per_thread><<<block_nums, tile_size>>>(a, b, res, N);
  const int block_nums_naive = (N + tile_size - 1) / tile_size;
  naive_kernel<<<block_nums_naive, tile_size>>>(a, b, res, N);
}