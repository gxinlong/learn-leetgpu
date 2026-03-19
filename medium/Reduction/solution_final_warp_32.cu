#include <cuda_runtime.h>
#include "../../include/cuda_ide.h"  // 仅供 IDE：threadIdx/blockIdx/blockDim 桩声明，nvcc 编译时无效

#define BLOCK_SIZE_N 256
#define ITEMS_PER_THREAD 8
#define ITEMS_PER_BLOCK (BLOCK_SIZE_N*ITEMS_PER_THREAD)

#define FULL_MASK 0xffffffff

__device__ void final_warp_32(float& sum) {
  sum += __shfl_down_sync(FULL_MASK, sum, 16);
  sum += __shfl_down_sync(FULL_MASK, sum, 8);
  sum += __shfl_down_sync(FULL_MASK, sum, 4);
  sum += __shfl_down_sync(FULL_MASK, sum, 2);
  sum += __shfl_down_sync(FULL_MASK, sum, 1);
}

__global__ void reduce_kernel(const float* input, float* output, int N) {
  const int tx = threadIdx.x;
  const int b_base = blockIdx.x * ITEMS_PER_BLOCK;
  
  __shared__ float s_input[BLOCK_SIZE_N];

  // 预取数据
  float sum = 0;
  for (int i = tx; i < ITEMS_PER_BLOCK; i += BLOCK_SIZE_N) {
    const int input_idx = b_base + i;
    sum += (input_idx < N) ? input[input_idx] : 0.0f;
  }
  s_input[tx] = sum;
  __syncthreads();
  
  // 计算
  for (int ceil = BLOCK_SIZE_N / 2; ceil >= 32; ceil >>= 1) {
    if (tx < ceil) {
      sum += s_input[tx + ceil];
      s_input[tx] = sum;
    }
    __syncthreads();
  }

  if (tx < 32) {
    final_warp_32(sum);
  }

  // 输出
  if (tx == 0) {
    atomicAdd(output, sum);
  }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
  int threadsPerBlock = BLOCK_SIZE_N;
  int blocksPerGrid = (N + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK;

  reduce_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
}
