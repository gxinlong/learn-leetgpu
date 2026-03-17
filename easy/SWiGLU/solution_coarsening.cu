#include <cuda_runtime.h>
#include "../../include/cuda_ide.h"  // 仅供 IDE：threadIdx/blockIdx/blockDim 桩声明，nvcc 编译时无效

#define ITEMS_PER_THREAD 8
#define BLOCK 256
#define ITEMS_PER_BLOCK (BLOCK*ITEMS_PER_THREAD)

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
  const int bidx_base = blockIdx.x * ITEMS_PER_BLOCK;

  #pragma unroll 4
  for (int i = threadIdx.x; i < ITEMS_PER_BLOCK; i += BLOCK) {
    const int value_idx = bidx_base + i;
    if (value_idx < halfN) {
      float left = input[value_idx];
      float right = input[value_idx + halfN];
      output[value_idx] = left * right / (1.0f + expf(-left));
    }
  }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = BLOCK;
    int blocksPerGrid = (halfN + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
