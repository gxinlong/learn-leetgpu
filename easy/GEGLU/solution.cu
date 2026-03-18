#include <cuda_runtime.h>
#include "../../include/cuda_ide.h"  // 仅供 IDE：threadIdx/blockIdx/blockDim 桩声明，nvcc 编译时无效

__global__ void geglu_kernel(const float* input, float* output, int halfN) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < halfN) {
    float left = input[tid];
    float right = input[tid + halfN];
    output[tid] = left * (0.5f * right * (1.0f + erff(right * 0.70710678118f)));
  }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
  int halfN = N / 2;
  int threadsPerBlock = 256;
  int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

  geglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
}
