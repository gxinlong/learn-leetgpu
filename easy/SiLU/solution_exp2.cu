#include <cuda_runtime.h>
#include "../../include/cuda_ide.h"  // 仅供 IDE：threadIdx/blockIdx/blockDim 桩声明，nvcc 编译时无效

__global__ void silu_kernel(const float* input, float* output, int N) {
  const int gid = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (gid < N) {
    float value = input[gid];
    output[gid] = value / (1 + exp2f(-value * 1.4426950408889634f));
  }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
