#include <cuda_runtime.h>
#include "../../include/cuda_ide.h"  // 仅供 IDE：threadIdx/blockIdx/blockDim 桩声明，nvcc 编译时无效

__global__ void interleave_kernel(const float* A, const float* B, float2* output2, int N) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < N) {
    output2[tid] = {A[tid], B[tid]};
  }
}

// A, B, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float2* output2 = (float2*)output;

    interleave_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, output2, N);
    cudaDeviceSynchronize();
}
