#include <cuda_runtime.h>
#include "../../include/cuda_ide.h"  // 仅供 IDE：threadIdx/blockIdx/blockDim 桩声明，nvcc 编译时无效

#define B 256

__global__ void relu_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
  const int tx = threadIdx.x;
  const int global_id = blockIdx.x * blockDim.x + tx;

  if (global_id < N) {
    float value = input[global_id];
    output[global_id] = max(value, 0.0);
  }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = B;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
