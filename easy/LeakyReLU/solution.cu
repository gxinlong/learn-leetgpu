#include <cuda_runtime.h>
#include "../../include/cuda_ide.h"  // 仅供 IDE：threadIdx/blockIdx/blockDim 桩声明，nvcc 编译时无效

__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
  const int tx = threadIdx.x;
  const int g_id = blockDim.x * blockIdx.x + tx;

  if (g_id < N) {
    float value = input[g_id];
    output[g_id] = (value > 0) ? value : (value * float(0.01));
  }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
