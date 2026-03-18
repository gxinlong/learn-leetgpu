#include <cuda_runtime.h>
#include <math.h>
#include "../../include/cuda_ide.h"  // 仅供 IDE：threadIdx/blockIdx/blockDim 桩声明，nvcc 编译时无效

__global__ void sigmoid_kernel(const float* X, float* Y, int N) {
  const int gid = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (gid < N) {
    float value = X[gid];
    Y[gid] = 1.0f / (1.0f + __expf(-value));
  }
}

// X, Y are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* X, float* Y, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y, N);
    cudaDeviceSynchronize();
}
