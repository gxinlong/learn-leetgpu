#include <cuda_runtime.h>
#include "../../include/cuda_ide.h"  // 仅供 IDE：threadIdx/blockIdx/blockDim 桩声明，nvcc 编译时无效

#define B 128

__global__ void relu_kernel(const float* input, float* output, int N) {
  const int tx = threadIdx.x;
  const int global_id = (blockIdx.x * blockDim.x + tx) * 4;

  if (global_id + 3 < N) {
    float4 value = (reinterpret_cast<const float4*>(&(input[global_id])))[0];
    output[global_id] = (value.x > 0) ? value.x : 0.0;
    output[global_id + 1] = (value.y > 0) ? value.y : 0.0;
    output[global_id + 2] = (value.z > 0) ? value.z : 0.0;
    output[global_id + 3] = (value.w > 0) ? value.w : 0.0;
  }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = B;
    int blocksPerGrid = (N / 4 + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
