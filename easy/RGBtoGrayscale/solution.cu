#include <cuda_runtime.h>
#include "../../include/cuda_ide.h"  // 仅供 IDE：threadIdx/blockIdx/blockDim 桩声明，nvcc 编译时无效

__global__ void rgb_to_grayscale_kernel(const float3* input, float* output, int N) {
  const int gid = blockDim.x * blockIdx.x + threadIdx.x;

  if (gid < N) {
    float3 value = input[gid];
    output[gid] = 0.299f * value.x + 0.587f * value.y + 0.114f * value.z;
  }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int width, int height) {
    int total_pixels = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    const float3* input_f3 = (const float3*)input;

    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_f3, output, total_pixels);
    cudaDeviceSynchronize();
}
