#include <cuda_runtime.h>
#include "../../include/cuda_ide.h"  // 仅供 IDE：threadIdx/blockIdx/blockDim 桩声明，nvcc 编译时无效

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output, int input_size, int kernel_size, int output_size) {
  const int tx = threadIdx.x;
  const int global_id = blockIdx.x * blockDim.x + tx;

  if (global_id < output_size) {
    float sum = 0;
    for (int i = 0; i < kernel_size; ++i) {
      sum += input[global_id + i] * kernel[i];
    }
    output[global_id] = sum;
  }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size,
                      int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size,
                                                              kernel_size, output_size);
    cudaDeviceSynchronize();
}
