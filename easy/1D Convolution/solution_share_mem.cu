#include <cuda_runtime.h>
#include "../../include/cuda_ide.h"  // 仅供 IDE：threadIdx/blockIdx/blockDim 桩声明，nvcc 编译时无效

#define BLOCK 256
#define MAX_KERNEL_SIZE 2047

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output, int input_size, int kernel_size, int output_size) {
  const int tx = threadIdx.x;
  const int block_st = blockIdx.x * blockDim.x;
  const int output_id = block_st + tx;

  const int share_input_size = kernel_size + BLOCK;

  __shared__ float s_input[MAX_KERNEL_SIZE + BLOCK];
  __shared__ float s_kernel[MAX_KERNEL_SIZE];

  // 加载 global mem -> share mem
  for (int i = tx; i < share_input_size; i += BLOCK) {
    if (block_st + i < input_size) {
      s_input[i] = input[block_st + i];
    }
  }
  for (int i = tx; i < kernel_size; i += BLOCK) {
    s_kernel[i] = kernel[i];
  }
  __syncthreads();

  // 计算
  if (output_id < output_size) {
    float sum = 0;
    for (int i = 0; i < kernel_size; ++i) {
      sum += s_input[i + tx] * s_kernel[i];
    }
    output[output_id] = sum;
  }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size,
                      int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = BLOCK;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size,
                                                              kernel_size, output_size);
    cudaDeviceSynchronize();
}
