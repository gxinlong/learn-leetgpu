#include <cuda_runtime.h>
#include "../../include/cuda_ide.h"  // 仅供 IDE：threadIdx/blockIdx/blockDim 桩声明，nvcc 编译时无效

#define BLOCK 256
#define MAX_KERNEL_SIZE 2047
#define TILE 4     // 每个 thread 处理多少个 float

#define PAD_FLOAT4(x) (x - ((x) % 4) + 4)

#define MAX_SMEM_INPUT_SIZE (MAX_KERNEL_SIZE + BLOCK*TILE - 1)

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__constant__ float d_kernel[MAX_KERNEL_SIZE];

__global__ void convolution_1d_kernel(float* input, float* kernel, float* output, int input_size, int kernel_size, int output_size) {
  const int tx = threadIdx.x;
  const int block_st = (blockIdx.x * blockDim.x) * TILE;
  const int output_base = block_st + tx * TILE;

  input = &(input[block_st]);   // 偏移输入地址到 block

  float reg_input[TILE*2];
  float reg_kernel[TILE];
  float reg_output[TILE] = {0};

  __shared__ float s_input[PAD_FLOAT4(MAX_SMEM_INPUT_SIZE)];
  // __shared__ float s_kernel[PAD_FLOAT4(MAX_KERNEL_SIZE)];

  const int share_input_size = PAD_FLOAT4(kernel_size + BLOCK*TILE - 1); // share input padding float4 0 后的大小
  const int share_kernel_size = PAD_FLOAT4(kernel_size);
  // 预加载 global mem -> share mem
  for (int i = tx; i < share_input_size; i += BLOCK) {
    s_input[i] = (block_st + i < input_size) ? input[i] : 0.0;
  }
  // for (int i = tx; i < share_kernel_size; i += BLOCK) {
  //   s_kernel[i] = (i < kernel_size) ? kernel[i] : 0.0;
  // }
  __syncthreads();

  // 计算
  if (output_base < output_size) {
    const int thread_st = tx * TILE;
    for (int k = 0; k < share_kernel_size; k += TILE) {
      FETCH_FLOAT4(reg_input[0]) = FETCH_FLOAT4(s_input[thread_st + k]);
      FETCH_FLOAT4(reg_input[TILE]) = FETCH_FLOAT4(s_input[thread_st + k + TILE]);
      FETCH_FLOAT4(reg_kernel[0]) = FETCH_FLOAT4(d_kernel[k]);

      for (int i = 0; i < TILE; ++i) {
        for (int j = 0; j < TILE; ++j) {
          reg_output[i] += reg_kernel[j] * reg_input[i + j];
        }
      }
    }

    if (output_base + TILE < output_size) {
      FETCH_FLOAT4(output[output_base]) = FETCH_FLOAT4(reg_output[0]);
    } else {
      for (int i = output_base; i < output_size; ++i) {
        output[i] = reg_output[i - output_base];
      }
    }
  }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size,
                      int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = BLOCK;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    cudaMemcpyToSymbol(d_kernel, kernel, kernel_size * sizeof(float));

    // PAD_FLOAT4 可能扩展到 kernel_size 之外，d_kernel 全局常量内存会残留上次测试数据
    // 必须将 padding 区域显式置零
    int padded_kernel_size = PAD_FLOAT4(kernel_size);
    int pad_count = padded_kernel_size - kernel_size;
    if (pad_count > 0) {
        float zeros[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        cudaMemcpyToSymbol(d_kernel, zeros, pad_count * sizeof(float), kernel_size * sizeof(float));
    }

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>((float*)input, (float*)kernel, output, input_size,
                                                              kernel_size, output_size);
    cudaDeviceSynchronize();
}
