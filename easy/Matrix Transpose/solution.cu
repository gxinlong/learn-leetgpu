#include <cuda_runtime.h>
#include "../../include/cuda_ide.h"  // 仅供 IDE：threadIdx/blockIdx/blockDim 桩声明，nvcc 编译时无效

#define BM 16
#define BN 16

__global__ void matrix_transpose_kernel(const float* input, float* output, int M, int N) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  const int input_block_row_st = by * BM;
  const int input_block_col_st = bx * BN;
  const int output_block_row_st = input_block_col_st;
  const int output_block_col_st = input_block_row_st;

  const int input_thread_row = input_block_row_st + ty;
  const int input_thread_col = input_block_col_st + tx;
  const int output_thread_row = output_block_row_st + tx;
  const int output_thread_col = output_block_col_st + ty;
  if (input_thread_row < M && input_thread_col < N)
    output[(output_thread_row) * M + output_thread_col] = input[(input_thread_row) * N + input_thread_col];
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int M, int N) {
    dim3 threadsPerBlock(BM, BN);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, M, N);
    cudaDeviceSynchronize();
}
