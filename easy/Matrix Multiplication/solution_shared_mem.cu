#include <cuda_runtime.h>
#include "../../include/cuda_ide.h"  // 仅供 IDE：threadIdx/blockIdx/blockDim 桩声明，nvcc 编译时无效

// shared memory 优化版本

#define BLOCK_K 256
#define BLOCK_M 16
#define BLOCK_N 16

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  int idx_x = tid_x + blockIdx.x * blockDim.x;
  int idx_y = tid_y + blockIdx.y * blockDim.y;

  bool in_bounds = (idx_x < N && idx_y < M);

  __shared__ float sdata_A[BLOCK_M][BLOCK_K];
  __shared__ float sdata_B[BLOCK_K][BLOCK_N];

  float sum = 0;
  for (int ki = 0; ki < K; ki += BLOCK_K) {
    // 搬数据
    if (in_bounds) {
      for (int kk = tid_x; kk < BLOCK_K; kk += BLOCK_N) {
        sdata_A[tid_y][kk] = (ki + kk < K) ? A[idx_y * K + ki + kk] : 0.0f;
      }
      for (int kk = tid_y; kk < BLOCK_K; kk += BLOCK_M) {
        sdata_B[kk][tid_x] = (ki + kk < K) ? B[(ki + kk) * N + idx_x] : 0.0f;
      }
    }
    __syncthreads();

    // 运算
    if (in_bounds) {
      for (int kk = 0; kk < BLOCK_K && ki + kk < K; ++kk) {
        sum += sdata_A[tid_y][kk] * sdata_B[kk][tid_x];
      }
    }
    __syncthreads();
  }

  if (in_bounds) {
    C[idx_y * N + idx_x] = sum;
  }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int K, int N) {
    dim3 threadsPerBlock(BLOCK_N, BLOCK_M);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}
