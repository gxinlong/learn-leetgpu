#include <cuda_runtime.h>
#include "../../include/cuda_ide.h"  // 仅供 IDE：threadIdx/blockIdx/blockDim 桩声明，nvcc 编译时无效


__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
  int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
  int tid_y = threadIdx.y + blockIdx.y * blockDim.y;

  if (tid_x >= N || tid_y >= M) return;

  float sum = 0;
  for (int k = 0; k < K; ++k) {
    sum += A[tid_y * K + k] * B[k * N + tid_x];
  }
  C[tid_y * N + tid_x] = sum;
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int K, int N) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}
