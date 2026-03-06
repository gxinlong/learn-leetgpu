#include <cuda_runtime.h>
#include "../../include/cuda_ide.h"  // 仅供 IDE：threadIdx/blockIdx/blockDim 桩声明，nvcc 编译时无效

// 1. shared memory
// 2. A 矩阵转置避免 bank conflict
// 3. register tiling: 每个 thread 计算 TILE_REG x TILE_REG 的输出子块

#define BK 128
#define BM 16
#define BN 16

#define TILE_REG 2

#define OUT_M (TILE_REG * BM)
#define OUT_N (TILE_REG * BN)

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;

  __shared__ float S_A[BK][OUT_M];
  __shared__ float S_B[BK][OUT_N];

  float reg_A[TILE_REG];
  float reg_B[TILE_REG];
  float reg_C[TILE_REG][TILE_REG] = {0};

  int out_base_row = tid_y * TILE_REG + blockIdx.y * OUT_M;
  int out_base_col = tid_x * TILE_REG + blockIdx.x * OUT_N;

  for (int k_base = 0; k_base < K; k_base += BK) {
    // global data -> shared memory
    for (int m = tid_y; m < OUT_M; m += BM) {
      int global_row = blockIdx.y * OUT_M + m;
      for (int k = tid_x; k < BK; k += BN) {
        S_A[k][m] = (global_row < M && k_base + k < K) ? A[global_row * K + k_base + k] : 0;
      }
    }
    for (int k = tid_y; k < BK; k += BM) {
      for (int n = tid_x; n < OUT_N; n += BN) {
        int global_col = blockIdx.x * OUT_N + n;
        S_B[k][n] = (k_base + k < K && global_col < N) ? B[(k_base + k) * N + global_col] : 0;
      }
    }
    __syncthreads();

    // shared memory -> register
    for (int k = 0; k < BK; ++k) {
      #pragma unroll
      for (int r = 0; r < TILE_REG; ++r) {
        reg_A[r] = S_A[k][tid_y * TILE_REG + r];
      }
      #pragma unroll
      for (int c = 0; c < TILE_REG; ++c) {
        reg_B[c] = S_B[k][tid_x * TILE_REG + c];
      }
      
      #pragma unroll
      for (int r = 0; r < TILE_REG; ++r) {
        #pragma unroll
        for (int c = 0; c < TILE_REG; ++c) {
          reg_C[r][c] += reg_A[r] * reg_B[c];
        }
      }
    }

    __syncthreads();
  }
  #pragma unroll
  for (int r = 0; r < TILE_REG; ++r) {
    #pragma unroll
    for (int c = 0; c < TILE_REG; ++c) {
      if (out_base_row + r < M && out_base_col + c < N) {
        C[(out_base_row + r) * N + out_base_col + c] = reg_C[r][c];
      }
    }
  }

}


// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int K, int N) {
    dim3 threadsPerBlock(BN, BM);
    dim3 blocksPerGrid((N + OUT_N - 1) / OUT_N,
                       (M + OUT_M - 1) / OUT_M);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}
