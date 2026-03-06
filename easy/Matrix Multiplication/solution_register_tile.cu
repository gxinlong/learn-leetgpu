#include <cuda_runtime.h>
#include "../../include/cuda_ide.h"  // 仅供 IDE：threadIdx/blockIdx/blockDim 桩声明，nvcc 编译时无效

// 1. shared memory
// 2. A 矩阵转置避免 bank conflict
// 3. register tiling: 每个 thread 计算 TILE_REG x TILE_REG 的输出子块

#define BLOCK_K 128
#define BLOCK_M 16
#define BLOCK_N 16

#define TILE_REG 2

// 每个 block 覆盖的输出尺寸
#define OUT_M (BLOCK_M * TILE_REG)
#define OUT_N (BLOCK_N * TILE_REG)

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;

  int out_row_base = blockIdx.y * OUT_M + tid_y * TILE_REG;
  int out_col_base = blockIdx.x * OUT_N + tid_x * TILE_REG;

  __shared__ float sdata_A[BLOCK_K][OUT_M + 1];
  __shared__ float sdata_B[BLOCK_K][OUT_N];

  float reg_a[TILE_REG];
  float reg_b[TILE_REG];
  float reg_C[TILE_REG][TILE_REG] = {0};

  for (int ki = 0; ki < K; ki += BLOCK_K) {
    // 搬数据: 每个 thread 协作加载整个 block 需要的 A/B tile
    // sdata_A[k][m] = A[block_row + m][ki + k]，转置存储
    // sdata_B[k][n] = B[ki + k][block_col + n]
    for (int m = tid_y; m < OUT_M; m += BLOCK_M) {
      int global_row = blockIdx.y * OUT_M + m;
      for (int kk = tid_x; kk < BLOCK_K; kk += BLOCK_N) {
        float val = 0.0f;
        if (global_row < M && ki + kk < K)
          val = A[global_row * K + ki + kk];
        sdata_A[kk][m] = val;
      }
    }
    for (int n = tid_x; n < OUT_N; n += BLOCK_N) {
      int global_col = blockIdx.x * OUT_N + n;
      for (int kk = tid_y; kk < BLOCK_K; kk += BLOCK_M) {
        float val = 0.0f;
        if (ki + kk < K && global_col < N)
          val = B[(ki + kk) * N + global_col];
        sdata_B[kk][n] = val;
      }
    }
    __syncthreads();

    // 运算: 遍历 K 维度做外积累加
    for (int kk = 0; kk < BLOCK_K; ++kk) {
      #pragma unroll
      for (int ri = 0; ri < TILE_REG; ++ri)
        reg_a[ri] = sdata_A[kk][tid_y * TILE_REG + ri];
      #pragma unroll
      for (int ci = 0; ci < TILE_REG; ++ci)
        reg_b[ci] = sdata_B[kk][tid_x * TILE_REG + ci];

      #pragma unroll
      for (int ri = 0; ri < TILE_REG; ++ri)
        #pragma unroll
        for (int ci = 0; ci < TILE_REG; ++ci)
          reg_C[ri][ci] += reg_a[ri] * reg_b[ci];
    }
    __syncthreads();
  }

  // 写回
  for (int ri = 0; ri < TILE_REG; ++ri) {
    for (int ci = 0; ci < TILE_REG; ++ci) {
      int row = out_row_base + ri;
      int col = out_col_base + ci;
      if (row < M && col < N)
        C[row * N + col] = reg_C[ri][ci];
    }
  }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int K, int N) {
    dim3 threadsPerBlock(BLOCK_N, BLOCK_M);
    dim3 blocksPerGrid((N + OUT_N - 1) / OUT_N,
                       (M + OUT_M - 1) / OUT_M);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}
