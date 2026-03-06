#include <cuda_runtime.h>
#include "../../include/cuda_ide.h"  // 仅供 IDE：threadIdx/blockIdx/blockDim 桩声明，nvcc 编译时无效

// 1. shared memory
// 2. A 矩阵转置避免 bank conflict
// 3. register tiling: 每个 thread 计算 TILE_REG x TILE_REG 的输出子块
// 4. prefetch + ldg.128

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define BLOCK_M 16
#define BLOCK_N 16

#define TILE_REG 8
#define BLOCK_K TILE_REG

// 每个 block 覆盖的输出尺寸
#define OUT_M (BLOCK_M * TILE_REG)
#define OUT_N (BLOCK_N * TILE_REG)

__global__ void matrix_multiplication_kernel(float* A, float* B, float* C, int M, int K, int N) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  const int tid = ty * BLOCK_N + tx;  // 线程块内的线程 id
  const int block_thread_num = BLOCK_M * BLOCK_N; // 线程块线程总数

  const int out_row_base = blockIdx.y * OUT_M + ty * TILE_REG;
  const int out_col_base = blockIdx.x * OUT_N + tx * TILE_REG;

  __shared__ float sdata_A[2][BLOCK_K][OUT_M];
  __shared__ float sdata_B[2][BLOCK_K][OUT_N];

  const int float4_ld_per_thread = OUT_M * BLOCK_K / 4 / block_thread_num;  // global mem -> reg 的时候, 每个线程负责加载float4的数量

  float load_reg_a[float4_ld_per_thread * 4];
  float load_reg_b[float4_ld_per_thread * 4];

  float reg_a[2][TILE_REG];
  float reg_b[2][TILE_REG];
  float reg_C[TILE_REG][TILE_REG] = {0};

  A = &(A[(by * OUT_M) * K]);
  B = &(B[bx * OUT_N]);

  // 预先加载数据到 0 号 sdata 和 load_reg
  const int ldg_A_thread_per_row = BLOCK_K / 4;
  const int ldg_A_row = tid / ldg_A_thread_per_row;
  const int ldg_A_col = tid % ldg_A_thread_per_row;
  FETCH_FLOAT4(load_reg_a[0]) = FETCH_FLOAT4(A[ldg_A_row * K + ldg_A_col * 4]);
  const int ldg_B_thread_per_row = OUT_N / 4;
  const int ldg_B_row = tid / ldg_B_thread_per_row;
  const int ldg_B_col = tid % ldg_B_thread_per_row;
  FETCH_FLOAT4(load_reg_b[0]) = FETCH_FLOAT4(B[ldg_B_row * N + ldg_B_col * 4]);

  // k 奇数行对 m/n 索引 ^64，错位存储以消除 bank conflict
  sdata_A[0][ldg_A_col * 4    ][ldg_A_row      ] = load_reg_a[0];
  sdata_A[0][ldg_A_col * 4 + 1][ldg_A_row ^ 64 ] = load_reg_a[1];
  sdata_A[0][ldg_A_col * 4 + 2][ldg_A_row      ] = load_reg_a[2];
  sdata_A[0][ldg_A_col * 4 + 3][ldg_A_row ^ 64 ] = load_reg_a[3];
  FETCH_FLOAT4(sdata_B[0][ldg_B_row][ldg_B_col * 4 ^ ((ldg_B_row & 1) * 64)]) = FETCH_FLOAT4(load_reg_b[0]);
  __syncthreads();

  // 0 号 reg tile 数据写入 reg
  const int smem_A_row = ty * TILE_REG; // smem 中本线程应该处理的 sdata_A 的行, float4地址
  FETCH_FLOAT4(reg_a[0][0]) = FETCH_FLOAT4(sdata_A[0][0][smem_A_row]);
  FETCH_FLOAT4(reg_a[0][4]) = FETCH_FLOAT4(sdata_A[0][0][smem_A_row + 4]);
  const int smem_B_col = tx * TILE_REG; // smem 中本线程应该处理的 sdata_B 的列, float4的地址
  FETCH_FLOAT4(reg_b[0][0]) = FETCH_FLOAT4(sdata_B[0][0][smem_B_col]);
  FETCH_FLOAT4(reg_b[0][4]) = FETCH_FLOAT4(sdata_B[0][0][smem_B_col + 4]);

  int write_idx = 1;  // 当前预加载写入的 idx
  int load_idx = 0;   // 当前消费的 idx
  int smem_tile_idx = 0;
  do {
    smem_tile_idx += BLOCK_K;
    // 预加载下一次数据，global to smem
    if (smem_tile_idx < K) {
      FETCH_FLOAT4(load_reg_a[0]) = FETCH_FLOAT4(A[ldg_A_row * K + ldg_A_col * 4 + smem_tile_idx]);

      FETCH_FLOAT4(load_reg_b[0]) = FETCH_FLOAT4(B[(ldg_B_row + smem_tile_idx) * N + ldg_B_col * 4]);
    }
    
    // 迭代 smem 内的数据, 第0个已经预取, 最后一个留着不算
    // 过程中首先预取 smem to reg, 然后消费 reg 进行计算
    #pragma unroll
    for (int i = 0; i < BLOCK_K - 1; ++i) {
      const int smem_off = ((i + 1) & 1) * 64;
      FETCH_FLOAT4(reg_a[(i+1)%2][0]) = FETCH_FLOAT4(sdata_A[load_idx][i+1][(smem_A_row    ) ^ smem_off]);
      FETCH_FLOAT4(reg_a[(i+1)%2][4]) = FETCH_FLOAT4(sdata_A[load_idx][i+1][(smem_A_row + 4) ^ smem_off]);
      FETCH_FLOAT4(reg_b[(i+1)%2][0]) = FETCH_FLOAT4(sdata_B[load_idx][i+1][(smem_B_col    ) ^ smem_off]);
      FETCH_FLOAT4(reg_b[(i+1)%2][4]) = FETCH_FLOAT4(sdata_B[load_idx][i+1][(smem_B_col + 4) ^ smem_off]);
      #pragma unroll
      for (int r = 0; r < TILE_REG; ++r) {
        #pragma unroll
        for (int c = 0; c < TILE_REG; ++c) {
          reg_C[r][c] += reg_a[i%2][r] * reg_b[i%2][c];
        }
      }
    }

    // 预加载数据写入，load reg to smem
    if (smem_tile_idx < K) {
      sdata_A[write_idx][ldg_A_col * 4    ][ldg_A_row      ] = load_reg_a[0];
      sdata_A[write_idx][ldg_A_col * 4 + 1][ldg_A_row ^ 64 ] = load_reg_a[1];
      sdata_A[write_idx][ldg_A_col * 4 + 2][ldg_A_row      ] = load_reg_a[2];
      sdata_A[write_idx][ldg_A_col * 4 + 3][ldg_A_row ^ 64 ] = load_reg_a[3];
      FETCH_FLOAT4(sdata_B[write_idx][ldg_B_row][ldg_B_col * 4 ^ ((ldg_B_row & 1) * 64)]) = FETCH_FLOAT4(load_reg_b[0]);
      write_idx ^= 1;
    }
    __syncthreads();
    load_idx ^= 1;

    FETCH_FLOAT4(reg_a[0][0]) = FETCH_FLOAT4(sdata_A[load_idx][0][smem_A_row]);
    FETCH_FLOAT4(reg_a[0][4]) = FETCH_FLOAT4(sdata_A[load_idx][0][smem_A_row + 4]);
    FETCH_FLOAT4(reg_b[0][0]) = FETCH_FLOAT4(sdata_B[load_idx][0][smem_B_col]);
    FETCH_FLOAT4(reg_b[0][4]) = FETCH_FLOAT4(sdata_B[load_idx][0][smem_B_col + 4]);

    #pragma unroll
    for (int r = 0; r < TILE_REG; ++r) {
      #pragma unroll
      for (int c = 0; c < TILE_REG; ++c) {
        reg_C[r][c] += reg_a[1][r] * reg_b[1][c];
      }
    }
  } while (smem_tile_idx < K);

  for (int r = 0; r < TILE_REG; ++r) {
    for (int c = 0; c < TILE_REG; ++c) {
      int row = out_row_base + r;
      int col = out_col_base + c;
      if (row < M && col < N) {
        C[row * N + col] = reg_C[r][c];
      }
    }
  }
}


__global__ void matrix_multiplication_kernel_small(const float* A, const float* B, float* C, int M, int K, int N) {
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

    // 快速 kernel 要求：
    //   M % OUT_M == 0：A 的行 tile 不越界
    //   N % OUT_N == 0：B 的列 tile 不越界（ldg_B_col 最大访问到 OUT_N-4 列）
    //   K % BLOCK_K == 0：K 方向无不完整 tile，避免最后一个 tile 读入垃圾
    bool use_fast = (M % OUT_M == 0) && (N % OUT_N == 0) && (K % BLOCK_K == 0);
    if (use_fast) {
        float* AA = (float*)(A);
        float* BB = (float*)(B);
        matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(AA, BB, C, M, K, N);
    } else {
        matrix_multiplication_kernel_small<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    }
    cudaDeviceSynchronize();
}
