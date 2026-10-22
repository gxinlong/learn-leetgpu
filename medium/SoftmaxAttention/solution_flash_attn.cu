#include "cuda_runtime.h"
#include "float.h"


__global__ void softmax_attn(const float* Q, const float* K, const float* V, float* output, int M, int N, int d, int tile_r, int tile_c) {
  extern __shared__ float smem[];
  const int tx = threadIdx.x;
  const int q_r_idx = blockDim.x * blockIdx.x + tx;
  const float d_sqrt = 1.0f/sqrtf((float)d);

  float* s_Q = smem;
  float* s_K = s_Q + tile_r * d;
  float* s_V = s_K + tile_c * d;
  float* s_O = s_V + tile_c * d;

  for (int i = 0; i < d; ++i) {
    s_O[tx * d + i] = 0.0f;
  }

  float exp_qkj_sum = 0.0f;
  float qkj_max = -FLT_MAX;

  for (int iter_c = 0; iter_c < N; iter_c += tile_c) {
    for (int dd = 0; dd < d; ++dd) {
      if (q_r_idx < M) {
        const int q_idx = q_r_idx * d + dd;
        s_Q[tx * d + dd] = Q[q_idx];
      }
      for (int j = 0; j < tile_c; j += tile_r) {
        const int idx = iter_c + tx + j;
        if (idx < N) {
          const int kv_idx = (idx) * d + dd;
          s_K[(tx + j) * d + dd] = K[kv_idx];
          s_V[(tx + j) * d + dd] = V[kv_idx];
        }
      }
    }
    __syncthreads();
    
    if (q_r_idx < M) {
      for (int j = 0; j < tile_c && iter_c + j < N; ++j) {
        float qkj = 0.0f;
        for (int dd = 0; dd < d; ++dd) {
          qkj += s_Q[tx * d + dd] * s_K[j * d + dd];
        }
        qkj *= d_sqrt;
        float old_max = qkj_max;
        qkj_max = fmaxf(qkj, qkj_max);
        float exp_qkj = __expf(qkj - qkj_max);

        float alpha = __expf(old_max - qkj_max);
        // 更新分母
        exp_qkj_sum = exp_qkj_sum * alpha + exp_qkj;
        // 更新分子
        for (int dd = 0; dd < d; ++dd) {
          s_O[tx * d + dd] = s_O[tx * d + dd] * alpha + exp_qkj * s_V[j * d + dd];
        }
      }
    }
    __syncthreads();
  }

  if (q_r_idx < M) {
    exp_qkj_sum = 1.0f / exp_qkj_sum;
    for (int dd = 0; dd < d; ++dd) {
      const int output_idx = q_r_idx * d + dd;
      output[output_idx] = s_O[tx * d + dd] * exp_qkj_sum;
    }
  }
}


extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
  int smem_item_cnt = 16384 / d;
  int tile_r = smem_item_cnt / 4;
  int threadPerBlock = tile_r;
  int blockPerGrid = (M + threadPerBlock - 1) / threadPerBlock;

  int smem_bytes = sizeof(float) * 2 * (tile_r * d + TILE_C_SIZE * d);

  softmax_attn<<<blockPerGrid, threadPerBlock, smem_bytes>>>(Q, K, V, output, M, N, d, tile_r, tile_c);
}