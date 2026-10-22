#include "cuda_runtime.h"
#include "float.h"


__device__ void insert_value(float* values, int* indexs, float value, int index, int k) {
  if (value < values[0]) {
    return;
  }
  for (int i = 1; i < k; ++i) {
    if (value > values[i]) {
      values[i-1] = values[i];
      indexs[i-1] = indexs[i];
      continue;
    }
    values[i-1] = value;
    indexs[i-1] = index;
    return;
  }
  values[k-1] = value;
  indexs[k-1] = index;
}

__device__ void softmax(float* values, int k) {
  float sum = 0.0f;
  float max_v = values[k-1];
  for (int i = 0; i < k; ++i) {
    values[i] = __expf(values[i] - max_v);
    sum += values[i];
  }
  for (int i = 0; i < k; ++i) {
    values[i] /= sum;
  }
}


__device__ __forceinline__ void swap(float& a, float& b) {
  float tmp = a;
  a = b;
  b = tmp;
}

template<unsigned int TOPK>
__global__ void kernel(const float* logits, float* weights, int* indices, int M, int E, int k) {
  const int tx = threadIdx.x;
  const int tr = blockDim.x * blockIdx.x + tx;
  const int input_i_base = tr * E;
  const int output_i_base = tr * TOPK;
  if (tr >= M) {
    return;
  }

  float values[TOPK] = {-FLT_MAX};
  int indexs[TOPK] = {int(-1)};
  #pragma unroll
  for (int i = 0; i < TOPK; ++i) {
    values[i] = -FLT_MAX;
    indexs[i] = -1;
  }

  float value = 0.0f;
  for (int i = 0; i < E; ++i) {
    const int input_i = input_i_base + i;
    // value = logits[input_i];

    // insert_value(values, indexs, value, (int)i, TOPK);

    // LDL 低效率实现
    // if (value < values[0]) {
    //   continue;;
    // }
    // int j = 0;
    // #pragma unroll
    // for (j = 1; j < TOPK; ++j) {
    //   if (value > values[j]) {
    //     values[j-1] = values[j];
    //     indexs[j-1] = indexs[j];
    //     continue;
    //   }
    //   break;
    // }
    // values[j-1] = value;
    // indexs[j-1] = i;

    float candidate = logits[input_i];
    int candidate_idx = i;
    #pragma unroll
    for (int p = TOPK - 1; p >= 0; --p) {
      const bool take = candidate > values[p];

      const float old_value = values[p];
      const int old_idx = indexs[p];

      values[p] = take ? candidate : old_value;
      indexs[p] = take ? candidate_idx : old_idx;

      candidate = take ? old_value : candidate;
      candidate_idx = take ? old_idx : candidate_idx;
    }
  }

  // softmax(values, TOPK);
  // float value = 0.0f;
  value = 0.0f;
  float max_v = values[TOPK-1];
  for (int i = 0; i < TOPK; ++i) {
    values[i] = __expf(values[i] - max_v);
    value += values[i];
  }
  for (int i = 0; i < TOPK; ++i) {
    values[i] /= value;
  }

  #pragma unroll
  for (int i = 0; i < TOPK; ++i) {
    const int output_i = output_i_base + i;
    weights[output_i] = values[TOPK-i-1];
    indices[output_i] = (int)(indexs[TOPK-i-1]);
  }
}

extern "C" void solve(const float* logits, float* weights, int* indices, int M, int E, int k) {
  int tile_size = 128;
  int block_nums = (M + tile_size - 1) / tile_size;
  if (k == 1) {
    kernel<1><<<block_nums, tile_size>>>(logits, weights, indices, M, E, k);
  } else if (k == 2) {
    kernel<2><<<block_nums, tile_size>>>(logits, weights, indices, M, E, k);
  } else if (k <= 4) {
    kernel<4><<<block_nums, tile_size>>>(logits, weights, indices, M, E, k);
  } else if (k <= 8) {
    kernel<8><<<block_nums, tile_size>>>(logits, weights, indices, M, E, k);
  } else if (k <= 16) {
    kernel<16><<<block_nums, tile_size>>>(logits, weights, indices, M, E, k);
  } else if (k <= 32) {
    kernel<32><<<block_nums, tile_size>>>(logits, weights, indices, M, E, k);
  } else if (k <= 64) {
    kernel<64><<<block_nums, tile_size>>>(logits, weights, indices, M, E, k);
  } else if (k <= 128) {
    int tile_size = 64;
    int block_nums = (M + tile_size - 1) / tile_size;
    kernel<128><<<block_nums, tile_size>>>(logits, weights, indices, M, E, k);
  } else if (k <= 256) {
    int tile_size = 64;
    int block_nums = (M + tile_size - 1) / tile_size;
    kernel<256><<<block_nums, tile_size>>>(logits, weights, indices, M, E, k);
  } 
}