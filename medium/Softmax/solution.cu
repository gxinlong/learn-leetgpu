#include <cuda_runtime.h>
#include "float.h"
#include <cooperative_groups.h>

#define TILE_SIZE 256
#define GRID_SIZE 160

#define FULL_MASK 0xffffffff

template<unsigned int SIZE>
__device__ void warp_reduce(float& sum, float& max_v) {
  if (SIZE >= 32) {
    float h_sum = __shfl_down_sync(FULL_MASK, sum, 16);
    float h_max_v = __shfl_down_sync(FULL_MASK, max_v, 16);
    float new_max = fmaxf(max_v, h_max_v);
    sum = sum * __expf(max_v - new_max) + h_sum * __expf(h_max_v - new_max);
    max_v = new_max;
  }
  if (SIZE >= 16) {
    float h_sum = __shfl_down_sync(FULL_MASK, sum, 8);
    float h_max_v = __shfl_down_sync(FULL_MASK, max_v, 8);
    float new_max = fmaxf(max_v, h_max_v);
    sum = sum * __expf(max_v - new_max) + h_sum * __expf(h_max_v - new_max);
    max_v = new_max;
  }
  if (SIZE >= 8) {
    float h_sum = __shfl_down_sync(FULL_MASK, sum, 4);
    float h_max_v = __shfl_down_sync(FULL_MASK, max_v, 4);
    float new_max = fmaxf(max_v, h_max_v);
    sum = sum * __expf(max_v - new_max) + h_sum * __expf(h_max_v - new_max);
    max_v = new_max;
  }
  if (SIZE >= 4) {
    float h_sum = __shfl_down_sync(FULL_MASK, sum, 2);
    float h_max_v = __shfl_down_sync(FULL_MASK, max_v, 2);
    float new_max = fmaxf(max_v, h_max_v);
    sum = sum * __expf(max_v - new_max) + h_sum * __expf(h_max_v - new_max);
    max_v = new_max;
  }
  if (SIZE >= 2) {
    float h_sum = __shfl_down_sync(FULL_MASK, sum, 1);
    float h_max_v = __shfl_down_sync(FULL_MASK, max_v, 1);
    float new_max = fmaxf(max_v, h_max_v);
    sum = sum * __expf(max_v - new_max) + h_sum * __expf(h_max_v - new_max);
    max_v = new_max;
  }
}

__device__ float sum_output[1024];
__device__ float max_v_output[1024];

__global__ void softmax_reduce_sum(const float* input, float* output, int N) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();

  const int tx = threadIdx.x;
  const int tid = blockDim.x * blockIdx.x + tx;
  const int stride = gridDim.x * blockDim.x;
  
  float sum = 0.0f;
  float max_v = -FLT_MAX;
  
  __shared__ float s_sum[TILE_SIZE];
  __shared__ float s_max_v[TILE_SIZE];

  for (int i = tid; i < N; i += stride) {
    float value = input[i];
    float new_max = fmaxf(max_v, value);
    value = __expf(value - new_max);
    if (new_max == max_v) {
      sum = sum + value;
    } else {
      sum = sum * __expf(max_v - new_max) + value;
    }
    max_v = new_max;
  }
  s_sum[tx] = sum;
  s_max_v[tx] = max_v;
  __syncthreads();


  for (int ceil = TILE_SIZE / 2; ceil >= 32; ceil >>= 1) {
    if (tx < ceil) {
      float max_2 = s_max_v[tx + ceil];
      float new_max = fmaxf(max_v, max_2);
      if (new_max == max_v) {
        sum = sum + s_sum[tx + ceil] * exp(max_2 - new_max);
      } else {
        sum = sum * __expf(max_v - new_max) + s_sum[tx + ceil];
      }
      s_sum[tx] = sum;
      max_v = new_max;
      s_max_v[tx] = max_v;
    }
    __syncthreads();
  }

  if (tx < 32) {
    warp_reduce<32>(sum, max_v);
  }

  if (tx == 0) {
    sum_output[blockIdx.x] = sum;
    max_v_output[blockIdx.x] = max_v;
  }
  grid.sync();

  __shared__ float final_sum_sub, final_max_v;

  if (tx < gridDim.x) {
    sum = sum_output[tx];
    max_v = max_v_output[tx];
    
    warp_reduce<32>(sum, max_v);
    const int warpid = tx / warpSize;
    if (tx % 32 == 0) {
      s_sum[warpid] = sum;
      s_max_v[warpid] = max_v;
    }
  }
  __syncthreads();

  int valid_warps = (gridDim.x + 31) / 32;
  if (tx < 32) {
    sum = tx < valid_warps ? s_sum[tx] : 0.0f;
    max_v = tx < valid_warps ? s_max_v[tx] : -FLT_MAX;
    warp_reduce<32>(sum, max_v);
  }
  if (tx == 0) {
    final_sum_sub = 1.0 / sum;
    final_max_v = max_v;
  }
  __syncthreads();

  for (int i = tid; i < N; i += stride) {
    output[i] = __expf(input[i] - final_max_v) * final_sum_sub;
  }
}



extern "C" void solve(const float* input, float* output, int N) {
  // int blocksPerSM = 0;
  // int device = 0;
  
  // cudaDeviceProp prop;
  // cudaGetDeviceProperties(&prop, device);

  // cudaOccupancyMaxActiveBlocksPerMultiprocessor(
  //   &blocksPerSM,
  //   softmax_reduce_sum,
  //   TILE_SIZE,
  //   0
  // );
  // int maxCooperativeBlocks = blocksPerSM * prop.multiProcessorCount;

  int threadPerBlock = TILE_SIZE;
  int blockPerGrid = 160;
  dim3 gridDim(blockPerGrid, 1, 1);
  dim3 blockDim(threadPerBlock, 1, 1);
  void* args[] = {(void*)(&input), (void*)(&output), (void*)(&N)};
  cudaLaunchCooperativeKernel(
    (void*)softmax_reduce_sum,
    gridDim,
    blockDim,
    args,
    0,0
  );
}