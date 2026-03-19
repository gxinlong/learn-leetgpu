#include <cuda_runtime.h>
#include "../../include/cuda_ide.h"  // 仅供 IDE：threadIdx/blockIdx/blockDim 桩声明，nvcc 编译时无效

#define BLOCK_SIZE_N 256

__global__ void reduce_kernel(const float* input, float* output, int N) {
  const int tx = threadIdx.x;
  const int tid = blockIdx.x * blockDim.x + tx;
  
  __shared__ float s_input[BLOCK_SIZE_N];

  // 预取数据
  s_input[tx] = (tid < N) ? input[tid] : 0.0f;
  __syncthreads();
  
  // 计算
  for (int ceil = BLOCK_SIZE_N / 2; ceil > 0; ceil >>= 1) {
    if (tx < ceil) {
      s_input[tx] += s_input[tx + ceil];
    }
    __syncthreads();
  }

  // 输出
  if (tx == 0) {
    atomicAdd(output, s_input[tx]);
  }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
  int threadsPerBlock = BLOCK_SIZE_N;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  reduce_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
}
