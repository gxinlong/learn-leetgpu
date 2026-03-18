#include <cuda_runtime.h>
#include "../../include/cuda_ide.h"  // 仅供 IDE：threadIdx/blockIdx/blockDim 桩声明，nvcc 编译时无效

__global__ void clip_kernel(const float* input, float* output, float lo, float hi, int N) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (tid < N) {
    float value = input[tid];
    if (value < lo) {
      value = lo;
    } else if (value > hi) {
      value = hi;
    }
    output[tid] = value;
  }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, float lo, float hi, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    clip_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, lo, hi, N);
}
