#include <cuda_runtime.h>
#include "../../include/cuda_ide.h"  // 仅供 IDE：threadIdx/blockIdx/blockDim 桩声明，nvcc 编译时无效

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
  const int gid = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (gid < halfN) {
    float left = input[gid];
    float right = input[gid + halfN];
    output[gid] = left * right / (1.0f + expf(-left));
  }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
}
