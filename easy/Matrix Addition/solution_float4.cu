#include <cuda_runtime.h>
#include "../../include/cuda_ide.h"  // 仅供 IDE：threadIdx/blockIdx/blockDim 桩声明，nvcc 编译时无效

__global__ void matrix_add(const float4* A, const float4* B, float4* C, int N) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < N * N / 4) {
    float4 a = A[idx], b = B[idx];
    C[idx] = {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
  }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int total = N * N / 4;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    float4* fp4_a = (float4*)(A);
    float4* fp4_b = (float4*)(B);
    float4* fp4_c = (float4*)(C);

    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(fp4_a, fp4_b, fp4_c, N);
    cudaDeviceSynchronize();
}
