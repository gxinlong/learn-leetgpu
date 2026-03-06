#include <cuda_runtime.h>
#include "../../include/cuda_ide.h"  // 仅供 IDE：threadIdx/blockIdx/blockDim 桩声明，nvcc 编译时无效

__global__ void vector_add_float4(const float4* A, const float4* B, float* C, int N_vec4, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int base = idx * 4;
  if (idx < N_vec4) {
      float4 a = A[idx];
      float4 b = B[idx];
      if (base + 0 < N) C[base + 0] = a.x + b.x;
      if (base + 1 < N) C[base + 1] = a.y + b.y;
      if (base + 2 < N) C[base + 2] = a.z + b.z;
      if (base + 3 < N) C[base + 3] = a.w + b.w;
  }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
  int N_vec4 = (N + 3) / 4; // Number of float4 elements
  int threadsPerBlock = 256;
  int blocksPerGrid = (N_vec4 + threadsPerBlock - 1) / threadsPerBlock;

  // Cast float* to float4*
  const float4* A4 = reinterpret_cast<const float4*>(A);
  const float4* B4 = reinterpret_cast<const float4*>(B);

  vector_add_float4<<<blocksPerGrid, threadsPerBlock>>>(A4, B4, C, N_vec4, N);
  //cudaDeviceSynchronize();
}
