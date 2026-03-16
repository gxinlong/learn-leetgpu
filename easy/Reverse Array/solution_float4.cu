#include <cuda_runtime.h>
#include "../../include/cuda_ide.h"  // 仅供 IDE：threadIdx/blockIdx/blockDim 桩声明，nvcc 编译时无效

// 只进行性能测试，为了方便，假设 N % 8 == 0

#define B 256
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__global__ void reverse_array(float* input, int N, int pivot, int N_float4) {
  const int tx = threadIdx.x;
  const int global_id = blockIdx.x * blockDim.x + tx;
  const int left_idx = global_id * 4;
  const int right_idx = N - global_id * 4 - 4;

  float left_reg[4], right_reg[4];

  if (left_idx < pivot) {
    FETCH_FLOAT4(left_reg[0]) = FETCH_FLOAT4(input[left_idx]);
    FETCH_FLOAT4(right_reg[0]) = FETCH_FLOAT4(input[right_idx]);
    FETCH_FLOAT4(input[left_idx]) = FETCH_FLOAT4(right_reg[0]);
    FETCH_FLOAT4(input[right_idx]) = FETCH_FLOAT4(left_reg[0]);
  }
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = B;
    int pivot = N / 2;
    int blocksPerGrid = (pivot + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N, pivot, N / 4);
    cudaDeviceSynchronize();
}
