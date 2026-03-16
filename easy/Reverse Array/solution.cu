#include <cuda_runtime.h>
#include "../../include/cuda_ide.h"  // 仅供 IDE：threadIdx/blockIdx/blockDim 桩声明，nvcc 编译时无效


#define B 262144

__global__ void reverse_array(float* input, int N, int pivot) {
  const int tx = threadIdx.x;
  const int global_id = blockIdx.x * blockDim.x + tx;

  const int target_idx = N - global_id - 1;
  if (global_id < pivot) {
    float left = input[global_id];
    float right = input[target_idx];
    input[global_id] = right;
    input[target_idx] = left;
  }

}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = B;
    int pivot = N / 2;
    int blocksPerGrid = (pivot + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N, pivot);
    cudaDeviceSynchronize();
}
