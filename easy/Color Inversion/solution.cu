#include <cuda_runtime.h>
#include "../../include/cuda_ide.h"  // 仅供 IDE：threadIdx/blockIdx/blockDim 桩声明，nvcc 编译时无效


__global__ void invert_kernel(unsigned char* image, int N) {
  const int tx = threadIdx.x;
  const int block_st = blockIdx.x * blockDim.x;
  const int global_id = block_st + tx;

  if (global_id < N) {
    if (((global_id + 1) & 3) != 0) {
      const unsigned char value = image[global_id];
      image[global_id] = 255 - value;
    }
  }
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    const int N = width * height * 4;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, N);
    cudaDeviceSynchronize();
}
