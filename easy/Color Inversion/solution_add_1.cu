#include <cuda_runtime.h>
#include "../../include/cuda_ide.h"  // 仅供 IDE：threadIdx/blockIdx/blockDim 桩声明，nvcc 编译时无效

#define MASK 0x00FFFFFF

__global__ void invert_kernel(unsigned int* int_img, int N) {
  const int tx = threadIdx.x;
  const int block_st = blockIdx.x * blockDim.x;
  const int global_id = block_st + tx;

  if (global_id < N) {
    const unsigned int value = int_img[global_id];
    int_img[global_id] = value ^ MASK;
  }
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int N = width * height;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    unsigned int* int_img = (unsigned int*) image;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(int_img, N);
    cudaDeviceSynchronize();
}
