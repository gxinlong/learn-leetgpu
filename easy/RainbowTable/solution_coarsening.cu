#include <cuda_runtime.h>
#include "../../include/cuda_ide.h"  // 仅供 IDE：threadIdx/blockIdx/blockDim 桩声明，nvcc 编译时无效

#define BLOCK 256
#define ITEMS_PER_THREAD 8

__device__ unsigned int fnv1a_hash(int input) {
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;

    unsigned int hash = OFFSET_BASIS;

    for (int byte_pos = 0; byte_pos < 4; byte_pos++) {
        unsigned char byte = (input >> (byte_pos * 8)) & 0xFF;
        hash = (hash ^ byte) * FNV_PRIME;
    }

    return hash;
}

__global__ void fnv1a_hash_kernel(int* input, unsigned int* output, int N, int R) {
  const int tx = threadIdx.x;
  const int gid_base = blockIdx.x * blockDim.x;
  const int bidx_base = gid_base * ITEMS_PER_THREAD;
  
  input = &(input[bidx_base]);
  output = &(output[bidx_base]);

  #pragma unroll
  for (int i = tx; i < blockDim.x * ITEMS_PER_THREAD; i += blockDim.x) {
    if (bidx_base + i < N) {
        int value = input[i];
        #pragma unroll
        for (int j = 0; j < R; ++j) {
            value = fnv1a_hash(value);
        }
        output[i] = value;
    }
  }

}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, unsigned int* output, int N, int R) {
    int threadsPerBlock = BLOCK;
    const int items_per_block = threadsPerBlock * ITEMS_PER_THREAD;
    int blocksPerGrid = (N + items_per_block) / items_per_block;

    int* input_tmp = (int*)input;

    fnv1a_hash_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_tmp, output, N, R);
    cudaDeviceSynchronize();
}
