#include <cuda_runtime.h>

__global__ void swiglu(const float* __restrict__ input, float* __restrict__ output, int N) {
  int tid = threadIdx.x + (blockDim.x * blockIdx.x);
  float inp, inp2;
  for(int i = tid; i < N; i += (gridDim.x * blockDim.x)) {
    inp = input[i];
    inp2 = input[i + N];
    inp = inp / (1.0f + __expf(-inp));
    inp *= inp2;
    output[i] = inp;
  }
}
// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
  int numTBlocks = 80;
  int numThreads = 512;
  swiglu<<<numTBlocks, numThreads>>>(input, output, N / 2);
}