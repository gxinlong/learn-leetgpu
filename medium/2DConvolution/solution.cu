#include "cuda_runtime.h"

#define GET_IN_WIDTH(out_width, kernel_width) (out_width + kernel_width - 1)

template<unsigned int TILE_ROW, unsigned int TILE_COL>
__global__ void conv_kernel(const float* input, const float* kernel, float* output, 
  int input_rows, int input_cols, int kernel_rows, int kernel_cols, int input_smem_rows, int input_smem_cols, int output_rows, int output_cols) {
  extern __shared__ float smem[];

  float* s_kernel = smem;
  float* s_input = smem + (kernel_rows * kernel_cols);

  const int tr = threadIdx.y;
  const int tc = threadIdx.x;
  const int out_r = blockDim.y * blockIdx.y + threadIdx.y;
  const int out_c = blockDim.x * blockIdx.x + threadIdx.x;
  constexpr int block_items = TILE_ROW * TILE_COL;
  const int tid = tr * TILE_COL + tc;
  const int kernel_items = kernel_rows * kernel_cols;

  // load kernel
  for (int i = tid; i < kernel_items; i += block_items) {
    s_kernel[i] = kernel[i];
  }
  // for (int r = 0; r < kernel_rows; r += TILE_ROW) {
  //   for (int c = 0; c < kernel_cols; c += TILE_COL) {
  //     if (r + tr < kernel_rows && c + tc < kernel_cols) {
  //       const int kernel_idx = (r + tr) * kernel_cols + c + tc;
  //       s_kernel[kernel_idx] = kernel[kernel_idx];
  //     }
  //   }
  // }
  // load input
  for (int r = 0; r < input_smem_rows; r += TILE_ROW) {
    for (int c = 0; c < input_smem_cols; c += TILE_COL) {
      if (r + tr < input_smem_rows && c + tc < input_smem_cols) {
        const int input_s_idx = (r + tr) * input_smem_cols + c + tc;
        const int input_g_idx = (out_r + r) * input_cols + out_c + c;
        if (out_r + r < input_rows && out_c + c < input_cols) {
          s_input[input_s_idx] = input[input_g_idx];
        } else {
          s_input[input_s_idx] = 0.0f;
        }
      }
    }
  }
  __syncthreads();

  // per thread
  if (out_r < output_rows && out_c < output_cols) {
    float out_sum = 0.0f;
    for (int r = 0; r < kernel_rows; ++r) {
      for (int c = 0; c < kernel_cols; ++c) {
        const int input_s_idx = (tr + r) * input_smem_cols + tc + c;
        const int kernel_s_idx = r * kernel_cols + c;
        out_sum += s_input[input_s_idx] * s_kernel[kernel_s_idx];
      }
    }
    output[out_r * output_cols + out_c] = out_sum;
  }
}

#define TILE_R 32
#define TILE_C 32

extern "C" void solve(const float* input, const float* kernel, float* output, 
  int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
  
  const int output_rows = input_rows - kernel_rows + 1;
  const int output_cols = input_cols - kernel_cols + 1;
  const int input_smem_rows = TILE_R + kernel_rows - 1;
  const int input_smem_cols = TILE_C + kernel_cols - 1;
  dim3 blockDim(TILE_C, TILE_R, 1);
  dim3 gridDim((output_cols + TILE_C - 1) / TILE_C, (output_rows + TILE_R - 1) / TILE_R, 1);
  const int smem_bytes = (input_smem_rows * input_smem_cols + kernel_rows * kernel_cols) * sizeof(float);
  conv_kernel<TILE_R, TILE_C><<<gridDim, blockDim, smem_bytes>>>(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols, input_smem_rows, input_smem_cols, output_rows, output_cols);
}