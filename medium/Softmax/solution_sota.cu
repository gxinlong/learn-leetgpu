#include <cuda_runtime.h>
#include <float.h>
#include <cooperative_groups.h>

template <unsigned int blockSize>
__device__ void softmax_warp_reduce(float exp_max, float exp_sum, float* __restrict__ exp_max_output, float* __restrict__ exp_sum_output) {
    if (blockSize >= 32) {
        float exp_max_tmp = __shfl_down_sync(0xffffffff, exp_max, 16);
        float exp_sum_tmp = __shfl_down_sync(0xffffffff, exp_sum, 16);
        float delta1 = exp_max < exp_max_tmp ? exp_max - exp_max_tmp : 0;
        float delta2 = exp_max_tmp < exp_max ? exp_max_tmp - exp_max : 0;
        exp_max = fmaxf(exp_max, exp_max_tmp);
        exp_sum = exp_sum * __expf(delta1) + exp_sum_tmp * __expf(delta2);
    }

    if (blockSize >= 16) {
        float exp_max_tmp = __shfl_down_sync(0xffffffff, exp_max, 8);
        float exp_sum_tmp = __shfl_down_sync(0xffffffff, exp_sum, 8);
        float delta1 = exp_max < exp_max_tmp ? exp_max - exp_max_tmp : 0;
        float delta2 = exp_max_tmp < exp_max ? exp_max_tmp - exp_max : 0;
        exp_max = fmaxf(exp_max, exp_max_tmp);
        exp_sum = exp_sum * __expf(delta1) + exp_sum_tmp * __expf(delta2);
    }

    if (blockSize >= 8) {
        float exp_max_tmp = __shfl_down_sync(0xffffffff, exp_max, 4);
        float exp_sum_tmp = __shfl_down_sync(0xffffffff, exp_sum, 4);
        float delta1 = exp_max < exp_max_tmp ? exp_max - exp_max_tmp : 0;
        float delta2 = exp_max_tmp < exp_max ? exp_max_tmp - exp_max : 0;
        exp_max = fmaxf(exp_max, exp_max_tmp);
        exp_sum = exp_sum * __expf(delta1) + exp_sum_tmp * __expf(delta2);
    }

    if (blockSize >= 4) {
        float exp_max_tmp = __shfl_down_sync(0xffffffff, exp_max, 2);
        float exp_sum_tmp = __shfl_down_sync(0xffffffff, exp_sum, 2);
        float delta1 = exp_max < exp_max_tmp ? exp_max - exp_max_tmp : 0;
        float delta2 = exp_max_tmp < exp_max ? exp_max_tmp - exp_max : 0;
        exp_max = fmaxf(exp_max, exp_max_tmp);
        exp_sum = exp_sum * __expf(delta1) + exp_sum_tmp * __expf(delta2);
    }

    if (blockSize >= 2) {
        float exp_max_tmp = __shfl_down_sync(0xffffffff, exp_max, 1);
        float exp_sum_tmp = __shfl_down_sync(0xffffffff, exp_sum, 1);
        float delta1 = exp_max < exp_max_tmp ? exp_max - exp_max_tmp : 0;
        float delta2 = exp_max_tmp < exp_max ? exp_max_tmp - exp_max : 0;
        exp_max = fmaxf(exp_max, exp_max_tmp);
        exp_sum = exp_sum * __expf(delta1) + exp_sum_tmp * __expf(delta2);
    }

    exp_max_output[0] = exp_max;
    exp_sum_output[0] = exp_sum;
}

__device__ float exp_sum_grid[1024];
__device__ float exp_max_grid[1024];

__global__ void softmax_block_reduce(const float* __restrict__ input, float* __restrict__ output,  int N) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    
    __shared__ float exp_sum_output[16];
    __shared__ float exp_max_output[16];

    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    int stride = gridDim.x * blockDim.x; // 480 * 256
    // stride = (N + stride - 1) / stride;
    int idx = (blockDim.x * blockIdx.x + threadIdx.x);

    float exp_max = -FLT_MAX;
    float exp_sum = 0;

    #pragma unroll
    for (int i = idx; i < N; i += stride) {
        float tmp = input[i];
        float tmp_max = fmaxf(exp_max, tmp);
        exp_sum = exp_sum * __expf(exp_max - tmp_max) + __expf(tmp - tmp_max);
        exp_max = tmp_max;
    }
    
    float exp_max_o;
    float exp_sum_o;
    softmax_warp_reduce<32>(exp_max, exp_sum, &exp_max_o, &exp_sum_o);

    if (lane_id == 0) {
        exp_sum_output[warp_id] = exp_sum_o;
        exp_max_output[warp_id] = exp_max_o;
    }

    __syncthreads();
    if (warp_id == 0) {
        exp_max = exp_max_output[lane_id];
        exp_sum = exp_sum_output[lane_id];
        softmax_warp_reduce<8>(exp_max, exp_sum, &exp_max_o, &exp_sum_o);
    }

    if (threadIdx.x == 0) {
        exp_max_grid[blockIdx.x] = exp_max_o;
        exp_sum_grid[blockIdx.x] = exp_sum_o;
    }

    grid.sync();

    __shared__ float final_exp_max, final_exp_sum;

    if (threadIdx.x < 32) {
        exp_max_o = -FLT_MAX;
        exp_sum_o = 0.0f;
        for (int i = threadIdx.x; i < gridDim.x; i+=warpSize) {
            float exp_max_o_tmp = exp_max_o;
            exp_max_o = fmaxf(exp_max_o_tmp, exp_max_grid[i]);
            exp_sum_o = exp_sum_o * __expf(exp_max_o_tmp - exp_max_o) + exp_sum_grid[i] * __expf(exp_max_grid[i] - exp_max_o);
        }

        softmax_warp_reduce<32>(exp_max_o, exp_sum_o, &exp_max_o, &exp_sum_o);
        if (threadIdx.x == 0) {
            final_exp_max = exp_max_o;
            final_exp_sum = 1. / exp_sum_o;
        }
    }
    __syncthreads();

    stride = gridDim.x * blockDim.x;
    idx = (blockDim.x * blockIdx.x + threadIdx.x);
    for (int i = idx; i < N; i += stride) {
        output[i] = __expf(input[i] - final_exp_max) * final_exp_sum;
    }
}


__global__ void softmax_grid_reduce(const float* __restrict__ exp_max_input, const float* __restrict__ exp_sum_input, float* __restrict__ exp_max_output_block, float* __restrict__ exp_sum_output_block, int N) {
    __shared__ float exp_sum_output[8];
    __shared__ float exp_max_output[8];

    int idx = blockDim.x * blockIdx.x * 2 + threadIdx.x;
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    
    float max_input1 = (idx < N) ? exp_max_input[idx] : 0.0f;
    float max_input2 = (idx + 256 < N) ? exp_max_input[idx + 256] : 0.0f;

    float sum_input1 = (idx < N) ? exp_sum_input[idx] : 0.0f;
    float sum_input2 = (idx + 256 < N) ? exp_sum_input[idx + 256] : 0.0f;

    float exp_max = max_input1 > max_input2 ? max_input1 : max_input2;
    float exp_sum =  sum_input1 * __expf(max_input1 - exp_max) + sum_input2 * __expf(max_input2 - exp_max);

    float exp_max_o;
    float exp_sum_o;

    softmax_warp_reduce<32>(exp_max, exp_sum, &exp_max_o, &exp_sum_o);

    if (lane_id == 0) {
        exp_sum_output[warp_id] = exp_sum_o;
        exp_max_output[warp_id] = exp_max_o;
    }

    __syncthreads();
    if (warp_id == 0) {
        exp_max = exp_max_output[lane_id];
        exp_sum = exp_sum_output[lane_id];
        softmax_warp_reduce<8>(exp_max, exp_sum, &exp_max_o, &exp_sum_o);
    }

    if (threadIdx.x == 0) {
        exp_max_output_block[0] = exp_max_o;
        exp_sum_output_block[0] = 1.0 / exp_sum_o;
    }
}

__global__ void softmax_kernel(const float* input, float *output, float* sum_exp, float* max_exp, int N) {
    int stride = gridDim.x * blockDim.x;
    int idx = (blockDim.x * blockIdx.x + threadIdx.x);
    for (int i = idx; i < N; i += stride) {
        output[i] = __expf(input[i] - max_exp[0]) * sum_exp[0];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = 160;

    void* args[] = { (void*)&input, (void*)&output, (void*)&N };

    dim3 gridDim(blocksPerGrid, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);
    (cudaLaunchCooperativeKernel(
        (void*)softmax_block_reduce,
        gridDim,
        blockDim,
        args,
        0,    
        0        
    ));
}
