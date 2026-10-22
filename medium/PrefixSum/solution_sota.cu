#include <cuda_runtime.h>
#include <cooperative_groups.h>

#define CDIV(X, Y) ((X + Y - 1) / Y)
#define FLOAT4(X) (*(reinterpret_cast<float4*>(&X)))
#define CONST_FLOAT4(X) (*(reinterpret_cast<const float4*>(&X)))

namespace cg = cooperative_groups;

__device__ float prefix_sum_grid[1024];

__device__ float prefix_sum_warp_kernel(float v) {
    int lane_id = threadIdx.x % 32;
    float t;
    t = __shfl_up_sync(0xffffffff, v, 1);
    if (lane_id >= 1) {
        v += t;
    }
        t = __shfl_up_sync(0xffffffff, v, 2);
    if (lane_id >= 2) {
        v += t;
    }
        t = __shfl_up_sync(0xffffffff, v, 4);
    if (lane_id >= 4) {
        v += t;
    }
        t = __shfl_up_sync(0xffffffff, v, 8);
    if (lane_id >= 8) {
        v += t;
    }
        t = __shfl_up_sync(0xffffffff, v, 16);
    if (lane_id >= 16) {
        v += t;
    }
    return v;
}

__global__ void prefix_sum_kernel(const float* input, float *output, int N) {
    cg::grid_group grid = cg::this_grid();
    __shared__ float prefix_sum_block[8];
    int tid = threadIdx.x;
    int offset = blockIdx.x * (12 * 256) + tid * 12;

    alignas(16) float input_thread[12];

    if (offset + 11 < N) {
        FLOAT4(input_thread[0]) = CONST_FLOAT4(input[offset]);
        FLOAT4(input_thread[4]) = CONST_FLOAT4(input[offset + 4]);
        FLOAT4(input_thread[8]) = CONST_FLOAT4(input[offset + 8]);
    } else {
        for (int i = 0; i < 12; i++) {
            if (offset + i < N) {
                input_thread[i] = input[offset + i];
            } else {
                input_thread[i] = 0;
            }
        }
    }

    for (int i = 1; i < 12; i++) {
        input_thread[i] = input_thread[i - 1] + input_thread[i];
    }

    // warp内求和 128
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    float prefix_sum_warp = prefix_sum_warp_kernel(input_thread[11]);
    
    if (lane_id == 31) {
        prefix_sum_block[warp_id] = prefix_sum_warp;
    }

    __syncthreads();

    if (warp_id == 0) {
        float prefix = lane_id < 8 ? prefix_sum_block[lane_id] : 0;
        prefix = prefix_sum_warp_kernel(prefix);
        if (lane_id < 8) {
            prefix_sum_block[lane_id] = prefix;
        }
    }

    __syncthreads(); 

    if (warp_id == 0 && lane_id == 0) {
        prefix_sum_grid[blockIdx.x] = prefix_sum_block[7];
    }

    grid.sync();

    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //     for (int i = 1; i < 82; i++) {
    //         prefix_sum_grid[i] += prefix_sum_grid[i - 1];
    //     }
    // }

    if (blockIdx.x == 0 && warp_id == 0) {
        int l = lane_id * 3;
        prefix_sum_grid[l + 1] += prefix_sum_grid[l];
        prefix_sum_grid[l + 2] += prefix_sum_grid[l + 1];
        float prefix = prefix_sum_grid[l + 2];
        prefix = prefix_sum_warp_kernel(prefix);
        prefix_sum_grid[l] += prefix - prefix_sum_grid[l + 2];
        prefix_sum_grid[l + 1] += prefix - prefix_sum_grid[l + 2];
        prefix_sum_grid[l + 2] = prefix;
    }

    grid.sync();

    for (int i = 0; i < 12; i++) {
        input_thread[i] = input_thread[i] + prefix_sum_warp - input_thread[11];
        if (warp_id > 0) {
            input_thread[i] += prefix_sum_block[warp_id - 1];
        }
        if (blockIdx.x > 0) {
            input_thread[i] += prefix_sum_grid[blockIdx.x - 1];
        }
    }

    if (offset + 11 < N) {
        FLOAT4(output[offset]) = FLOAT4(input_thread[0]);
        FLOAT4(output[offset + 4]) = FLOAT4(input_thread[4]);
        FLOAT4(output[offset + 8]) = FLOAT4(input_thread[8]);
    } else {
        for (int i = 0; i < 12; i++) {
            if (offset + i < N) {
                output[offset + i] = input_thread[i];
            }
        }
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    // 4, 32 * 4 = 128, 128 * 8 = 1024
    // 256 000 / 1024 = 250
    // int blocks = 240;
    int threads = 256;
    int nums_per_block = 256 * 5;
    int needed_blocks = 120;
    void *args[] = { &input, &output, &N };
    cudaLaunchCooperativeKernel(
        (void*)prefix_sum_kernel,
        needed_blocks, threads,
        args, 0, 0
    );
}
