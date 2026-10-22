#include <cuda_runtime.h>

// 简单的插入排序，用于维护一个小的 Top-K 数组
__device__ void insert_val(float* data, float val, int k) {
    if (val <= data[k-1]) return;
    int i = k - 2;
    while (i >= 0 && data[i] < val) {
        data[i+1] = data[i];
        i--;
    }
    data[i+1] = val;
}

__global__ void top_k_warp_kernel(const float* input, float* output, int N, int k) {
    // 现在的共享内存：每个 Warp 拥有一块 K 大小的区域
    // 每个 Block 有 256/32 = 8 个 Warp
    // 8 * 100 * 4 = 3.2KB，远小于 48KB 限制！
    extern __shared__ float smem_pool[];
    
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    float* my_warp_q = smem_pool + warp_id * k;

    // 1. 初始化 Warp 共享区域 (由每个 Warp 的前几个线程分担)
    if (lane_id < k) {
        for(int i = lane_id; i < k; i += 32) my_warp_q[i] = -1e38f;
    }
    __syncwarp();

    // 2. 数据处理：由于不能每个线程存 K 个，我们让每个线程只存 1 个最大值做初步筛选
    float local_max = -1e38f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        if (input[i] > local_max) local_max = input[i];
        // 如果数据量巨大，这里可以每隔一段就往 my_warp_q 里塞一次，
        // 但为了逻辑清晰，我们先做简单处理。
    }

    // 3. 关键：将每个线程攒下的 local_max 插入到所属 Warp 的共享列表中
    // 注意：插入共享内存需要互斥或简单的顺序逻辑
    // 这里使用简单的原子锁或让 Warp 内轮流插入（由于 K 小，性能损失可接受）
    for (int j = 0; j < 32; j++) {
        if (lane_id == j) {
            insert_val(my_warp_q, local_max, k);
        }
        __syncwarp(); 
    }
    __syncthreads();

    // 4. 最后由每个 Block 的 0 号线程汇总所有 Warp 的结果 (8个 Warp * K)
    if (threadIdx.x == 0) {
        // 直接在 smem_pool[0...k-1] 归约剩下的几个 Warp
        for (int w = 1; w < (blockDim.x / 32); w++) {
            float* other_warp_q = smem_pool + w * k;
            for (int i = 0; i < k; i++) {
                insert_val(smem_pool, other_warp_q[i], k);
            }
        }
        // 写回全局内存
        for (int i = 0; i < k; i++) {
            output[blockIdx.x * k + i] = smem_pool[i];
        }
    }
}

extern "C" void solve(const float* input, float* output, int N, int k) {
    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = 256; 
    
    // 现在的共享内存大小：8条 Warp * 100个 float
    size_t smem_size = warps_per_block * k * sizeof(float);

    float* d_temp;
    cudaMalloc(&d_temp, blocks * k * sizeof(float));

    // 第一遍：N -> blocks * k
    top_k_warp_kernel<<<blocks, threads, smem_size>>>(input, d_temp, N, k);
    
    // 第二遍：blocks * k -> k (只启动一个 block)
    top_k_warp_kernel<<<1, threads, smem_size>>>(d_temp, output, blocks * k, k);

    cudaFree(d_temp);
}