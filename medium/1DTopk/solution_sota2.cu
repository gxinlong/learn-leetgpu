#include <cuda_runtime.h>
#include <cfloat>
__device__ void insert_value(float* data,float val, int k) {
    if(data[k-1] >= val) return;
    int idx = k-2;
    while(idx>=0 && data[idx] < val) {
        data[idx+1] = data[idx];
        idx--;
    }
    data[idx+1] = val;
}

__global__ void topk(const float* input, float* output, int N,int k) {
    extern __shared__ float candidate[];

    int warpID = threadIdx.x / 32;
    int laneID = threadIdx.x % 32;
    int gid = blockIdx.x* blockDim.x+threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int num_warps = blockDim.x / 32;
    float* warp_q = &candidate[warpID*k];

    if (laneID < k) {
        for(int i = laneID;i<k;i+=32) {
            warp_q[i] = -FLT_MAX;
        }
    }
     __syncwarp();

     float local_max = -FLT_MAX;
     for (int i = gid; i < N; i += stride) {
        local_max = fmaxf(local_max, input[i]);
    }

    for(int j =0;j<32;j++){
        if(laneID == j) {
            insert_value(warp_q,local_max,k);
        }
        __syncwarp();
    }
    __syncthreads();

    if(threadIdx.x==0) {
        for(int w = 1; w < (blockDim.x)/32;w++){
            float* other_warp_q = &candidate[w*k];
            for(int i=0;i<k;i++) {
                insert_value(candidate,other_warp_q[i],k);
            }
        }
        for(int i =0;i<k;i++){
            output[blockIdx.x*k+i] = candidate[i];
        }
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int k) {
    dim3 block(256);
    dim3 grid(200);

    int threads = 256;
    int warp_per_block = threads/32;
    int blocks = 128;

    size_t  smem_size = warp_per_block*k*sizeof(float);

    float* d_tmp;
    cudaMalloc(&d_tmp, blocks * k * sizeof(float));

    topk<<<blocks,threads,smem_size>>>(input, d_tmp, N, k);
    topk<<<1, threads, smem_size>>>(d_tmp, output, blocks * k, k);

    cudaFree(d_tmp);
}
