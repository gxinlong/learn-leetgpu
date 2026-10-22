#include <cuda_runtime.h>
#include <cmath>

#define CEIL(N, DIV) ((N+DIV-1)/(DIV))
// Transpose config
constexpr int TRAN_BW = 32;

// Matmul config
constexpr int BW = 64;      // Block Width: The tiled linear size (the pixels)
constexpr int BW_bit = 6;
constexpr int TW = 4;       // Tile Width: The local size each thread responsible for
constexpr int TW_bit = 2;
constexpr int TC = BW/TW;   // Thread Count: The thread count along one dimension
constexpr int TC_bit = BW/TW;

// Softmax config
constexpr int SOFTMAX_BW = 256;
constexpr int WARP_SIZE = 32;

__global__ void transpose_kernel(const float* input, float* output, int rows, int cols){
    __shared__ float tile[TRAN_BW][TRAN_BW+1];
    int in_x = blockIdx.x * blockDim.x + threadIdx.x;
    int in_y = blockIdx.y * blockDim.y + threadIdx.y;


    if(in_y<rows && in_x<cols){
        tile[threadIdx.y][threadIdx.x] = input[in_y*cols + in_x];
    }

    __syncthreads();
    int out_x = in_y;
    int out_y = in_x;

    if(out_y<cols && out_x<rows){
        output[out_y*rows + out_x] = tile[threadIdx.y][threadIdx.x];
    }
}

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K){
    __shared__ float sA[BW][BW+1];
    __shared__ float sB[BW][BW+1];
    float tile[TW][TW] = {0.0};
    float m_reg[TW];
    float n_reg[TW];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int tid = threadIdx.y*blockDim.x + threadIdx.x;

    int pixelCount = BW*BW;
    int threadCount = blockDim.y*blockDim.x;

    int phaseCount = CEIL(N, BW);

    // Tiled linear loop
    for(int ph=0; ph<phaseCount; ++ph){
        // The start addr of this phase
        int phaseBase = ph << BW_bit;

        // 1. Load data from Global to Shared
        //    Each thread is responsible for (pixelCount/threadCount) pixels
        for(int i=0; i<pixelCount; i+=threadCount){
            int offset = tid + i;
            int local_x = offset & (BW-1);
            int local_y = offset >> BW_bit;
            int global_x = (blockIdx.x << BW_bit) + local_x;
            int global_y = (blockIdx.y << BW_bit) + local_y;

            sA[local_y][local_x] = (global_y<M && phaseBase+local_x<N) ? A[global_y*N + (phaseBase+local_x)]
                                                                        : 0.0f;
            sB[local_y][local_x] = (phaseBase+local_y<N && global_x<K) ? B[(phaseBase+local_y)*K + global_x]
                                                                        : 0.0f;
        }

        __syncthreads();
        
        // 2. Load data from Shared to Register
        //    And accumulate the 8x8 outer product
        for(int k=0; k<BW; ++k){
            // Load the m_reg and n_reg
            #pragma unroll
            for(int i=0; i<TW; ++i){
                m_reg[i] = sA[(ty<<TW_bit)+i][k];
                n_reg[i] = sB[k][(tx<<TW_bit)+i];
            }

            // Accumulate the outer product
            for(int i=0; i<TW; ++i){
                #pragma unroll
                for(int j=0; j<TW; ++j){
                    tile[i][j] += m_reg[i] * n_reg[j];
                }
            }
        }

        __syncthreads();   
    }

    // 3. Write back from Register to Global
    int blockBase_y = blockIdx.y << BW_bit;
    int blockBase_x = blockIdx.x << BW_bit;
    for(int i=0; i<TW; ++i){
        int global_y = blockBase_y + (ty<<TW_bit) + i;
        for(int j=0; j<TW; ++j){
            int global_x = blockBase_x + (tx<<TW_bit) + j;
            
            if(global_y<M && global_x<K){
                C[global_y*K + global_x] = tile[i][j];
            }
        }
    }
}

template <bool IS_MAX>
__inline__ __device__ void blockReduce(const float* input, float* output, int N){
    __shared__ float sdata[SOFTMAX_BW];
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    // 1. Reduce 256*16 pixels into 256 pixels
    float val = IS_MAX ? -INFINITY : 0.0f;

    for(int i=tid; i<N; i+=bdim){
        if(i<N){
            if(IS_MAX){
                val = fmaxf(val, input[i]);
            }
            else{
                val += input[i];
            }
        }
    }

    sdata[tid] = val;
    __syncthreads();

    // 2. Reduce 256 pixels into 32 pixels (WARP_SIZE)
    if(tid<WARP_SIZE){
        val = IS_MAX ? -INFINITY : 0.0f;
        for(int i=tid; i<bdim; i+=WARP_SIZE){
            if (IS_MAX) {
                val = fmaxf(val, sdata[i]);
            } else {
                val += sdata[i];
            }
        }
    }

    // 3. Reduce 32 pixels into 1 pixel using shfl_sync
    if(tid<WARP_SIZE){
        if(IS_MAX){
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, 16));
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, 8));
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, 4));
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, 2));
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, 1));
        }
        else{
            val += __shfl_down_sync(0xFFFFFFFF, val, 16);
            val += __shfl_down_sync(0xFFFFFFFF, val, 8);
            val += __shfl_down_sync(0xFFFFFFFF, val, 4);
            val += __shfl_down_sync(0xFFFFFFFF, val, 2);
            val += __shfl_down_sync(0xFFFFFFFF, val, 1);
        }
    }

    if(tid==0){
        *output = val;
    }
    __syncthreads();
}


__global__ void softmax_kernel(float* input, float scale, int M, int N){
    
    __shared__ float s_row_max;
    __shared__ float s_row_sum;

    int row = blockIdx.x;
    if(row>=M) return;
    int tid = threadIdx.x;

    float* row_input = input + row*N;

    // 1. Scale the score
    for(int i=tid; i<N; i+=blockDim.x){
        row_input[i] = row_input[i] * scale;
    }
    __syncthreads();

    // 2. Find the row max and 
    blockReduce<true>(row_input, &s_row_max, N);

    // 3. Compute safe exponent
    for(int i=tid; i<N; i+=blockDim.x){
        row_input[i] = expf(row_input[i] - s_row_max); 
    }
    __syncthreads();

    // 4. Get the accumulation
    blockReduce<false>(row_input, &s_row_sum, N);
    if(tid==0){
        s_row_sum = 1.0 / s_row_sum;
    }

    __syncthreads();

    // 5. Normalization
    for(int i=tid; i<N; i+=blockDim.x){
        row_input[i] = row_input[i] * s_row_sum; 
    }
    __syncthreads();

}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d){
    float *KT, *score;
    cudaMalloc(&KT, sizeof(float) * d * N);
    cudaMalloc(&score, sizeof(float) * M * N);

    // 1. Transpose the K (Nxd)
    dim3 TRANSPOSE_threadsPerBlock(TRAN_BW, TRAN_BW);
    dim3 TRANSPOSE_blockPerGrid(CEIL(d, TRAN_BW), CEIL(N, TRAN_BW));
    transpose_kernel<<<TRANSPOSE_blockPerGrid, TRANSPOSE_threadsPerBlock>>>(K, KT, N, d);

    // 2. QK^T Matmul (M*d * d*N)
    dim3 MATMUL_threadsPerBlock(BW/TW, BW/TW);
    dim3 MATMUL_blockPerGrid(CEIL(N, BW), CEIL(M, BW));
    matmul_kernel<<<MATMUL_blockPerGrid, MATMUL_threadsPerBlock>>>(Q, KT, score, M, d, N);

    // 3. Softmax
    int SOFTMAX_threadsPerBlock = SOFTMAX_BW;
    int SOFTMAX_blockPerGrid = M;
    float scale = 1.0 / sqrtf(d);
    softmax_kernel<<<SOFTMAX_blockPerGrid, SOFTMAX_threadsPerBlock>>>(score, scale, M, N);

    // 4. Score * V (M*N * N*d)
    MATMUL_threadsPerBlock = dim3(BW/TW, BW/TW);
    MATMUL_blockPerGrid = dim3(CEIL(d, BW), CEIL(M, BW));
    matmul_kernel<<<MATMUL_blockPerGrid, MATMUL_threadsPerBlock>>>(score, V, output, M, N, d);

    cudaFree(KT);
    cudaFree(score);

}
