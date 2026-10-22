#include "cuda_runtime.h"
#include "float.h"
#include "cooperative_groups.h"

#define FULL_MASK 0xffffffffu
#define BLOCK_NUMS 80

constexpr int WARP_SIZE = 32;

template<unsigned int TOPK>
__device__ __forceinline__ void insert(float (&values)[TOPK], float candidate) {
  bool greater = false;
  float old_value = 0.0f;
  #pragma unroll
  for (int j = 0; j < TOPK; ++j) {
    greater = candidate > values[j];
    old_value = values[j];
    values[j] = greater ? candidate : old_value;
    candidate = greater ? old_value : candidate;
  }
}

template<unsigned int TOPK, unsigned int stride>
__device__ __forceinline__ void reduce(float (&values)[TOPK]) {
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  // Keep this loop rolled. Unrolling TOPK candidates and the TOPK-sized
  // insertion together would generate O(TOPK^2) instructions per stage.
  #pragma unroll 1
  for (int i = 0; i < TOPK; ++i) {
    const float candidate = __shfl_down_sync(FULL_MASK, values[i], stride);
    if (lane < stride) {
      insert(values, candidate);
    }
  }
}

template<unsigned int TOPK>
__device__ __forceinline__ void for_reduce_all(float (&values)[TOPK]) {
  reduce<TOPK, 16>(values);
  reduce<TOPK, 8>(values);
  reduce<TOPK, 4>(values);
  reduce<TOPK, 2>(values);
  reduce<TOPK, 1>(values);
}

__device__ float g_topk[BLOCK_NUMS*100];

template<unsigned int TOPK, int THREADS_PER_BLOCK>
__global__ __launch_bounds__(THREADS_PER_BLOCK, 2)
void topk_kernel(const float* input, float* output, int N, int k) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  const int tx = threadIdx.x;
  const int lane = tx & (WARP_SIZE - 1);
  const int warp_id = tx / WARP_SIZE;
  constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;
  const int block_base = blockIdx.x * THREADS_PER_BLOCK;
  constexpr int grid_stride = BLOCK_NUMS * THREADS_PER_BLOCK;

  float values[TOPK];
  extern __shared__ float s_topk[];
  #pragma unroll
  for (int i = 0; i < TOPK; ++i) {
    values[i] = -FLT_MAX;
  }

  // Each thread builds a private sorted Top-K list.
  for (int i = block_base + tx; i < N; i += grid_stride) {
    float candidate = input[i];
    insert(values, candidate);
  }

  // First reduce each warp to its lane 0, then publish one Top-K list per
  // warp. This keeps dynamic shared memory at WARPS_PER_BLOCK * TOPK.
  for_reduce_all(values);
  if (lane == 0) {
    #pragma unroll
    for (int i = 0; i < TOPK; ++i) {
      s_topk[warp_id * TOPK + i] = values[i];
    }
  }
  __syncthreads();

  // Warp 0 merges the per-warp lists into one Top-K list for this block.
  if (warp_id == 0) {
    #pragma unroll
    for (int i = 0; i < TOPK; ++i) {
      values[i] = -FLT_MAX;
    }
    for (int i = lane; i < WARPS_PER_BLOCK * TOPK; i += WARP_SIZE) {
      insert(values, s_topk[i]);
    }
    for_reduce_all(values);
    if (lane == 0) {
      const int output_base = blockIdx.x * TOPK;
      #pragma unroll
      for (int i = 0; i < TOPK; ++i) {
        g_topk[output_base + i] = values[i];
      }
    }
  }

  grid.sync();

  // Block 0 warp 0 performs the final global merge. Reset its previous
  // block-local state so block 0 is not counted twice.
  if (blockIdx.x == 0 && warp_id == 0) {
    #pragma unroll
    for (int i = 0; i < TOPK; ++i) {
      values[i] = -FLT_MAX;
    }
    for (int i = lane; i < BLOCK_NUMS * TOPK; i += WARP_SIZE) {
      insert(values, g_topk[i]);
    }
    for_reduce_all(values);
    if (lane == 0) {
      #pragma unroll
      for (int i = 0; i < TOPK; ++i) {
        output[i] = values[i];
      }
    }
  }
}

// Generic correctness fallback for k != 100.
//
// Sort all input values with four stable 8-bit LSD radix passes, then copy the
// largest k values in reverse order.  The float-to-key transform preserves the
// numeric order of all non-NaN IEEE-754 float values.
constexpr int FALLBACK_RADIX_BITS = 8;
constexpr int FALLBACK_RADIX_SIZE = 1 << FALLBACK_RADIX_BITS;
constexpr int FALLBACK_THREADS = 32;
constexpr int FALLBACK_ITEMS_PER_CHUNK = 4096;

__device__ __forceinline__ unsigned int float_to_ordered_key(float value) {
  const unsigned int bits = __float_as_uint(value);
  const unsigned int mask =
      (bits & 0x80000000u) ? 0xffffffffu : 0x80000000u;
  return bits ^ mask;
}

// Produce one 256-bin histogram for each contiguous input chunk.
__global__ void fallback_radix_histogram(
    const float* input,
    unsigned int* block_histograms,
    int N,
    int shift) {
  __shared__ unsigned int histogram[FALLBACK_RADIX_SIZE];
  const int lane = threadIdx.x;

  for (int bin = lane; bin < FALLBACK_RADIX_SIZE;
       bin += FALLBACK_THREADS) {
    histogram[bin] = 0;
  }
  __syncwarp();

  const int chunk_begin = blockIdx.x * FALLBACK_ITEMS_PER_CHUNK;
  const int chunk_end = min(chunk_begin + FALLBACK_ITEMS_PER_CHUNK, N);
  for (int index = chunk_begin + lane; index < chunk_end;
       index += FALLBACK_THREADS) {
    const unsigned int key = float_to_ordered_key(input[index]);
    const unsigned int digit =
        (key >> shift) & (FALLBACK_RADIX_SIZE - 1);
    atomicAdd(&histogram[digit], 1u);
  }
  __syncwarp();

  for (int bin = lane; bin < FALLBACK_RADIX_SIZE;
       bin += FALLBACK_THREADS) {
    block_histograms[blockIdx.x * FALLBACK_RADIX_SIZE + bin] =
        histogram[bin];
  }
}

// Convert per-chunk counts into stable global output offsets.  One thread owns
// one radix bin, so accesses for each chunk remain coalesced across the block.
__global__ void fallback_radix_offsets(
    unsigned int* block_histograms,
    int chunk_count) {
  __shared__ unsigned int totals[FALLBACK_RADIX_SIZE];
  __shared__ unsigned int bucket_bases[FALLBACK_RADIX_SIZE];
  const int bin = threadIdx.x;

  unsigned int total = 0;
  for (int chunk = 0; chunk < chunk_count; ++chunk) {
    total += block_histograms[chunk * FALLBACK_RADIX_SIZE + bin];
  }
  totals[bin] = total;
  __syncthreads();

  if (bin == 0) {
    unsigned int base = 0;
    for (int i = 0; i < FALLBACK_RADIX_SIZE; ++i) {
      bucket_bases[i] = base;
      base += totals[i];
    }
  }
  __syncthreads();

  unsigned int offset = bucket_bases[bin];
  for (int chunk = 0; chunk < chunk_count; ++chunk) {
    const int position = chunk * FALLBACK_RADIX_SIZE + bin;
    const unsigned int count = block_histograms[position];
    block_histograms[position] = offset;
    offset += count;
  }
}

// Stable scatter within each chunk.  __match_any_sync groups equal radix
// digits in a warp; lane order supplies the stable rank inside each group.
__global__ void fallback_radix_scatter(
    const float* input,
    float* output,
    const unsigned int* block_offsets,
    int N,
    int shift) {
  __shared__ unsigned int emitted[FALLBACK_RADIX_SIZE];
  const int lane = threadIdx.x;

  for (int bin = lane; bin < FALLBACK_RADIX_SIZE;
       bin += FALLBACK_THREADS) {
    emitted[bin] = 0;
  }
  __syncwarp();

  const int chunk_begin = blockIdx.x * FALLBACK_ITEMS_PER_CHUNK;
  const int chunk_end = min(chunk_begin + FALLBACK_ITEMS_PER_CHUNK, N);
  for (int round_begin = chunk_begin; round_begin < chunk_end;
       round_begin += FALLBACK_THREADS) {
    const int index = round_begin + lane;
    const bool active = index < chunk_end;
    const unsigned int active_mask = __ballot_sync(FULL_MASK, active);

    if (active) {
      const float value = input[index];
      const unsigned int key = float_to_ordered_key(value);
      const unsigned int digit =
          (key >> shift) & (FALLBACK_RADIX_SIZE - 1);
      const unsigned int group_mask =
          __match_any_sync(active_mask, digit);
      const int leader = __ffs(group_mask) - 1;

      unsigned int group_base = 0;
      if (lane == leader) {
        group_base = emitted[digit];
        emitted[digit] += __popc(group_mask);
      }
      group_base = __shfl_sync(group_mask, group_base, leader);
      const unsigned int lower_lanes =
          group_mask & ((1u << lane) - 1u);
      const unsigned int rank = __popc(lower_lanes);
      const unsigned int output_index =
          block_offsets[blockIdx.x * FALLBACK_RADIX_SIZE + digit] +
          group_base + rank;
      output[output_index] = value;
    }

    // Makes updates to emitted[] visible before the next input round.
    __syncwarp();
  }
}

__global__ void fallback_copy_topk_descending(
    const float* sorted_input,
    float* output,
    int N,
    int k) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < k) {
    output[index] = sorted_input[N - 1 - index];
  }
}

void launch_generic_topk(
    const float* input,
    float* output,
    int N,
    int k) {
  const int chunk_count =
      (N + FALLBACK_ITEMS_PER_CHUNK - 1) / FALLBACK_ITEMS_PER_CHUNK;

  float* buffer_a = nullptr;
  float* buffer_b = nullptr;
  unsigned int* block_histograms = nullptr;
  cudaMalloc(&buffer_a, static_cast<size_t>(N) * sizeof(float));
  cudaMalloc(&buffer_b, static_cast<size_t>(N) * sizeof(float));
  cudaMalloc(
      &block_histograms,
      static_cast<size_t>(chunk_count) * FALLBACK_RADIX_SIZE *
          sizeof(unsigned int));

  const float* source = input;
  float* destination = buffer_a;
  for (int shift = 0; shift < 32; shift += FALLBACK_RADIX_BITS) {
    fallback_radix_histogram<<<chunk_count, FALLBACK_THREADS>>>(
        source, block_histograms, N, shift);
    fallback_radix_offsets<<<1, FALLBACK_RADIX_SIZE>>>(
        block_histograms, chunk_count);
    fallback_radix_scatter<<<chunk_count, FALLBACK_THREADS>>>(
        source, destination, block_histograms, N, shift);

    source = destination;
    destination = destination == buffer_a ? buffer_b : buffer_a;
  }

  constexpr int COPY_THREADS = 256;
  const int copy_blocks = (k + COPY_THREADS - 1) / COPY_THREADS;
  fallback_copy_topk_descending<<<copy_blocks, COPY_THREADS>>>(
      source, output, N, k);

  cudaFree(block_histograms);
  cudaFree(buffer_b);
  cudaFree(buffer_a);
}


extern "C" void solve(const float* input, float* output, int N, int k) {
  if (k != 100) {
    launch_generic_topk(input, output, N, k);
    return;
  }

  constexpr int TOPK = 100;
  constexpr int thread_per_block = 128;
  constexpr int warps_per_block = thread_per_block / WARP_SIZE;
  constexpr int shared_memory_bytes =
      warps_per_block * TOPK * sizeof(float);

  dim3 grid(BLOCK_NUMS, 1, 1);
  dim3 block(thread_per_block, 1, 1);
  cudaFuncSetAttribute(
      topk_kernel<TOPK, thread_per_block>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      shared_memory_bytes);

  void* args[4] = {(void*)(&input), (void*)(&output), (void*)(&N), (void*)(&k)};
  cudaLaunchCooperativeKernel(
    (void*)topk_kernel<TOPK, thread_per_block>,
    grid, block,
    args,
    shared_memory_bytes,
    0
  );
}
