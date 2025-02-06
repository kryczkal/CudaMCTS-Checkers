#include "cuda/cuda_utils.cuh"
#include "cuda/reductions.cuh"

#include <cuda_runtime.h>
#include <algorithm>

/**
 * @brief First-pass kernel: each block sums part of the input array into a
 *        single partial sum stored in g_blockSums[blockIdx.x].
 */
__global__ void blockSumKernel(const u8* __restrict__ g_input, u64* __restrict__ g_blockSums, size_t n)
{
    extern __shared__ u64 sdata[];  // dynamic shared memory for partial sums

    // Global thread index
    size_t tid  = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid = gridDim.x * blockDim.x;

    // Accumulate a local sum in 64-bit
    u64 local_sum = 0;
    // Stride loop to collect all values for this thread
    while (tid < n) {
        local_sum += g_input[tid];
        tid += grid;
    }

    // Write to shared memory
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    // Tree reduction in shared memory
    const u32 block_size = blockDim.x;
    for (u32 stride = block_size / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Write block’s partial sum to g_blockSums
    if (threadIdx.x == 0) {
        g_blockSums[blockIdx.x] = sdata[0];
    }
}

/**
 * @brief Final pass kernel: sums up the block partial sums into a single value.
 */
__global__ void finalSumKernel(const u64* __restrict__ g_blockSums, u64* __restrict__ g_out, size_t n)
{
    extern __shared__ u64 sdata[];

    // Each thread loads one partial sum
    const size_t kTid = threadIdx.x;
    u64 val           = 0;
    if (kTid < n) {
        val = g_blockSums[kTid];
    }
    sdata[threadIdx.x] = val;
    __syncthreads();

    // Perform standard reduction in shared memory
    const u32 block_size = blockDim.x;
    for (u32 stride = block_size / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        // Write final sum
        g_out[0] = sdata[0];
    }
}

/**
 * @brief Host function to sum n bytes in d_in, returning a 64-bit integer result.
 */
u64 DeviceSumU8(const u8* d_in, size_t n)
{
    if (n == 0) {
        return 0;
    }

    // Decide block and grid size
    const u64 kBlockSize = 256;
    const u64 kGridSize  = static_cast<int>((n + kBlockSize - 1) / kBlockSize);

    // Allocate partial sums (one per block)
    u64* d_block_sums = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_block_sums, kGridSize * sizeof(u64)));

    // First pass: partial sums per block
    size_t shared_bytes = kBlockSize * sizeof(u64);
    blockSumKernel<<<kGridSize, kBlockSize, shared_bytes>>>(d_in, d_block_sums, n);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_LAST_CUDA_ERROR();

    // If we have only one block, we are already done
    if (kGridSize == 1) {
        // Just copy back result
        u64 host_sum;
        CHECK_CUDA_ERROR(cudaMemcpy(&host_sum, d_block_sums, sizeof(u64), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaFree(d_block_sums));
        return host_sum;
    }

    // Final pass: sum the partial sums in a single block
    u64* d_out = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_out, sizeof(u64)));

    // Launch one block with enough threads to handle partial sums
    const u64 final_threads = std::max((u64)1, std::min(kBlockSize, kGridSize));
    const size_t shared_2   = final_threads * sizeof(u64);
    finalSumKernel<<<1, final_threads, shared_2>>>(d_block_sums, d_out, kGridSize);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_LAST_CUDA_ERROR();

    // Copy final sum to host
    u64 host_sum;
    CHECK_CUDA_ERROR(cudaMemcpy(&host_sum, d_out, sizeof(u64), cudaMemcpyDeviceToHost));

    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_block_sums));
    CHECK_CUDA_ERROR(cudaFree(d_out));

    return host_sum;
}
