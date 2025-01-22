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
    u64 localSum = 0;
    // Stride loop to collect all values for this thread
    while (tid < n) {
        localSum += g_input[tid];
        tid += grid;
    }

    // Write to shared memory
    sdata[threadIdx.x] = localSum;
    __syncthreads();

    // Tree reduction in shared memory
    u32 blockSize = blockDim.x;
    for (u32 stride = blockSize / 2; stride > 0; stride >>= 1) {
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
 *        For simplicity, we run only one block here.
 */
__global__ void finalSumKernel(const u64* __restrict__ g_blockSums, u64* __restrict__ g_out, size_t n)
{
    extern __shared__ u64 sdata[];

    // Each thread loads one partial sum
    size_t tid = threadIdx.x;
    u64 val    = 0;
    if (tid < n) {
        val = g_blockSums[tid];
    }
    sdata[threadIdx.x] = val;
    __syncthreads();

    // Perform standard reduction in shared memory
    u32 blockSize = blockDim.x;
    for (u32 stride = blockSize / 2; stride > 0; stride >>= 1) {
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
    const u64 blockSize = 256;
    const u64 gridSize  = static_cast<int>((n + blockSize - 1) / blockSize);

    // Allocate partial sums (one per block)
    u64* d_blockSums = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_blockSums, gridSize * sizeof(u64)));

    // First pass: partial sums per block
    size_t sharedBytes = blockSize * sizeof(u64);
    blockSumKernel<<<gridSize, blockSize, sharedBytes>>>(d_in, d_blockSums, n);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_LAST_CUDA_ERROR();

    // If we have only one block, we are already done
    if (gridSize == 1) {
        // Just copy back result
        u64 hostSum;
        CHECK_CUDA_ERROR(cudaMemcpy(&hostSum, d_blockSums, sizeof(u64), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaFree(d_blockSums));
        return hostSum;
    }

    // Final pass: sum the partial sums in a single block
    u64* d_out = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_out, sizeof(u64)));

    // Launch one block with enough threads to handle gridSize partial sums
    u64 finalThreads = std::max((u64)1, std::min(blockSize, gridSize));
    size_t shared2   = finalThreads * sizeof(u64);
    finalSumKernel<<<1, finalThreads, shared2>>>(d_blockSums, d_out, gridSize);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_LAST_CUDA_ERROR();

    // Copy final sum to host
    u64 hostSum;
    CHECK_CUDA_ERROR(cudaMemcpy(&hostSum, d_out, sizeof(u64), cudaMemcpyDeviceToHost));

    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_blockSums));
    CHECK_CUDA_ERROR(cudaFree(d_out));

    return hostSum;
}
