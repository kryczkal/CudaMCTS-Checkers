#include <gtest/gtest.h>
#include <algorithm>
#include <unordered_map>
#include <vector>

#include "common/checkers_defines.hpp"
#include "cpu/board_helpers.hpp"
#include "cpu/launchers.hpp"
#include "cuda/board_helpers.cuh"
#include "cuda/cuda_utils.cuh"
#include "cuda/launchers.cuh"

/**
 * This test suite verifies the correctness of the `IsOnEdge` function
 * for both CPU and GPU implementations in the checkers engine.
 * It uses Google Test's Typed Tests to ensure both implementations
 * are tested uniformly.
 */

namespace
{

/**
 * @brief CUDA kernel to test IsOnEdge on the device.
 *
 * @param d_indices Device array of board indices.
 * @param d_results Device array to store results.
 * @param n         Number of elements.
 */
__global__ static void IsOnEdgeKernel(const checkers::board_index_t* d_indices, char* d_results, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_results[idx] = (checkers::gpu::move_gen::IsOnEdge(d_indices[idx]) != 0) ? 1 : 0;
    }
}

/**
 * CPU Implementation Wrapper for IsOnEdge Testing
 */
struct CPUISEdgeTestImpl {
    using BoardType = checkers::cpu::Board;

    /**
     * @brief Checks if a given index is on the edge for the CPU implementation.
     *
     * @param index The board index to check.
     * @return True if the index is on the edge, False otherwise.
     */
    static bool IsOnEdge(checkers::board_index_t index) { return checkers::cpu::move_gen::IsOnEdge(index) != 0; }

    /**
     * @brief Creates a dummy board (not used in this test).
     */
    static BoardType MakeBoard() { return BoardType{}; }

    /**
     * @brief Sets a piece on the board (not used in this test).
     */
    static void SetPiece(BoardType& board, checkers::board_index_t idx, char pieceType)
    {
        board.setPieceAt(idx, pieceType);
    }
};

/**
 * GPU Implementation Wrapper for IsOnEdge Testing
 */
struct GPUISEdgeTestImpl {
    using BoardType = checkers::gpu::launchers::GpuBoard;

    /**
     * @brief Checks if a given index is on the edge for the GPU implementation.
     *
     * @param indices Host array of board indices to check.
     * @param results Host array to store the results.
     */
    static void CheckIsOnEdgeGPU(const std::vector<checkers::board_index_t>& indices, std::vector<char>& results)
    {
        size_t n = indices.size();
        checkers::board_index_t* d_indices;
        char* d_results;

        // Allocate device memory
        cudaMalloc(&d_indices, n * sizeof(checkers::board_index_t));
        cudaMalloc(&d_results, n * sizeof(char));

        // Copy indices to device
        cudaMemcpy(d_indices, indices.data(), n * sizeof(checkers::board_index_t), cudaMemcpyHostToDevice);

        // Launch kernel
        int threads = 256;
        int blocks  = (n + threads - 1) / threads;
        IsOnEdgeKernel<<<blocks, threads>>>(d_indices, d_results, n);
        cudaDeviceSynchronize();

        // Check for kernel launch errors
        CudaUtils::CheckLastCudaError(__FILE__, __LINE__);

        // Copy results back to host
        results.resize(n);
        cudaMemcpy(results.data(), d_results, n * sizeof(char), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_indices);
        cudaFree(d_results);
    }

    /**
     * @brief Creates a dummy board (not used in this test).
     */
    static BoardType MakeBoard() { return BoardType{}; }

    /**
     * @brief Sets a piece on the board (not used in this test).
     */
    static void SetPiece(BoardType& board, checkers::board_index_t idx, char pieceType)
    {
        board.setPieceAt(idx, pieceType);
    }
};

/**
 * @brief Helper function to convert index to string for better error messages.
 */
std::string IndexToString(checkers::board_index_t index)
{
    // Assuming standard 8x4 board (32 squares)
    // Convert index to algebraic notation (a1 to h8)
    const char columns[] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'};
    size_t row           = index / 4;
    size_t col           = index % 4;
    // Adjust column based on row parity
    if (row % 2 == 0) {
        col = col * 2 + 1;
    } else {
        col = col * 2;
    }
    return std::string(1, columns[col]) + std::to_string(row + 1);
}

/**
 * @brief Typed Test Fixture for IsOnEdge Tests
 */
template <typename Impl>
class IsOnEdgeTest : public ::testing::Test
{
    protected:
};

using IsOnEdgeImplementations = ::testing::Types<CPUISEdgeTestImpl, GPUISEdgeTestImpl>;
TYPED_TEST_SUITE(IsOnEdgeTest, IsOnEdgeImplementations);

/**
 * @brief Test that certain indices are correctly identified as being on the edge.
 */
TYPED_TEST(IsOnEdgeTest, IdentifiesEdgeIndices)
{
    using ImplType = TypeParam;

    // Define edge indices for a standard 32-square checkers board
    // Assuming indices 0-31, with 0-7 being top row, 24-31 being bottom row
    std::vector<checkers::board_index_t> edgeIndices = {
        0,  1,  2,  3,   // Top edge
        0,  8,  16, 24,  // Left edge
        7,  15, 23, 31,  // Right edge
        28, 29, 30, 31   // Bottom edge
    };

    // Remove duplicates and sort
    std::vector<checkers::board_index_t> uniqueEdgeIndices;
    for (auto idx : edgeIndices) {
        if (std::find(uniqueEdgeIndices.begin(), uniqueEdgeIndices.end(), idx) == uniqueEdgeIndices.end()) {
            uniqueEdgeIndices.push_back(idx);
        }
    }

    // Create non-edge indices by excluding edgeIndices from all indices
    std::vector<checkers::board_index_t> allIndices(32);
    for (size_t i = 0; i < 32; ++i) {
        allIndices[i] = static_cast<checkers::board_index_t>(i);
    }

    std::vector<checkers::board_index_t> nonEdgeIndices;
    for (auto idx : allIndices) {
        if (std::find(uniqueEdgeIndices.begin(), uniqueEdgeIndices.end(), idx) == uniqueEdgeIndices.end()) {
            nonEdgeIndices.push_back(idx);
        }
    }

    // Test for CPU Implementation
    if constexpr (std::is_same_v<ImplType, CPUISEdgeTestImpl>) {
        // Test edge indices
        for (auto idx : uniqueEdgeIndices) {
            bool isEdge = ImplType::IsOnEdge(idx);
            EXPECT_TRUE(isEdge) << "Index " << IndexToString(idx) << " should be on the edge.";
        }

        // Test non-edge indices
        for (auto idx : nonEdgeIndices) {
            bool isEdge = ImplType::IsOnEdge(idx);
            EXPECT_FALSE(isEdge) << "Index " << IndexToString(idx) << " should NOT be on the edge.";
        }
    }
    // Test for GPU Implementation
    else if constexpr (std::is_same_v<ImplType, GPUISEdgeTestImpl>) {
        // Combine edge and non-edge indices
        std::vector<checkers::board_index_t> testIndices = uniqueEdgeIndices;
        testIndices.insert(testIndices.end(), nonEdgeIndices.begin(), nonEdgeIndices.end());

        std::vector<char> results;
        ImplType::CheckIsOnEdgeGPU(testIndices, results);

        ASSERT_EQ(testIndices.size(), results.size());

        // Check edge indices
        for (size_t i = 0; i < uniqueEdgeIndices.size(); ++i) {
            bool isEdge = results[i] != 0;
            EXPECT_TRUE(isEdge) << "GPU: Index " << IndexToString(testIndices[i]) << " should be on the edge.";
        }

        // Check non-edge indices
        for (size_t i = uniqueEdgeIndices.size(); i < testIndices.size(); ++i) {
            bool isEdge = results[i] != 0;
            EXPECT_FALSE(isEdge) << "GPU: Index " << IndexToString(testIndices[i]) << " should NOT be on the edge.";
        }
    }
}

}  // namespace
