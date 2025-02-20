#include "common/checkers_defines.hpp"
#include "cpu/capture_lookup_table.hpp"
#include "cuda/capture_lookup_table.cuh"
#include "cuda/cuda_utils.cuh"

namespace checkers::gpu::apply_move
{
bool is_initialized = false;

__constant__ board_t d_kCaptureLookUpTable[BoardConstants::kBoardSize * BoardConstants::kBoardSize];

void InitializeCaptureLookupTable()
{
    if (is_initialized) {
        return;
    }
    // Flatten the 2D host array to a 1D array
    std::array<board_t, BoardConstants::kBoardSize * BoardConstants::kBoardSize> flat_table{};
    for (size_t i = 0; i < BoardConstants::kBoardSize; ++i) {
        for (size_t j = 0; j < BoardConstants::kBoardSize; ++j) {
            flat_table[i * BoardConstants::kBoardSize + j] = checkers::cpu::apply_move::h_kCaptureLookUpTable[i][j];
        }
    }

    // Copy the flattened data to constant memory on the device
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(
        d_kCaptureLookUpTable, flat_table.data(),
        sizeof(board_t) * BoardConstants::kBoardSize * BoardConstants::kBoardSize
    ));

    is_initialized = true;
}

}  // namespace checkers::gpu::apply_move
