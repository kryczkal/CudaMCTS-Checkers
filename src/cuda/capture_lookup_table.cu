#include "checkers_defines.hpp"
#include "cpu/capture_lookup_table.hpp"
#include "cuda/apply_move.cuh"
#include "cuda/board_helpers.cuh"
#include "cuda/capture_lookup_table.cuh"
#include "cuda/cuda_utils.cuh"

namespace checkers::gpu::apply_move
{
bool is_initialized = false;

__constant__ board_t d_kCaptureLookUpTable[move_gen::BoardConstants::kBoardSize * move_gen::BoardConstants::kBoardSize];

void InitializeCaptureLookupTable()
{
    if (is_initialized) {
        return;
    }
    // Flatten the 2D host array to a 1D array
    std::array<board_t, move_gen::BoardConstants::kBoardSize * move_gen::BoardConstants::kBoardSize> flatTable{};
    for (size_t i = 0; i < move_gen::BoardConstants::kBoardSize; ++i) {
        for (size_t j = 0; j < move_gen::BoardConstants::kBoardSize; ++j) {
            flatTable[i * move_gen::BoardConstants::kBoardSize + j] =
                checkers::cpu::apply_move::h_kCaptureLookUpTable[i][j];
        }
    }

    // Copy the flattened data to constant memory on the device
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(
        d_kCaptureLookUpTable, flatTable.data(),
        sizeof(board_t) * move_gen::BoardConstants::kBoardSize * move_gen::BoardConstants::kBoardSize
    ));
}
}  // namespace checkers::gpu::apply_move
