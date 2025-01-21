#include "array"
#include "checkers_defines.hpp"
#include "cuda/apply_move.cuh"
#include "cuda/board_helpers.cuh"
#include "cuda_utils.cuh"

namespace checkers::gpu::apply_move
{
extern std::array<std::array<board_t, BoardConstants::kBoardSize>, BoardConstants::kBoardSize> h_kCaptureLookUpTable;
__constant__ board_t d_kCaptureLookUpTable[BoardConstants::kBoardSize * BoardConstants::kBoardSize];

void InitializeCaptureLookupTable()
{
    // Flatten the 2D host array to a 1D array
    std::array<board_t, BoardConstants::kBoardSize * BoardConstants::kBoardSize> flatTable{};
    for (size_t i = 0; i < BoardConstants::kBoardSize; ++i) {
        for (size_t j = 0; j < BoardConstants::kBoardSize; ++j) {
            flatTable[i * BoardConstants::kBoardSize + j] = h_kCaptureLookUpTable[i][j];
        }
    }

    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(
        d_kCaptureLookUpTable, flatTable.data(),
        sizeof(board_t) * BoardConstants::kBoardSize * BoardConstants::kBoardSize
    ));
}
}  // namespace checkers::gpu::apply_move
