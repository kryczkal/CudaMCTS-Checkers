#ifndef MCTS_CHECKERS_INCLUDE_CUDA_MOVE_SELECTION_CUH_
#define MCTS_CHECKERS_INCLUDE_CUDA_MOVE_SELECTION_CUH_

#include "cuda/checkers_defines.hpp"
#include "cuda_runtime.h"
#include "types.hpp"

namespace checkers::gpu::move_selection
{
__global__ void SelectBestMoves(
    // Board States
    u32 *d_whites, u32 *d_blacks, u32 *d_kings,
    // Moves
    move_t *d_moves, u8 *d_move_counts, move_flags_t *d_move_capture_mask, move_flags_t *d_per_board_flags,
    // Number of boards to process
    u64 n_boards,
    // Output
    move_t *d_best_moves
);
}  // namespace checkers::gpu::move_selection

#endif  // MCTS_CHECKERS_INCLUDE_CUDA_MOVE_SELECTION_CUH_
