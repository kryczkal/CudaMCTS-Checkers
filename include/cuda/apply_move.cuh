#ifndef MCTS_CHECKERS_INCLUDE_CUDA_APPLY_MOVE_CUH_
#define MCTS_CHECKERS_INCLUDE_CUDA_APPLY_MOVE_CUH_

#include "checkers_defines.hpp"
#include "cuda_runtime.h"

namespace checkers::gpu::apply_move
{
template <Turn turn>
__global__ void ApplyMove(
    // Board States
    board_t *d_whites, board_t *d_blacks, board_t *d_kings,
    // Moves
    move_t *d_moves,
    // Number of boards to process
    u64 n_boards
);
}

#endif  // MCTS_CHECKERS_INCLUDE_CUDA_APPLY_MOVE_CUH_
