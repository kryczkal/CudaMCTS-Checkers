#ifndef MCTS_CHECKERS_INCLUDE_CUDA_APPLY_MOVE_CUH_
#define MCTS_CHECKERS_INCLUDE_CUDA_APPLY_MOVE_CUH_

#include "checkers_defines.hpp"
#include "cuda_runtime.h"

namespace checkers::gpu::apply_move
{

__device__ void ApplyMoveOnBoardIdx(
    const u64 board_idx,
    // Board States
    board_t *d_whites, board_t *d_blacks, board_t *d_kings,
    // Moves
    const move_t *d_moves,
    // Number of boards to process
    const u64 n_boards
);

__global__ void ApplyMove(
    // Board States
    board_t *d_whites, board_t *d_blacks, board_t *d_kings,
    // Moves
    const move_t *d_moves,
    // Number of boards to process
    const u64 n_boards
);
}  // namespace checkers::gpu::apply_move

#endif  // MCTS_CHECKERS_INCLUDE_CUDA_APPLY_MOVE_CUH_
