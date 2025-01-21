#ifndef MCTS_CHECKERS_INCLUDE_CUDA_APPLY_MOVE_CUH_
#define MCTS_CHECKERS_INCLUDE_CUDA_APPLY_MOVE_CUH_

#include "checkers_defines.hpp"
#include "cuda_runtime.h"

namespace checkers::gpu::apply_move
{

/**
 * \brief Applies a move on a specific board index.
 *
 * \param board_idx The index of the board to apply the move on.
 * \param d_whites Pointer to the array of white pieces on the boards.
 * \param d_blacks Pointer to the array of black pieces on the boards.
 * \param d_kings Pointer to the array of king pieces on the boards.
 * \param d_moves Pointer to the array of moves to be applied.
 * \param n_boards The number of boards to process.
 */
__device__ void ApplyMoveOnBoardIdx(
    const board_index_t board_idx,
    // Board States
    board_t *d_whites, board_t *d_blacks, board_t *d_kings,
    // Moves
    const move_t *d_moves,
    // Number of boards to process
    const u64 n_boards
);

/**
 * \brief CUDA kernel to apply moves on multiple boards.
 *
 * \param d_whites Pointer to the array of white pieces on the boards.
 * \param d_blacks Pointer to the array of black pieces on the boards.
 * \param d_kings Pointer to the array of king pieces on the boards.
 * \param d_moves Pointer to the array of moves to be applied.
 * \param n_boards The number of boards to process.
 */
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
