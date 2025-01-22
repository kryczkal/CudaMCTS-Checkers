#ifndef MCTS_CHECKERS_INCLUDE_CUDA_APPLY_MOVE_CUH_
#define MCTS_CHECKERS_INCLUDE_CUDA_APPLY_MOVE_CUH_

#include "checkers_defines.hpp"
#include "cuda_runtime.h"

namespace checkers::gpu::apply_move
{
/**
 * @brief Applies a single move on a board. This function operates on pre-offset
 *        pointers or direct references to the board’s bitmasks, rather than computing
 *        global indices.
 *
 * @param move The move to apply (encoded as from/to).
 * @param white_bits Reference to the bitmask for white pieces.
 * @param black_bits Reference to the bitmask for black pieces.
 * @param king_bits Reference to the bitmask for king pieces.
 */
__device__ void ApplyMoveOnSingleBoard(move_t move, board_t& white_bits, board_t& black_bits, board_t& king_bits);

/**
 * @brief Kernel that applies one move per board, for n_boards, using global indexing.
 *
 * @param d_whites Device pointer to array of white bitmasks, size n_boards.
 * @param d_blacks Device pointer to array of black bitmasks, size n_boards.
 * @param d_kings Device pointer to array of king bitmasks, size n_boards.
 * @param d_moves Device pointer to array of moves, size n_boards (1 move per board).
 * @param n_boards Number of boards.
 */
__global__ void ApplyMove(
    board_t* d_whites, board_t* d_blacks, board_t* d_kings, const move_t* d_moves, const u64 n_boards
);

}  // namespace checkers::gpu::apply_move

#endif  // MCTS_CHECKERS_INCLUDE_CUDA_APPLY_MOVE_CUH_
