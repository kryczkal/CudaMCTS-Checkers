#ifndef MCTS_CHECKERS_INCLUDE_CUDA_MOVE_SELECTION_CUH_
#define MCTS_CHECKERS_INCLUDE_CUDA_MOVE_SELECTION_CUH_

#include "checkers_defines.hpp"
#include "cuda_runtime.h"
#include "types.hpp"

namespace checkers::gpu::move_selection
{
/**
 * @brief Selects a "best" move (or randomly chosen move) for each board.
 *
 * The kernel expects exactly one thread per board. It uses the associated
 * seed in \p seeds[board_idx] to pick a random piece that has one or more moves,
 * and then to pick a random sub-move among that piece's valid moves.
 *
 * @param d_whites     The white piece bitmask per board (unused in random selection, but included for future
 * expansions).
 * @param d_blacks     The black piece bitmask per board (unused in random selection).
 * @param d_kings      The king bitmask per board (unused in random selection).
 * @param d_moves      Flattened array of all generated moves for all boards.
 *                     Size = n_boards * 32 * kNumMaxMovesPerPiece.
 * @param d_move_counts  For each board, for each of the 32 squares, how many moves are valid.
 *                       Size = n_boards * 32.
 * @param d_move_capture_mask  Per-square capture mask (unused here, but included for expansions).
 *                             Size = n_boards * 32.
 * @param d_per_board_flags    Additional flags per board (unused in random selection).
 * @param n_boards             Number of boards to process.
 * @param seeds                One random byte per board.
 * @param d_best_moves         Output: the chosen move per board.
 */
__global__ void SelectBestMoves(
    const u32* d_whites, const u32* d_blacks, const u32* d_kings, const move_t* d_moves, const u8* d_move_counts,
    const move_flags_t* d_move_capture_mask, const move_flags_t* d_per_board_flags, const u64 n_boards, const u8* seeds,
    move_t* d_best_moves
);
}  // namespace checkers::gpu::move_selection

#endif  // MCTS_CHECKERS_INCLUDE_CUDA_MOVE_SELECTION_CUH_
