#ifndef MCTS_CHECKERS_INCLUDE_CUDA_MOVE_SELECTION_CUH_
#define MCTS_CHECKERS_INCLUDE_CUDA_MOVE_SELECTION_CUH_

#include "common/checkers_defines.hpp"
#include "cuda_runtime.h"
#include "types.hpp"

namespace checkers::gpu::move_selection
{
/**
 * @brief Selects a single move (randomly) for a single board, given that
 *        all indexing is already offset to this board. If a capture is found
 *        in the per-board flags, randomly pick among capturing moves only.
 *
 * @param white_bits  (unused by the random policy, but included for future expansions).
 * @param black_bits  (unused by the random policy, but included for future expansions).
 * @param king_bits   (unused by the random policy, but included for future expansions).
 * @param moves       Pointer to a flattened array of size: (32 * kNumMaxMovesPerPiece).
 * @param move_counts Pointer to an array of 32 counters, each telling how many moves a square has.
 * @param capture_masks Pointer to an array of 32 bitmasks. If bit i is set in capture_masks[sq],
 *                     that sub-move is a capture.
 * @param per_board_flags The combined flags for the entire board, e.g. MoveFlagsConstants::kCaptureFound, etc.
 * @param seed        Random seed reference for generating the random pick.
 *
 * @return Chosen move. If no moves are found, returns kInvalidMove.
 */
__device__ move_t SelectRandomMoveForSingleBoard(
    const board_t white_bits, const board_t black_bits, const board_t king_bits, const move_t* moves,
    const u8* move_counts, const move_flags_t* capture_masks, const move_flags_t per_board_flags, u8& seed
);

/**
 * @brief Selects the best move for a single board, given that all indexing is already offset to this board.
 *
 * @param white_bits  (unused by the best move policy, but included for future expansions).
 * @param black_bits  (unused by the best move policy, but included for future expansions).
 * @param king_bits   (unused by the best move policy, but included for future expansions).
 * @param moves       Pointer to a flattened array of size: (32 * kNumMaxMovesPerPiece).
 * @param move_counts Pointer to an array of 32 counters, each telling how many moves a square has.
 * @param capture_masks Pointer to an array of 32 bitmasks. If bit i is set in capture_masks[sq],
 *                     that sub-move is a capture.
 * @param per_board_flags The combined flags for the entire board, e.g. MoveFlagsConstants::kCaptureFound, etc.
 * @param seed        Random seed reference for generating the random pick.
 *
 * @return Chosen move. If no moves are found, returns kInvalidMove.
 */
__device__ move_t SelectBestMoveForSingleBoard(
    const board_t white_bits, const board_t black_bits, const board_t king_bits, const move_t* moves,
    const u8* move_counts, const move_flags_t* capture_masks, const move_flags_t per_board_flags, u8& seed
);

/**
 * @brief Kernel that picks one move for each board. The caller ensures that
 *        d_moves/d_move_counts/d_move_capture_mask/d_per_board_flags
 *        have valid data. The chosen moves are returned in d_best_moves.
 *
 * @param d_whites White bitmasks per board.
 * @param d_blacks Black bitmasks per board.
 * @param d_kings King bitmasks per board.
 * @param d_moves Flattened array of size n_boards * (32*kNumMaxMovesPerPiece).
 * @param d_move_counts Flattened array of size n_boards * 32.
 * @param d_move_capture_mask Flattened array of size n_boards * 32.
 * @param d_per_board_flags Size = n_boards.
 * @param n_boards Number of boards.
 * @param d_seeds One random byte per board.
 * @param d_best_moves One move_t per board for the final chosen move.
 */
__global__ void SelectBestMoves(
    const board_t* d_whites, const board_t* d_blacks, const board_t* d_kings, const move_t* d_moves,
    const u8* d_move_counts, const move_flags_t* d_move_capture_mask, const move_flags_t* d_per_board_flags,
    const u64 n_boards, u8* d_seeds, move_t* d_best_moves
);

}  // namespace checkers::gpu::move_selection

#endif  // MCTS_CHECKERS_INCLUDE_CUDA_MOVE_SELECTION_CUH_
