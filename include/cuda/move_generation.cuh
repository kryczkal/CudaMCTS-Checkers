#ifndef MCTS_CHECKERS_INCLUDE_MOVE_GENERATION_CUH_
#define MCTS_CHECKERS_INCLUDE_MOVE_GENERATION_CUH_

#include "cuda_runtime.h"
#include "types.hpp"

#include "common/checkers_defines.hpp"

namespace checkers::gpu::move_gen
{
/**
 * @brief Generates moves for a single piece. This function does NOT compute its
 *        global index but instead operates on already-offset pointers for the
 *        piece’s data within arrays.
 *
 * @tparam turn Current side to move (White or Black).
 *
 * @param figure_idx Index of the piece (0..31) in the 32-square board.
 * @param white_pieces Bitmask of white pieces for the board.
 * @param black_pieces Bitmask of black pieces for the board.
 * @param kings Bitmask of kings for the board.
 * @param out_moves Pointer to an array that can hold up to kNumMaxMovesPerPiece moves for this piece.
 * @param out_move_count Reference to a single counter that will be incremented to the number of valid sub-moves found.
 * @param out_capture_mask Reference to a bitmask indicating which of the sub-moves in out_moves are captures.
 * @param per_board_flags Pointer (or reference) to the flags for the entire board. Bits MoveFound/CaptureFound may be
 * set.
 */
template <Turn turn>
__device__ __forceinline__ void GenerateMovesForSinglePiece(
    const board_index_t figure_idx, const board_t white_pieces, const board_t black_pieces, const board_t kings,
    move_t* out_moves,  // size >= kNumMaxMovesPerPiece
    u8& out_move_count, move_flags_t& out_capture_mask, move_flags_t& per_board_flags
);

/**
 * @brief Stand-alone kernel that generates moves for multiple boards, with 32 threads per board.
 *
 * Each thread handles exactly one figure index on a board. In total, we produce
 * up to kNumMaxMovesPerPiece sub-moves per piece.
 *
 * @tparam turn Which side to move (White or Black).
 *
 * @param d_whites Array of white bitmasks, one per board.
 * @param d_blacks Array of black bitmasks, one per board.
 * @param d_kings Array of king bitmasks, one per board.
 * @param d_moves Flattened array of size (n_boards * 32 * kNumMaxMovesPerPiece).
 * @param d_move_counts Flattened array of size (n_boards * 32).
 * @param d_move_capture_mask Flattened array of size (n_boards * 32). Each element is a bitmask for captures among that
 * piece's moves.
 * @param d_per_board_move_flags Array of size n_boards with per-board flags (MoveFound/CaptureFound).
 * @param n_boards How many boards we are generating moves for.
 */
template <Turn turn>
__global__ void GenerateMoves(
    const board_t* d_whites, const board_t* d_blacks, const board_t* d_kings, move_t* d_moves, u8* d_move_counts,
    move_flags_t* d_move_capture_mask, move_flags_t* d_per_board_move_flags, const u64 n_boards
);

}  // namespace checkers::gpu::move_gen

#include "move_generation.tpp"

#endif  // MCTS_CHECKERS_INCLUDE_MOVE_GENERATION_CUH_
