#ifndef MCTS_CHECKERS_INCLUDE_MOVE_GENERATION_CUH_
#define MCTS_CHECKERS_INCLUDE_MOVE_GENERATION_CUH_

#include "cuda_runtime.h"
#include "types.hpp"

#include "checkers_defines.hpp"

namespace checkers::gpu::move_gen
{
template <Turn turn>
__global__ void GenerateMoves(
    // Board States
    const board_t *d_whites, const board_t *d_blacks, const board_t *d_kings,
    // Moves
    move_t *d_moves, u8 *d_move_counts, move_flags_t *d_move_capture_mask, move_flags_t *d_per_board_move_flags,
    // Number of boards to process
    const u64 n_boards
);

template <Turn turn>
__device__ __forceinline__ void GenerateMovesForBoardIdxFigureIdx(
    const u64 board_idx, board_index_t figure_idx,
    // Board States
    const board_t *d_whites, const board_t *d_blacks, const board_t *d_kings,
    // Moves
    move_t *d_moves, u8 *d_move_counts, move_flags_t *d_move_capture_mask, move_flags_t *d_per_board_move_flags,
    // Number of boards to process
    const u64 n_boards
);

}  // namespace checkers::gpu::move_gen

#include "move_generation.tpp"

#endif  // MCTS_CHECKERS_INCLUDE_MOVE_GENERATION_CUH_
