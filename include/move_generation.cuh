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
    const u32 *d_whites, const u32 *d_blacks, const u32 *d_kings,
    // Moves
    move_t *d_moves, u8 *d_move_counts, move_flags_t *d_move_capture_mask
);

}  // namespace checkers::gpu::move_gen

#include "move_generation.tpp"

#endif  // MCTS_CHECKERS_INCLUDE_MOVE_GENERATION_CUH_
