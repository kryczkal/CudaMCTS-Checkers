#ifndef MCTS_CHECKERS_INCLUDE_MOVE_GENERATION_CUH_
#define MCTS_CHECKERS_INCLUDE_MOVE_GENERATION_CUH_

#include "types.hpp"
#include "cuda_runtime.h"

#include "checkers_defines.hpp"

namespace checkers::mcts::gpu {

    template<Turn turn>
    __global__ void GenerateMoves(
            // Board States
            const u32 *d_whites,
            const u32 *d_blacks,
            const u32 *d_kings,
            // Moves
            u16 *d_moves,
            u8* d_move_counts,
            u8* d_move_flags
            );


} // checkers::mcts::gpu

#include "move_generation.tpp"

#endif // MCTS_CHECKERS_INCLUDE_MOVE_GENERATION_CUH_