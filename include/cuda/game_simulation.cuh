#ifndef MCTS_CHECKERS_INCLUDE_CUDA_GAME_SIMULATION_CUH_
#define MCTS_CHECKERS_INCLUDE_CUDA_GAME_SIMULATION_CUH_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "checkers_defines.hpp"

namespace checkers::gpu
{

/**
 * \brief Outcome encoding in scores[]:
 *  0 = in progress (not used at the end, but can be intermediate)
 *  1 = White wins
 *  2 = Black wins
 *  3 = Draw
 */
static constexpr u8 kOutcomeWhite = 1;
static constexpr u8 kOutcomeBlack = 2;
static constexpr u8 kOutcomeDraw  = 3;
static constexpr u8 kOutcomeNone  = 0;

// -----------------------------------------------------------------------------
// We define how many boards we process per block. Each board gets exactly 32 threads.
//
// Example: kNumBoardsPerBlock = 4 means each block has 4 * 32 = 128 threads.
// -----------------------------------------------------------------------------
static constexpr int kNumBoardsPerBlock = 10;

__global__ void SimulateCheckersGamesOneBoardPerBlock(
    const board_t* d_whites,         // [n_simulation_counts]
    const board_t* d_blacks,         // [n_simulation_counts]
    const board_t* turn_black,       // [n_simulation_counts]
    const u8* d_start_turns,         // [n_simulation_counts] (0=White, 1=Black)
    const u64* d_simulation_counts,  // [n_simulation_counts]
    const u64 n_simulation_counts,   // how many distinct board/turn combos
    u8* d_scores,                    // [n_total_simulations] final results
    u8* d_seeds,                     // [n_total_simulations] random seeds
    const int max_iterations,
    const u64 n_total_simulations  // sum of all d_simCounts[i]);
);
}  // namespace checkers::gpu

#endif  // MCTS_CHECKERS_INCLUDE_CUDA_GAME_SIMULATION_CUH_
