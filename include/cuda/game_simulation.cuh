#ifndef MCTS_CHECKERS_INCLUDE_CUDA_GAME_SIMULATION_CUH_
#define MCTS_CHECKERS_INCLUDE_CUDA_GAME_SIMULATION_CUH_

#include "common/checkers_defines.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace checkers::gpu
{

// -----------------------------------------------------------------------------
// We define how many boards we process per block. Each board gets a fixed number of threads.
// kThreadsPerBoardInSimulation defines how many threads are assigned per board (between 1 and 32).
// For example: if kThreadsPerBoardInSimulation = 6, each board is processed by 6 threads,
// and these threads will loop over all 32 board squares.
// -----------------------------------------------------------------------------
static constexpr int kNumBoardsPerBlock           = 1;   // 1 is empirically the fastest
static constexpr int kThreadsPerBoardInSimulation = 32;  // Change this value between 1 and 32 as needed

static_assert(
    kThreadsPerBoardInSimulation > 0 && kThreadsPerBoardInSimulation <= 32,
    "kThreadsPerBoardInSimulation must be between 1 and 32"
);
static_assert(
    kNumBoardsPerBlock == 1 || kThreadsPerBoardInSimulation == 32,
    "kNumBoardsPerBlock must be 1 if kThreadsPerBoardInSimulation is not 32"
);

__global__ void SimulateCheckersGames(
    const board_t* d_whites,         // [n_simulation_counts]
    const board_t* d_blacks,         // [n_simulation_counts]
    const board_t* d_kings,          // [n_simulation_counts]
    const u8* d_start_turns,         // [n_simulation_counts] (0=White, 1=Black)
    const u64* d_simulation_counts,  // [n_simulation_counts]
    const u64 n_simulation_counts,   // how many distinct board/turn combos
    u8* d_scores,                    // [n_total_simulations] final results
    u8* d_seeds,                     // [n_total_simulations] random seeds
    const int max_iterations,
    const u64 n_total_simulations  // sum of all d_simulation_counts[i]
);
}  // namespace checkers::gpu

#endif  // MCTS_CHECKERS_INCLUDE_CUDA_GAME_SIMULATION_CUH_
