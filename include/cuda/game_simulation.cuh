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

/**
 * @brief Kernel that simulates multiple checkers games, with 1 board per block
 *        and 32 threads per block. Each thread in a block handles 1 square. We
 *        do up to max_iterations half-moves, then store the final outcome in d_scores.
 *
 * @param d_whites      White bitmasks, one per board.
 * @param d_blacks      Black bitmasks, one per board.
 * @param d_kings       King bitmasks, one per board.
 * @param d_scores      Output array: 1=White,2=Black,3=Draw,0=In progress.
 * @param d_seeds       One random seed per board.
 * @param max_iterations If we reach that many half-moves, we declare a draw.
 * @param n_boards      Number of boards to simulate.
 */
__global__ void SimulateCheckersGamesOneBoardPerBlock(
    const board_t* d_whites, const board_t* d_blacks, const board_t* d_kings, u8* d_scores, const u8* d_seeds,
    const int max_iterations, const u64 n_boards
);

}  // namespace checkers::gpu

#endif  // MCTS_CHECKERS_INCLUDE_CUDA_GAME_SIMULATION_CUH_
