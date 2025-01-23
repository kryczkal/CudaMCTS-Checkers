#ifndef MCTS_CHECKERS_INCLUDE_CUDA_LAUNCHERS_CUH_
#define MCTS_CHECKERS_INCLUDE_CUDA_LAUNCHERS_CUH_

#include <algorithm>
#include <cstring>
#include <vector>

#include "common/checkers_defines.hpp"
#include "cpu/board.hpp"
#include "cuda/cuda_utils.cuh"
#include "cuda/move_generation.cuh"
#include "mcts/simulation_results.hpp"

namespace checkers::gpu::launchers
{

/**
 * @brief Holds a simple board definition for host usage. We store
 *        bitmasks for white/black pieces, plus king flags.
 */
using GpuBoard = checkers::cpu::Board;

/**
 * @brief This function allocates device memory for a vector of boards,
 *        copies data to device, launches the GPU kernel, and retrieves results.
 *
 * @tparam turn Whether to generate moves for White or Black.
 * @param boards Vector of host GpuBoard objects (white/black/kings bitmasks).
 * @param turn Whether to generate moves for White or Black.
 * @return Vector of MoveGenResult objects, one per board.
 */
std::vector<MoveGenResult> HostGenerateMoves(const std::vector<GpuBoard>& boards, Turn turn);

/**
 * @brief Host function to apply a single move per board on the GPU.
 *        The size of @p moves must match the size of @p boards (one move per board).
 *        Returns updated board states after each move is applied.
 */
std::vector<GpuBoard> HostApplyMoves(const std::vector<GpuBoard>& boards, const std::vector<move_t>& moves);

/**
 * @brief Host function to select a "best" move (or randomly chosen move) per board.
 *
 * @param boards           Board states for each position.
 * @param moves            Flattened moves for all boards; size = n_boards * 32 * kNumMaxMovesPerPiece.
 * @param move_counts      Number of sub-moves per square for each board; size = n_boards * 32.
 * @param capture_masks    For each square, bitmask indicating which sub-moves are captures; size = n_boards * 32.
 * @param per_board_flags  Additional flags (bitwise MoveFlagsConstants) per board; size = n_boards.
 * @param seeds            One random byte per board (used for random selection).
 *
 * @return A vector of size n_boards, each element is the chosen move_t for that board.
 */
std::vector<move_t> HostSelectBestMoves(
    const std::vector<GpuBoard>& boards, const std::vector<move_t>& moves, const std::vector<u8>& move_counts,
    const std::vector<move_flags_t>& capture_masks, const std::vector<move_flags_t>& per_board_flags,
    const std::vector<u8>& seeds
);

/**
 * @brief Updated function that:
 *   - Accepts a vector of SimulationParam.
 *   - Allocates GPU memory for the board definitions, calls the kernel that
 *     simulates all the games in one go, storing partial results in d_scores.
 *   - For each batch, calls an optimized GPU reduction kernel to sum up
 *     the outcomes in d_scores for that batch.
 *   - Returns a vector of SimulationResult, containing final .score
 *     (sum/2.0) and the number of simulations for that batch.
 *
 * @param params          Vector of SimulationParam structures.
 * @param max_iterations  If we reach that many half-moves, declare a draw.
 * @return A vector of size `params.size()`, each entry is SimulationResult.
 */
std::vector<SimulationResult> HostSimulateCheckersGames(const std::vector<SimulationParam>& params, int max_iterations);
}  // namespace checkers::gpu::launchers

#endif  // MCTS_CHECKERS_INCLUDE_CUDA_LAUNCHERS_CUH_
