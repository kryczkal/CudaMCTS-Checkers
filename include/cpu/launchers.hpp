#ifndef MCTS_CHECKERS_INCLUDE_CPU_LAUNCHERS_HPP_
#define MCTS_CHECKERS_INCLUDE_CPU_LAUNCHERS_HPP_

#include <vector>
#include "board.hpp"
#include "common/checkers_defines.hpp"
#include "mcts/simulation_results.hpp"

namespace checkers::cpu::launchers
{

std::vector<MoveGenResult> HostGenerateMoves(const std::vector<Board> &boards, Turn turn);

/**
 * @brief CPU version of "HostApplyMoves": applies exactly one move per board.
 */
std::vector<Board> HostApplyMoves(const std::vector<Board> &boards, const std::vector<move_t> &moves);

/**
 * @brief CPU version of "HostSelectBestMoves".
 */
std::vector<move_t> HostSelectBestMoves(
    const std::vector<Board> &boards, const std::vector<move_t> &moves, const std::vector<u8> &move_counts,
    const std::vector<move_flags_t> &capture_masks, const std::vector<move_flags_t> &per_board_flags,
    const std::vector<u8> &seeds
);

/**
 * @brief CPU version of "HostSimulateCheckersGames", doing random playouts for each board.
 */
std::vector<SimulationResult> HostSimulateCheckersGames(const std::vector<SimulationParam> &params, int max_iterations);

}  // namespace checkers::cpu::launchers

#endif  // MCTS_CHECKERS_INCLUDE_CPU_LAUNCHERS_HPP_
