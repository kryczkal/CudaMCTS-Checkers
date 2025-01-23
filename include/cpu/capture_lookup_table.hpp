#ifndef MCTS_CHECKERS_INCLUDE_CPU_CAPTURE_LOOKUP_TABLE_HPP_
#define MCTS_CHECKERS_INCLUDE_CPU_CAPTURE_LOOKUP_TABLE_HPP_

#include "array"
#include "board_helpers.hpp"
#include "common/checkers_defines.hpp"

namespace checkers::cpu::apply_move
{

extern std::array<std::array<board_t, BoardConstants::kBoardSize>, BoardConstants::kBoardSize> h_kCaptureLookUpTable;

}  // namespace checkers::cpu::apply_move

#endif  // MCTS_CHECKERS_INCLUDE_CPU_CAPTURE_LOOKUP_TABLE_HPP_
