#ifndef MCTS_CHECKERS_INCLUDE_CPU_APPLY_MOVE_HPP_
#define MCTS_CHECKERS_INCLUDE_CPU_APPLY_MOVE_HPP_

#include "common/checkers_defines.hpp"

namespace checkers::cpu::apply_move
{
void ApplyMoveOnSingleBoard(move_t move, board_t& white_bits, board_t& black_bits, board_t& king_bits);
}

#endif
