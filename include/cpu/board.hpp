#ifndef MCTS_CHECKERS_INCLUDE_CPU_CPU_BOARD_HPP_
#define MCTS_CHECKERS_INCLUDE_CPU_CPU_BOARD_HPP_

#include "common/checkers_defines.hpp"

namespace checkers::cpu
{
struct Board {
    public:
    board_t white_pieces_;
    board_t black_pieces_;
    board_t kings_;
};

}  // namespace checkers::cpu

#endif  // MCTS_CHECKERS_INCLUDE_CPU_CPU_BOARD_HPP_
