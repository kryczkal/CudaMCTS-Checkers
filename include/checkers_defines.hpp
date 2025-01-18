#ifndef MCTS_CHECKERS_INCLUDE_CHECKERS_DEFINES_HPP_
#define MCTS_CHECKERS_INCLUDE_CHECKERS_DEFINES_HPP_

namespace checkers {
    enum class Turn {
        kWhite,
        kBlack
    };

    using board_t = u32;
    using move_t = u16;
    using board_index_t = u8;
}


#endif // MCTS_CHECKERS_INCLUDE_CHECKERS_DEFINES_HPP_