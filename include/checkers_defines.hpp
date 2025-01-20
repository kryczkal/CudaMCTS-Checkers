#ifndef MCTS_CHECKERS_INCLUDE_CHECKERS_DEFINES_HPP_
#define MCTS_CHECKERS_INCLUDE_CHECKERS_DEFINES_HPP_

#include "types.hpp"

namespace checkers
{
/////////////////////////////////// Types ////////////////////////////////////

enum class Turn { kWhite, kBlack };

using board_t       = u32;
using move_t        = u16;
using board_index_t = u8;
using move_flags_t  = u16;

///////////////////////////////// Constants //////////////////////////////////

namespace gpu::move_gen
{
static constexpr u8 kNumMaxMovesPerPiece = 13;
}  // namespace gpu::move_gen

class MoveConstants
{
    public:
    static constexpr move_t kInvalidMove = 0xFFFF;
};

class MoveFlagsConstants
{
    public:
    static constexpr u8 kMoveFound    = 0;
    static constexpr u8 kCaptureFound = 1;
};

class BoardConstants
{
    public:
    static constexpr u8 kBoardEdgeLength = 4;
    static constexpr u8 kBoardSize       = 32;

    static constexpr board_t kLeftBoardEdgeMask = []() constexpr {
        board_t mask = 0;
        for (u8 i = 0; i < kBoardEdgeLength; ++i) {
            mask |= 1 << (i * kBoardEdgeLength * 2);
        }
        return mask;
    }();

    static constexpr board_t kRightBoardEdgeMask = []() constexpr {
        board_t mask = 0;
        for (u8 i = 0; i < kBoardEdgeLength; ++i) {
            mask |= 1 << (i * kBoardEdgeLength * 2 + kBoardEdgeLength - 1);
        }
        return mask;
    }();

    static constexpr board_t kTopBoardEdgeMask = []() constexpr {
        board_t mask = 0;
        for (u8 i = 0; i < kBoardEdgeLength; ++i) {
            mask |= 1 << i;
        }
        return mask;
    }();

    static constexpr board_t kBottomBoardEdgeMask = []() constexpr {
        board_t mask = 0;
        for (u8 i = 0; i < kBoardEdgeLength; ++i) {
            mask |= 1 << (kBoardSize - kBoardEdgeLength + i);
        }
        return mask;
    }();

    static constexpr board_t kEdgeMask =
        kLeftBoardEdgeMask | kRightBoardEdgeMask | kTopBoardEdgeMask | kBottomBoardEdgeMask;
};

}  // namespace checkers

#endif  // MCTS_CHECKERS_INCLUDE_CHECKERS_DEFINES_HPP_
