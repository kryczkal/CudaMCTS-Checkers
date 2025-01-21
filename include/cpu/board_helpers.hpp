#ifndef MCTS_CHECKERS_INCLUDE_CPU_BOARD_HELPERS_HPP_
#define MCTS_CHECKERS_INCLUDE_CPU_BOARD_HELPERS_HPP_

#include "assert.h"
#include "checkers_defines.hpp"

namespace checkers::cpu::move_gen
{
enum class RowParity { kEven, kOdd };
enum class Direction { kUpLeft, kUpRight, kDownLeft, kDownRight };

constexpr move_t EncodeMove(const board_index_t from, const board_index_t to) { return (from | ((move_t)to << 8)); }
board_index_t DecodeMoveFrom(move_t move)
{
    static constexpr move_t kFromMask = (u8)~0;
    return move & kFromMask;
}

board_index_t DecodeMoveTo(move_t move)
{
    static constexpr move_t kToMask = (u8)~0 << 8;
    return (move & kToMask) >> 8;
}

template <typename UnsignedFlagType>
constexpr u8 ReadFlag(const UnsignedFlagType flags, const u8 index)
{
    return ((flags >> index) & 1);
}

constexpr u8 IsOnEdge(const board_t edge_mask, const board_index_t index) { return ((edge_mask >> index) & 1); }

constexpr u8 IsPieceAt(board_t board, board_index_t index) { return ((board >> index) & 1); }

constexpr i8 GetParityOffset(RowParity parity) { return parity == RowParity::kEven ? -1 : 0; }

constexpr RowParity GetRowParity(board_index_t index)
{
    //    assert(index < BoardConstants::kBoardSize);

    // TODO: Validate in assembly that this modulo is optmized to & with a bitmask
    return (index % (2 * BoardConstants::kBoardEdgeLength)) >= BoardConstants::kBoardEdgeLength ? RowParity::kOdd
                                                                                                : RowParity::kEven;
}

template <Direction direction>
constexpr board_index_t GetAdjacentIndex(board_index_t index)
{
    switch (direction) {
        case Direction::kUpLeft:
            return index - BoardConstants::kBoardEdgeLength + GetParityOffset(GetRowParity(index));
        case Direction::kUpRight:
            return index - BoardConstants::kBoardEdgeLength + GetParityOffset(GetRowParity(index)) + 1;
        case Direction::kDownLeft:
            return index + BoardConstants::kBoardEdgeLength + GetParityOffset(GetRowParity(index));
        case Direction::kDownRight:
            return index + BoardConstants::kBoardEdgeLength + GetParityOffset(GetRowParity(index)) + 1;
        default:
            assert(false);
            return (board_index_t)~0;
    }
}
}  // namespace checkers::cpu::move_gen

#endif  // MCTS_CHECKERS_INCLUDE_CPU_BOARD_HELPERS_HPP_
