#ifndef MCTS_CHECKERS_INCLUDE_CPU_BOARD_HELPERS_HPP_
#define MCTS_CHECKERS_INCLUDE_CPU_BOARD_HELPERS_HPP_

#include "assert.h"
#include "checkers_defines.hpp"

namespace checkers::cpu::move_gen
{
enum class RowParity { kEven, kOdd };
enum class Direction { kUpLeft, kUpRight, kDownLeft, kDownRight };
enum class MovePart { From, To };

constexpr move_t EncodeMove(const board_index_t from, const board_index_t to) { return (from | ((move_t)to << 8)); }

/**
 * @brief Templated decode function. Allows specifying whether we want the `From` or `To` part of a move.
 */
template <MovePart part>
constexpr board_index_t DecodeMove(const move_t move)
{
    if constexpr (part == MovePart::From) {
        // Lower 8 bits
        constexpr move_t kFromMask = 0x00FFU;
        return static_cast<board_index_t>(move & kFromMask);
    } else {
        // Upper 8 bits
        constexpr move_t kToMask = 0xFF00U;
        return static_cast<board_index_t>((move & kToMask) >> 8);
    }
}

template <typename UnsignedFlagType>
constexpr u8 ReadFlag(const UnsignedFlagType flags, const u8 index)
{
    return ((flags >> index) & 1);
}

constexpr u8 IsOnEdge(const board_t edge_mask, const board_index_t index) { return ((edge_mask >> index) & 1); }

/**
 * @brief Returns 1 if `index` is on *any* edge (left/right/top/bottom).
 *        Uses the precomputed BoardConstants::kEdgeMask.
 */
constexpr u8 IsOnEdge(board_index_t index)
{
    // This specialization checks the entire edge mask
    return static_cast<u8>((gpu::move_gen::BoardConstants::kEdgeMask >> index) & 1U);
}

/**
 * @brief For a given Direction, returns 1 if `index` is on the relevant edges
 *        for that direction, otherwise 0.
 */
template <Direction direction>
constexpr u8 IsOnEdge(board_index_t index)
{
    if constexpr (direction == Direction::kUpLeft) {
        // top or left edges
        constexpr board_t kMask =
            gpu::move_gen::BoardConstants::kTopBoardEdgeMask | gpu::move_gen::BoardConstants::kLeftBoardEdgeMask;
        return static_cast<u8>((kMask >> index) & 1U);
    } else if constexpr (direction == Direction::kUpRight) {
        // top or right edges
        constexpr board_t kMask =
            gpu::move_gen::BoardConstants::kTopBoardEdgeMask | gpu::move_gen::BoardConstants::kRightBoardEdgeMask;
        return static_cast<u8>((kMask >> index) & 1U);
    } else if constexpr (direction == Direction::kDownLeft) {
        // bottom or left edges
        constexpr board_t kMask =
            gpu::move_gen::BoardConstants::kBottomBoardEdgeMask | gpu::move_gen::BoardConstants::kLeftBoardEdgeMask;
        return static_cast<u8>((kMask >> index) & 1U);
    } else {  // Direction::kDownRight
        // bottom or right edges
        constexpr board_t kMask =
            gpu::move_gen::BoardConstants::kBottomBoardEdgeMask | gpu::move_gen::BoardConstants::kRightBoardEdgeMask;
        return static_cast<u8>((kMask >> index) & 1U);
    }
}

constexpr u8 IsPieceAt(board_t board, board_index_t index) { return ((board >> index) & 1); }

constexpr i8 GetParityOffset(RowParity parity) { return parity == RowParity::kEven ? -1 : 0; }

constexpr RowParity GetRowParity(board_index_t index)
{
    //    assert(index < BoardConstants::kBoardSize);

    // TODO: Validate in assembly that this modulo is optmized to & with a bitmask
    return (index % (2 * gpu::move_gen::BoardConstants::kBoardEdgeLength)) >=
                   gpu::move_gen::BoardConstants::kBoardEdgeLength
               ? RowParity::kOdd
               : RowParity::kEven;
}

template <Direction direction>
constexpr board_index_t GetAdjacentIndex(board_index_t index)
{
    switch (direction) {
        case Direction::kUpLeft:
            return index - gpu::move_gen::BoardConstants::kBoardEdgeLength + GetParityOffset(GetRowParity(index));
        case Direction::kUpRight:
            return index - gpu::move_gen::BoardConstants::kBoardEdgeLength + GetParityOffset(GetRowParity(index)) + 1;
        case Direction::kDownLeft:
            return index + gpu::move_gen::BoardConstants::kBoardEdgeLength + GetParityOffset(GetRowParity(index));
        case Direction::kDownRight:
            return index + gpu::move_gen::BoardConstants::kBoardEdgeLength + GetParityOffset(GetRowParity(index)) + 1;
        default:
            assert(false);
            return (board_index_t)~0;
    }
}
}  // namespace checkers::cpu::move_gen

#endif  // MCTS_CHECKERS_INCLUDE_CPU_BOARD_HELPERS_HPP_
