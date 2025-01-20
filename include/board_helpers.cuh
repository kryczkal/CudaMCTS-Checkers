#ifndef MCTS_CHECKERS_INCLUDE_BOARD_HELPERS_CUH_
#define MCTS_CHECKERS_INCLUDE_BOARD_HELPERS_CUH_

#include "assert.h"
#include "checkers_defines.hpp"

namespace checkers::gpu::move_gen
{
enum class RowParity { kEven, kOdd };
enum class Direction { kUpLeft, kUpRight, kDownLeft, kDownRight };

__device__ __forceinline__ constexpr move_t EncodeMove(const board_index_t from, const board_index_t to)
{
    return (from | ((move_t)to << 8));
}

template <typename UnsignedFlagType>
__device__ __forceinline__ constexpr u8 ReadFlag(const UnsignedFlagType flags, const u8 index)
{
    return ((flags >> index) & 1);
}

__device__ __forceinline__ constexpr u8 IsOnEdge(const board_t edge_mask, const board_index_t index)
{
    return ((edge_mask >> index) & 1);
}

__device__ __forceinline__ constexpr u8 IsPieceAt(board_t board, board_index_t index) { return ((board >> index) & 1); }

__device__ __forceinline__ constexpr i8 GetParityOffset(RowParity parity)
{
    return parity == RowParity::kEven ? -1 : 0;
}

__device__ __forceinline__ constexpr RowParity GetRowParity(board_index_t index)
{
    //    assert(index < BoardConstants::kBoardSize);

    // TODO: Validate in assembly that this modulo is optmized to & with a bitmask
    return (index % (2 * BoardConstants::kBoardEdgeLength)) >= BoardConstants::kBoardEdgeLength ? RowParity::kOdd
                                                                                                : RowParity::kEven;
}

template <Direction direction>
__device__ __forceinline__ constexpr board_index_t GetAdjacentIndex(board_index_t index)
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
}  // namespace checkers::gpu::move_gen

#endif  // <MCTS_CHECKERS_INCLUDE_BOARD_HELPERS_CUH_
