#ifndef CUDA_MCTS_CHECKRS_INCLUDE_BOARD_TPP_
#define CUDA_MCTS_CHECKRS_INCLUDE_BOARD_TPP_

#include <bitset>
#include <cassert>
#include <cpp_defines.hpp>
#include <move.hpp>
#include <types.hpp>
#include "board.hpp"

namespace CudaMctsCheckers
{
constexpr i8 Board::ParityOffset(RowParity parity) { return parity == RowParity::kEven ? -1 : 0; }

template <BoardCheckType type>
constexpr BoardCheckType Board::GetOppositeType()
{
    switch (type) {
        case BoardCheckType::kWhite:
            return BoardCheckType::kBlack;
        case BoardCheckType::kBlack:
            return BoardCheckType::kWhite;
        default:
            assert(false);
            return BoardCheckType::kAll;
    }
}

template <BoardCheckType type>
FORCE_INLINE bool Board::IsPieceAt(IndexType index) const
{
    assert(index <= kHalfBoardSize);

    HalfBoard pieces;
    switch (type) {
        case BoardCheckType::kWhite:
            pieces = white_pieces;
            break;
        case BoardCheckType::kBlack:
            pieces = black_pieces;
            break;
        case BoardCheckType::kKings:
            pieces = kings;
            break;
        case BoardCheckType::kAll:
            pieces = white_pieces | black_pieces;
            break;
        default:
            assert(false);
            return false;
    }
    HalfBoard mask   = 1 << (index & ~kInvalidIndex);
    HalfBoard result = (pieces & mask);
    return result != 0;
}

template <BoardCheckType type>
FORCE_INLINE void Board::SetPieceAt(IndexType index)
{
    assert(index != kInvalidIndex);
    assert(index < kHalfBoardSize);
    switch (type) {
        case BoardCheckType::kWhite:
            white_pieces |= (1 << index);
            break;
        case BoardCheckType::kBlack:
            black_pieces |= (1 << index);
            break;
        case BoardCheckType::kKings:
            kings |= (1 << index);
            break;
        case BoardCheckType::kAll:
            white_pieces |= (1 << index);
            black_pieces |= (1 << index);
            break;
        default:
            assert(false);
            break;
    }
}

template <BoardCheckType type>
FORCE_INLINE void Board::UnsetPieceAt(IndexType index)
{
    assert(index != kInvalidIndex);
    assert(index < kHalfBoardSize);
    switch (type) {
        case BoardCheckType::kWhite:
            white_pieces &= ~(1 << index);
            break;
        case BoardCheckType::kBlack:
            black_pieces &= ~(1 << index);
            break;
        case BoardCheckType::kKings:
            kings &= ~(1 << index);
            break;
        case BoardCheckType::kAll:
            white_pieces &= ~(1 << index);
            black_pieces &= ~(1 << index);
            break;
        default:
            assert(false);
            break;
    }
}

template <BoardCheckType type>
FORCE_INLINE void Board::MovePiece(IndexType from, IndexType to)
{
    assert(IsPieceAt<type>(from));
    assert(!IsPieceAt<type>(to));

    UnsetPieceAt<type>(from);
    SetPieceAt<type>(to);
}

template <BoardCheckType type>
FORCE_INLINE bool Board::PieceReachedEnd(IndexType index) const
{
    assert(type != BoardCheckType::kAll);
    assert(type != BoardCheckType::kKings);
    assert(index != kInvalidIndex);
    assert(index < kHalfBoardSize);
    switch (type) {
        case BoardCheckType::kWhite:
            return index < kHalfBoardEdgeLength;
        case BoardCheckType::kBlack:
            return index >= kHalfBoardSize - kHalfBoardEdgeLength;
        default:
            assert(false);
            return false;
    }
}

FORCE_INLINE bool Board::IsAtLeftEdge(IndexType index)
{
    assert(index != kInvalidIndex);
    assert(index < kHalfBoardSize);
    return index % kEdgeLength == 0;
}

FORCE_INLINE bool Board::IsAtRightEdge(IndexType index)
{
    assert(index != kInvalidIndex);
    assert(index < kHalfBoardSize);
    return index % kEdgeLength == kEdgeLength - 1;
}

FORCE_INLINE Board::IndexType Board::InvalidateOutBoundsIndex(IndexType index)
{
    return index >= kHalfBoardSize ? kInvalidIndex
                                   : index;  // Going sub zero will wrap around so this is correct
}

FORCE_INLINE RowParity Board::GetRowParity(IndexType index)
{
    assert(index != kInvalidIndex);
    assert(index < kHalfBoardSize);
    return (index % kEdgeLength) >= kHalfBoardEdgeLength ? RowParity::kOdd : RowParity::kEven;
}

template <MoveDirection direction>
FORCE_INLINE Board::IndexType Board::GetRelativeMoveIndex(IndexType index) const
{
    assert(index != kInvalidIndex);
    assert(index < kHalfBoardSize);

    if (direction == MoveDirection::kDownLeft || direction == MoveDirection::kUpLeft) {
        if (IsAtLeftEdge(index)) {
            return kInvalidIndex;
        }
    }

    if (direction == MoveDirection::kDownRight || direction == MoveDirection::kUpRight) {
        if (IsAtRightEdge(index)) {
            return kInvalidIndex;
        }
    }

    switch (direction) {
        case MoveDirection::kUpLeft:
            return InvalidateOutBoundsIndex(
                index - kHalfBoardEdgeLength + ParityOffset(GetRowParity(index))
            );
        case MoveDirection::kUpRight:
            return InvalidateOutBoundsIndex(
                index - kHalfBoardEdgeLength + ParityOffset(GetRowParity(index)) + 1
            );
        case MoveDirection::kDownLeft:
            return InvalidateOutBoundsIndex(
                index + kHalfBoardEdgeLength + ParityOffset(GetRowParity(index))
            );
        case MoveDirection::kDownRight:
            return InvalidateOutBoundsIndex(
                index + kHalfBoardEdgeLength + ParityOffset(GetRowParity(index)) + 1
            );
        default:
            assert(false);
            return kInvalidIndex;
    }
}

/**
 * Get the index of the field the piece would move to if it moved left.
 * This does not check if the move is valid.
 */
template <BoardCheckType type>
FORCE_INLINE Board::IndexType Board::GetPieceLeftMoveIndex(IndexType index) const
{
    assert(type != BoardCheckType::kAll);
    assert(type != BoardCheckType::kKings);
    assert(index != kInvalidIndex);
    assert(index < kHalfBoardSize);

    switch (type) {
        case BoardCheckType::kWhite:
            return GetRelativeMoveIndex<MoveDirection::kUpLeft>(index);
        case BoardCheckType::kBlack:
            return GetRelativeMoveIndex<MoveDirection::kDownLeft>(index);
        default:
            assert(false);
            return kInvalidIndex;
    }
}

/**
 * Get the index of the field the piece would move to if it moved left.
 * This does not check if the move is valid.
 */
template <BoardCheckType type>
FORCE_INLINE Board::IndexType Board::GetPieceRightMoveIndex(IndexType index) const
{
    assert(type != BoardCheckType::kAll);
    assert(type != BoardCheckType::kKings);
    assert(index != kInvalidIndex);
    assert(index < kHalfBoardSize);

    switch (type) {
        case BoardCheckType::kWhite:
            return GetRelativeMoveIndex<MoveDirection::kUpRight>(index);
        case BoardCheckType::kBlack:
            return GetRelativeMoveIndex<MoveDirection::kDownRight>(index);
        default:
            assert(false);
            return kInvalidIndex;
    }
}

}  // namespace CudaMctsCheckers

#endif  // CUDA_MCTS_CHECKRS_INCLUDE_BOARD_TPP_
