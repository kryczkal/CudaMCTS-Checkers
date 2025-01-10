#ifndef CUDA_MCTS_CHECKRS_INCLUDE_BOARD_HPP_
#define CUDA_MCTS_CHECKRS_INCLUDE_BOARD_HPP_

#include <bitset>
#include <cassert>
#include <cpp_defines.hpp>
#include <types.hpp>

enum class BoardCheckType { kWhite, kBlack, kKings, kAll };

namespace CudaMctsCheckers
{

struct PACK Board {
    using HalfBoard = u32;
    using IndexType = u8;
    //------------------------------------------------------------------------------//
    //                                Static Fields                                 //
    //------------------------------------------------------------------------------//
    static constexpr u8 kEdgeLength    = 8;  // Board size in the x direction
    static constexpr u8 kSizeTotal     = kEdgeLength * kEdgeLength;  // Total board size
    static constexpr u8 kHalfBoardSize = kSizeTotal / 2;             // Board size used by pieces
    static constexpr u8 kHalfBoardEdgeLength =
        kEdgeLength / 2;                                    // Half board size in the x direction
    static constexpr u8 kDiagonalSize        = 8;           // Size of the diagonal
    static constexpr u8 kNumPiecesPerPlayer  = 12;          // Number of pieces per player
    static constexpr IndexType kInvalidIndex = kSizeTotal;  // Invalid index

    //------------------------------------------------------------------------------//
    //                                    Fields                                    //
    //------------------------------------------------------------------------------//
    HalfBoard white_pieces;  // Bitset of white pieces
    HalfBoard black_pieces;  // Bitset of black pieces
    HalfBoard kings;         // Bitset of kings

    template <BoardCheckType type>
    static constexpr BoardCheckType GetOppositeType();

    template <BoardCheckType type>
    FORCE_INLINE bool IsPieceAt(IndexType index) const;

    template <BoardCheckType type>
    FORCE_INLINE void SetPieceAt(IndexType index);

    template <BoardCheckType type>
    FORCE_INLINE void UnsetPieceAt(IndexType index);

    template <BoardCheckType type>
    FORCE_INLINE void MovePiece(IndexType from, IndexType to);

    template <BoardCheckType type>
    FORCE_INLINE bool PieceReachedEnd(IndexType index) const;

    // TODO: This didn't take into account i operate on a half board and not a full board
    // This means going every other row is misaligned by 1
    // Example:
    // x x x x x
    //  x x x x
    // x x x x x
    // So all my index calculations are off

    template <BoardCheckType type>
    FORCE_INLINE IndexType GetRelativeTopLeftIndex(IndexType index) const;

    template <BoardCheckType type>
    FORCE_INLINE IndexType GetRelativeTopRightIndex(IndexType index) const;

    template <BoardCheckType type>
    FORCE_INLINE IndexType GetRelativeBottomLeftIndex(IndexType index) const;

    template <BoardCheckType type>
    FORCE_INLINE IndexType GetRelativeBottomRightIndex(IndexType index) const;

    template <BoardCheckType type>
    FORCE_INLINE IndexType GetPieceLeftMoveIndex(IndexType index) const;

    template <BoardCheckType type>
    FORCE_INLINE IndexType GetPieceRightMoveIndex(IndexType index) const;
};

}  // namespace CudaMctsCheckers

#include <board.tpp>

#endif  // CUDA_MCTS_CHECKRS_INCLUDE_BOARD_HPP_