#ifndef CUDA_MCTS_CHECKRS_INCLUDE_BOARD_HPP_
#define CUDA_MCTS_CHECKRS_INCLUDE_BOARD_HPP_

#include <bitset>
#include <cassert>
#include <cpp_defines.hpp>
#include <move_direction.hpp>
#include <types.hpp>

class MoveGenerationOutput;
namespace CudaMctsCheckers
{
enum class BoardCheckType { kWhite, kBlack, kKings, kAll };
enum class RowParity { kEven, kOdd };

class PACK Board
{
    public:
    using HalfBoard = u32;
    using IndexType = u8;
    //------------------------------------------------------------------------------//
    //                                Static Fields                                 //
    //------------------------------------------------------------------------------//
    static constexpr u8 kEdgeLength    = 8;  // Board size in the x direction
    static constexpr u8 kSizeTotal     = kEdgeLength * kEdgeLength;  // Total board size
    static constexpr u8 kHalfBoardSize = kSizeTotal / 2;             // Board size used by pieces
    static constexpr u8 kHalfBoardEdgeLength =
        kEdgeLength / 2;                            // Half board size in the x direction
    static constexpr u8 kDiagonalSize        = 8;   // Size of the diagonal
    static constexpr u8 kNumPiecesPerPlayer  = 12;  // Number of pieces per player
    static constexpr IndexType kInvalidIndex = kHalfBoardSize;  // Invalid index

    //------------------------------------------------------------------------------//
    //                                    Fields                                    //
    //------------------------------------------------------------------------------//
    HalfBoard white_pieces;  // Bitset of white pieces (starting from bottom)
    HalfBoard black_pieces;  // Bitset of black pieces (starting from top)
    HalfBoard kings;         // Bitset of kings
    u8 time_from_non_reversible_move;

    //------------------------------------------------------------------------------//
    //                                Constructors                                  //
    //------------------------------------------------------------------------------//
    Board();

    //------------------------------------------------------------------------------//
    //                                Public Methods                                //
    //------------------------------------------------------------------------------//

    static constexpr i8 ParityOffset(RowParity parity);

    template <BoardCheckType type>
    static constexpr BoardCheckType GetOppositeType();

    template <BoardCheckType type>
    [[nodiscard]] FORCE_INLINE bool IsPieceAt(IndexType index) const;

    template <BoardCheckType type>
    FORCE_INLINE void SetPieceAt(IndexType index);

    template <BoardCheckType type>
    FORCE_INLINE void UnsetPieceAt(IndexType index);

    template <BoardCheckType type>
    FORCE_INLINE void MovePiece(IndexType from, IndexType to);

    template <BoardCheckType type>
    static constexpr FORCE_INLINE bool PieceReachedEnd(IndexType index);

    static constexpr FORCE_INLINE bool IsAtLeftEdge(IndexType index);
    static constexpr FORCE_INLINE bool IsAtRightEdge(IndexType index);
    static constexpr FORCE_INLINE RowParity GetRowParity(IndexType index);
    static constexpr FORCE_INLINE IndexType InvalidateOutBoundsIndex(IndexType index);

    template <MoveDirection direction>
    static constexpr FORCE_INLINE IndexType GetRelativeMoveIndex(IndexType index);

    template <BoardCheckType type>
    static constexpr FORCE_INLINE IndexType GetPieceLeftMoveIndex(IndexType index);

    template <BoardCheckType type>
    static constexpr FORCE_INLINE IndexType GetPieceRightMoveIndex(IndexType index);

    template <BoardCheckType type>
    FORCE_INLINE void ApplyMove(IndexType from, IndexType to, bool is_capture);

    FORCE_INLINE void PromoteAll()
    {
        constexpr HalfBoard white_mask = (1u << kHalfBoardEdgeLength) - 1;
        constexpr HalfBoard black_mask = ((1u << kHalfBoardEdgeLength) - 1)
                                         << (kHalfBoardSize - kHalfBoardEdgeLength);
        HalfBoard promotion;

        promotion = white_pieces & white_mask;
        kings |= promotion;
        promotion = black_pieces & black_mask;
        kings |= promotion;
    }

    //------------------------------------------------------------------------------//
    //                                Friend Functions                              //
    //------------------------------------------------------------------------------//
    friend std::ostream& operator<<(std::ostream& os, const Board& board);
};

}  // namespace CudaMctsCheckers

#include <board.tpp>

#endif  // CUDA_MCTS_CHECKRS_INCLUDE_BOARD_HPP_
