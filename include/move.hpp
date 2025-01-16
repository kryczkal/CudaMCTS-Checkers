#ifndef CUDA_MCTS_CHECKRS_INCLUDE_MOVE_HPP_
#define CUDA_MCTS_CHECKRS_INCLUDE_MOVE_HPP_

#include <board.hpp>
#include <cpp_defines.hpp>

namespace CudaMctsCheckers
{

class Move
{
    public:
    static constexpr u32 kNumMaxPossibleMovesPerPiece = Board::kDiagonalSize * 2 - 1;
    static constexpr u32 kNumMoveArrayForPlayerSize =
        kNumMaxPossibleMovesPerPiece * Board::kHalfBoardSize;

    enum PieceMoveIndexes { kLeft, kRight, kKnownIndexesEnd };
    static_assert(kNumMaxPossibleMovesPerPiece >= kKnownIndexesEnd);

    using Type = Board::IndexType;  // Move type is actually an index of the board to move to
    using MoveArrayForPlayer                 = std::array<Type, kNumMoveArrayForPlayerSize>;
    static constexpr Move::Type kInvalidMove = Board::kInvalidIndex;

    static FORCE_INLINE u32 DecodeOriginIndex(const u32 index_in_moves_array)
    {
        return index_in_moves_array / kNumMaxPossibleMovesPerPiece;
    }
};
}  // namespace CudaMctsCheckers

#endif  // CUDA_MCTS_CHECKRS_INCLUDE_MOVE_HPP_
