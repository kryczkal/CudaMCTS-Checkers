#ifndef CUDA_MCTS_CHECKRS_INCLUDE_MOVE_HPP_
#define CUDA_MCTS_CHECKRS_INCLUDE_MOVE_HPP_

#include <board.hpp>

namespace CudaMctsCheckers
{

class Move
{
    public:
    static constexpr u32 kNumMaxPossibleMovesPerPiece = Board::kDiagonalSize * 2 - 1;
    static constexpr u32 kNumMaxPossibleMovesPerPlayer =
        kNumMaxPossibleMovesPerPiece * Board::kNumPiecesPerPlayer;
    static constexpr u32 kNumMaxPossibleMovesTotal = kNumMaxPossibleMovesPerPlayer * 2;

    enum PieceMoveIndexes { kLeft, kRight, kLeftCapture, kRightCapture, kKnownIndexesEnd };
    static_assert(kNumMaxPossibleMovesPerPiece >= kKnownIndexesEnd);

    using Type = Board::IndexType;  // Move type is actually an index of the board to move to
    using MoveArrayForPlayer = std::array<Type, kNumMaxPossibleMovesPerPlayer>;

    static constexpr Move::Type kInvalidMove = Board::kSizeTotal;
};
}  // namespace CudaMctsCheckers

#endif  // CUDA_MCTS_CHECKRS_INCLUDE_MOVE_HPP_