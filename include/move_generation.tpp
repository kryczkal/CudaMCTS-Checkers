#ifndef CUDA_MCTS_CHECKRS_INCLUDE_MOVE_GENERATION_TPP_
#define CUDA_MCTS_CHECKRS_INCLUDE_MOVE_GENERATION_TPP_

#include <array>
#include <iostream>
#include <move_generation.hpp>

namespace CudaMctsCheckers
{

// TODO: Multi-threaded move generation

template <BoardCheckType type>
MoveGenerationOutput MoveGenerator::GenerateMovesForPlayerCpu(const Board &board)
{
    assert(type != BoardCheckType::kAll);
    assert(type != BoardCheckType::kKings);

    MoveGenerationOutput output = {};
    for (u32 i = 0; i < Move::kNumMaxPossibleMovesPerPlayer; ++i) {
        output.possible_moves[i] = Move::kInvalidMove;
    }

    for (u32 i = 0; i < Board::kHalfBoardSize; ++i) {  // TODO: count leading zeros __builtin_clz
        if (board.IsPieceAt<type>(i)) {  // TODO: Just do 2 loops for pieces, and for kings
            if (!board.IsPieceAt<BoardCheckType::kKings>(i)) {
                // Try to move forward
                Board::IndexType left_move_index  = board.GetPieceLeftMoveIndex<type>(i);
                Board::IndexType right_move_index = board.GetPieceRightMoveIndex<type>(i);
                output.possible_moves
                    [i * Move::kNumMaxPossibleMovesPerPlayer + Move::PieceMoveIndexes::kLeft] =
                    left_move_index;
                output.possible_moves
                    [i * Move::kNumMaxPossibleMovesPerPlayer + Move::PieceMoveIndexes::kRight] =
                    right_move_index;

                // Detect capture
                if (left_move_index != Board::kInvalidIndex &&
                    board.IsPieceAt<GetOppositeType<type>()>(left_move_index) &&
                    !board.IsPieceAt<BoardCheckType::kAll>(
                        board.GetPieceLeftMoveIndex<type>(left_move_index)
                    )) {
                    output.possible_moves
                        [i * Move::kNumMaxPossibleMovesPerPlayer +
                         Move::PieceMoveIndexes::kLeftCapture] =
                        board.GetPieceLeftMoveIndex<type>(left_move_index);
                    output.detected_capture = true;
                }
                if (right_move_index != Board::kInvalidIndex &&
                    board.IsPieceAt<GetOppositeType<type>()>(right_move_index) &&
                    !board.IsPieceAt<BoardCheckType::kAll>(
                        board.GetPieceRightMoveIndex<type>(right_move_index)
                    )) {
                    output.possible_moves
                        [i * Move::kNumMaxPossibleMovesPerPlayer +
                         Move::PieceMoveIndexes::kRightCapture] =
                        board.GetPieceRightMoveIndex<type>(right_move_index);
                    output.detected_capture = true;
                }
            } else {
                // Check every diagonal

                // Top left
                Board::IndexType top_left_index = board.GetPieceLeftMoveIndex<type>(i);
                while (top_left_index != Board::kInvalidIndex) {
                }
            }
        }
    }

    return output;
}

}  // namespace CudaMctsCheckers

#endif  // CUDA_MCTS_CHECKRS_INCLUDE_MOVE_GENERATION_TPP_
