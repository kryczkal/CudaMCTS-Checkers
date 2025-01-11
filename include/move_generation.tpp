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
    for (u32 i = 0; i < Move::kNumMoveArrayForPlayerSize; ++i) {
        output.possible_moves[i] = Move::kInvalidMove;
    }

    for (Board::IndexType i = 0; i < Board::kHalfBoardSize;
         ++i) {                          // TODO: count leading zeros __builtin_clz
        if (board.IsPieceAt<type>(i)) {  // TODO: Just do 2 loops for pieces, and for kings
            if (!board.IsPieceAt<BoardCheckType::kKings>(i)) {
                // Try to move forward
                Board::IndexType left_move_index  = board.GetPieceLeftMoveIndex<type>(i);
                Board::IndexType right_move_index = board.GetPieceRightMoveIndex<type>(i);
                output.possible_moves
                    [i * Move::kNumMaxPossibleMovesPerPiece + Move::PieceMoveIndexes::kLeft] =
                    left_move_index;
                output.possible_moves
                    [i * Move::kNumMaxPossibleMovesPerPiece + Move::PieceMoveIndexes::kRight] =
                    right_move_index;

                // Detect capture
                if (left_move_index != Board::kInvalidIndex &&
                    board.IsPieceAt<Board::GetOppositeType<type>()>(left_move_index) &&
                    !board.IsPieceAt<BoardCheckType::kAll>(
                        board.GetPieceLeftMoveIndex<type>(left_move_index)
                    )) {
                    output.possible_moves
                        [i * Move::kNumMaxPossibleMovesPerPiece +
                         Move::PieceMoveIndexes::kLeftCapture] =
                        board.GetPieceLeftMoveIndex<type>(left_move_index);
                    output.detected_capture = true;
                }
                if (right_move_index != Board::kInvalidIndex &&
                    board.IsPieceAt<Board::GetOppositeType<type>()>(right_move_index) &&
                    !board.IsPieceAt<BoardCheckType::kAll>(
                        board.GetPieceRightMoveIndex<type>(right_move_index)
                    )) {
                    output.possible_moves
                        [i * Move::kNumMaxPossibleMovesPerPiece +
                         Move::PieceMoveIndexes::kRightCapture] =
                        board.GetPieceRightMoveIndex<type>(right_move_index);
                    output.detected_capture = true;
                }
            } else {
                Board::IndexType CurrentMoveIndex = i * Move::kNumMaxPossibleMovesPerPiece;
                GenerateMovesDiagonalCpu<type, MoveDirection::kUpLeft>(
                    board, output, i, CurrentMoveIndex
                );
                GenerateMovesDiagonalCpu<type, MoveDirection::kUpRight>(
                    board, output, i, CurrentMoveIndex
                );
                GenerateMovesDiagonalCpu<type, MoveDirection::kDownLeft>(
                    board, output, i, CurrentMoveIndex
                );
                GenerateMovesDiagonalCpu<type, MoveDirection::kDownRight>(
                    board, output, i, CurrentMoveIndex
                );
            }
        }
    }

    return output;
}

template <BoardCheckType type, MoveDirection direction>
void MoveGenerator::GenerateMovesDiagonalCpu(
    const Board &board, MoveGenerationOutput &output, Board::IndexType index,
    Board::IndexType &move_index
)
{
    Board::IndexType board_index = index;
    while (board_index != Board::kInvalidIndex) {
        if (board.IsPieceAt<Board::GetOppositeType<type>()>(board_index)) {
            // Try to capture
            Board::IndexType capture_index =
                board.GetRelativeMoveIndex<MoveDirection::kUpLeft>(board_index);
            if (board.IsPieceAt<BoardCheckType::kAll>(capture_index)) {
                break;
            }
            output.possible_moves[move_index] = capture_index;
            move_index++;
            output.detected_capture = true;
        } else if (!board.IsPieceAt<type>(board_index)) {
            output.possible_moves[move_index] = board_index;
            move_index++;
        } else {
            break;
        }
    }
}

}  // namespace CudaMctsCheckers

#endif  // CUDA_MCTS_CHECKRS_INCLUDE_MOVE_GENERATION_TPP_
