#ifndef CUDA_MCTS_CHECKRS_INCLUDE_MOVE_GENERATION_TPP_
#define CUDA_MCTS_CHECKRS_INCLUDE_MOVE_GENERATION_TPP_

#include <array>
#include <board.hpp>
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
    for (u32 i = 0; i < Move::kNumMoveArrayForPlayerSize; ++i) {  // TODO: Unoptimal
        output.possible_moves[i] = Move::kInvalidMove;
    }
    output.no_moves = true;

    for (Board::IndexType i = 0; i < Board::kHalfBoardSize;
         ++i) {                          // TODO: count leading zeros __builtin_clz
        if (board.IsPieceAt<type>(i)) {  // TODO: Just do 2 loops for pieces, and for kings
            u32 current_move_index = i * Move::kNumMaxPossibleMovesPerPiece;
            if (!board.IsPieceAt<BoardCheckType::kKings>(i)) {
                // Try to move forward
                GenerateMovesPieceCpu<type>(board, output, i, current_move_index);

            } else {
                GenerateMovesKingCpu<type>(board, output, i, current_move_index);
            }
        }
    }

    return output;
}

template <BoardCheckType type>
void MoveGenerator::GenerateMovesKingCpu(
    const Board &board, MoveGenerationOutput &output, Board::IndexType i, u32 current_move_index
)
{
    GenerateMovesDiagonalCpu<type, MoveDirection::kUpLeft>(board, output, i, current_move_index);
    GenerateMovesDiagonalCpu<type, MoveDirection::kUpRight>(board, output, i, current_move_index);
    GenerateMovesDiagonalCpu<type, MoveDirection::kDownLeft>(board, output, i, current_move_index);
    GenerateMovesDiagonalCpu<type, MoveDirection::kDownRight>(board, output, i, current_move_index);
}

template <BoardCheckType type>
void MoveGenerator::GenerateMovesPieceCpu(
    const Board &board, MoveGenerationOutput &output, Board::IndexType i, u32 current_move_index
)
{
    Board::IndexType left_move_index  = board.GetPieceLeftMoveIndex<type>(i);
    Board::IndexType right_move_index = board.GetPieceRightMoveIndex<type>(i);
    output.possible_moves[current_move_index + Move::kLeft] =
        board.template IsPieceAt<BoardCheckType::kAll>(left_move_index) ? Move::kInvalidMove
                                                                        : left_move_index;
    output.possible_moves[current_move_index + Move::kRight] =
        board.template IsPieceAt<BoardCheckType::kAll>(right_move_index) ? Move::kInvalidMove
                                                                         : right_move_index;

    // Detect capture
    if (left_move_index != Board::kInvalidIndex &&
        board.GetPieceLeftMoveIndex<type>(left_move_index) != Board::kInvalidIndex &&
        board.IsPieceAt<Board::GetOppositeType<type>()>(left_move_index) &&
        !board.IsPieceAt<BoardCheckType::kAll>(board.GetPieceLeftMoveIndex<type>(left_move_index)
        )) {
        output.possible_moves[current_move_index + Move::kLeft] =
            board.GetPieceLeftMoveIndex<type>(left_move_index);
        output.capture_moves_bitmask[current_move_index + Move::kLeft]       = true;
        output.capture_moves_bitmask[MoveGenerationOutput::CaptureFlagIndex] = true;
    }
    if (right_move_index != Board::kInvalidIndex &&
        board.GetPieceRightMoveIndex<type>(right_move_index) != Board::kInvalidIndex &&
        board.IsPieceAt<Board::GetOppositeType<type>()>(right_move_index) &&
        !board.IsPieceAt<BoardCheckType::kAll>(board.GetPieceRightMoveIndex<type>(right_move_index)
        )) {
        output.possible_moves[current_move_index + Move::kRight] =
            board.GetPieceRightMoveIndex<type>(right_move_index);
        output.capture_moves_bitmask[current_move_index + Move::kRight]      = true;
        output.capture_moves_bitmask[MoveGenerationOutput::CaptureFlagIndex] = true;
    }
    if (output.no_moves) {
        output.no_moves =
            output.possible_moves[current_move_index + Move::kLeft] == Board::kInvalidIndex &&
            output.possible_moves[current_move_index + Move::kRight] == Board::kInvalidIndex;
    }
}

template <BoardCheckType type, MoveDirection direction>
void MoveGenerator::GenerateMovesDiagonalCpu(
    const Board &board, MoveGenerationOutput &output, Board::IndexType index,
    u32 &current_move_index
)
{
    Board::IndexType board_index = board.template GetRelativeMoveIndex<direction>(index);
    bool is_capturing            = false;
    while (board_index != Board::kInvalidIndex) {
        if (board.IsPieceAt<Board::GetOppositeType<type>()>(board_index)) {
            // Try to capture
            board_index = board.GetRelativeMoveIndex<direction>(board_index);
            if (board_index == Board::kInvalidIndex ||
                board.IsPieceAt<BoardCheckType::kAll>(board_index)) {
                break;
            }
            output.possible_moves[current_move_index]                            = board_index;
            output.capture_moves_bitmask[current_move_index]                     = true;
            output.capture_moves_bitmask[MoveGenerationOutput::CaptureFlagIndex] = true;
            is_capturing                                                         = true;
            output.no_moves                                                      = false;
            current_move_index++;
        } else if (!board.IsPieceAt<type>(board_index)) {
            output.possible_moves[current_move_index]        = board_index;
            output.capture_moves_bitmask[current_move_index] = is_capturing;
            output.no_moves                                  = false;
            current_move_index++;
        } else {
            break;
        }
        board_index = board.template GetRelativeMoveIndex<direction>(board_index);
    }
}

}  // namespace CudaMctsCheckers

#endif  // CUDA_MCTS_CHECKRS_INCLUDE_MOVE_GENERATION_TPP_
