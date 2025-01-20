#ifndef MCTS_CHECKERS_INCLUDE_MOVE_GENERATION_TPP_
#define MCTS_CHECKERS_INCLUDE_MOVE_GENERATION_TPP_

#include "board_helpers.cuh"

namespace checkers::mcts::gpu
{
// Flags
static constexpr u8 kIsPieceOnBoardFlagIndex = 0;
// Adjacent pieces
static constexpr u8 kIsUpperLeftMoveInvalid  = 1;
static constexpr u8 kIsUpperRightMoveInvalid = 2;
static constexpr u8 kIsLowerLeftMoveInvalid  = 3;
static constexpr u8 kIsLowerRightMoveInvalid = 4;

static constexpr u8 kOnlyIsPieceOnBoardMask = 1 << kIsPieceOnBoardFlagIndex;

template <Turn turn>
void GenerateMoves(
    // Board States
    const board_t *d_whites, const board_t *d_blacks, const board_t *d_kings,
    // Moves
    move_t *d_moves, u8 *d_move_counts, u8 &d_move_flags,
    // Number of boards to process
    const u64 n_boards
)
{
    // TODO: Use shmem
    const board_t board_index        = 0;
    const board_index_t figure_index = 0;
    u32 move_index                   = 0;

    u8 flags     = 0;
    u8 num_moves = 0;

    const board_t current_player_pieces = turn == Turn::kWhite ? d_whites[board_index] : d_blacks[board_index];
    const board_t all_pieces            = d_whites[board_index] | d_blacks[board_index];

    /////////////////////////////// Regular Pieces ///////////////////////////////

    const board_t no_kings = current_player_pieces & ~d_kings[board_index];
    flags |= IsPieceAt(no_kings, figure_index) << kIsPieceOnBoardFlagIndex;

    const board_index_t upper_left_index  = GetAdjacentIndex<Direction::kUpLeft>(figure_index);
    const board_index_t upper_right_index = GetAdjacentIndex<Direction::kUpRight>(figure_index);
    const board_index_t lower_left_index  = GetAdjacentIndex<Direction::kDownLeft>(figure_index);
    const board_index_t lower_right_index = GetAdjacentIndex<Direction::kDownRight>(figure_index);

    ////////////////////////////// Regular Movement //////////////////////////////

    u16 move;
    switch (turn) {
        case Turn::kWhite: {
            flags |=
                (IsOnEdge(BoardConstants::kLeftBoardEdgeMask, figure_index) | IsPieceAt(all_pieces, upper_left_index))
                << kIsUpperLeftMoveInvalid;
            move                = EncodeMove(figure_index, upper_left_index);
            d_moves[move_index] = ReadFlag(flags, kIsUpperLeftMoveInvalid) ? MoveConstants::kInvalidMove : move;
            move_index          = ReadFlag(flags, kIsUpperLeftMoveInvalid) ? move_index : move_index + 1;
            num_moves           = ReadFlag(flags, kIsUpperLeftMoveInvalid) ? num_moves : num_moves + 1;
            d_move_flags |= ~ReadFlag(flags, kIsUpperLeftMoveInvalid) << MoveFlagsConstants::kMoveFound;

            flags |=
                (IsOnEdge(BoardConstants::kRightBoardEdgeMask, figure_index) | IsPieceAt(all_pieces, upper_right_index))
                << kIsUpperRightMoveInvalid;
            move                = EncodeMove(figure_index, upper_right_index);
            d_moves[move_index] = ReadFlag(flags, kIsUpperRightMoveInvalid) ? MoveConstants::kInvalidMove : move;
            move_index          = ReadFlag(flags, kIsUpperRightMoveInvalid) ? move_index : move_index + 1;
            num_moves           = ReadFlag(flags, kIsUpperRightMoveInvalid) ? num_moves : num_moves + 1;
            d_move_flags |= ~ReadFlag(flags, kIsUpperRightMoveInvalid) << MoveFlagsConstants::kMoveFound;

            break;
        }
        case Turn::kBlack: {
            flags |=
                (IsOnEdge(BoardConstants::kLeftBoardEdgeMask, figure_index) | IsPieceAt(all_pieces, lower_left_index))
                << kIsLowerLeftMoveInvalid;
            move                = EncodeMove(figure_index, lower_left_index);
            d_moves[move_index] = ReadFlag(flags, kIsLowerLeftMoveInvalid) ? MoveConstants::kInvalidMove : move;
            move_index          = ReadFlag(flags, kIsLowerLeftMoveInvalid) ? move_index : move_index + 1;
            num_moves           = ReadFlag(flags, kIsLowerLeftMoveInvalid) ? num_moves : num_moves + 1;
            d_move_flags |= ~ReadFlag(flags, kIsLowerLeftMoveInvalid) << MoveFlagsConstants::kMoveFound;

            flags |=
                (IsOnEdge(BoardConstants::kRightBoardEdgeMask, figure_index) | IsPieceAt(all_pieces, lower_right_index))
                << kIsLowerRightMoveInvalid;
            move                = EncodeMove(figure_index, lower_right_index);
            d_moves[move_index] = ReadFlag(flags, kIsLowerRightMoveInvalid) ? MoveConstants::kInvalidMove : move;
            move_index          = ReadFlag(flags, kIsLowerRightMoveInvalid) ? move_index : move_index + 1;
            num_moves           = ReadFlag(flags, kIsLowerRightMoveInvalid) ? num_moves : num_moves + 1;
            d_move_flags |= ~ReadFlag(flags, kIsLowerRightMoveInvalid) << MoveFlagsConstants::kMoveFound;

            break;
        }
    }

    ////////////////////////////////// Captures //////////////////////////////////

    flags &= kOnlyIsPieceOnBoardMask;

    const board_index_t upper_left_jump_index  = GetAdjacentIndex<Direction::kUpLeft>(upper_left_index);
    const board_index_t upper_right_jump_index = GetAdjacentIndex<Direction::kUpRight>(upper_right_index);
    const board_index_t lower_left_jump_index  = GetAdjacentIndex<Direction::kDownLeft>(lower_left_index);
    const board_index_t lower_right_jump_index = GetAdjacentIndex<Direction::kDownRight>(lower_right_index);

    board_t enemy_pieces = turn == Turn ::kWhite ? d_blacks[board_index] : d_whites[board_index];
    flags |= (~IsOnEdge(BoardConstants::kLeftBoardEdgeMask, figure_index) &
              ~IsOnEdge(BoardConstants::kLeftBoardEdgeMask, upper_left_index) &
              IsPieceAt(enemy_pieces, upper_left_index) & ~IsPieceAt(all_pieces, upper_left_jump_index))
             << kIsUpperLeftMoveInvalid;

    flags |= (~IsOnEdge(BoardConstants::kLeftBoardEdgeMask, figure_index) &
              ~IsOnEdge(BoardConstants::kLeftBoardEdgeMask, lower_left_index) &
              IsPieceAt(enemy_pieces, lower_left_index) & ~IsPieceAt(all_pieces, lower_left_jump_index))
             << kIsLowerLeftMoveInvalid;

    flags |= (~IsOnEdge(BoardConstants::kRightBoardEdgeMask, figure_index) &
              ~IsOnEdge(BoardConstants::kRightBoardEdgeMask, upper_right_index) &
              IsPieceAt(enemy_pieces, upper_right_index) & ~IsPieceAt(all_pieces, upper_right_jump_index))
             << kIsUpperRightMoveInvalid;

    flags |= (~IsOnEdge(BoardConstants::kRightBoardEdgeMask, figure_index) &
              ~IsOnEdge(BoardConstants::kRightBoardEdgeMask, lower_right_index) &
              IsPieceAt(enemy_pieces, lower_right_index) & ~IsPieceAt(all_pieces, lower_right_jump_index))
             << kIsLowerRightMoveInvalid;

    move                = EncodeMove(figure_index, upper_left_jump_index);
    d_moves[move_index] = ReadFlag(flags, kIsUpperLeftMoveInvalid) ? MoveConstants::kInvalidMove : move;
    move_index          = ReadFlag(flags, kIsUpperLeftMoveInvalid) ? move_index : move_index + 1;
    num_moves           = ReadFlag(flags, kIsUpperLeftMoveInvalid) ? num_moves : num_moves + 1;
    d_move_flags |= ~ReadFlag(flags, kIsUpperLeftMoveInvalid) << MoveFlagsConstants::kMoveFound;
    d_move_flags |= ~ReadFlag(flags, kIsUpperLeftMoveInvalid) << MoveFlagsConstants::kCaptureFound;

    move                = EncodeMove(figure_index, upper_right_jump_index);
    d_moves[move_index] = ReadFlag(flags, kIsUpperRightMoveInvalid) ? MoveConstants::kInvalidMove : move;
    move_index          = ReadFlag(flags, kIsUpperRightMoveInvalid) ? move_index : move_index + 1;
    num_moves           = ReadFlag(flags, kIsUpperRightMoveInvalid) ? num_moves : num_moves + 1;
    d_move_flags |= ~ReadFlag(flags, kIsUpperRightMoveInvalid) << MoveFlagsConstants::kMoveFound;

    move                = EncodeMove(figure_index, lower_left_jump_index);
    d_moves[move_index] = ReadFlag(flags, kIsLowerLeftMoveInvalid) ? MoveConstants::kInvalidMove : move;
    move_index          = ReadFlag(flags, kIsLowerLeftMoveInvalid) ? move_index : move_index + 1;
    num_moves           = ReadFlag(flags, kIsLowerLeftMoveInvalid) ? num_moves : num_moves + 1;
    d_move_flags |= ~ReadFlag(flags, kIsLowerLeftMoveInvalid) << MoveFlagsConstants::kMoveFound;

    move                = EncodeMove(figure_index, lower_right_jump_index);
    d_moves[move_index] = ReadFlag(flags, kIsLowerRightMoveInvalid) ? MoveConstants::kInvalidMove : move;
    move_index          = ReadFlag(flags, kIsLowerRightMoveInvalid) ? move_index : move_index + 1;
    num_moves           = ReadFlag(flags, kIsLowerRightMoveInvalid) ? num_moves : num_moves + 1;
    d_move_flags |= ~ReadFlag(flags, kIsLowerRightMoveInvalid) << MoveFlagsConstants::kMoveFound;


}
}  // namespace checkers::mcts::gpu

#endif  // MCTS_CHECKERS_INCLUDE_MOVE_GENERATION_TPP_
