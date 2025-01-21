#ifndef MCTS_CHECKERS_INCLUDE_MOVE_GENERATION_TPP_
#define MCTS_CHECKERS_INCLUDE_MOVE_GENERATION_TPP_

#include "board_helpers.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"

// Flags
static constexpr u8 kIsPieceOnBoardFlagIndex = 0;
// Adjacent pieces
static constexpr u8 kIsUpperLeftMoveInvalid  = 1;
static constexpr u8 kIsUpperRightMoveInvalid = 2;
static constexpr u8 kIsLowerLeftMoveInvalid  = 3;
static constexpr u8 kIsLowerRightMoveInvalid = 4;

static constexpr u8 kOnlyIsPieceOnBoardMask = 1 << kIsPieceOnBoardFlagIndex;

namespace checkers::gpu::move_gen
{

template <Turn turn>
__device__ __forceinline__ void TryMoveForward(
    board_index_t figure_idx, board_t all_pieces, move_t *d_moves, u32 &move_idx, u8 &num_moves,
    move_flags_t &d_move_capture_mask, move_flags_t &d_per_board_move_flags, u8 &flags
);

template <Turn turn>
__device__ __forceinline__ void TryCapture(
    board_index_t figure_idx, board_t all_pieces, board_t enemy_pieces, move_t *d_moves, u32 &move_idx, u8 &num_moves,
    move_flags_t &d_move_capture_mask, move_flags_t &d_per_board_move_flags, u8 &flags
);

template <Direction direction>
__device__ __forceinline__ void TryDiagonal(
    board_index_t figure_idx, board_t all_pieces, board_t enemy_pieces, board_index_t start_idx, move_t *d_moves,
    u32 &move_idx, u8 &num_moves, move_flags_t &d_move_capture_masks, move_flags_t &d_per_board_move_flags, u8 &flags
);

__device__ __forceinline__ void TryKingMoves(
    board_index_t figure_idx, board_t all_pieces, board_t enemy_pieces, move_t *d_moves, u32 &move_index, u8 &num_moves,
    move_flags_t &d_move_capture_mask, move_flags_t &d_per_board_move_flags, u8 &flags
);

// TODO: Split flags into two bools to eliminate race conditions

////////////////////////////////////////////////////////////////////////////////
//                           MAIN GENERATE MOVES                              //
////////////////////////////////////////////////////////////////////////////////

template <Turn turn>
__global__ void GenerateMoves(
    // Board States
    const board_t *d_whites, const board_t *d_blacks, const board_t *d_kings,
    // Moves
    move_t *d_moves, u8 *d_move_counts, move_flags_t *d_move_capture_mask, move_flags_t *d_per_board_move_flags,
    // Number of boards to process
    const u64 n_boards
)
{
    // Each thread handles exactly one figure index on a board, but handles potentially many boards
    //  -> 32 threads per board.

    for (u64 global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x; global_thread_idx < n_boards * 32;
         global_thread_idx += gridDim.x * blockDim.x) {
        // Compute board index and figure index
        const u64 board_idx            = global_thread_idx / 32;
        const board_index_t figure_idx = (board_index_t)(global_thread_idx % 32);
        GenerateMovesForBoardIdxFigureIdx<turn>(
            board_idx, figure_idx, d_whites, d_blacks, d_kings, d_moves, d_move_counts, d_move_capture_mask,
            d_per_board_move_flags, n_boards
        );
    }
}

template <Turn turn>
__device__ __forceinline__ void GenerateMovesForBoardIdxFigureIdx(
    const u64 board_idx, board_index_t figure_idx,
    // Board States
    const board_t *d_whites, const board_t *d_blacks, const board_t *d_kings,
    // Moves
    move_t *d_moves, u8 *d_move_counts, move_flags_t *d_move_capture_mask, move_flags_t *d_per_board_move_flags,
    // Number of boards to process
    const u64 n_boards
)
{
    // Initialize tracking
    u8 flags     = 0;
    u8 num_moves = 0;
    u32 move_idx = (board_idx * BoardConstants::kBoardSize + figure_idx) * (kNumMaxMovesPerPiece);

    move_flags_t &move_flags              = d_move_capture_mask[(board_idx * BoardConstants::kBoardSize + figure_idx)];
    move_flags_t &per_board_move_flag_set = d_per_board_move_flags[board_idx];
    move_flags                            = 0;

    // Access the bitmasks
    const board_t white_pieces = d_whites[board_idx];
    const board_t black_pieces = d_blacks[board_idx];
    const board_t kings        = d_kings[board_idx];

    // Current player's pieces and the union
    const board_t current_player_pieces = (turn == Turn::kWhite) ? white_pieces : black_pieces;
    const board_t all_pieces            = white_pieces | black_pieces;
    const board_t enemy_pieces          = (turn == Turn::kWhite) ? black_pieces : white_pieces;

    const board_t board_without_kings = current_player_pieces & ~kings;
    flags |= IsPieceAt(board_without_kings, figure_idx) << kIsPieceOnBoardFlagIndex;

    //
    // 1) Try forward moves
    //
    TryMoveForward<turn>(
        figure_idx, all_pieces, d_moves, move_idx, num_moves, move_flags, per_board_move_flag_set, flags
    );
    flags &= kOnlyIsPieceOnBoardMask;

    //
    // 2) Try captures
    //

    // Keep only the kIsPieceOnBoardFlagIndex from previous flags
    TryCapture<turn>(
        figure_idx, all_pieces, enemy_pieces, d_moves, move_idx, num_moves, move_flags, per_board_move_flag_set, flags
    );
    flags &= kOnlyIsPieceOnBoardMask;

    //
    // 3) Try kings
    //
    flags = 0;
    flags |= IsPieceAt(kings, figure_idx) << kIsPieceOnBoardFlagIndex;
    TryKingMoves(
        figure_idx, all_pieces, enemy_pieces, d_moves, move_idx, num_moves, move_flags, per_board_move_flag_set, flags
    );

    d_move_counts[board_idx * BoardConstants::kBoardSize + figure_idx] = num_moves;
}

////////////////////////////////////////////////////////////////////////////////
//                            FORWARD MOVES                                   //
////////////////////////////////////////////////////////////////////////////////

template <Turn turn>
__device__ __forceinline__ void TryMoveForward(
    const board_index_t figure_idx, const board_t all_pieces, move_t *d_moves, u32 &move_idx, u8 &num_moves,
    move_flags_t &d_move_capture_mask, move_flags_t &d_per_board_move_flags, u8 &flags
)
{
    const board_index_t upper_left_index  = GetAdjacentIndex<Direction::kUpLeft>(figure_idx);
    const board_index_t upper_right_index = GetAdjacentIndex<Direction::kUpRight>(figure_idx);
    const board_index_t lower_left_index  = GetAdjacentIndex<Direction::kDownLeft>(figure_idx);
    const board_index_t lower_right_index = GetAdjacentIndex<Direction::kDownRight>(figure_idx);

    // Helper lambda for writing a move to the d_moves array
    auto writeMove = [&](bool is_move_invalid, board_index_t to_idx) {
        is_move_invalid |= !ReadFlag(flags, kIsPieceOnBoardFlagIndex);
        if (!is_move_invalid) {
            printf("[%d] Move to %d\n", figure_idx, to_idx);
        }
        const move_t move = EncodeMove(figure_idx, to_idx);
        d_moves[move_idx] = is_move_invalid ? MoveConstants::kInvalidMove : move;
        move_idx          = is_move_invalid ? move_idx : move_idx + 1;
        num_moves         = is_move_invalid ? num_moves : num_moves + 1;
        if (!is_move_invalid) {
            d_per_board_move_flags |= (1 << MoveFlagsConstants::kMoveFound);
        }
    };

    if constexpr (turn == Turn::kWhite) {
        flags |= (IsOnEdge<Direction::kUpLeft>(figure_idx) | IsPieceAt(all_pieces, upper_left_index))
                 << kIsUpperLeftMoveInvalid;
        writeMove(ReadFlag(flags, kIsUpperLeftMoveInvalid), upper_left_index);

        flags |= (IsOnEdge<Direction::kUpRight>(figure_idx) | IsPieceAt(all_pieces, upper_right_index))
                 << kIsUpperRightMoveInvalid;
        writeMove(ReadFlag(flags, kIsUpperRightMoveInvalid), upper_right_index);

    } else {
        flags |= (IsOnEdge<Direction::kDownLeft>(figure_idx) | IsPieceAt(all_pieces, lower_left_index))
                 << kIsLowerLeftMoveInvalid;
        writeMove(ReadFlag(flags, kIsLowerLeftMoveInvalid), lower_left_index);

        flags |= (IsOnEdge<Direction::kDownLeft>(figure_idx) | IsPieceAt(all_pieces, lower_right_index))
                 << kIsLowerRightMoveInvalid;
        writeMove(ReadFlag(flags, kIsLowerRightMoveInvalid), lower_right_index);
    }
}

////////////////////////////////////////////////////////////////////////////////
//                            CAPTURE MOVES                                   //
////////////////////////////////////////////////////////////////////////////////

template <Turn turn>
__device__ __forceinline__ void TryCapture(
    const board_index_t figure_idx, const board_t all_pieces, const board_t enemy_pieces, move_t *d_moves,
    u32 &move_idx, u8 &num_moves, move_flags_t &d_move_capture_mask, move_flags_t &d_per_board_move_flags, u8 &flags
)
{
    // Indices for adjacent squares
    const board_index_t ul = GetAdjacentIndex<Direction::kUpLeft>(figure_idx);
    const board_index_t ur = GetAdjacentIndex<Direction::kUpRight>(figure_idx);
    const board_index_t ll = GetAdjacentIndex<Direction::kDownLeft>(figure_idx);
    const board_index_t lr = GetAdjacentIndex<Direction::kDownRight>(figure_idx);

    // Indices for jump squares
    const board_index_t ul_jump = GetAdjacentIndex<Direction::kUpLeft>(ul);
    const board_index_t ur_jump = GetAdjacentIndex<Direction::kUpRight>(ur);
    const board_index_t ll_jump = GetAdjacentIndex<Direction::kDownLeft>(ll);
    const board_index_t lr_jump = GetAdjacentIndex<Direction::kDownRight>(lr);

    // Build up capture flags
    flags |= (IsOnEdge<Direction::kUpLeft>(figure_idx) | IsOnEdge<Direction::kUpLeft>(ul) |
              !IsPieceAt(enemy_pieces, ul) | IsPieceAt(all_pieces, ul_jump))
             << kIsUpperLeftMoveInvalid;

    flags |= (IsOnEdge<Direction::kDownLeft>(figure_idx) | IsOnEdge<Direction::kDownLeft>(ll) |
              !IsPieceAt(enemy_pieces, ll) | IsPieceAt(all_pieces, ll_jump))
             << kIsLowerLeftMoveInvalid;

    flags |= (IsOnEdge<Direction::kUpRight>(figure_idx) | IsOnEdge<Direction::kUpRight>(ur) |
              !IsPieceAt(enemy_pieces, ur) | IsPieceAt(all_pieces, ur_jump))
             << kIsUpperRightMoveInvalid;

    flags |= (IsOnEdge<Direction::kDownRight>(figure_idx) | IsOnEdge<Direction::kDownRight>(lr) |
              !IsPieceAt(enemy_pieces, lr) | IsPieceAt(all_pieces, lr_jump))
             << kIsLowerRightMoveInvalid;

    // Helper lambda to record captures
    auto writeCapture = [&](bool is_capture_invalid, board_index_t to_idx) {
        const move_t move = EncodeMove(figure_idx, to_idx);
        is_capture_invalid |= !ReadFlag(flags, kIsPieceOnBoardFlagIndex);
        if (!is_capture_invalid) {
            printf("[%d] Capture to %d\n", figure_idx, to_idx);
        }
        d_moves[move_idx] = is_capture_invalid ? MoveConstants::kInvalidMove : move;
        d_move_capture_mask |= is_capture_invalid ? 0 : (1 << num_moves);
        move_idx  = is_capture_invalid ? move_idx : move_idx + 1;
        num_moves = is_capture_invalid ? num_moves : num_moves + 1;
        if (!is_capture_invalid) {
            d_per_board_move_flags |= (1 << MoveFlagsConstants::kMoveFound);
            d_per_board_move_flags |= (1 << MoveFlagsConstants::kCaptureFound);
        }
    };

    // Write all four capture attempts
    writeCapture(ReadFlag(flags, kIsUpperLeftMoveInvalid), ul_jump);
    writeCapture(ReadFlag(flags, kIsUpperRightMoveInvalid), ur_jump);
    writeCapture(ReadFlag(flags, kIsLowerLeftMoveInvalid), ll_jump);
    writeCapture(ReadFlag(flags, kIsLowerRightMoveInvalid), lr_jump);
}

////////////////////////////////////////////////////////////
// TryKingMoves: calls TryDiagonal in each diagonal dir   //
////////////////////////////////////////////////////////////
__device__ __forceinline__ void TryKingMoves(
    const board_index_t figure_idx, const board_t all_pieces, const board_t enemy_pieces, move_t *d_moves,
    u32 &move_index, u8 &num_moves, move_flags_t &d_move_capture_mask, move_flags_t &d_per_board_move_flags, u8 &flags
)
{
    // For a king, we just attempt all four diagonal directions
    TryDiagonal<Direction::kUpLeft>(
        figure_idx, all_pieces, enemy_pieces, figure_idx, d_moves, move_index, num_moves, d_move_capture_mask,
        d_per_board_move_flags, flags
    );
    flags &= kOnlyIsPieceOnBoardMask;
    TryDiagonal<Direction::kUpRight>(
        figure_idx, all_pieces, enemy_pieces, figure_idx, d_moves, move_index, num_moves, d_move_capture_mask,
        d_per_board_move_flags, flags
    );
    flags &= kOnlyIsPieceOnBoardMask;
    TryDiagonal<Direction::kDownLeft>(
        figure_idx, all_pieces, enemy_pieces, figure_idx, d_moves, move_index, num_moves, d_move_capture_mask,
        d_per_board_move_flags, flags
    );
    flags &= kOnlyIsPieceOnBoardMask;
    TryDiagonal<Direction::kDownRight>(
        figure_idx, all_pieces, enemy_pieces, figure_idx, d_moves, move_index, num_moves, d_move_capture_mask,
        d_per_board_move_flags, flags
    );
    flags &= kOnlyIsPieceOnBoardMask;
}

//////////////////////////////////////////////////////////////////
// TryDiagonal: scans outward for a king in one diagonal path   //
//////////////////////////////////////////////////////////////////
template <Direction direction>
__device__ __forceinline__ void TryDiagonal(
    const board_index_t figure_idx, const board_t all_pieces, const board_t enemy_pieces, board_index_t start_idx,
    move_t *d_moves, u32 &move_idx, u8 &num_moves, move_flags_t &d_move_capture_masks,
    move_flags_t &d_per_board_move_flags, u8 &flags
)
{
    static constexpr u8 kContinueFlagIndex    = 1;
    static constexpr u8 kInvalidNextFlagIndex = 2;
    static constexpr u8 kInvalidJumpFlagIndex = 3;

    flags &= kOnlyIsPieceOnBoardMask;
    const bool shouldStop = !ReadFlag(flags, kIsPieceOnBoardFlagIndex) || IsOnEdge<direction>(start_idx);
    flags |= (!shouldStop) << kContinueFlagIndex;
    board_index_t current_idx = start_idx;

    auto writeKingMove = [&](bool is_capture_invalid, bool is_capture, board_index_t to_idx) {
        const move_t move = EncodeMove(figure_idx, to_idx);
        is_capture_invalid |= !ReadFlag(flags, kIsPieceOnBoardFlagIndex);
        if (!is_capture_invalid) {
            if (is_capture) {
                printf("[%d] Capture to %d\n", figure_idx, to_idx);
            } else {
                printf("[%d] Move to %d\n", figure_idx, to_idx);
            }
        }
        d_moves[move_idx] = is_capture_invalid ? MoveConstants::kInvalidMove : move;
        d_move_capture_masks |= is_capture_invalid || !is_capture ? 0 : (1 << num_moves);
        move_idx  = is_capture_invalid ? move_idx : move_idx + 1;
        num_moves = is_capture_invalid ? num_moves : num_moves + 1;
        if (!is_capture_invalid) {
            d_per_board_move_flags |= (1 << MoveFlagsConstants::kMoveFound);
            if (is_capture) {
                d_per_board_move_flags |= (1 << MoveFlagsConstants::kCaptureFound);
            }
        }
    };

    bool capturing = false;
    while (ReadFlag(flags, kContinueFlagIndex)) {
        const board_index_t next_idx      = GetAdjacentIndex<direction>(current_idx);
        const board_index_t next_jump_idx = GetAdjacentIndex<direction>(next_idx);

        // Try moving
        flags |= (IsOnEdge<direction>(current_idx) | IsPieceAt(all_pieces, next_idx)) << kInvalidNextFlagIndex;
        writeKingMove(ReadFlag(flags, kInvalidNextFlagIndex), capturing, next_idx);

        // Try capturing
        flags |= (IsOnEdge<direction>(current_idx) | IsOnEdge<direction>(next_jump_idx) |
                  !IsPieceAt(enemy_pieces, next_idx) | IsPieceAt(all_pieces, next_jump_idx))
                 << kInvalidJumpFlagIndex;
        capturing = capturing || !ReadFlag(flags, kInvalidJumpFlagIndex);
        if (capturing) {
            printf("[%d] Capturing to %d\n", figure_idx, next_jump_idx);
        }

        current_idx            = next_idx;
        const bool shouldStop2 = IsOnEdge<direction>(next_idx) ||
                                 (ReadFlag(flags, kInvalidNextFlagIndex) && ReadFlag(flags, kInvalidJumpFlagIndex));
        flags &= kOnlyIsPieceOnBoardMask;
        flags |= (!shouldStop2 << kContinueFlagIndex);
    }
}

}  // namespace checkers::gpu::move_gen

#endif  // MCTS_CHECKERS_INCLUDE_MOVE_GENERATION_TPP_
