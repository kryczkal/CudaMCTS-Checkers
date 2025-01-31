#ifndef MCTS_CHECKERS_INCLUDE_MOVE_GENERATION_TPP_
#define MCTS_CHECKERS_INCLUDE_MOVE_GENERATION_TPP_

#include <iostream>
#include "board_helpers.cuh"
#include "common/checkers_defines.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "types.hpp"

namespace checkers::gpu::move_gen
{

// Flags
static constexpr u8 kIsPieceOnBoardFlagIndex = 0;
// Adjacent pieces
static constexpr u8 kIsUpperLeftMoveInvalid  = 1;
static constexpr u8 kIsUpperRightMoveInvalid = 2;
static constexpr u8 kIsLowerLeftMoveInvalid  = 3;
static constexpr u8 kIsLowerRightMoveInvalid = 4;

static constexpr u8 kOnlyIsPieceOnBoardMask = 1 << kIsPieceOnBoardFlagIndex;

///////////////////////////////////////////////////////////////////////////////
//                      FORWARD DECLARATIONS OF HELPER FUNCS                  //
///////////////////////////////////////////////////////////////////////////////

template <Turn turn>
__device__ __forceinline__ void TryMoveForward(
    board_index_t figure_idx, board_t all_pieces, move_t* out_moves, u8& out_num_moves, move_flags_t& out_capture_mask,
    move_flags_t& per_board_flags, u8& flags
);

template <Turn turn>
__device__ __forceinline__ void TryCapture(
    board_index_t figure_idx, board_t all_pieces, board_t enemy_pieces, move_t* out_moves, u8& out_num_moves,
    move_flags_t& out_capture_mask, move_flags_t& per_board_flags, u8& flags
);

__device__ __forceinline__ void TryKingMoves(
    board_index_t figure_idx, board_t all_pieces, board_t enemy_pieces, board_t kings, move_t* out_moves,
    u8& out_num_moves, move_flags_t& out_capture_mask, move_flags_t& per_board_flags, u8& flags
);

template <Direction direction>
__device__ __forceinline__ void TryDiagonal(
    board_index_t figure_idx, board_t all_pieces, board_t enemy_pieces, board_index_t start_idx, move_t* out_moves,
    u8& out_num_moves, move_flags_t& out_capture_mask, move_flags_t& per_board_flags, u8& flags
);

///////////////////////////////////////////////////////////////////////////////
//                 CORE FUNCTION: GenerateMovesForSinglePiece                //
///////////////////////////////////////////////////////////////////////////////
template <Turn turn>
__device__ __forceinline__ void GenerateMovesForSinglePiece(
    const board_index_t figure_idx, const board_t white_pieces, const board_t black_pieces, const board_t kings,
    move_t* out_moves,  // size >= kNumMaxMovesPerPiece
    u8& out_move_count, move_flags_t& out_capture_mask, move_flags_t& per_board_flags
)
{
    // We'll track partial flags in a local variable
    u8 flags = 0;

    // Identify if this piece is a normal man for the current turn
    const board_t current_player_pieces = (turn == Turn::kWhite) ? white_pieces : black_pieces;
    const board_t all_pieces            = white_pieces | black_pieces;
    const board_t enemy_pieces          = (turn == Turn::kWhite) ? black_pieces : white_pieces;

    // For each new call, out_move_count & out_capture_mask are assumed to be zeroed by the caller
    // We'll increment out_move_count as we find valid moves, and set bits in out_capture_mask.

    // If not a piece for this turn, we do nothing
    const bool is_piece = ReadFlag(current_player_pieces, figure_idx);
    if (!is_piece) {
        return;
    }

    // We set a bit flag if the piece is present
    flags |= 1 << kIsPieceOnBoardFlagIndex;

    // Count how many moves we've added so far
    u8 num_moves = 0;

    // if it's a non-king piece for the current turn, try forward moves and captures
    const bool is_king_piece = ReadFlag(kings, figure_idx);
    if (!is_king_piece) {
        // Try forward moves
        TryMoveForward<turn>(figure_idx, all_pieces, out_moves, num_moves, out_capture_mask, per_board_flags, flags);
        // Keep only the kIsPieceOnBoardFlagIndex
        flags &= kOnlyIsPieceOnBoardMask;

        // Try captures
        TryCapture<turn>(
            figure_idx, all_pieces, enemy_pieces, out_moves, num_moves, out_capture_mask, per_board_flags, flags
        );
        flags &= kOnlyIsPieceOnBoardMask;
    } else {
        flags = 0;
        flags |= 1 << kIsPieceOnBoardFlagIndex;

        TryKingMoves(
            figure_idx, all_pieces, enemy_pieces, kings, out_moves, num_moves, out_capture_mask, per_board_flags, flags
        );
    }

    // Finally, store how many we found
    out_move_count = num_moves;
}

////////////////////////////////////////////////////////////////////////////////
//                   STAND-ALONE KERNEL: GenerateMoves                        //
////////////////////////////////////////////////////////////////////////////////
template <Turn turn>
__global__ void GenerateMoves(
    const board_t* d_whites, const board_t* d_blacks, const board_t* d_kings, move_t* d_moves, u8* d_move_counts,
    move_flags_t* d_move_capture_mask, move_flags_t* d_per_board_move_flags, const u64 n_boards
)
{
    // Each thread handles exactly one (board, figure_idx). 32 threads per board => figure_idx in [0..31].
    // The global thread index is used to map to (board_idx, figure_idx).
    u64 global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (global_thread_idx < n_boards * 32ULL) {
        const u64 board_idx            = global_thread_idx / 32ULL;
        const u64 figure_idx_int       = global_thread_idx % 32ULL;
        const board_index_t figure_idx = static_cast<board_index_t>(figure_idx_int);

        // Identify the output location(s) for this piece
        // We have up to kNumMaxMovesPerPiece moves stored for each piece.
        const u64 base_moves_idx =
            (board_idx * BoardConstants::kBoardSize + figure_idx) * static_cast<u64>(kNumMaxMovesPerPiece);

        move_t* out_moves_ptr          = &d_moves[base_moves_idx];
        u8* out_move_count_ptr         = &d_move_counts[board_idx * BoardConstants::kBoardSize + figure_idx];
        move_flags_t* out_capture_mask = &d_move_capture_mask[board_idx * BoardConstants::kBoardSize + figure_idx];
        move_flags_t& per_board_flags  = d_per_board_move_flags[board_idx];

        // Clear the outputs for this piece
        *out_move_count_ptr = 0;
        *out_capture_mask   = 0;

        // Retrieve the piece bitmasks for this board
        board_t white_pieces = d_whites[board_idx];
        board_t black_pieces = d_blacks[board_idx];
        board_t kings        = d_kings[board_idx];

        // Generate
        GenerateMovesForSinglePiece<turn>(
            figure_idx, white_pieces, black_pieces, kings, out_moves_ptr, *out_move_count_ptr, *out_capture_mask,
            per_board_flags
        );

        global_thread_idx += gridDim.x * blockDim.x;
    }
}

////////////////////////////////////////////////////////////////////////////////
//                         HELPER IMPLEMENTATIONS                             //
////////////////////////////////////////////////////////////////////////////////
template <Turn turn>
__device__ __forceinline__ void TryMoveForward(
    board_index_t figure_idx, board_t all_pieces, move_t* out_moves, u8& out_num_moves, move_flags_t& out_capture_mask,
    move_flags_t& per_board_flags, u8& flags
)
{
    const board_index_t upper_left_index  = GetAdjacentIndex<Direction::kUpLeft>(figure_idx);
    const board_index_t upper_right_index = GetAdjacentIndex<Direction::kUpRight>(figure_idx);
    const board_index_t lower_left_index  = GetAdjacentIndex<Direction::kDownLeft>(figure_idx);
    const board_index_t lower_right_index = GetAdjacentIndex<Direction::kDownRight>(figure_idx);

    auto WriteMove = [&](bool is_move_invalid, board_index_t to_idx) {
        is_move_invalid |= !ReadFlag(flags, kIsPieceOnBoardFlagIndex);
        const move_t mv = EncodeMove(figure_idx, to_idx);
        if (!is_move_invalid) {
            out_moves[out_num_moves] = mv;
            out_num_moves++;
            per_board_flags |= (1 << MoveFlagsConstants::kMoveFound);
        }
    };

    if constexpr (turn == Turn::kWhite) {
        flags |= (IsOnEdge<Direction::kUpLeft>(figure_idx) | IsPieceAt(all_pieces, upper_left_index))
                 << kIsUpperLeftMoveInvalid;
        WriteMove(ReadFlag(flags, kIsUpperLeftMoveInvalid), upper_left_index);

        flags |= (IsOnEdge<Direction::kUpRight>(figure_idx) | IsPieceAt(all_pieces, upper_right_index))
                 << kIsUpperRightMoveInvalid;
        WriteMove(ReadFlag(flags, kIsUpperRightMoveInvalid), upper_right_index);

    } else {
        flags |= (IsOnEdge<Direction::kDownLeft>(figure_idx) | IsPieceAt(all_pieces, lower_left_index))
                 << kIsLowerLeftMoveInvalid;
        WriteMove(ReadFlag(flags, kIsLowerLeftMoveInvalid), lower_left_index);

        flags |= (IsOnEdge<Direction::kDownRight>(figure_idx) | IsPieceAt(all_pieces, lower_right_index))
                 << kIsLowerRightMoveInvalid;
        WriteMove(ReadFlag(flags, kIsLowerRightMoveInvalid), lower_right_index);
    }
}

template <Turn turn>
__device__ __forceinline__ void TryCapture(
    board_index_t figure_idx, board_t all_pieces, board_t enemy_pieces, move_t* out_moves, u8& out_num_moves,
    move_flags_t& out_capture_mask, move_flags_t& per_board_flags, u8& flags
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

    auto WriteCapture = [&](bool is_capture_invalid, board_index_t to_idx) {
        const move_t mv = EncodeMove(figure_idx, to_idx);
        is_capture_invalid |= !ReadFlag(flags, kIsPieceOnBoardFlagIndex);
        if (!is_capture_invalid) {
            out_moves[out_num_moves] = mv;
            out_capture_mask |= (1 << out_num_moves);
            out_num_moves++;
            per_board_flags |= (1 << MoveFlagsConstants::kMoveFound);
            per_board_flags |= (1 << MoveFlagsConstants::kCaptureFound);
        }
    };

    WriteCapture(ReadFlag(flags, kIsUpperLeftMoveInvalid), ul_jump);
    WriteCapture(ReadFlag(flags, kIsUpperRightMoveInvalid), ur_jump);
    WriteCapture(ReadFlag(flags, kIsLowerLeftMoveInvalid), ll_jump);
    WriteCapture(ReadFlag(flags, kIsLowerRightMoveInvalid), lr_jump);
}

__device__ __forceinline__ void TryKingMoves(
    board_index_t figure_idx, board_t all_pieces, board_t enemy_pieces, board_t /* kings */, move_t* out_moves,
    u8& out_num_moves, move_flags_t& out_capture_mask, move_flags_t& per_board_flags, u8& flags
)
{
    TryDiagonal<Direction::kUpLeft>(
        figure_idx, all_pieces, enemy_pieces, figure_idx, out_moves, out_num_moves, out_capture_mask, per_board_flags,
        flags
    );
    flags &= kOnlyIsPieceOnBoardMask;

    TryDiagonal<Direction::kUpRight>(
        figure_idx, all_pieces, enemy_pieces, figure_idx, out_moves, out_num_moves, out_capture_mask, per_board_flags,
        flags
    );
    flags &= kOnlyIsPieceOnBoardMask;

    TryDiagonal<Direction::kDownLeft>(
        figure_idx, all_pieces, enemy_pieces, figure_idx, out_moves, out_num_moves, out_capture_mask, per_board_flags,
        flags
    );
    flags &= kOnlyIsPieceOnBoardMask;

    TryDiagonal<Direction::kDownRight>(
        figure_idx, all_pieces, enemy_pieces, figure_idx, out_moves, out_num_moves, out_capture_mask, per_board_flags,
        flags
    );
    flags &= kOnlyIsPieceOnBoardMask;
}

template <Direction direction>
__device__ __forceinline__ void TryDiagonal(
    board_index_t figure_idx, board_t all_pieces, board_t enemy_pieces, board_index_t start_idx, move_t* out_moves,
    u8& out_num_moves, move_flags_t& out_capture_mask, move_flags_t& per_board_flags, u8& flags
)
{
    static constexpr u8 kContinueFlagIndex    = 1;
    static constexpr u8 kInvalidNextFlagIndex = 2;
    static constexpr u8 kInvalidJumpFlagIndex = 3;

    flags &= kOnlyIsPieceOnBoardMask;

    // If not on board or on edge, we stop
    const bool immediate_stop = !ReadFlag(flags, kIsPieceOnBoardFlagIndex) || IsOnEdge<direction>(start_idx);
    flags |= (!immediate_stop) << kContinueFlagIndex;

    board_index_t current_idx = start_idx;
    bool capturing            = false;

    auto WriteKingMove = [&](bool is_capture_invalid, bool is_capture, board_index_t to_idx) {
        is_capture_invalid |= !ReadFlag(flags, kIsPieceOnBoardFlagIndex);
        if (!is_capture_invalid) {
            move_t mv                = EncodeMove(figure_idx, to_idx);
            out_moves[out_num_moves] = mv;
            // If it's a capture, set the relevant bit
            if (is_capture) {
                out_capture_mask |= (1 << out_num_moves);
                per_board_flags |= (1 << MoveFlagsConstants::kCaptureFound);
            }
            per_board_flags |= (1 << MoveFlagsConstants::kMoveFound);
            out_num_moves++;
        }
    };

    while (ReadFlag(flags, kContinueFlagIndex)) {
        board_index_t next_idx      = GetAdjacentIndex<direction>(current_idx);
        board_index_t next_jump_idx = GetAdjacentIndex<direction>(next_idx);

        // Attempt a simple move
        flags |= (IsOnEdge<direction>(current_idx) | IsPieceAt(all_pieces, next_idx)) << kInvalidNextFlagIndex;
        WriteKingMove(ReadFlag(flags, kInvalidNextFlagIndex), capturing, next_idx);

        // Attempt a capture
        flags |= (IsOnEdge<direction>(current_idx) | IsOnEdge<direction>(next_idx) |
                  !IsPieceAt(enemy_pieces, next_idx) | IsPieceAt(all_pieces, next_jump_idx))
                 << kInvalidJumpFlagIndex;
        const bool can_capture = !ReadFlag(flags, kInvalidJumpFlagIndex);
        capturing              = capturing || can_capture;

        current_idx = next_idx;

        const bool next_stop = IsOnEdge<direction>(next_idx) ||
                               (ReadFlag(flags, kInvalidNextFlagIndex) && ReadFlag(flags, kInvalidJumpFlagIndex));
        flags &= kOnlyIsPieceOnBoardMask;  // keep only piece presence
        flags |= (!next_stop << kContinueFlagIndex);
    }
}

}  // namespace checkers::gpu::move_gen

#endif  // MCTS_CHECKERS_INCLUDE_MOVE_GENERATION_TPP_
