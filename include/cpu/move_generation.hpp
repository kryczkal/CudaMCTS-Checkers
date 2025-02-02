#ifndef MCTS_CHECKERS_INCLUDE_CPU_MOVE_GENERATION_HPP_
#define MCTS_CHECKERS_INCLUDE_CPU_MOVE_GENERATION_HPP_

#include "algorithm"
#include "execution"
#include "numeric"
#include "vector"

#include "cpu/board_helpers.hpp"

namespace checkers::cpu::move_gen
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
static inline void TryMoveForward(
    board_index_t figure_idx, board_t all_pieces, move_t *out_moves, u8 &out_num_moves, move_flags_t &out_capture_mask,
    move_flags_t &per_board_flags, u8 &flags
)
{
    UNUSED(out_moves);
    UNUSED(out_num_moves);
    UNUSED(out_capture_mask);
    UNUSED(per_board_flags);

    using namespace checkers::cpu;
    using namespace checkers::cpu::move_gen;

    const board_index_t ul = GetAdjacentIndex<Direction::kUpLeft>(figure_idx);
    const board_index_t ur = GetAdjacentIndex<Direction::kUpRight>(figure_idx);
    const board_index_t ll = GetAdjacentIndex<Direction::kDownLeft>(figure_idx);
    const board_index_t lr = GetAdjacentIndex<Direction::kDownRight>(figure_idx);

    // Helper for writing a move if not invalid
    auto WriteMove = [&](bool isInvalid, board_index_t to_idx) {
        isInvalid |= !ReadFlag(flags, kIsPieceOnBoardFlagIndex);
        if (!isInvalid) {
            out_moves[out_num_moves] = EncodeMove(figure_idx, to_idx);
            out_num_moves++;
            per_board_flags |= (1 << MoveFlagsConstants::kMoveFound);
        }
    };

    if constexpr (turn == Turn::kWhite) {
        flags |= (IsOnEdge<Direction::kUpLeft>(figure_idx) || IsPieceAt(all_pieces, ul)) << kIsUpperLeftMoveInvalid;
        flags |= (IsOnEdge<Direction::kUpRight>(figure_idx) || IsPieceAt(all_pieces, ur)) << kIsUpperRightMoveInvalid;

        WriteMove(ReadFlag(flags, kIsUpperLeftMoveInvalid), ul);
        WriteMove(ReadFlag(flags, kIsUpperRightMoveInvalid), ur);

    } else {
        flags |= (IsOnEdge<Direction::kDownLeft>(figure_idx) || IsPieceAt(all_pieces, ll)) << kIsLowerLeftMoveInvalid;
        flags |= (IsOnEdge<Direction::kDownRight>(figure_idx) || IsPieceAt(all_pieces, lr)) << kIsLowerRightMoveInvalid;

        WriteMove(ReadFlag(flags, kIsLowerLeftMoveInvalid), ll);
        WriteMove(ReadFlag(flags, kIsLowerRightMoveInvalid), lr);
    }
}

/**
 * @brief Attempt captures for a single piece on CPU.
 */
template <Turn turn>
static inline void TryCapture(
    board_index_t figure_idx, board_t all_pieces, board_t enemy_pieces, move_t *out_moves, u8 &out_num_moves,
    move_flags_t &out_capture_mask, move_flags_t &per_board_flags, u8 &flags
)
{
    using namespace checkers::cpu::move_gen;
    using namespace checkers::cpu;

    const board_index_t ul = GetAdjacentIndex<Direction::kUpLeft>(figure_idx);
    const board_index_t ur = GetAdjacentIndex<Direction::kUpRight>(figure_idx);
    const board_index_t ll = GetAdjacentIndex<Direction::kDownLeft>(figure_idx);
    const board_index_t lr = GetAdjacentIndex<Direction::kDownRight>(figure_idx);

    const board_index_t ul_jump = GetAdjacentIndex<Direction::kUpLeft>(ul);
    const board_index_t ur_jump = GetAdjacentIndex<Direction::kUpRight>(ur);
    const board_index_t ll_jump = GetAdjacentIndex<Direction::kDownLeft>(ll);
    const board_index_t lr_jump = GetAdjacentIndex<Direction::kDownRight>(lr);

    // Each direction might be invalid if on edge, or if missing an enemy piece, or if the jump square is occupied.
    flags |= (IsOnEdge<Direction::kUpLeft>(figure_idx) || IsOnEdge<Direction::kUpLeft>(ul) ||
              !IsPieceAt(enemy_pieces, ul) || IsPieceAt(all_pieces, ul_jump))
             << kIsUpperLeftMoveInvalid;
    flags |= (IsOnEdge<Direction::kUpRight>(figure_idx) || IsOnEdge<Direction::kUpRight>(ur) ||
              !IsPieceAt(enemy_pieces, ur) || IsPieceAt(all_pieces, ur_jump))
             << kIsUpperRightMoveInvalid;
    flags |= (IsOnEdge<Direction::kDownLeft>(figure_idx) || IsOnEdge<Direction::kDownLeft>(ll) ||
              !IsPieceAt(enemy_pieces, ll) || IsPieceAt(all_pieces, ll_jump))
             << kIsLowerLeftMoveInvalid;
    flags |= (IsOnEdge<Direction::kDownRight>(figure_idx) || IsOnEdge<Direction::kDownRight>(lr) ||
              !IsPieceAt(enemy_pieces, lr) || IsPieceAt(all_pieces, lr_jump))
             << kIsLowerRightMoveInvalid;

    // Helper for writing a capturing move
    auto WriteCapture = [&](bool invalid, board_index_t to_idx) {
        invalid |= !ReadFlag(flags, kIsPieceOnBoardFlagIndex);
        if (!invalid) {
            move_t mv                = EncodeMove(figure_idx, to_idx);
            out_moves[out_num_moves] = mv;
            // Mark it as capture
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

/**
 * @brief Attempt king moves for a single piece on CPU.
 */
template <Direction direction>
static inline void TryDiagonal(
    board_index_t figure_idx, board_t all_pieces, board_t enemy_pieces, board_index_t start_idx, move_t *out_moves,
    u8 &out_num_moves, move_flags_t &out_capture_mask, move_flags_t &per_board_flags, u8 &flags
)
{
    using namespace checkers::cpu;
    using namespace checkers::cpu::move_gen;

    static constexpr u8 kIsPieceOnBoardFlagIndex = 0;
    static constexpr u8 kContinueFlagIndex       = 1;
    static constexpr u8 kInvalidNextFlagIndex    = 2;
    static constexpr u8 kInvalidJumpFlagIndex    = 3;

    const bool immediate_stop = !ReadFlag(flags, kIsPieceOnBoardFlagIndex) || IsOnEdge<direction>(start_idx);
    flags |= (!immediate_stop) << kContinueFlagIndex;

    board_index_t current_idx = start_idx;
    bool has_captured_so_far  = false;

    auto WriteKingMove = [&](bool is_capture_invalid, bool is_capture, board_index_t to_idx) {
        is_capture_invalid |= !ReadFlag(flags, kIsPieceOnBoardFlagIndex);
        if (!is_capture_invalid) {
            move_t mv                = EncodeMove(figure_idx, to_idx);
            out_moves[out_num_moves] = mv;
            if (is_capture) {
                out_capture_mask |= (1 << out_num_moves);
                has_captured_so_far = true;
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
        WriteKingMove(ReadFlag(flags, kInvalidNextFlagIndex), has_captured_so_far, next_idx);

        // Attempt a capture
        flags |= (IsOnEdge<direction>(current_idx) | IsOnEdge<direction>(next_idx) |
                  !IsPieceAt(enemy_pieces, next_idx) | IsPieceAt(all_pieces, next_jump_idx))
                 << kInvalidJumpFlagIndex;
        const bool can_capture = !ReadFlag(flags, kInvalidJumpFlagIndex);
        has_captured_so_far    = has_captured_so_far || can_capture;

        current_idx = next_idx;

        const bool next_stop = IsOnEdge<direction>(next_idx) ||
                               (ReadFlag(flags, kInvalidNextFlagIndex) && ReadFlag(flags, kInvalidJumpFlagIndex));

        flags &= kOnlyIsPieceOnBoardMask;  // keep only piece presence
        flags |= (!next_stop << kContinueFlagIndex);
    }
}

static inline void TryKingMoves(
    board_index_t figure_idx, board_t all_pieces, board_t enemy_pieces, board_t /*kings*/, move_t *out_moves,
    u8 &out_num_moves, move_flags_t &out_capture_mask, move_flags_t &per_board_flags, u8 &flags
)
{
    using namespace checkers::cpu::move_gen;

    TryDiagonal<Direction::kUpLeft>(
        figure_idx, all_pieces, enemy_pieces, figure_idx, out_moves, out_num_moves, out_capture_mask, per_board_flags,
        flags
    );
    TryDiagonal<Direction::kUpRight>(
        figure_idx, all_pieces, enemy_pieces, figure_idx, out_moves, out_num_moves, out_capture_mask, per_board_flags,
        flags
    );
    TryDiagonal<Direction::kDownLeft>(
        figure_idx, all_pieces, enemy_pieces, figure_idx, out_moves, out_num_moves, out_capture_mask, per_board_flags,
        flags
    );
    TryDiagonal<Direction::kDownRight>(
        figure_idx, all_pieces, enemy_pieces, figure_idx, out_moves, out_num_moves, out_capture_mask, per_board_flags,
        flags
    );
}

/**
 * @brief CPU function that tries to generate all moves for a single piece (normal or king),
 *        setting bits for capturing moves, etc.
 */
template <Turn turn>
static inline void GenerateMovesForSinglePiece(
    board_index_t figure_idx, board_t white_pieces, board_t black_pieces, board_t kings,
    move_t *out_moves,  // size >= kNumMaxMovesPerPiece
    u8 &out_move_count, move_flags_t &out_capture_mask, move_flags_t &per_board_flags
)
{
    using checkers::cpu::ReadFlag;

    // We'll track partial flags in a local variable
    u8 flags = 0;

    // Identify if this piece is from the current turn
    board_t current_player_pieces = (turn == Turn::kWhite) ? white_pieces : black_pieces;

    // If not a piece for this turn, do nothing
    const bool is_piece = ReadFlag(current_player_pieces, figure_idx);
    if (!is_piece) {
        return;
    }

    board_t all_pieces   = white_pieces | black_pieces;
    board_t enemy_pieces = (turn == Turn::kWhite) ? black_pieces : white_pieces;

    // Mark the piece as on board in "flags"
    flags |= (1 << kIsPieceOnBoardFlagIndex);

    // Count how many moves we've added so far
    u8 num_moves = 0;

    const bool is_king_piece = ReadFlag(kings, figure_idx);
    if (!is_king_piece) {
        // normal piece
        TryMoveForward<turn>(figure_idx, all_pieces, out_moves, num_moves, out_capture_mask, per_board_flags, flags);
        // keep only the "piece on board" bit
        flags &= kOnlyIsPieceOnBoardMask;

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

    out_move_count = num_moves;
}

template <Turn turn>
void GenerateMoves(
    const board_t *d_whites, const board_t *d_blacks, const board_t *d_kings, move_t *d_moves, u8 *d_move_counts,
    move_flags_t *d_move_capture_mask, move_flags_t *d_per_board_move_flags, const u64 n_boards
)
{
    // Create a container with board indices [0, n_boards)
    std::vector<u64> board_indices(n_boards);
    std::iota(board_indices.begin(), board_indices.end(), 0);

    // Use parallel execution to process each board concurrently.
    std::for_each(std::execution::par, board_indices.begin(), board_indices.end(), [=](u64 board_idx) {
        // Retrieve the piece bitmasks for this board.
        board_t white_pieces = d_whites[board_idx];
        board_t black_pieces = d_blacks[board_idx];
        board_t kings        = d_kings[board_idx];

        // Process each piece on the board sequentially.
        for (u64 figure_idx_int = 0; figure_idx_int < BoardConstants::kBoardSize; figure_idx_int++) {
            const board_index_t figure_idx = static_cast<board_index_t>(figure_idx_int);

            // Calculate the base index for the moves of the current piece.
            const u64 base_moves_idx =
                (board_idx * BoardConstants::kBoardSize + figure_idx) * static_cast<u64>(kNumMaxMovesPerPiece);

            move_t *out_moves_ptr          = &d_moves[base_moves_idx];
            u8 *out_move_count_ptr         = &d_move_counts[board_idx * BoardConstants::kBoardSize + figure_idx];
            move_flags_t *out_capture_mask = &d_move_capture_mask[board_idx * BoardConstants::kBoardSize + figure_idx];
            move_flags_t &per_board_flags  = d_per_board_move_flags[board_idx];

            // Clear the outputs for this piece.
            *out_move_count_ptr = 0;
            *out_capture_mask   = 0;

            // Generate moves for this single piece.
            GenerateMovesForSinglePiece<turn>(
                figure_idx, white_pieces, black_pieces, kings, out_moves_ptr, *out_move_count_ptr, *out_capture_mask,
                per_board_flags
            );
        }
    });
}
}  // namespace checkers::cpu::move_gen

#endif  // MCTS_CHECKERS_INCLUDE_CPU_MOVE_GENERATION_HPP_
