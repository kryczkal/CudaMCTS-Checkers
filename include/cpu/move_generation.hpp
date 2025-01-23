#ifndef MCTS_CHECKERS_INCLUDE_CPU_MOVE_GENERATION_HPP_
#define MCTS_CHECKERS_INCLUDE_CPU_MOVE_GENERATION_HPP_

#include "cpu/board_helpers.hpp"

namespace checkers::cpu::move_gen
{

// TODO: Split into .tpp

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
        isInvalid |= !ReadFlag(flags, 0);  // 0 is kIsPieceOnBoardFlagIndex
        if (!isInvalid) {
            out_moves[out_num_moves] = EncodeMove(figure_idx, to_idx);
            out_num_moves++;
            per_board_flags |= (1 << MoveFlagsConstants::kMoveFound);
        }
    };

    if constexpr (turn == Turn::kWhite) {
        bool ul_invalid = (IsOnEdge<Direction::kUpLeft>(figure_idx) || IsPieceAt(all_pieces, ul));
        bool ur_invalid = (IsOnEdge<Direction::kUpRight>(figure_idx) || IsPieceAt(all_pieces, ur));

        // store bits in flags
        if (ul_invalid) {
            flags |= (1 << 1);  // kIsUpperLeftMoveInvalid  TODO: add these as constants
        }
        if (ur_invalid) {
            flags |= (1 << 2);  // kIsUpperRightMoveInvalid
        }

        WriteMove(ul_invalid, ul);
        WriteMove(ur_invalid, ur);
    } else {
        // turn == Turn::kBlack
        bool ll_invalid = (IsOnEdge<Direction::kDownLeft>(figure_idx) || IsPieceAt(all_pieces, ll));
        bool lr_invalid = (IsOnEdge<Direction::kDownRight>(figure_idx) || IsPieceAt(all_pieces, lr));

        if (ll_invalid) {
            flags |= (1 << 3);  // kIsLowerLeftMoveInvalid
        }
        if (lr_invalid) {
            flags |= (1 << 4);  // kIsLowerRightMoveInvalid
        }

        WriteMove(ll_invalid, ll);
        WriteMove(lr_invalid, lr);
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

    // We replicate the bit-flag logic in a straightforward manner:
    // Each direction might be invalid if on edge, or if missing an enemy piece, or if the jump square is occupied.
    bool ul_invalid =
        (IsOnEdge<Direction::kUpLeft>(figure_idx) || IsOnEdge<Direction::kUpLeft>(ul) || !IsPieceAt(enemy_pieces, ul) ||
         IsPieceAt(all_pieces, ul_jump));
    bool ur_invalid =
        (IsOnEdge<Direction::kUpRight>(figure_idx) || IsOnEdge<Direction::kUpRight>(ur) ||
         !IsPieceAt(enemy_pieces, ur) || IsPieceAt(all_pieces, ur_jump));
    bool ll_invalid =
        (IsOnEdge<Direction::kDownLeft>(figure_idx) || IsOnEdge<Direction::kDownLeft>(ll) ||
         !IsPieceAt(enemy_pieces, ll) || IsPieceAt(all_pieces, ll_jump));
    bool lr_invalid =
        (IsOnEdge<Direction::kDownRight>(figure_idx) || IsOnEdge<Direction::kDownRight>(lr) ||
         !IsPieceAt(enemy_pieces, lr) || IsPieceAt(all_pieces, lr_jump));

    // write bits into flags for debug if needed
    if (ul_invalid) {
        flags |= (1 << 1);
    }
    if (ur_invalid) {
        flags |= (1 << 2);
    }
    if (ll_invalid) {
        flags |= (1 << 3);
    }
    if (lr_invalid) {
        flags |= (1 << 4);
    }

    // Helper for writing a capturing move
    auto WriteCapture = [&](bool invalid, board_index_t to_idx) {
        invalid |= !ReadFlag(flags, 0);  // if piece not on board or already flagged
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

    WriteCapture(ul_invalid, ul_jump);
    WriteCapture(ur_invalid, ur_jump);
    WriteCapture(ll_invalid, ll_jump);
    WriteCapture(lr_invalid, lr_jump);
}

/**
 * @brief Attempt king moves for a single piece on CPU.
 */
static inline void TryDiagonal(
    checkers::cpu::move_gen::Direction direction, board_index_t figure_idx, board_t all_pieces, board_t enemy_pieces,
    board_index_t start_idx, move_t *out_moves, u8 &out_num_moves, move_flags_t &out_capture_mask,
    move_flags_t &per_board_flags, u8 &flags
)
{
    using namespace checkers::cpu;
    using namespace checkers::cpu::move_gen;
    // We replicate the logic from the GPU's TryDiagonal.

    static constexpr u8 kIsPieceOnBoardIndex = 0;
    if (!ReadFlag(flags, kIsPieceOnBoardIndex)) {
        return;  // the piece is not present for some reason
    }

    bool stop                 = false;
    board_index_t current_idx = start_idx;
    bool hasCapturedSoFar     = false;  // track if we do a capturing move

    while (!stop) {
        // next in diagonal
        if (IsOnEdge<Direction::kUpLeft>(current_idx) && direction == Direction::kUpLeft) {
            break;
        }
        if (IsOnEdge<Direction::kUpRight>(current_idx) && direction == Direction::kUpRight) {
            break;
        }
        if (IsOnEdge<Direction::kDownLeft>(current_idx) && direction == Direction::kDownLeft) {
            break;
        }
        if (IsOnEdge<Direction::kDownRight>(current_idx) && direction == Direction::kDownRight) {
            break;
        }

        board_index_t next_idx =
            (direction == Direction::kUpLeft)     ? GetAdjacentIndex<Direction::kUpLeft>(current_idx)
            : (direction == Direction::kUpRight)  ? GetAdjacentIndex<Direction::kUpRight>(current_idx)
            : (direction == Direction::kDownLeft) ? GetAdjacentIndex<Direction::kDownLeft>(current_idx)
                                                  : GetAdjacentIndex<Direction::kDownRight>(current_idx);

        // 1) Try normal move
        if (!IsPieceAt(all_pieces, next_idx)) {
            // valid normal move
            move_t mv                = EncodeMove(figure_idx, next_idx);
            out_moves[out_num_moves] = mv;
            if (hasCapturedSoFar) {
                // if we had captured before, this move is also a capture
                out_capture_mask |= (1 << out_num_moves);
                per_board_flags |= (1 << MoveFlagsConstants::kCaptureFound);
            }
            out_num_moves++;
            per_board_flags |= (1 << MoveFlagsConstants::kMoveFound);
        }

        // 2) Try capture
        // Check next_idx has enemy piece, and next jump is free.
        if (IsPieceAt(enemy_pieces, next_idx)) {
            // jump square
            if (!IsOnEdge<Direction::kUpLeft>(next_idx) && direction == Direction::kUpLeft) {
                board_index_t jump_idx = GetAdjacentIndex<Direction::kUpLeft>(next_idx);
                if (!IsPieceAt(all_pieces, jump_idx) && !IsOnEdge<Direction::kUpLeft>(next_idx)) {
                    // can capture
                    move_t mv                = EncodeMove(figure_idx, jump_idx);
                    out_moves[out_num_moves] = mv;
                    out_capture_mask |= (1 << out_num_moves);
                    hasCapturedSoFar = true;
                    out_num_moves++;
                    per_board_flags |= (1 << MoveFlagsConstants::kMoveFound);
                    per_board_flags |= (1 << MoveFlagsConstants::kCaptureFound);
                }
            }
            // same for the other directions...
            else if (!IsOnEdge<Direction::kUpRight>(next_idx) && direction == Direction::kUpRight) {
                board_index_t jump_idx = GetAdjacentIndex<Direction::kUpRight>(next_idx);
                if (!IsPieceAt(all_pieces, jump_idx) && !IsOnEdge<Direction::kUpRight>(next_idx)) {
                    move_t mv                = EncodeMove(figure_idx, jump_idx);
                    out_moves[out_num_moves] = mv;
                    out_capture_mask |= (1 << out_num_moves);
                    hasCapturedSoFar = true;
                    out_num_moves++;
                    per_board_flags |= (1 << MoveFlagsConstants::kMoveFound);
                    per_board_flags |= (1 << MoveFlagsConstants::kCaptureFound);
                }
            } else if (!IsOnEdge<Direction::kDownLeft>(next_idx) && direction == Direction::kDownLeft) {
                board_index_t jump_idx = GetAdjacentIndex<Direction::kDownLeft>(next_idx);
                if (!IsPieceAt(all_pieces, jump_idx) && !IsOnEdge<Direction::kDownLeft>(next_idx)) {
                    move_t mv                = EncodeMove(figure_idx, jump_idx);
                    out_moves[out_num_moves] = mv;
                    out_capture_mask |= (1 << out_num_moves);
                    hasCapturedSoFar = true;
                    out_num_moves++;
                    per_board_flags |= (1 << MoveFlagsConstants::kMoveFound);
                    per_board_flags |= (1 << MoveFlagsConstants::kCaptureFound);
                }
            } else if (!IsOnEdge<Direction::kDownRight>(next_idx) && direction == Direction::kDownRight) {
                board_index_t jump_idx = GetAdjacentIndex<Direction::kDownRight>(next_idx);
                if (!IsPieceAt(all_pieces, jump_idx) && !IsOnEdge<Direction::kDownRight>(next_idx)) {
                    move_t mv                = EncodeMove(figure_idx, jump_idx);
                    out_moves[out_num_moves] = mv;
                    out_capture_mask |= (1 << out_num_moves);
                    hasCapturedSoFar = true;
                    out_num_moves++;
                    per_board_flags |= (1 << MoveFlagsConstants::kMoveFound);
                    per_board_flags |= (1 << MoveFlagsConstants::kCaptureFound);
                }
            }

            // Once we detect an enemy, we typically break scanning further squares
            // (like normal checkers king logic). For simplicity, let's break here.
            break;
        }

        // If next square is occupied by your side, stop
        if (IsPieceAt(all_pieces, next_idx)) {
            break;
        }

        // otherwise continue scanning
        current_idx = next_idx;
    }
}

static inline void TryKingMoves(
    board_index_t figure_idx, board_t all_pieces, board_t enemy_pieces, board_t /*kings*/, move_t *out_moves,
    u8 &out_num_moves, move_flags_t &out_capture_mask, move_flags_t &per_board_flags, u8 &flags
)
{
    using namespace checkers::cpu::move_gen;

    // We'll just call TryDiagonal in each direction
    TryDiagonal(
        Direction::kUpLeft, figure_idx, all_pieces, enemy_pieces, figure_idx, out_moves, out_num_moves,
        out_capture_mask, per_board_flags, flags
    );
    TryDiagonal(
        Direction::kUpRight, figure_idx, all_pieces, enemy_pieces, figure_idx, out_moves, out_num_moves,
        out_capture_mask, per_board_flags, flags
    );
    TryDiagonal(
        Direction::kDownLeft, figure_idx, all_pieces, enemy_pieces, figure_idx, out_moves, out_num_moves,
        out_capture_mask, per_board_flags, flags
    );
    TryDiagonal(
        Direction::kDownRight, figure_idx, all_pieces, enemy_pieces, figure_idx, out_moves, out_num_moves,
        out_capture_mask, per_board_flags, flags
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
    board_t all_pieces            = white_pieces | black_pieces;
    board_t enemy_pieces          = (turn == Turn::kWhite) ? black_pieces : white_pieces;

    // If not a piece for this turn, do nothing
    bool is_piece = (current_player_pieces >> figure_idx) & 1U;
    if (!is_piece) {
        return;
    }

    // Mark the piece as on board in "flags"
    flags |= (1 << 0);

    // Count how many moves we've added so far
    u8 num_moves = 0;

    bool is_king_piece = (kings >> figure_idx) & 1U;
    if (!is_king_piece) {
        // normal piece
        TryMoveForward<turn>(figure_idx, all_pieces, out_moves, num_moves, out_capture_mask, per_board_flags, flags);
        // keep only the "piece on board" bit
        flags &= 1;

        TryCapture<turn>(
            figure_idx, all_pieces, enemy_pieces, out_moves, num_moves, out_capture_mask, per_board_flags, flags
        );
        flags &= 1;
    } else {
        // it's a king
        TryKingMoves(
            figure_idx, all_pieces, enemy_pieces, kings, out_moves, num_moves, out_capture_mask, per_board_flags, flags
        );
    }

    out_move_count = num_moves;
}

}  // namespace checkers::cpu::move_gen

#endif  // MCTS_CHECKERS_INCLUDE_CPU_MOVE_GENERATION_HPP_
