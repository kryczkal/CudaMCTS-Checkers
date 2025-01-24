#include "game/checkers_engine.hpp"
#include <algorithm>
#include <iostream>
#include "cpu/board_helpers.hpp"
#include "cpu/launchers.hpp"  // For CPU-based generation/apply
#include "cpu/move_generation.hpp"
#include "cuda/launchers.cuh"  // If you want GPU-based generation/apply
#include "cuda/move_generation.tpp"

namespace checkers
{

CheckersEngine::CheckersEngine(const checkers::cpu::Board &board, checkers::Turn turn)
    : board_(board), current_turn_(turn)
{
    last_moves_.h_moves.resize(
        checkers::MoveGenResult::kTotalSquares * checkers::MoveGenResult::kMovesPerPiece, checkers::kInvalidMove
    );
    last_moves_.h_move_counts.resize(checkers::MoveGenResult::kTotalSquares, 0);
    last_moves_.h_capture_masks.resize(checkers::MoveGenResult::kTotalSquares, 0);
    last_moves_.h_per_board_flags.resize(1, 0);
}

checkers::cpu::Board CheckersEngine::GetBoard() const { return board_; }

checkers::Turn CheckersEngine::GetCurrentTurn() const { return current_turn_; }

void CheckersEngine::GenerateMovesCPU()
{
    auto results = checkers::cpu::launchers::HostGenerateMoves({board_}, current_turn_);
    last_moves_  = results[0];
    has_no_moves_ =
        checkers::gpu::ReadFlag(last_moves_.h_per_board_flags[0], checkers::MoveFlagsConstants::kMoveFound) == 0;
}

void CheckersEngine::GenerateMovesGPU()
{
    auto results = checkers::gpu::launchers::HostGenerateMoves({board_}, current_turn_);
    last_moves_  = results[0];
    has_no_moves_ =
        checkers::cpu::ReadFlag(last_moves_.h_per_board_flags[0], checkers::MoveFlagsConstants::kMoveFound) == 0;
}

const checkers::MoveGenResult &CheckersEngine::GetLastMoveGenResult() const { return last_moves_; }

bool CheckersEngine::HasNoMoves() const { return has_no_moves_; }

bool CheckersEngine::ApplyMove(checkers::move_t move, bool do_validate)
{
    // Optionally re-generate all moves for validation
    if (do_validate) {
        GenerateMovesCPU();
        if (has_no_moves_) {
            return false;
        }
        if (!IsMoveValid(move)) {
            return false;
        }
    }

    // Apply the move
    auto updated = checkers::cpu::launchers::HostApplyMoves({board_}, {move});
    board_       = updated[0];

    // Handle promotion
    board_.kings |= (board_.white & checkers::BoardConstants::kTopBoardEdgeMask);
    board_.kings |= (board_.black & checkers::BoardConstants::kBottomBoardEdgeMask);

    // If it was a capture, reset the non-reversible counter; otherwise increment.
    bool was_capture = IsCaptureMove(move);
    checkers::board_index_t from_sq =
        checkers::cpu::move_gen::DecodeMove<checkers::cpu::move_gen::MovePart::From>(move);
    UpdateNonReversibleCount(was_capture, from_sq);
    SwitchTurnIfNeeded(move);

    return true;
}

void CheckersEngine::SwitchTurnIfNeeded(checkers::move_t last_move)
{
    bool was_capture = IsCaptureMove(last_move);
    if (!was_capture) {
        // Normal move => simply switch
        current_turn_ = (current_turn_ == Turn::kWhite) ? Turn::kBlack : Turn::kWhite;
        return;
    }

    // -------------------------------------------
    //  If the last move was a capture, see if the
    //  piece can continue capturing from 'to_sq'.
    // -------------------------------------------
    board_index_t to_sq = cpu::move_gen::DecodeMove<cpu::move_gen::MovePart::To>(last_move);

    // We'll generate moves for that single square only,
    // resetting last_moves_ to store partial results.
    std::fill(last_moves_.h_move_counts.begin(), last_moves_.h_move_counts.end(), 0);
    std::fill(last_moves_.h_capture_masks.begin(), last_moves_.h_capture_masks.end(), 0);
    std::fill(last_moves_.h_moves.begin(), last_moves_.h_moves.end(), kInvalidMove);
    last_moves_.h_per_board_flags[0] = 0;

    // local references
    board_t w = board_.white;
    board_t b = board_.black;
    board_t k = board_.kings;

    // Perform a single-piece move generation
    if (current_turn_ == Turn::kWhite) {
        cpu::move_gen::GenerateMovesForSinglePiece<Turn::kWhite>(
            to_sq, w, b, k, &last_moves_.h_moves[to_sq * kNumMaxMovesPerPiece], last_moves_.h_move_counts[to_sq],
            last_moves_.h_capture_masks[to_sq], last_moves_.h_per_board_flags[0]
        );
    } else {
        cpu::move_gen::GenerateMovesForSinglePiece<Turn::kBlack>(
            to_sq, b, w, k, &last_moves_.h_moves[to_sq * kNumMaxMovesPerPiece], last_moves_.h_move_counts[to_sq],
            last_moves_.h_capture_masks[to_sq], last_moves_.h_per_board_flags[0]
        );
    }

    // -------------------------------------------
    // Check if further capture is possible
    // -------------------------------------------
    bool more_captures_available = cpu::ReadFlag(last_moves_.h_per_board_flags[0], MoveFlagsConstants::kCaptureFound);

    if (!more_captures_available) {
        // No additional capture => switch sides
        current_turn_ = (current_turn_ == Turn::kWhite) ? Turn::kBlack : Turn::kWhite;
    }
    // else remain on same side to do another capture
}

GameResult CheckersEngine::CheckGameResult() const
{
    // If no moves => other side wins
    if (has_no_moves_) {
        // The side that cannot move loses
        if (current_turn_ == checkers::Turn::kWhite) {
            return GameResult::kBlackWin;
        } else {
            return GameResult::kWhiteWin;
        }
    }
    // If no pieces for black => white wins
    if (board_.black == 0) {
        return GameResult::kWhiteWin;
    }
    // If no pieces for white => black wins
    if (board_.white == 0) {
        return GameResult::kBlackWin;
    }
    // 40-move rule
    if (non_reversible_count_ >= 40) {
        return GameResult::kDraw;
    }
    // Not done
    return GameResult::kInProgress;
}

void CheckersEngine::UpdateNonReversibleCount(bool was_capture, checkers::board_index_t from_sq)
{
    const bool kFromIsKing = checkers::cpu::ReadFlag(board_.kings, from_sq);

    if (was_capture || !kFromIsKing) {
        non_reversible_count_ = 0;
    } else {
        non_reversible_count_++;
    }
}

bool CheckersEngine::IsMoveValid(checkers::move_t mv) const
{
    const bool kBoardHasCapture =
        checkers::cpu::ReadFlag(last_moves_.h_per_board_flags[0], checkers::MoveFlagsConstants::kCaptureFound);
    checkers::board_index_t from_sq = checkers::gpu::move_gen::DecodeMove<checkers::gpu::move_gen::MovePart::From>(mv);

    // Is in bounds?
    if (from_sq >= checkers::BoardConstants::kBoardSize) {
        return false;
    }

    // How many sub-moves for that square?
    u8 count = last_moves_.h_move_counts[from_sq];
    if (count == 0)
        return false;

    const size_t kBaseIdx = from_sq * checkers::kNumMaxMovesPerPiece;

    for (u8 sub = 0; sub < count; sub++) {
        checkers::move_t candidate = last_moves_.h_moves[kBaseIdx + sub];
        if (candidate == mv) {
            // If captures exist for the board, ensure this move is also a capture
            if (kBoardHasCapture) {
                bool isCapture = checkers::cpu::ReadFlag(last_moves_.h_capture_masks[from_sq], sub);
                return isCapture;
            }
            return true;
        }
    }
    return false;
}

bool CheckersEngine::IsCaptureMove(checkers::move_t mv) const
{
    checkers::board_index_t from_sq = checkers::gpu::move_gen::DecodeMove<checkers::gpu::move_gen::MovePart::From>(mv);

    const size_t kBaseIdx = from_sq * checkers::kNumMaxMovesPerPiece;
    u8 count              = last_moves_.h_move_counts[from_sq];
    for (u8 i = 0; i < count; i++) {
        if (last_moves_.h_moves[kBaseIdx + i] == mv) {
            bool is_cap = ((last_moves_.h_capture_masks[from_sq] >> i) & 1U) != 0;
            return is_cap;
        }
    }
    return false;
}

}  // namespace checkers
