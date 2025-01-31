#include "game/checkers_engine.hpp"
#include <algorithm>
#include <iostream>

#include "common/checkers_defines.hpp"
#include "cpu/apply_move.hpp"
#include "cpu/board_helpers.hpp"
#include "cpu/launchers.hpp"
#include "cpu/move_generation.hpp"

namespace checkers
{

CheckersEngine::CheckersEngine(const checkers::cpu::Board &board, checkers::Turn turn)
    : board_(board), current_turn_(turn)
{
}

checkers::cpu::Board CheckersEngine::GetBoard() const { return board_; }

checkers::Turn CheckersEngine::GetCurrentTurn() const { return current_turn_; }

bool CheckersEngine::IsTerminal() const
{
    // If the game result says it's not in progress, it's terminal
    return (game_result_ != GameResult::kInProgress);
}

GameResult CheckersEngine::CheckGameResult()
{
    if (game_result_ != GameResult::kInProgress) {
        return game_result_;
    }

    // If either side has no pieces, the other side wins
    if (board_.white == 0) {
        game_result_ = GameResult::kBlackWin;
        return game_result_;
    }
    if (board_.black == 0) {
        game_result_ = GameResult::kWhiteWin;
        return game_result_;
    }

    // 40-move rule for a draw
    if (non_reversible_count_ >= 40) {
        game_result_ = GameResult::kDraw;
        return game_result_;
    }

    MoveGenResult result = GenerateMoves();
    if (!cpu::ReadFlag(result.h_per_board_flags[0], MoveFlagsConstants::kMoveFound)) {
        // The side with no moves loses
        if (current_turn_ == Turn::kWhite) {
            game_result_ = GameResult::kBlackWin;
            return game_result_;
        } else {
            game_result_ = GameResult::kWhiteWin;
            return game_result_;
        }
    }
    // Otherwise, not done
    return game_result_;
}

MoveGenResult CheckersEngine::GenerateMoves()
{
    using namespace checkers::cpu::move_gen;

    MoveGenResult result{};
    if (current_turn_ == Turn::kBlack) {
        cpu::move_gen::GenerateMoves<Turn::kBlack>(
            &board_.white, &board_.black, &board_.kings, result.h_moves.data(), result.h_move_counts.data(),
            result.h_capture_masks.data(), result.h_per_board_flags.data(), 1
        );
    } else {
        cpu::move_gen::GenerateMoves<Turn::kWhite>(
            &board_.white, &board_.black, &board_.kings, result.h_moves.data(), result.h_move_counts.data(),
            result.h_capture_masks.data(), result.h_per_board_flags.data(), 1
        );
    }
    if (!cpu::ReadFlag(result.h_per_board_flags[0], MoveFlagsConstants::kMoveFound)) {
        game_result_ = (current_turn_ == Turn::kWhite ? GameResult::kBlackWin : GameResult::kWhiteWin);
    }
    return result;
}

bool CheckersEngine::ApplyMove(move_t mv, bool validate)
{
    using namespace checkers::cpu::move_gen;
    using namespace checkers::cpu::apply_move;

    // Basic check if the move is feasible. We'll generate all possible single-step or single-jump moves.
    if (validate) {
        MoveGenResult result = GenerateMoves();
        auto it              = std::find(result.h_moves.begin(), result.h_moves.end(), mv);
        if (it == result.h_moves.end()) {
            return false;
        }
        // Check if the move is a capture if we have captures
        auto from_sq = DecodeMove<MovePart::From>(mv);
        if (cpu::ReadFlag(result.h_per_board_flags[0], MoveFlagsConstants::kCaptureFound)) {
            auto idx = (it - result.h_moves.begin()) % MoveGenResult::kMovesPerPiece;
            if (!cpu::ReadFlag(result.h_capture_masks[from_sq], idx)) {
                return false;
            }
        }
    }

    // We apply exactly one single-jump or step
    board_index_t from_sq = DecodeMove<MovePart::From>(mv);

    // Save old board to detect a capture
    board_t old_enemy_board = (current_turn_ == Turn::kWhite ? board_.black : board_.white);

    // Actually apply the move to the bitmasks
    ApplyMoveOnSingleBoard(mv, board_.white, board_.black, board_.kings);

    // Check if it was a capture
    bool was_capture = false;
    {
        board_t enemy_board = (current_turn_ == Turn::kWhite ? board_.black : board_.white);
        if (enemy_board < old_enemy_board) {
            was_capture = true;
        }
    }

    UpdateNonReversibleCount(was_capture, from_sq);

    // If capturing, check if we can continue from 'to_sq'. If yes, do NOT switch turn.
    bool multi_cap_continues = false;
    if (was_capture) {
        multi_cap_continues = CheckAndMaybeContinueCapture(mv);
    }

    if (!multi_cap_continues) {
        // Switch side
        current_turn_ = (current_turn_ == Turn::kWhite ? Turn::kBlack : Turn::kWhite);
        HandlePromotions();
    }

    return true;
}

bool CheckersEngine::CheckAndMaybeContinueCapture(move_t last_move)
{
    using namespace checkers::cpu::move_gen;

    board_index_t to_sq = DecodeMove<MovePart::To>(last_move);

    // generate moves for to_sq
    // If there's a capturing move from that square, we remain on the same turn.
    move_t local_moves[kNumMaxMovesPerPiece];
    u8 local_count                  = 0;
    move_flags_t local_capture_mask = 0;
    move_flags_t dummy_flags        = 0;

    board_t w = board_.white;
    board_t b = board_.black;
    board_t k = board_.kings;

    if (current_turn_ == Turn::kWhite) {
        GenerateMovesForSinglePiece<Turn::kWhite>(
            to_sq, w, b, k, local_moves, local_count, local_capture_mask, dummy_flags
        );
    } else {
        GenerateMovesForSinglePiece<Turn::kBlack>(
            to_sq, w, b, k, local_moves, local_count, local_capture_mask, dummy_flags
        );
    }

    // If no sub-moves or no capture in them, we cannot continue
    if (local_count == 0) {
        return false;
    }
    if (local_capture_mask == 0) {
        return false;
    }

    // There's at least one capturing sub-move => we do NOT switch turns
    return true;
}

void CheckersEngine::HandlePromotions()
{
    // TODO: Test this
    board_.kings |= (board_.white & BoardConstants::kTopBoardEdgeMask);
    board_.kings |= (board_.black & BoardConstants::kBottomBoardEdgeMask);
}

void CheckersEngine::UpdateNonReversibleCount(bool was_capture, checkers::board_index_t from_sq)
{
    using namespace checkers::cpu;
    // If a capture or the from-square was not a king, reset to 0
    const bool from_was_king = ReadFlag(board_.kings, from_sq);

    if (was_capture || !from_was_king) {
        non_reversible_count_ = 0;
    } else {
        // It's a king move that didn't capture => increment
        non_reversible_count_++;
    }
}

}  // namespace checkers
