#include "checkers_engine.hpp"
#include <cstdlib>  // for rand()
#include <fstream>
#include <iostream>
#include <sstream>

namespace CudaMctsCheckers
{

CheckersEngine::CheckersEngine(const Board &board, Turn turn) : board_(board), current_turn_(turn)
{
}

const Board &CheckersEngine::GetBoard() const { return board_; }

Turn CheckersEngine::GetCurrentTurn() const { return current_turn_; }

MoveGenerationOutput CheckersEngine::GenerateCurrentPlayerMoves() const
{
    if (current_turn_ == Turn::kWhite) {
        return MoveGenerator::GenerateMovesForPlayerCpu<BoardCheckType::kWhite>(board_);
    } else {
        return MoveGenerator::GenerateMovesForPlayerCpu<BoardCheckType::kBlack>(board_);
    }
}

bool CheckersEngine::ApplyMove(
    Board::IndexType from_idx, Board::IndexType to_idx, bool force_capture
)
{
    // Generate moves for the current player
    auto moves_output = GenerateCurrentPlayerMoves();

    bool found            = false;
    bool capture_happened = false;

    // Search through generated moves to see if (from_idx -> to_idx) is valid
    for (u32 i = 0; i < Move::kNumMoveArrayForPlayerSize; ++i) {
        if (Move::DecodeOriginIndex(i) == from_idx && moves_output.possible_moves[i] == to_idx) {
            // If caller forces a capture, skip if it’s not a capturing move
            if (force_capture && !moves_output.capture_moves_bitmask[i]) {
                continue;
            }
            found            = true;
            capture_happened = moves_output.capture_moves_bitmask[i];
            break;
        }
    }

    if (!found) {
        return false;
    }

    // Actually apply
    if (current_turn_ == Turn::kWhite) {
        board_.ApplyMove<BoardCheckType::kWhite>(from_idx, to_idx, capture_happened);
    } else {
        board_.ApplyMove<BoardCheckType::kBlack>(from_idx, to_idx, capture_happened);
    }

    // Promote
    PromoteAndUpdateReversibleCount(!capture_happened);

    // Switch turn if we are not continuing a multi-capture
    SwitchTurnIfNeeded(capture_happened);

    return true;
}

bool CheckersEngine::ApplyRandomMove()
{
    auto moves_output = GenerateCurrentPlayerMoves();

    // Check if any valid moves
    bool any_valid_moves = false;
    for (u32 i = 0; i < Move::kNumMoveArrayForPlayerSize; ++i) {
        if (moves_output.possible_moves[i] != Move::kInvalidMove) {
            any_valid_moves = true;
            break;
        }
    }
    if (!any_valid_moves) {
        return false;
    }

    // Check if a capture is required
    bool capture_required =
        moves_output.capture_moves_bitmask[MoveGenerationOutput::CaptureFlagIndex];

    // We’ll pick a random move that satisfies capture conditions if needed
    bool found_move       = false;
    u32 random_move_index = 0;
    for (u32 attempts = 0; attempts < Move::kNumMoveArrayForPlayerSize; attempts++) {
        random_move_index       = rand() % Move::kNumMoveArrayForPlayerSize;
        Board::IndexType to_idx = moves_output.possible_moves[random_move_index];
        if (to_idx == Move::kInvalidMove) {
            continue;
        }
        bool is_capture = moves_output.capture_moves_bitmask[random_move_index];
        if (capture_required && !is_capture) {
            continue;
        }
        found_move = true;
        break;
    }

    if (!found_move) {
        // no suitable move found => none or forced capture not possible
        return false;
    }

    // Apply the chosen move
    Board::IndexType from_idx = Move::DecodeOriginIndex(random_move_index);
    Board::IndexType to_idx   = moves_output.possible_moves[random_move_index];
    bool capture_happened     = moves_output.capture_moves_bitmask[random_move_index];

    if (current_turn_ == Turn::kWhite) {
        board_.ApplyMove<BoardCheckType::kWhite>(from_idx, to_idx, capture_happened);
    } else {
        board_.ApplyMove<BoardCheckType::kBlack>(from_idx, to_idx, capture_happened);
    }

    // Promote
    PromoteAndUpdateReversibleCount(!capture_happened);

    // Switch turn
    SwitchTurnIfNeeded(capture_happened);

    return true;
}

void CheckersEngine::SwitchTurnIfNeeded(bool capture_performed)
{
    // If you want multi-capture in a single turn, you might skip flipping the turn
    // in some rule sets. For now, we always flip after 1 move in this engine:
    if (!capture_performed) {
        current_turn_ = (current_turn_ == Turn::kWhite) ? Turn::kBlack : Turn::kWhite;
    } else {
        // If you do not want immediate multi-jump, also flip here:
        current_turn_ = (current_turn_ == Turn::kWhite) ? Turn::kBlack : Turn::kWhite;
    }
}

void CheckersEngine::PromoteAndUpdateReversibleCount(bool was_non_reversible)
{
    if (was_non_reversible) {
        // e.g. a piece advanced but no capture
        board_.time_from_non_reversible_move++;
    } else {
        // reset for capturing or certain moves
        board_.time_from_non_reversible_move = 0;
    }
    board_.PromoteAll();
}

GameResult CheckersEngine::CheckGameResult() const { return board_.CheckGameResult(); }

bool CheckersEngine::RestoreFromHistoryFile(
    const std::string &history_file, std::string &error_message
)
{
    std::ifstream infile(history_file);
    if (!infile.is_open()) {
        error_message = "Unable to open history file: " + history_file;
        return false;
    }

    std::string move_str;
    size_t line_number = 0;
    while (std::getline(infile, move_str)) {
        line_number++;
        // Trim whitespace
        move_str.erase(0, move_str.find_first_not_of(" \t\r\n"));
        move_str.erase(move_str.find_last_not_of(" \t\r\n") + 1);

        if (move_str.empty()) {
            continue;  // Skip empty lines
        }

        // Split move_str into fields separated by ':' or '-'
        std::vector<std::string> fields;
        std::stringstream ss(move_str);
        std::string part;
        while (std::getline(ss, part, ':')) {
            // Now split by '-'
            std::stringstream ss_inner(part);
            while (std::getline(ss_inner, part, '-')) {
                if (!part.empty()) {
                    fields.push_back(part);
                }
            }
        }

        if (fields.size() < 2) {
            error_message =
                "Invalid move format on line " + std::to_string(line_number) + ": " + move_str;
            return false;
        }

        // Apply each move segment
        for (size_t i = 0; i < fields.size() - 1; ++i) {
            std::string from_field = fields[i];
            std::string to_field   = fields[i + 1];

            Board::IndexType from_idx = ConvertNotationToIndex(from_field);
            Board::IndexType to_idx   = ConvertNotationToIndex(to_field);

            if (from_idx == Board::kInvalidIndex || to_idx == Board::kInvalidIndex) {
                error_message = "Invalid square in move on line " + std::to_string(line_number) +
                                ": " + move_str;
                return false;
            }

            // Determine if this is a capture move
            bool force_capture = (fields.size() > 2);  // Any multi-segment move requires captures

            bool success = ApplyMove(from_idx, to_idx, force_capture);
            if (!success) {
                error_message =
                    "Failed to apply move on line " + std::to_string(line_number) + ": " + move_str;
                return false;
            }
        }
    }

    return true;
}

Board::IndexType CheckersEngine::ConvertNotationToIndex(const std::string &field) const
{
    if (field.size() < 2) {
        return Board::kInvalidIndex;
    }
    char fileChar = static_cast<char>(std::tolower(field[0]));
    if (fileChar < 'a' || fileChar > 'h') {
        return Board::kInvalidIndex;
    }
    int file = fileChar - 'a';

    if (!std::isdigit(field[1])) {
        return Board::kInvalidIndex;
    }
    int rank = field[1] - '0';  // '1' => 1
    if (rank < 1 || rank > 8) {
        return Board::kInvalidIndex;
    }

    // row=8-rank
    int row         = 8 - rank;
    bool isPlayable = (row % 2 == 0) ? (file % 2 == 0) : (file % 2 != 0);
    if (!isPlayable) {
        return Board::kInvalidIndex;
    }

    int colOffset          = (row % 2 == 0) ? (file / 2) : ((file - 1) / 2);
    Board::IndexType index = static_cast<Board::IndexType>(row * 4 + colOffset);
    if (index >= Board::kHalfBoardSize) {
        return Board::kInvalidIndex;
    }
    return index;
}

}  // namespace CudaMctsCheckers
