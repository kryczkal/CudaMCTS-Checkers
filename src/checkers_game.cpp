#include "checkers_game.hpp"
#include <cctype>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

namespace CudaMctsCheckers
{

CheckersGame::CheckersGame() : human_turn_(Turn::kWhite)
{
    Board initial_board;
    // Initialize standard checkers layout
    SetupStandardBoard(initial_board);

    // Start with White
    engine_ = std::make_unique<CheckersEngine>(initial_board, Turn::kWhite);
}

CheckersGame::CheckersGame(Turn humanTurn) : human_turn_(humanTurn)
{
    Board initial_board;
    SetupStandardBoard(initial_board);

    // By default, White starts
    engine_ = std::make_unique<CheckersEngine>(initial_board, Turn::kWhite);
}

CheckersGame::CheckersGame(const Board &board, Turn humanTurn) : human_turn_(humanTurn)
{
    // Also starts with White
    engine_ = std::make_unique<CheckersEngine>(board, Turn::kWhite);
}

void CheckersGame::SetTimeLimit(f32 seconds) { time_limit_per_move_ = seconds; }

void CheckersGame::SetTimeLimitAi(f32 seconds) { time_limit_per_move_ai_ = seconds; }

void CheckersGame::SetGui(std::shared_ptr<ICheckersGui> gui) { gui_ = gui; }

void CheckersGame::Play(const std::string &output_file)
{
    assert(engine_);
    assert(gui_);
    if (!gui_) {
        std::cerr << "No GUI set";
        return;
    }

    // Display initial board
    gui_->DisplayBoard(engine_->GetBoard());

    // Main loop
    GameResult result;
    while ((result = engine_->CheckGameResult()) == GameResult::kInProgress) {
        gui_->DisplayMessage(
            (engine_->GetCurrentTurn() == Turn::kWhite ? "White" : "Black") +
            std::string(" to move.")
        );

        if (IsHumanTurn()) {
            // Time-limited user input
            auto start_time = std::chrono::steady_clock::now();
            std::string move_str;
            move_str      = gui_->PromptForMove();
            auto end_time = std::chrono::steady_clock::now();
            float elapsed = std::chrono::duration<float>(end_time - start_time).count();

            if (elapsed > time_limit_per_move_) {
                // Time out => lose
                gui_->DisplayMessage("You took too long. You lose!");
                result =
                    (human_turn_ == Turn::kWhite) ? GameResult::kBlackWin : GameResult::kWhiteWin;
                break;
            }

            // Attempt the move
            auto [ok, err_msg] = AttemptMoveViaEngine(move_str);
            if (!ok) {
                gui_->DisplayMessage("Invalid move: " + err_msg);
            } else {
                gui_->DisplayBoard(engine_->GetBoard());
            }
        } else {
            // AI
            gui_->DisplayMessage("AI thinking...");

            auto start_time = std::chrono::steady_clock::now();

            // Build a tree from current engine state
            MonteCarloTree tree(engine_->GetBoard(), engine_->GetCurrentTurn());
            TrieDecodedMoveAsPair best_move = tree.Run(time_limit_per_move_ai_ - 0.1f);

            auto end_time = std::chrono::steady_clock::now();
            float elapsed = std::chrono::duration<float>(end_time - start_time).count();

            //            if (elapsed > time_limit_per_move_ai_) {
            //                // Time out => lose
            //                gui_->DisplayMessage("AI took too long. You win!");
            //                result = (human_turn_ == Turn::kWhite) ? GameResult::kWhiteWin :
            //                GameResult::kBlackWin; break;
            //            }

            // Apply
            bool success =
                engine_->ApplyMove<ApplyMoveType::kNoValidate>(best_move.first, best_move.second);
            if (!success) {
                // If AI somehow provided an invalid move, treat it as no moves => lose
                if (gui_) {
                    gui_->DisplayMessage("AI has no valid moves and loses!");
                }
                result =
                    (human_turn_ == Turn::kWhite) ? GameResult::kWhiteWin : GameResult::kBlackWin;
                break;
            }

            // Convert that to notation
            auto indexToNotation = [&](Board::IndexType idx) -> std::string {
                // same logic from your code
                int row       = idx / 4;
                int colOffset = idx % 4;
                int col       = (row % 2 == 0) ? (colOffset * 2) : (colOffset * 2 + 1);
                int rank      = 8 - row;  // row=0 => rank=8
                char fileChar = 'a' + col;
                std::ostringstream oss;
                oss << fileChar << rank;
                return oss.str();
            };

            // TODO: Chain captures
            std::string notation =
                indexToNotation(best_move.first) + "-" + indexToNotation(best_move.second);
            move_history_.push_back(notation);

            gui_->DisplayMessage("AI move: " + notation);
            gui_->DisplayBoard(engine_->GetBoard());
        }
    }

    gui_->DisplayBoard(engine_->GetBoard());

    // Announce
    std::string msg;
    if (result == GameResult::kWhiteWin) {
        msg = "Game over! White wins.";
    } else if (result == GameResult::kBlackWin) {
        msg = "Game over! Black wins.";
    } else {
        msg = "Game over! It's a draw.";
    }
    if (gui_) {
        gui_->DisplayMessage(msg);
    } else {
        std::cout << msg << std::endl;
    }
    // Save record
    SaveGameRecord(output_file);
}

bool CheckersGame::IsHumanTurn() const { return (engine_->GetCurrentTurn() == human_turn_); }

void CheckersGame::SetupStandardBoard(Board &board)
{
    // Clear
    board.white_pieces                  = 0;
    board.black_pieces                  = 0;
    board.kings                         = 0;
    board.time_from_non_reversible_move = 0;

    // Standard checkers initial setup:
    // Black => top rows (indices 0..11)
    // White => bottom rows (indices 20..31)
    for (Board::IndexType i = 0; i < 12; ++i) {
        board.SetPieceAt<BoardCheckType::kBlack>(i);
    }
    for (Board::IndexType i = 20; i < 32; ++i) {
        board.SetPieceAt<BoardCheckType::kWhite>(i);
    }
}

std::pair<bool, std::string> CheckersGame::AttemptMoveViaEngine(const std::string &move_str)
{
    // Delimiter might be ':' for capture or '-' for normal. Could also chain multiple captures.
    char delim = (move_str.find(':') != std::string::npos) ? ':' : '-';

    std::vector<std::string> fields;
    {
        std::stringstream ss(move_str);
        std::string part;
        while (std::getline(ss, part, delim)) {
            if (!part.empty()) {
                fields.push_back(part);
            }
        }
    }
    if (fields.size() < 2) {
        return {false, "Not enough fields in the move."};
    }

    // If multi-capture
    if (fields.size() > 2) {
        // Attempt multi-capture
        if (!ApplyMultiCaptureMoveViaEngine(fields)) {
            return {false, "Multi-capture sequence invalid or not allowed."};
        }
        move_history_.push_back(move_str);
        return {true, ""};
    }

    // Single move
    Board::IndexType from_idx = ConvertNotationToIndex(fields[0]);
    Board::IndexType to_idx   = ConvertNotationToIndex(fields[1]);
    if (from_idx == Board::kInvalidIndex || to_idx == Board::kInvalidIndex) {
        return {false, "Invalid squares in notation."};
    }

    // Attempt
    bool success = engine_->ApplyMove<ApplyMoveType::kValidate>(from_idx, to_idx);
    if (!success) {
        return {false, "Move not valid for current player."};
    }

    // Record notation
    move_history_.push_back(move_str);
    return {true, ""};
}

bool CheckersGame::ApplyMultiCaptureMoveViaEngine(const std::vector<std::string> &fields)
{
    // fields e.g. ["d2","f4","d6"]
    for (size_t i = 0; i + 1 < fields.size(); ++i) {
        Board::IndexType from_idx = ConvertNotationToIndex(fields[i]);
        Board::IndexType to_idx   = ConvertNotationToIndex(fields[i + 1]);
        if (from_idx == Board::kInvalidIndex || to_idx == Board::kInvalidIndex) {
            return false;
        }
        // Force capture
        bool success = engine_->ApplyMove<ApplyMoveType::kValidate>(from_idx, to_idx);
        if (!success) {
            return false;
        }
    }
    return true;
}

Board::IndexType CheckersGame::ConvertNotationToIndex(const std::string &field) const
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

void CheckersGame::SaveGameRecord(const std::string &output_file) const
{
    if (output_file.empty()) {
        std::cout << "\n===== Game Record =====\n";
        for (const auto &move : move_history_) {
            std::cout << move << std::endl;
        }
        return;
    }

    std::ofstream ofs(output_file);
    if (!ofs.is_open()) {
        std::cerr << "Cannot open " << output_file << " for writing.\n";
        return;
    }
    for (size_t i = 0; i < move_history_.size(); ++i) {
        ofs << (i + 1) << ". " << move_history_[i] << "\n";
    }
    ofs.close();
}

bool CheckersGame::LoadGameRecord(const std::string &input_file)
{
    assert(engine_);

    std::string error_message;
    bool success = engine_->RestoreFromHistoryFile(input_file, error_message);
    if (!success) {
        std::cerr << "Error loading history file: " << error_message << std::endl;
    }
    return success;
}

}  // namespace CudaMctsCheckers
