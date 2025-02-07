#include "game/checkers_game.hpp"
#include "common/checkers_defines.hpp"
#include "common/parallel.hpp"
#include "cpu/board_helpers.hpp"
#include "game/checkers_engine.hpp"

#include <cctype>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

namespace checkers
{

namespace
{
static constexpr std::string kDumpBoardToken = "dump_board";
static constexpr std::string kSaveToken      = "save_quit";
static constexpr f64 time_error_margin       = 9e-1;
}  // namespace

Game::Game(const checkers::cpu::Board &initial_board, const GameTypeInfo &game_type_info)
{
    game_type_info_ = game_type_info;

    // Create the engine with the initial board
    engine_ = std::make_unique<checkers::CheckersEngine>(initial_board, game_type_info.start_side);

    // Set up the GUI
    gui_ = game_type_info.gui;
}

GameResult Game::Play(const std::string &record_file)
{
    if (!gui_) {
        std::cerr << "No GUI/CLI interface is set. Exiting.\n";
        return GameResult::kInProgress;
    }

    // Show initial board
    gui_->DisplayBoard(engine_->GetBoard(), kInvalidMove);
    GameResult res = GameResult::kInProgress;

    while (true) {
        // Check if game ended
        res = engine_->CheckGameResult();
        if (res != GameResult::kInProgress) {
            // Announce
            std::string msg;
            if (res == GameResult::kWhiteWin) {
                msg = "White wins!";
            } else if (res == GameResult::kBlackWin) {
                msg = "Black wins!";
            } else {
                msg = "Draw!";
            }
            gui_->DisplayMessage("Game Over: " + msg);
            break;
        }

        // Decide whose turn it is
        checkers::Turn side_to_move = engine_->GetCurrentTurn();
        bool is_ai                  = IsAI(side_to_move);

        move_t chosen_move = kInvalidMove;
        std::string notation_move;

        f64 elapsed = 0.0;
        if (is_ai) {
            gui_->DisplayMessage(
                std::string("AI ") + (side_to_move == checkers::Turn::kWhite ? "White" : "Black") + " - is thinking..."
            );
            mcts::Tree tree(
                engine_->GetBoard(), side_to_move,
                side_to_move == Turn::kWhite ? game_type_info_.white_backend.value()
                                             : game_type_info_.black_backend.value()
            );
            f64 time_budget =
                (side_to_move == Turn::kWhite) ? game_type_info_.white_time_limit : game_type_info_.black_time_limit;
            const auto start  = std::chrono::steady_clock::now();
            chosen_move       = tree.Run(time_budget - mcts::kRunCallOverhead, kNumThreadsCPU);
            const auto finish = std::chrono::steady_clock::now();
            elapsed           = std::chrono::duration<float>(finish - start).count();

            std::ostringstream oss;
            oss << tree.GetRunInfo();
            std::string info_str = oss.str();

            gui_->DisplayMessage(info_str);

        } else {
            const auto start  = std::chrono::steady_clock::now();
            notation_move     = gui_->PromptForMove();
            const auto finish = std::chrono::steady_clock::now();
            elapsed           = std::chrono::duration<float>(finish - start).count();
        }

        const f64 time_limit = GetSideTimeLimit(side_to_move);
        if (is_ai) {
            gui_->DisplayMessage(
                std::string("AI ") + (side_to_move == Turn::kWhite ? "White" : "Black") + " - took " +
                std::to_string(elapsed) + " seconds"
            );
        }
        if (elapsed > time_limit + time_error_margin) {
            gui_->DisplayMessage(
                "Time out! " + std::string(side_to_move == checkers::Turn::kWhite ? "White" : "Black") + " loses."
            );
            res = GetOppositeSideWin(side_to_move);
            break;
        }

        if (is_ai) {
            bool success = engine_->ApplyMove(chosen_move, false);
            if (!success) {
                gui_->DisplayMessage(
                    "Move invalid! " + std::string(side_to_move == checkers::Turn::kWhite ? "White" : "Black") +
                    " loses."
                );
                gui_->DisplayMessage("(Since it's an AI move, this is a bug.)");
                res = GetOppositeSideWin(side_to_move);
                break;
            }
        } else {
            auto [success, msg, move] = AttemptMoveFromNotation(notation_move);
            chosen_move               = move;
            if (!success) {
                gui_->DisplayMessage("Invalid Move: " + msg);
                continue;
            }
        }
        // Log move in notation
        std::string move_string = GetMoveString(chosen_move);

        move_history_.push_back(move_string);

        if (is_ai) {
            gui_->DisplayMessage(
                "AI " + std::string(side_to_move == checkers::Turn::kWhite ? "White" : "Black") + " - plays " +
                move_string
            );
        }
        gui_->DisplayBoard(engine_->GetBoard(), chosen_move);
    }

    SaveRecord(record_file);
    return res;
}
std::string Game::GetMoveString(move_t move)
{
    board_index_t from_sq = cpu::move_gen::DecodeMove<cpu::move_gen::MovePart::From>(move);
    board_index_t to_sq   = cpu::move_gen::DecodeMove<cpu::move_gen::MovePart::To>(move);

    std::string move_string = SquareToNotation(from_sq) + "-" + SquareToNotation(to_sq);
    return move_string;
}
f64 Game::GetSideTimeLimit(const Turn &side_to_move) const
{
    f64 time_limit =
        (side_to_move == Turn::kWhite) ? game_type_info_.white_time_limit : game_type_info_.black_time_limit;
    return time_limit;
}
bool Game::IsAI(const Turn &side_to_move) const
{
    bool is_ai = (side_to_move == Turn::kWhite && game_type_info_.white_player_type == PlayerType::kAi) ||
                 (side_to_move == Turn::kBlack && game_type_info_.black_player_type == PlayerType::kAi);
    return is_ai;
}
GameResult Game::GetOppositeSideWin(const Turn &side_to_move)
{
    GameResult res;
    if (side_to_move == Turn::kWhite) {
        res = GameResult::kBlackWin;
    } else {
        res = GameResult::kWhiteWin;
    }
    return res;
}
std::string Game::SquareToNotation(board_index_t sq)
{
    int row       = sq / 4;
    int colOffset = sq % 4;
    int col       = (row % 2 == 0) ? (colOffset * 2) : (colOffset * 2 + 1);
    int rank      = 8 - row;  // row=0 => rank=8
    char file     = 'a' + col;
    std::stringstream ss;
    ss << file << rank;
    return ss.str();
}

std::tuple<bool, std::string, move_t> Game::AttemptMoveFromNotation(const std::string &move_line)
{
    char delim = '-';
    if (move_line.find(':') != std::string::npos) {
        delim = ':';
    }

    // If multiple captures, we might have more than one delimiter
    // We'll split by either ':' or '-'
    // Then see how many positions we get
    std::vector<std::string> fields;
    {
        std::stringstream ss(move_line);
        std::string token;
        while (std::getline(ss, token, delim)) {
            if (!token.empty()) {
                // Special token for debugging // TODO: Remove this or make a more elegant solution
                if (token == kDumpBoardToken) {
                    checkers::cpu::Board board = engine_->GetBoard();
                    std::cout << "White: " << board.white << std::endl;
                    std::cout << "Black: " << board.black << std::endl;
                    std::cout << "Kings: " << board.kings << std::endl;
                    return {false, "Dumped board.", kInvalidMove};
                } else if (token == kSaveToken) {
                    SaveRecord("game_record.txt");
                    std::cout << "Game record saved to game_record.txt\n";
                    return {false, "Game record saved.", kInvalidMove};
                } else {
                    fields.push_back(token);
                }
            }
        }
    }

    if (fields.size() < 2) {
        return {false, "Less than two squares given.", kInvalidMove};
    }

    // If more than 2 => multi-capture
    if (fields.size() > 2) {
        // replacing all ':' with '-'
        std::string replaced = move_line;
        for (char &c : replaced) {
            if (c == ':')
                c = '-';
        }
        // now split on '-'
        fields.clear();
        {
            std::stringstream ss(replaced);
            std::string part;
            while (std::getline(ss, part, '-')) {
                if (!part.empty())
                    fields.push_back(part);
            }
        }
        // Apply multi-capture
        if (!ApplyMultiCapture(fields)) {
            return {false, "Invalid multi-capture.", kInvalidMove};
        }
        // If success, store in record
        move_history_.push_back(move_line);
        return {true, "", kInvalidMove};
    }

    // If exactly 2 fields => single step
    checkers::board_index_t fromIdx = NotationToIndex(fields[0]);
    checkers::board_index_t toIdx   = NotationToIndex(fields[1]);
    if (fromIdx >= checkers::BoardConstants::kBoardSize || toIdx >= checkers::BoardConstants::kBoardSize) {
        std::string msg = "Invalid square(s): from=" + fields[0] + ", to=" + fields[1];
        return {false, msg, kInvalidMove};
    }

    // Encode
    checkers::move_t mv = (fromIdx) | (toIdx << 8);
    bool ok             = engine_->ApplyMove(mv, true);
    if (!ok)
        return {false, "Move not valid according to engine.", kInvalidMove};

    // Add to record
    move_history_.push_back(move_line);
    return {true, "", mv};
}

bool Game::ApplyMultiCapture(const std::vector<std::string> &fields)
{
    // Suppose we have e.g. ["d2", "f4", "d6"]
    // We apply them in sequence. The engine allows partial captures.
    for (size_t i = 0; i + 1 < fields.size(); i++) {
        checkers::board_index_t fromIdx = NotationToIndex(fields[i]);
        checkers::board_index_t toIdx   = NotationToIndex(fields[i + 1]);
        if (fromIdx >= checkers::BoardConstants::kBoardSize || toIdx >= checkers::BoardConstants::kBoardSize) {
            return false;
        }
        checkers::move_t mv = checkers::cpu::move_gen::EncodeMove(fromIdx, toIdx);

        // Must apply with validation so engine can accept or reject
        bool ok = engine_->ApplyMove(mv, true);
        if (!ok) {
            return false;
        }
    }
    return true;
}

checkers::board_index_t Game::NotationToIndex(const std::string &cell) const
{
    if (cell.size() < 2)
        return ~0;

    char fileChar = static_cast<char>(std::tolower(cell[0]));
    if (fileChar < 'a' || fileChar > 'h')
        return ~0;
    int file = fileChar - 'a';

    if (!std::isdigit(cell[1]))
        return ~0;
    int rank = (cell[1] - '0');
    if (rank < 1 || rank > 8)
        return ~0;

    // row = 8-rank
    int row          = 8 - rank;
    bool is_playable = ((row % 2 == 0 && file % 2 == 0) || (row % 2 == 1 && file % 2 == 1));
    if (!is_playable)
        return ~0;

    int col_offset = (row % 2 == 0) ? (file / 2) : ((file - 1) / 2);
    int idx        = row * 4 + col_offset;
    if (idx < 0 || idx >= 32) {
        return ~0;
    }
    return (checkers::board_index_t)idx;
}

void Game::SaveRecord(const std::string &recordFile) const
{
    if (recordFile.empty()) {
        std::cout << "\n=== Move History ===\n";
        for (const auto &m : move_history_) {
            std::cout << m << std::endl;
        }
        return;
    }
    std::ofstream ofs(recordFile);
    if (!ofs.is_open()) {
        std::cerr << "Cannot open " << recordFile << " for saving moves.\n";
        return;
    }
    for (size_t i = 0; i < move_history_.size(); i++) {
        ofs << (i + 1) << ". " << move_history_[i] << "\n";
    }
    ofs.close();
}

bool Game::LoadGameRecord(const std::string &inputFile)
{
    std::ifstream ifs(inputFile);
    if (!ifs.is_open()) {
        std::cerr << "Cannot open " << inputFile << " for reading.\n";
        return false;
    }
    std::string line;
    while (std::getline(ifs, line)) {
        // Each line might be "1. d2-e3" or just "d2:e3". We can parse out the move portion
        // Example: "3. d2:f4:d6"
        std::stringstream ss(line);
        std::string token;
        // skip the move number or prefix
        // e.g. "3."
        if (std::getline(ss, token, ' ')) {
            // do nothing
        }
        // remainder is the move
        std::string move_str;
        if (std::getline(ss, move_str)) {
            // Attempt to apply
            auto [ok, msg, _] = AttemptMoveFromNotation(move_str);
            if (!ok) {
                std::cerr << "Line parse fail: " << msg << "\nLine was: " << line << std::endl;
                return false;
            }
        }
    }
    return true;
}
}  // namespace checkers
