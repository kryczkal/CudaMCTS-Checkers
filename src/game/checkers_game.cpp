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
#include <unordered_map>

namespace checkers
{

CheckersGame::CheckersGame(
    const checkers::cpu::Board &initialBoard, checkers::Turn startTurn, GameMode mode, checkers::Turn humanTurn
)
    : human_turn_(humanTurn), game_mode_(mode)
{
    engine_ = std::make_unique<checkers::CheckersEngine>(initialBoard, startTurn);
    InitializeCommandParser();
}

void CheckersGame::SetHumanTimeLimit(float seconds) { human_time_limit_ = seconds; }

void CheckersGame::SetAiTimeLimit(float seconds) { ai_time_limit_ = seconds; }

void CheckersGame::SetGui(std::shared_ptr<ICheckersGui> gui) { gui_ = gui; }

void CheckersGame::SetSimulationBackend(mcts::SimulationBackend backend) { simulation_backend_ = backend; }

void CheckersGame::Play(const std::string &recordFile)
{
    if (!gui_) {
        std::cerr << "No GUI/CLI interface is set. Exiting.\n";
        return;
    }

    // Show initial board.
    gui_->DisplayBoard(engine_->GetBoard(), kInvalidMove);

    while (!quit_) {
        // Check for game termination.
        GameResult res = engine_->CheckGameResult();
        if (res != GameResult::kInProgress) {
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

        checkers::Turn side_to_move = engine_->GetCurrentTurn();
        bool is_human_move          = false;
        switch (game_mode_) {
            case GameMode::HumanVsHuman:
                is_human_move = true;
                break;
            case GameMode::HumanVsAi:
                is_human_move = (side_to_move == human_turn_);
                break;
            case GameMode::AiVsAi:
                is_human_move = false;
                break;
        }

        if (is_human_move) {
            gui_->DisplayMessage(
                (side_to_move == checkers::Turn::kWhite ? "White" : "Black") + std::string(" (Human) to move.")
            );
            auto start        = std::chrono::steady_clock::now();
            std::string input = gui_->PromptForMove();
            auto finish       = std::chrono::steady_clock::now();
            float elapsed     = std::chrono::duration<float>(finish - start).count();
            if (elapsed > human_time_limit_) {
                gui_->DisplayMessage("Time out! You lose.");
                break;
            }

            // Process command if input matches a registered command.
            if (ProcessCommand(input)) {
                continue;
            }

            // Process move notation.
            auto [ok, msg, move] = AttemptMoveFromNotation(input);
            if (!ok) {
                gui_->DisplayMessage("Invalid Move: " + msg);
                continue;
            }
            gui_->DisplayBoard(engine_->GetBoard(), move);
        } else {
            gui_->DisplayMessage("AI is thinking...");
            auto moves = engine_->GenerateMoves();
            if (engine_->CheckGameResult() != GameResult::kInProgress) {
                break;
            }

            auto start = std::chrono::steady_clock::now();
            // Instantiate the MCTS tree with the chosen simulation backend.
            checkers::mcts::MonteCarloTree tree(engine_->GetBoard(), side_to_move, simulation_backend_);
            checkers::move_t best_move = tree.Run(ai_time_limit_, checkers::kNumThreadsCPU);
            auto end                   = std::chrono::steady_clock::now();
            float elapsed              = std::chrono::duration<float>(end - start).count();
            std::cout << "AI took " << elapsed << " seconds\n";

            bool success = engine_->ApplyMove(best_move, false);
            if (!success) {
                gui_->DisplayMessage("AI move invalid? Forcing loss.");
                break;
            }
            // Convert move to a human-readable notation.
            checkers::board_index_t from_sq =
                checkers::cpu::move_gen::DecodeMove<checkers::cpu::move_gen::MovePart::From>(best_move);
            checkers::board_index_t to_sq =
                checkers::cpu::move_gen::DecodeMove<checkers::cpu::move_gen::MovePart::To>(best_move);
            std::string move_string = SquareToNotation(from_sq) + "-" + SquareToNotation(to_sq);
            move_history_.push_back(move_string);
            gui_->DisplayMessage("AI plays " + move_string);
            gui_->DisplayBoard(engine_->GetBoard(), best_move);
        }
    }

    SaveRecord(recordFile);
}

std::tuple<bool, std::string, move_t> CheckersGame::AttemptMoveFromNotation(const std::string &move_line)
{
    char delim = '-';
    if (move_line.find(':') != std::string::npos) {
        delim = ':';
    }
    std::vector<std::string> fields;
    {
        std::stringstream ss(move_line);
        std::string token;
        while (std::getline(ss, token, delim)) {
            if (!token.empty()) {
                fields.push_back(token);
            }
        }
    }
    if (fields.size() < 2) {
        return {false, "Less than two squares given.", kInvalidMove};
    }
    // Handle multi-capture moves.
    if (fields.size() > 2) {
        std::string replaced = move_line;
        for (char &c : replaced) {
            if (c == ':')
                c = '-';
        }
        fields.clear();
        {
            std::stringstream ss(replaced);
            std::string part;
            while (std::getline(ss, part, '-')) {
                if (!part.empty())
                    fields.push_back(part);
            }
        }
        if (!ApplyMultiCapture(fields)) {
            return {false, "Invalid multi-capture.", kInvalidMove};
        }
        move_history_.push_back(move_line);
        return {true, "", kInvalidMove};
    }
    checkers::board_index_t from_idx = NotationToIndex(fields[0]);
    checkers::board_index_t toIdx    = NotationToIndex(fields[1]);
    if (from_idx >= checkers::BoardConstants::kBoardSize || toIdx >= checkers::BoardConstants::kBoardSize) {
        std::string msg = "Invalid square(s): from=" + fields[0] + ", to=" + fields[1];
        return {false, msg, kInvalidMove};
    }
    std::cout << "From: " << static_cast<unsigned>(from_idx) << ", To: " << static_cast<unsigned>(toIdx) << std::endl;
    checkers::move_t mv = (from_idx) | (toIdx << 8);
    bool ok             = engine_->ApplyMove(mv, true);
    if (!ok)
        return {false, "Move not valid according to engine.", kInvalidMove};
    move_history_.push_back(move_line);
    return {true, "", mv};
}

bool CheckersGame::ApplyMultiCapture(const std::vector<std::string> &fields)
{
    for (size_t i = 0; i + 1 < fields.size(); i++) {
        checkers::board_index_t from_idx = NotationToIndex(fields[i]);
        checkers::board_index_t toIdx    = NotationToIndex(fields[i + 1]);
        if (from_idx >= checkers::BoardConstants::kBoardSize || toIdx >= checkers::BoardConstants::kBoardSize) {
            return false;
        }
        checkers::move_t mv = checkers::cpu::move_gen::EncodeMove(from_idx, toIdx);
        bool ok             = engine_->ApplyMove(mv, true);
        if (!ok) {
            return false;
        }
    }
    return true;
}

checkers::board_index_t CheckersGame::NotationToIndex(const std::string &cell) const
{
    if (cell.size() < 2) {
        return ~0;
    }
    char fileChar = static_cast<char>(std::tolower(cell[0]));
    if (fileChar < 'a' || fileChar > 'h') {
        return ~0;
    }
    int file = fileChar - 'a';
    if (!std::isdigit(cell[1])) {
        return ~0;
    }
    int rank = (cell[1] - '0');
    if (rank < 1 || rank > 8) {
        return ~0;
    }
    int row          = 8 - rank;
    bool is_playable = ((row % 2 == 0 && file % 2 == 0) || (row % 2 == 1 && file % 2 == 1));
    if (!is_playable) {
        return ~0;
    }
    int col_offset = (row % 2 == 0) ? (file / 2) : ((file - 1) / 2);
    int idx        = row * 4 + col_offset;
    if (idx < 0 || idx >= 32) {
        return ~0;
    }
    return static_cast<checkers::board_index_t>(idx);
}

std::string CheckersGame::SquareToNotation(checkers::board_index_t sq) const
{
    int row        = sq / 4;
    int col_offset = sq % 4;
    int col        = (row % 2 == 0) ? (col_offset * 2) : (col_offset * 2 + 1);
    int rank       = 8 - row;
    char file      = 'a' + col;
    std::stringstream ss;
    ss << file << rank;
    return ss.str();
}

void CheckersGame::SaveRecord(const std::string &recordFile) const
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

bool CheckersGame::LoadGameRecord(const std::string &inputFile)
{
    std::ifstream ifs(inputFile);
    if (!ifs.is_open()) {
        std::cerr << "Cannot open " << inputFile << " for reading.\n";
        return false;
    }
    std::string line;
    while (std::getline(ifs, line)) {
        std::stringstream ss(line);
        std::string token;
        // Skip move number if present.
        if (std::getline(ss, token, ' ')) {
        }
        std::string move_str;
        if (std::getline(ss, move_str)) {
            auto [ok, msg, _] = AttemptMoveFromNotation(move_str);
            if (!ok) {
                std::cerr << "Line parse fail: " << msg << "\nLine was: " << line << std::endl;
                return false;
            }
        }
    }
    return true;
}

// ---------------------- Command Parser Functions ----------------------

void CheckersGame::InitializeCommandParser()
{
    command_map_.clear();
    command_map_["help"] = [this]() {
        CommandHelp();
    };
    command_map_["dump"] = [this]() {
        CommandDumpBoard();
    };
    command_map_["save"] = [this]() {
        CommandSave();
    };
    command_map_["quit"] = [this]() {
        CommandQuit();
    };
}

bool CheckersGame::ProcessCommand(const std::string &input)
{
    // Trim input and check if it exactly matches one of our commands.
    std::string trimmed;
    std::istringstream iss(input);
    iss >> trimmed;
    auto it = command_map_.find(trimmed);
    if (it != command_map_.end()) {
        it->second();
        return true;
    }
    return false;
}

void CheckersGame::CommandHelp()
{
    std::string helpText = "Available commands:\n";
    helpText += "  help - Display this help message\n";
    helpText += "  dump - Dump the current board state\n";
    helpText += "  save - Save the move history to 'game_record.txt'\n";
    helpText += "  quit - Quit the game\n";
    gui_->DisplayMessage(helpText);
}

void CheckersGame::CommandDumpBoard()
{
    checkers::cpu::Board board = engine_->GetBoard();
    std::stringstream ss;
    ss << "Board State:\n";
    ss << "White: " << board.white << "\n";
    ss << "Black: " << board.black << "\n";
    ss << "Kings: " << board.kings << "\n";
    gui_->DisplayMessage(ss.str());
}

void CheckersGame::CommandSave()
{
    SaveRecord("game_record.txt");
    gui_->DisplayMessage("Game record saved to 'game_record.txt'.");
}

void CheckersGame::CommandQuit()
{
    gui_->DisplayMessage("Quitting game.");
    quit_ = true;
}

}  // namespace checkers
