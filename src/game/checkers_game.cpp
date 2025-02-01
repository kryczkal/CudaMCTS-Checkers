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
static constexpr std::string kSaveQuitToken  = "save_quit";
}  // namespace

CheckersGame::CheckersGame(const checkers::cpu::Board &initialBoard, checkers::Turn startTurn, checkers::Turn humanTurn)
    : human_turn_(humanTurn)
{
    // Create the engine with the initial board
    engine_ = std::make_unique<checkers::CheckersEngine>(initialBoard, startTurn);
}

void CheckersGame::SetHumanTimeLimit(float seconds) { human_time_limit_ = seconds; }

void CheckersGame::SetAiTimeLimit(float seconds) { ai_time_limit_ = seconds; }

void CheckersGame::SetGui(std::shared_ptr<ICheckersGui> gui) { gui_ = gui; }

void CheckersGame::PlayAiAi(const std::string &recordFile)
{
    // TODO: Extract AI move logic to a separate function and Human move logic to another function
    auto SquareToNotation = [&](checkers::board_index_t sq) {
        int row       = sq / 4;
        int colOffset = sq % 4;
        int col       = (row % 2 == 0) ? (colOffset * 2) : (colOffset * 2 + 1);
        int rank      = 8 - row;  // row=0 => rank=8
        char file     = 'a' + col;
        std::stringstream ss;
        ss << file << rank;
        return ss.str();
    };  // TODO: Move this to a common place

    if (!gui_) {
        std::cerr << "No GUI/CLI interface is set. Exiting.\n";
        return;
    }

    // Show initial board
    gui_->DisplayBoard(engine_->GetBoard(), kInvalidMove);

    while (true) {
        // Check if game ended
        GameResult res = engine_->CheckGameResult();
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
        bool is_first_ai            = (side_to_move == human_turn_);

        // Prompt
        if (is_first_ai) {
            gui_->DisplayMessage("AI 1 is thinking...");

            auto moves = engine_->GenerateMoves();
            if (engine_->CheckGameResult() != GameResult::kInProgress) {
                break;
            }

            // Start MCTS
            auto start = std::chrono::steady_clock::now();
            checkers::mcts::MonteCarloTree tree(engine_->GetBoard(), side_to_move);
            float time_budget = ai_time_limit_;

            checkers::move_t best_move =
                tree.RunParallel(time_budget - 0.1f, checkers::kNumThreadsCPU);  // reserve 0.1s for overhead

            // Apply bestMove
            bool success = engine_->ApplyMove(best_move, false);
            if (!success) {
                // No moves => lose
                gui_->DisplayMessage("AI 1 move is invalid? (Probably a BUG) Forcing loss.");
                if (side_to_move == checkers::Turn::kWhite) {
                    res = GameResult::kBlackWin;
                } else {
                    res = GameResult::kWhiteWin;
                }
                break;
            }
            // Log move in notation
            checkers::board_index_t from_sq =
                checkers::cpu::move_gen::DecodeMove<checkers::cpu::move_gen::MovePart::From>(best_move);
            checkers::board_index_t toSq =
                checkers::cpu::move_gen::DecodeMove<checkers::cpu::move_gen::MovePart::To>(best_move);

            // Convert to e.g. "d2-e3"
            std::string move_string = SquareToNotation(from_sq) + "-" + SquareToNotation(toSq);
            move_history_.push_back(move_string);

            gui_->DisplayMessage("AI plays " + move_string);
            gui_->DisplayBoard(engine_->GetBoard(), best_move);
        } else {
            // AI logic
            gui_->DisplayMessage("AI 2 is thinking...");

            auto moves = engine_->GenerateMoves();
            if (engine_->CheckGameResult() != GameResult::kInProgress) {
                break;
            }

            // Start MCTS
            auto start = std::chrono::steady_clock::now();
            checkers::mcts::MonteCarloTree tree(engine_->GetBoard(), side_to_move);
            float time_budget = ai_time_limit_;

            checkers::move_t best_move =
                tree.RunParallel(time_budget - 0.1f, checkers::kNumThreadsCPU);  // reserve 0.1s for overhead
            // Apply bestMove
            bool success = engine_->ApplyMove(best_move, false);
            if (!success) {
                // No moves => lose
                gui_->DisplayMessage("AI 2 move is invalid? (Probably a BUG) Forcing loss.");
                if (side_to_move == checkers::Turn::kWhite) {
                    res = GameResult::kBlackWin;
                } else {
                    res = GameResult::kWhiteWin;
                }
                break;
            }
            // Log move in notation
            checkers::board_index_t from_sq =
                checkers::cpu::move_gen::DecodeMove<checkers::cpu::move_gen::MovePart::From>(best_move);
            checkers::board_index_t toSq =
                checkers::cpu::move_gen::DecodeMove<checkers::cpu::move_gen::MovePart::To>(best_move);

            std::string move_string = SquareToNotation(from_sq) + "-" + SquareToNotation(toSq);
            move_history_.push_back(move_string);

            gui_->DisplayMessage("AI 2 plays " + move_string);
            gui_->DisplayBoard(engine_->GetBoard(), best_move);
        }
    }

    SaveRecord(recordFile);
}
void CheckersGame::Play(const std::string &recordFile)
{
    if (!gui_) {
        std::cerr << "No GUI/CLI interface is set. Exiting.\n";
        return;
    }

    // Show initial board
    gui_->DisplayBoard(engine_->GetBoard(), kInvalidMove);

    while (true) {
        // Check if game ended
        GameResult res = engine_->CheckGameResult();
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
        bool isHuman                = (side_to_move == human_turn_);

        // Prompt
        if (isHuman) {
            gui_->DisplayMessage(
                (side_to_move == checkers::Turn::kWhite ? "White" : "Black") + std::string(" (Human) to move.")
            );

            // Time-limited move
            auto start                = std::chrono::steady_clock::now();
            std::string notation_move = gui_->PromptForMove();
            auto finish               = std::chrono::steady_clock::now();
            float elapsed             = std::chrono::duration<float>(finish - start).count();

            if (elapsed > human_time_limit_) {
                // Timeout => sideToMove loses
                gui_->DisplayMessage("Time out! You lose.");
                if (side_to_move == checkers::Turn::kWhite) {
                    res = GameResult::kBlackWin;
                } else {
                    res = GameResult::kWhiteWin;
                }
                break;
            }

            // Try applying move
            auto [ok, msg, move] = AttemptMoveFromNotation(notation_move);
            if (!ok) {
                gui_->DisplayMessage("Invalid Move: " + msg);
                continue;
            } else {
                // Move was successful, display board
                gui_->DisplayBoard(engine_->GetBoard(), move);
            }
        } else {
            // AI logic
            gui_->DisplayMessage("AI is thinking...");

            auto moves = engine_->GenerateMoves();
            if (engine_->CheckGameResult() != GameResult::kInProgress) {
                break;
            }

            // Start MCTS
            auto start = std::chrono::steady_clock::now();
            checkers::mcts::MonteCarloTree tree(engine_->GetBoard(), side_to_move);
            float time_budget = ai_time_limit_;

            checkers::move_t best_move =
                tree.RunParallel(time_budget - 0.1f, checkers::kNumThreadsCPU);  // reserve 0.1s for overhead
            // Apply bestMove
            bool success = engine_->ApplyMove(best_move, false);
            if (!success) {
                // No moves => lose
                gui_->DisplayMessage("AI move invalid? Forcing loss.");
                if (side_to_move == checkers::Turn::kWhite) {
                    res = GameResult::kBlackWin;
                } else {
                    res = GameResult::kWhiteWin;
                }
                break;
            }
            // Log move in notation
            checkers::board_index_t from_sq =
                checkers::cpu::move_gen::DecodeMove<checkers::cpu::move_gen::MovePart::From>(best_move);
            checkers::board_index_t toSq =
                checkers::cpu::move_gen::DecodeMove<checkers::cpu::move_gen::MovePart::To>(best_move);

            // Convert to e.g. "d2-e3"
            auto SquareToNotation = [&](checkers::board_index_t sq) {
                int row       = sq / 4;
                int colOffset = sq % 4;
                int col       = (row % 2 == 0) ? (colOffset * 2) : (colOffset * 2 + 1);
                int rank      = 8 - row;  // row=0 => rank=8
                char file     = 'a' + col;
                std::stringstream ss;
                ss << file << rank;
                return ss.str();
            };
            std::string move_string = SquareToNotation(from_sq) + "-" + SquareToNotation(toSq);
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
                } else if (token == kSaveQuitToken) {
                    SaveRecord("game_record.txt");
                    std::cout << "Game record saved to game_record.txt\n";
                    return {false, "Game saved.", kInvalidMove};
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
        // If success, store in moveHistory
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
    std::cout << "From: " << static_cast<u32>(fromIdx) << ", To: " << static_cast<u32>(toIdx) << std::endl;

    // Encode
    checkers::move_t mv = (fromIdx) | (toIdx << 8);
    bool ok             = engine_->ApplyMove(mv, true);
    if (!ok)
        return {false, "Move not valid according to engine.", kInvalidMove};

    // Add to record
    move_history_.push_back(move_line);
    return {true, "", mv};
}

bool CheckersGame::ApplyMultiCapture(const std::vector<std::string> &fields)
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

checkers::board_index_t CheckersGame::NotationToIndex(const std::string &cell) const
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
    int row = 8 - rank;
    // isPlayable => same logic. We'll directly do the col logic:
    bool is_playable = ((row % 2 == 0 && file % 2 == 0) || (row % 2 == 1 && file % 2 == 1));
    if (!is_playable)
        return ~0;

    int col_offset = (row % 2 == 0) ? (file / 2) : ((file - 1) / 2);
    int idx        = row * 4 + col_offset;
    if (idx < 0 || idx >= 32)
        return ~0;
    return (checkers::board_index_t)idx;
}

void CheckersGame::SaveRecord(const std::string &recordFile) const
{
    if (recordFile.empty()) {
        // Print to stdout
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
