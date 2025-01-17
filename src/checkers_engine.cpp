#include <checkers_engine.hpp>
#include <cstdlib>  // for rand()
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace CudaMctsCheckers
{

CheckersEngine::CheckersEngine(const Board &board, Turn turn) : board_(board), current_turn_(turn)
{
    last_moves_output_.no_moves = false;
}
CheckersEngine::CheckersEngine()
{
    last_moves_output_.no_moves = false;
    board_                      = Board();
    current_turn_               = Turn::kWhite;

    for (int i = 0; i < 12; i++) {
        board_.SetPieceAt<BoardCheckType::kWhite>(i + 20);
        board_.SetPieceAt<BoardCheckType::kBlack>(i);
    }
}

const Board &CheckersEngine::GetBoard() const { return board_; }

Turn CheckersEngine::GetCurrentTurn() const { return current_turn_; }

MoveGenerationOutput CheckersEngine::GenerateCurrentPlayerMoves()
{
    MoveGenerationOutput moves_output;
    if (current_turn_ == Turn::kWhite) {
        moves_output = MoveGenerator::GenerateMovesForPlayerCpu<BoardCheckType::kWhite>(board_);
    } else {
        moves_output = MoveGenerator::GenerateMovesForPlayerCpu<BoardCheckType::kBlack>(board_);
    }
    last_moves_output_ = moves_output;
    return moves_output;
}

bool CheckersEngine::IsMoveValid(
    Board::IndexType from_idx, Move::Type to_idx, const MoveGenerationOutput &possible_moves
)
{  // Validate move
    bool found = false;
    bool capture_possible =
        possible_moves.capture_moves_bitmask[MoveGenerationOutput::CaptureFlagIndex];

    for (u32 i = from_idx * Move::kNumMaxPossibleMovesPerPiece;
         i < (from_idx + 1) * Move::kNumMaxPossibleMovesPerPiece; ++i) {
        if (possible_moves.possible_moves[i] == to_idx) {
            if (capture_possible && !possible_moves.capture_moves_bitmask[i]) {
                break;
            }
            found = true;
            break;
        }
    }
    if (!found) {
        return false;
    } else {
        return true;
    }
}

void CheckersEngine::PlayMove(unsigned char from_idx, unsigned char to_idx, bool is_capture)
{
    if (current_turn_ == Turn::kWhite) {
        board_.ApplyMove<BoardCheckType::kWhite>(from_idx, to_idx, is_capture);
    } else {
        board_.ApplyMove<BoardCheckType::kBlack>(from_idx, to_idx, is_capture);
    }
}

bool CheckersEngine::ApplyRandomMove()
{
    auto moves_output      = GetPrecomputedMovesOrGenerate();
    has_precomputed_moves_ = false;
    if (moves_output.no_moves)
        return false;

    bool capture_possible =
        moves_output.capture_moves_bitmask[MoveGenerationOutput::CaptureFlagIndex];

    u32 random_move_index =
        (rand() % Move::kNumMoveArrayForPlayerSize) % Move::kNumMaxPossibleMovesPerPiece;
    u32 i;
    for (i = 0; i < Board::kHalfBoardSize; i++) {
        if (capture_possible) {
            while (moves_output.possible_moves[random_move_index] != Move::kInvalidMove &&
                   !moves_output.capture_moves_bitmask[random_move_index]) {
                random_move_index = (random_move_index + 1) % Move::kNumMoveArrayForPlayerSize;
            }
        }

        if (moves_output.possible_moves[random_move_index] != Move::kInvalidMove) {
            break;
        }
        if (random_move_index < Move::kNumMoveArrayForPlayerSize - 1 &&
            moves_output.possible_moves[random_move_index + 1] != Move::kInvalidMove) {
            random_move_index++;
            break;
        }

        random_move_index = (random_move_index + Move::kNumMaxPossibleMovesPerPiece -
                             (random_move_index % Move::kNumMaxPossibleMovesPerPiece)) %
                            Move::kNumMoveArrayForPlayerSize;
    }

    if (i >= Board::kHalfBoardSize) {
        return false;
    }

    u32 move_indexes_stack[Move::kNumMaxPossibleMovesPerPiece];
    u8 stack_size                  = 0;
    move_indexes_stack[stack_size] = random_move_index;
    stack_size++;

    random_move_index++;
    while (moves_output.possible_moves[random_move_index] != Move::kInvalidMove) {
        if (capture_possible && !moves_output.capture_moves_bitmask[random_move_index]) {
            random_move_index++;
            continue;
        }
        move_indexes_stack[stack_size] = random_move_index;
        stack_size++;
        random_move_index++;
    }

    random_move_index = move_indexes_stack[rand() % stack_size];

    Board::IndexType from = Move::DecodeOriginIndex(random_move_index);
    Move::Type to         = moves_output.possible_moves[random_move_index];

    assert(from != Board::kInvalidIndex && to != Move::kInvalidMove);

    return ApplyMove<ApplyMoveType::kNoValidate>(from, to);
}

void CheckersEngine::SwitchTurnIfNoChainCapture(bool capture_performed)
{
    if (capture_performed) {
        if ((GenerateCurrentPlayerMoves())
                .capture_moves_bitmask[MoveGenerationOutput::CaptureFlagIndex]) {
            has_precomputed_moves_ = true;
            return;
        }
        has_precomputed_moves_ = false;
    }

    current_turn_ = (current_turn_ == Turn::kWhite) ? Turn::kBlack : Turn::kWhite;
}

GameResult CheckersEngine::CheckGameResult() const
{
    if (last_moves_output_.no_moves) {
        return (current_turn_ == Turn::kWhite) ? GameResult::kBlackWin : GameResult::kWhiteWin;
    }

    if (board_.white_pieces == 0) {
        return GameResult::kBlackWin;
    } else if (board_.black_pieces == 0) {
        return GameResult::kWhiteWin;
    } else {
        if (board_.time_from_non_reversible_move >= 40) {
            return GameResult::kDraw;
        }
        return GameResult::kInProgress;
    }
}

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

            bool success = ApplyMove<ApplyMoveType::kValidate>(from_idx, to_idx);
            if (!success) {
                error_message =
                    "Failed to apply move on line " + std::to_string(line_number) + ": " + move_str;
                return false;
            }
            std::cout << board_;
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

void CheckersEngine::PromoteAll() { board_.PromoteAll(); }

void CheckersEngine::UpdateTimeFromNonReversibleMove(Move::Type played_move, bool was_capture)
{
    if (board_.IsPieceAt<BoardCheckType::kKings>(played_move) && !was_capture) {
        board_.time_from_non_reversible_move++;
    } else {
        board_.time_from_non_reversible_move = 0;
    }
}

MoveGenerationOutput CheckersEngine::GetPrecomputedMovesOrGenerate()
{
    return has_precomputed_moves_ ? last_moves_output_ : GenerateCurrentPlayerMoves();
}

}  // namespace CudaMctsCheckers
