#include "game/cli_gui.hpp"
#include <cctype>
#include <iomanip>
#include <sstream>
#include "cpu/board_helpers.hpp"

namespace checkers
{

/**
 * @brief Helper function to convert an index 0..31 to e.g. 'd2' in checkers notation for printing.
 */
static std::string IndexToNotation(checkers::board_index_t idx)
{
    int row        = idx / 4;  // from top
    int col_offset = idx % 4;
    int col        = (row % 2 == 0) ? (col_offset * 2) : (col_offset * 2 + 1);
    int rank       = 8 - row;
    char file_char = 'a' + col;
    std::stringstream ss;
    ss << file_char << rank;
    return ss.str();
}

// Define ANSI color codes for styling
const std::string RED_BG       = "\033[41m";
const std::string WHITE_BG     = "\033[47m";
const std::string WHITE_COLOR  = "\033[37m";
const std::string BLACK_COLOR  = "\033[30m";
const std::string RED_ON_BLACK = "\033[31;40m";
const std::string RESET        = "\033[0m";

// Define piece representations
const std::string WHITE_PIECE  = "w";
const std::string WHITE_KING   = "W";
const std::string BLACK_PIECE  = "b";
const std::string BLACK_KING   = "B";
const std::string EMPTY_SQUARE = " ";

void CliGui::DisplayBoard(const checkers::cpu::Board &board)
{
    // Print column headers with padding
    std::cout << "\n    a   b   c   d   e   f   g   h\n";

    // Iterate through each row
    for (int row = 0; row < 8; row++) {
        // Print horizontal separator before each row
        std::cout << "  +---+---+---+---+---+---+---+---+\n";

        // Print row number at the start
        std::cout << (8 - row) << " |";

        // Iterate through each column
        for (int col = 0; col < 8; col++) {
            // Determine if the square is playable
            bool is_playable = ((row % 2 == 0 && col % 2 == 0) || (row % 2 == 1 && col % 2 == 1));

            if (!is_playable) {
                std::cout << "   " << RESET << "|";
                continue;
            }

            // Convert row and column to bitboard index
            int r     = row;
            int cHalf = (r % 2 == 0) ? (col / 2) : ((col - 1) / 2);
            int idx   = r * 4 + cHalf;

            // Determine piece presence and type
            const bool is_white = cpu::ReadFlag(board.white, idx);
            const bool is_black = cpu::ReadFlag(board.black, idx);
            const bool is_king  = cpu::ReadFlag(board.kings, idx);

            std::string piece = EMPTY_SQUARE;

            std::cout << "\033[1m";
            std::cout << "\033[52m";
            std::cout << RED_ON_BLACK;
            if (is_white || is_black) {
                if (is_white) {
                    // Assign appropriate piece representation
                    piece = is_king ? WHITE_KING : WHITE_PIECE;
                    // Print white piece with color and padding
                    std::cout << " " << piece << " ";
                } else if (is_black) {
                    // Assign appropriate piece representation
                    piece = is_king ? BLACK_KING : BLACK_PIECE;
                    // Print black piece with color and padding
                    std::cout << " " << piece << " ";
                }
            } else {
                // Empty playable square with white background and padding
                std::cout << "   ";
            }
            std::cout << RESET << "|";
        }

        // Print row number at the end
        std::cout << " " << (8 - row) << "\n";
    }

    // Print the final horizontal separator after the last row
    std::cout << "  +---+---+---+---+---+---+---+---+\n";

    // Print column headers again with padding
    std::cout << "    a   b   c   d   e   f   g   h\n\n";
}

void CliGui::DisplayMessage(const std::string &msg) { std::cout << msg << std::endl; }

std::string CliGui::PromptForMove()
{
    std::cout << "Enter your move (ex: d2-e3 for normal, d2:f4:d6 for multi-capture): ";
    std::string input;
    std::getline(std::cin, input);
    return input;
}

}  // namespace checkers
