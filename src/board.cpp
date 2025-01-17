#include <board.hpp>
#include <iomanip>
#include <iostream>

namespace CudaMctsCheckers
{
Board::Board() : white_pieces(0), black_pieces(0), kings(0), time_from_non_reversible_move(0) {}

std::ostream& operator<<(std::ostream& os, const Board& board)
{
    // Define the symbols
    const char EMPTY      = '.';
    const char WHITE      = 'w';
    const char BLACK      = 'b';
    const char WHITE_KING = 'W';
    const char BLACK_KING = 'B';

    // Iterate over each row (0 to 7)
    for (int row = 0; row < 8; ++row) {
        // Print row number (8 to 1)
        os << (8 - row) << " | ";

        // Iterate over each column (0 to 7)
        for (int col = 0; col < 8; ++col) {
            bool isPlayable;
            if (row % 2 == 0) {
                isPlayable = (col % 2 == 0);
            } else {
                isPlayable = (col % 2 != 0);
            }

            if (isPlayable) {
                // Calculate the index
                int index   = row * 4 + (col / 2);
                char symbol = EMPTY;

                if (board.IsPieceAt<BoardCheckType::kWhite>(index)) {
                    symbol = (board.IsPieceAt<BoardCheckType::kKings>(index)) ? WHITE_KING : WHITE;
                } else if (board.IsPieceAt<BoardCheckType::kBlack>(index)) {
                    symbol = (board.IsPieceAt<BoardCheckType::kKings>(index)) ? BLACK_KING : BLACK;
                }

                os << symbol << " ";
            } else {
                // Non-playable squares
                os << "  ";
            }
        }
        os << "\n";
    }

    // Print column labels
    os << "    a b c d e f g h\n";

    return os;
}

}  // namespace CudaMctsCheckers
