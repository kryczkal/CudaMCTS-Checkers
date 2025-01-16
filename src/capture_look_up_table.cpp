#include <board.hpp>

namespace CudaMctsCheckers
{

std::array<std::array<Board::HalfBoard, Board::kHalfBoardSize>, Board::kHalfBoardSize>
    kCaptureLookUpTable = []() {
        std::array<std::array<Board::HalfBoard, Board::kHalfBoardSize>, Board::kHalfBoardSize>
            table{};
        for (Board::IndexType i = 0; i < Board::kHalfBoardSize; ++i) {
            for (Board::IndexType j = i + 1; j < Board::kHalfBoardSize; ++j) {
                Board::IndexType current;
                bool found;

                // Try to reach j from i
                table[i][j] = 0;
                current     = i;
                found       = false;

                while (current != Board::kInvalidIndex) {
                    table[i][j] |= Board::HalfBoard(1) << current;
                    if (current == j) {
                        found = true;
                        break;
                    }
                    Board::IndexType next =
                        Board::GetRelativeMoveIndex<MoveDirection::kDownLeft>(current);
                    current = next;
                }

                if (found) {
                    table[i][j] = ~table[i][j];
                    table[j][i] = table[i][j];
                    continue;
                }

                table[i][j] = 0;
                current     = i;
                found       = false;

                while (current != Board::kInvalidIndex) {
                    table[i][j] |= Board::HalfBoard(1) << current;
                    if (current == j) {
                        found = true;
                        break;
                    }
                    Board::IndexType next =
                        Board::GetRelativeMoveIndex<MoveDirection::kDownRight>(current);
                    current = next;
                }

                if (found) {
                    table[i][j] = ~table[i][j];
                    table[j][i] = table[i][j];
                    continue;
                }

                table[i][j] = ~0;
                table[j][i] = table[i][j];
            }
        }
        return table;
    }();

}  // namespace CudaMctsCheckers
