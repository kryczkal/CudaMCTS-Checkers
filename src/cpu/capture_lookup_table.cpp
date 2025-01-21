#include "cpu/capture_lookup_table.hpp"
#include <array>
#include "checkers_defines.hpp"
#include "cpu/board_helpers.hpp"

namespace checkers::cpu::move_gen
{
// Define an invalid index constant
constexpr board_index_t kInvalidIndex = static_cast<board_index_t>(~0u);

std::array<std::array<board_t, BoardConstants::kBoardSize>, BoardConstants::kBoardSize> h_kCaptureLookUpTable = []() {
    std::array<std::array<board_t, BoardConstants::kBoardSize>, BoardConstants::kBoardSize> table{};

    for (board_index_t from = 0; from < BoardConstants::kBoardSize; ++from) {
        for (board_index_t to = from + 1; to < BoardConstants::kBoardSize; ++to) {
            board_t mask = 0;
            bool found   = false;

            // Attempt to find a capture path in the DownLeft direction
            board_index_t current = from;
            while (current != kInvalidIndex) {
                mask |= (1u << current);

                if (current == to) {
                    found = true;
                    break;
                }

                // Get the next adjacent index in the DownLeft direction
                current = GetAdjacentIndex<Direction::kDownLeft>(current);
            }

            if (found) {
                // Remove the 'from' and 'to' positions from the mask
                mask &= ~(1u << from);
                mask &= ~(1u << to);

                // Invert the mask to indicate captured pieces
                mask = ~mask;

                table[from][to] = mask;
                table[to][from] = mask;  // Ensure symmetry
                continue;
            }

            // Reset mask and attempt in the DownRight direction
            mask    = 0;
            found   = false;
            current = from;

            while (current != kInvalidIndex) {
                mask |= (1u << current);

                if (current == to) {
                    found = true;
                    break;
                }

                // Get the next adjacent index in the DownRight direction
                current = GetAdjacentIndex<Direction::kDownRight>(current);
            }

            if (found) {
                // Remove the 'from' and 'to' positions from the mask
                mask &= ~(1u << from);
                mask &= ~(1u << to);

                // Invert the mask to indicate captured pieces
                mask = ~mask;

                table[from][to] = mask;
                table[to][from] = mask;  // Ensure symmetry
                continue;
            }

            // If no path is found in either direction, set mask to all bits
            table[from][to] = ~0u;
            table[to][from] = table[from][to];
        }
    }

    return table;
}();
}  // namespace checkers::cpu::move_gen
