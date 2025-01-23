#include "cpu/capture_lookup_table.hpp"
#include <array>
#include "common/checkers_defines.hpp"
#include "cpu/board_helpers.hpp"

namespace checkers::cpu::apply_move
{
std::array<std::array<board_t, BoardConstants::kBoardSize>, BoardConstants::kBoardSize> h_kCaptureLookUpTable = []() {
    std::array<std::array<board_t, BoardConstants::kBoardSize>, BoardConstants::kBoardSize> table{};

    for (board_index_t from = 0; from < BoardConstants::kBoardSize; ++from) {
        for (board_index_t to = from + 1; to < BoardConstants::kBoardSize; ++to) {
            // We'll try to walk from 'from' to 'to' in DownLeft, then DownRight. If neither hits 'to', mask = ~0u.
            board_t mask = 0;
            bool found   = false;

            //--------------- Try DownLeft ---------------
            {
                board_index_t current = from;
                while (true) {
                    // Add current to path
                    mask |= (1u << current);

                    // Check if we reached the 'to' square
                    if (current == to) {
                        found = true;
                        break;
                    }

                    // If on edge, we cannot keep going in this direction
                    if (move_gen::IsOnEdge<move_gen::Direction::kDownLeft>(current)) {
                        break;
                    }

                    // Get the next diagonal position
                    current = move_gen::GetAdjacentIndex<move_gen::Direction::kDownLeft>(current);
                }

                if (found) {
                    // Exclude 'from' and 'to' from the path bits so they are NOT captured
                    mask &= ~(1u << from);
                    mask &= ~(1u << to);

                    // Invert to indicate these path-squares should be captured (0 bits)
                    // and everything else is left intact (1 bits).
                    mask = ~mask;

                    table[from][to] = mask;
                    table[to][from] = mask;  // Symmetric
                    continue;                // Done for this pair
                }
            }

            //--------------- Try DownRight ---------------
            {
                // Reset mask for second attempt
                mask                  = 0;
                board_index_t current = from;
                while (true) {
                    // Add current to path
                    mask |= (1u << current);

                    // Check if we reached the 'to' square
                    if (current == to) {
                        found = true;
                        break;
                    }

                    // If on edge, we cannot keep going in this direction
                    if (move_gen::IsOnEdge<move_gen::Direction::kDownRight>(current)) {
                        break;
                    }

                    // Get the next diagonal position
                    current = move_gen::GetAdjacentIndex<move_gen::Direction::kDownRight>(current);
                }

                if (found) {
                    // Exclude 'from' and 'to'
                    mask &= ~(1u << from);
                    mask &= ~(1u << to);

                    // Invert mask
                    mask = ~mask;

                    table[from][to] = mask;
                    table[to][from] = mask;  // Symmetric
                    continue;
                }
            }

            // If neither direction found a path to, then no captures
            table[from][to] = ~0u;
            table[to][from] = ~0u;  // Symmetric
        }
    }

    return table;
}();

}  // namespace checkers::cpu::apply_move
