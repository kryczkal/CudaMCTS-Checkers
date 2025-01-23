#include "cpu/apply_move.hpp"
#include "cpu/board_helpers.hpp"
#include "cpu/capture_lookup_table.hpp"

namespace checkers::cpu::apply_move
{
void ApplyMoveOnSingleBoard(move_t move, board_t &white_bits, board_t &black_bits, board_t &king_bits)
{
    if (move == kInvalidMove) {
        // No move to apply
        return;
    }

    board_index_t from = move_gen::DecodeMove<move_gen::MovePart::From>(move);
    board_index_t to   = move_gen::DecodeMove<move_gen::MovePart::To>(move);

    // Identify which side's piece moves
    bool from_is_white = (white_bits >> from) & 1U;
    bool from_is_black = (black_bits >> from) & 1U;
    bool from_is_king  = (king_bits >> from) & 1U;

    // Move the bit from 'from' to 'to'
    if (from_is_white) {
        white_bits |= (1U << to);
    }
    if (from_is_black) {
        black_bits |= (1U << to);
    }
    if (from_is_king) {
        king_bits |= (1U << to);
    }

    // Clear the original square
    white_bits &= ~(1U << from);
    black_bits &= ~(1U << from);
    king_bits &= ~(1U << from);

    board_t capture_mask = h_kCaptureLookUpTable[from][to];

    white_bits &= capture_mask;
    black_bits &= capture_mask;
    king_bits &= capture_mask;
}
}  // namespace checkers::cpu::apply_move
