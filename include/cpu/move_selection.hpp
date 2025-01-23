#ifndef MCTS_CHECKERS_INCLUDE_CPU_MOVE_SELECTION_HPP_
#define MCTS_CHECKERS_INCLUDE_CPU_MOVE_SELECTION_HPP_

namespace checkers::cpu::move_selection
{

move_t SelectRandomMoveForSingleBoard(
    board_t white_bits, board_t black_bits, board_t king_bits, const move_t* moves, const u8* move_counts,
    const move_flags_t* capture_masks, move_flags_t per_board_flags, u8& seed
);

move_t SelectBestMoveForSingleBoard(
    board_t white_bits, board_t black_bits, board_t king_bits, const move_t* moves, const u8* move_counts,
    const move_flags_t* capture_masks, move_flags_t per_board_flags, u8& seed
);

}  // namespace checkers::cpu::move_selection

#endif
