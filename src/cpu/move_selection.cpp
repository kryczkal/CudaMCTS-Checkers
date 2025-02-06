#include "common/checkers_defines.hpp"
#include "cpu/board_helpers.hpp"

namespace checkers::cpu::move_selection
{
move_t SelectRandomMoveForSingleBoard(
    board_t /*white_bits*/, board_t /*black_bits*/, board_t /*king_bits*/, const move_t *moves, const u8 *move_counts,
    const move_flags_t *capture_masks, move_flags_t per_board_flags, u32 &seed
)
{
    using checkers::cpu::ReadFlag;

    bool capture_required = ReadFlag(per_board_flags, MoveFlagsConstants::kCaptureFound);

    move_t chosen_move = kInvalidMove;

    board_index_t start_sq = seed % BoardConstants::kBoardSize;

    for (board_index_t i = 0; i < BoardConstants::kBoardSize; i++) {
        board_index_t sq = (start_sq + i) % BoardConstants::kBoardSize;

        u8 count = move_counts[sq];
        if (count == 0) {
            continue;
        }

        if (capture_required) {
            move_flags_t cmask = capture_masks[sq];
            if (cmask == 0) {
                // no capturing sub-moves for this square
                continue;
            }

            // gather the sub-moves that are captures
            u8 capturing_indices[16];
            u8 capturing_count = 0;
            for (u8 sub = 0; sub < count; sub++) {
                bool isCap = ReadFlag(cmask, sub);
                if (isCap) {
                    capturing_indices[capturing_count++] = sub;
                }
            }
            if (capturing_count == 0) {
                continue;
            }

            // pick one capturing sub-move
            u8 chosen_sub = seed % capturing_count;
            chosen_move   = moves[sq * kNumMaxMovesPerPiece + capturing_indices[chosen_sub]];
            SimpleRand(seed);
            break;
        } else {
            // no capture forced => pick any sub-move
            u8 chosen_sub = seed % count;
            chosen_move   = moves[sq * kNumMaxMovesPerPiece + chosen_sub];
            SimpleRand(seed);
            break;
        }
    }

    return chosen_move;
}

move_t SelectBestMoveForSingleBoard(
    board_t white_bits, board_t black_bits, board_t king_bits, const move_t *moves, const u8 *move_counts,
    const move_flags_t *capture_masks, move_flags_t per_board_flags, u32 &seed
)
{
    return SelectRandomMoveForSingleBoard(
        white_bits, black_bits, king_bits, moves, move_counts, capture_masks, per_board_flags, seed
    );
}
}  // namespace checkers::cpu::move_selection
