#include <cassert>
#include <cstdio>
#include "common/checkers_defines.hpp"
#include "cuda/board_helpers.cuh"
#include "cuda/move_selection.cuh"

namespace checkers::gpu::move_selection
{

__device__ move_t SelectRandomMoveForSingleBoard(
    const board_t white_bits, const board_t black_bits, const board_t king_bits, const move_t* moves,
    const u8* move_counts, const move_flags_t* capture_masks, const move_flags_t per_board_flags, u8& seed
)
{
    using gpu::move_gen::kNumMaxMovesPerPiece;

    // Detect if the board has a capture flagged
    const bool capture_required = ReadFlag(per_board_flags, move_gen::MoveFlagsConstants::kCaptureFound);

    move_t chosen_move = kInvalidMove;

    // Attempt picking one piece in random wrap-around order
    const board_index_t initial_figure_idx = seed % BoardConstants::kBoardSize;

    for (board_index_t i = 0; i < BoardConstants::kBoardSize; i++) {
        board_index_t candidate_square = (initial_figure_idx + i) % BoardConstants::kBoardSize;

        // If no sub-moves for this square, skip
        u8 count_for_candidate = move_counts[candidate_square];
        if (count_for_candidate == 0) {
            continue;
        }

        // If capturing is required, check if this square has any capturing sub-move
        if (capture_required) {
            move_flags_t capture_mask = capture_masks[candidate_square];
            if (capture_mask == 0) {
                continue;  // no captures from this square
            }

            // Gather indices of sub-moves that are captures
            u8 capturing_sub_moves[16];
            u8 captuing_count = 0;
            for (u8 sub = 0; sub < count_for_candidate; sub++) {
                const bool is_capture = ReadFlag(capture_mask, sub);
                if (is_capture) {
                    capturing_sub_moves[captuing_count++] = sub;
                }
            }
            if (captuing_count == 0) {
                // Shouldn’t happen if capture_mask != 0, but just in case
                continue;
            }

            // Pick one capturing sub-move at random
            const u8 chosen_sub_idx = seed % captuing_count;
            const u8 sub_move_index = capturing_sub_moves[chosen_sub_idx];
            chosen_move             = moves[candidate_square * kNumMaxMovesPerPiece + sub_move_index];

            // Update the seed
            seed = static_cast<u8>(seed + 13);

            break;
        } else {
            // If no capture forced, pick any sub-move at random
            const u8 chosen_sub_idx = seed % count_for_candidate;
            chosen_move             = moves[candidate_square * kNumMaxMovesPerPiece + chosen_sub_idx];

            // Update the seed
            seed = static_cast<u8>(seed + 13);

            break;
        }
    }

    return chosen_move;
}

__device__ move_t SelectBestMoveForSingleBoard(
    const board_t white_bits, const board_t black_bits, const board_t king_bits, const move_t* moves,
    const u8* move_counts, const move_flags_t* capture_masks, const move_flags_t per_board_flags, u8& seed
)
{
    // Random move for now, but possibility of adding different types of choosing (heuristic, etc.)
    return SelectRandomMoveForSingleBoard(
        white_bits, black_bits, king_bits, moves, move_counts, capture_masks, per_board_flags, seed
    );
}

__global__ void SelectBestMoves(
    const board_t* d_whites, const board_t* d_blacks, const board_t* d_kings, const move_t* d_moves,
    const u8* d_move_counts, const move_flags_t* d_move_capture_mask, const move_flags_t* d_per_board_flags,
    const u64 n_boards, u8* d_seeds, move_t* d_best_moves
)
{
    using gpu::move_gen::kNumMaxMovesPerPiece;

    // Each thread handles exactly one board
    u64 idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < n_boards) {
        board_t white_bits = d_whites[idx];
        board_t black_bits = d_blacks[idx];
        board_t king_bits  = d_kings[idx];
        move_flags_t flags = d_per_board_flags[idx];
        u8& seedRef        = d_seeds[idx];

        const move_t* boardMoves          = &d_moves[idx * (BoardConstants::kBoardSize * kNumMaxMovesPerPiece)];
        const u8* boardMoveCounts         = &d_move_counts[idx * BoardConstants::kBoardSize];
        const move_flags_t* boardCaptures = &d_move_capture_mask[idx * BoardConstants::kBoardSize];

        move_t chosenMove = SelectBestMoveForSingleBoard(
            white_bits, black_bits, king_bits, boardMoves, boardMoveCounts, boardCaptures, flags, seedRef
        );

        d_best_moves[idx] = chosenMove;

        idx += gridDim.x * blockDim.x;
    }
}

}  // namespace checkers::gpu::move_selection
