#include <cassert>
#include <cstdio>
#include "cuda/board_helpers.cuh"
#include "cuda/move_selection.cuh"

namespace checkers::gpu::move_selection
{

__device__ __forceinline__ void RandomSelection(
    const board_index_t board_idx, const u32* d_whites, const u32* d_blacks, const u32* d_kings,
    // Moves
    const move_t* d_moves, const u8* d_move_counts, const move_flags_t* d_move_capture,
    const move_flags_t* d_per_board_flags,
    // Num Boards
    const u64 n_boards,
    // Seeds for randomness
    const u8* seeds,
    // Output
    move_t* d_best_moves
)
{
    // 1) Detect if this board has at least one capture flagged
    const move_flags_t flags_for_this_board = d_per_board_flags[board_idx];
    const bool capture_required = move_gen::ReadFlag(flags_for_this_board, MoveFlagsConstants::kCaptureFound) != 0U;

    // 2) Attempt to pick one piece in random wrap-around order
    const board_index_t board_fields_begin = board_idx * BoardConstants::kBoardSize;
    const board_index_t initial_figure_idx = seeds[board_idx] % BoardConstants::kBoardSize;

    move_t chosen_move = MoveConstants::kInvalidMove;

    // We'll do at most 32 iterations to find any piece with valid moves.
    for (board_index_t i = 0; i < BoardConstants::kBoardSize; i++) {
        board_index_t candidateSquare = (initial_figure_idx + i) % BoardConstants::kBoardSize;
        u8 count_for_candidate        = d_move_counts[board_fields_begin + candidateSquare];
        if (count_for_candidate == 0) {
            continue;
        }

        if (capture_required) {
            const move_flags_t capture_mask = d_move_capture[board_fields_begin + candidateSquare];
            if (capture_mask == 0) {
                continue;
            }

            // We know at least one sub-move is a capture => let's pick from them
            // We'll gather their sub-move indices into a small array.
            u8 capturingSubMoves[16];
            u8 capturingCount = 0;
            for (u8 sub = 0; sub < count_for_candidate; sub++) {
                // Is sub-th move a capture?
                bool is_capture = move_gen::ReadFlag(capture_mask, sub) != 0U;
                if (is_capture) {
                    capturingSubMoves[capturingCount++] = sub;
                }
            }

            // If capturingCount == 0, continue searching next square
            if (capturingCount == 0) {
                continue;
            }

            // Otherwise pick one capturing sub-move index at random
            u8 random_sub      = seeds[board_idx] % capturingCount;
            u8 chosen_sub_move = capturingSubMoves[random_sub];

            // Flatten out to find the actual move in d_moves
            const u64 moves_base  = (u64)board_idx * (BoardConstants::kBoardSize * move_gen::kNumMaxMovesPerPiece);
            const u64 piece_base  = (u64)candidateSquare * move_gen::kNumMaxMovesPerPiece;
            const u64 final_index = moves_base + piece_base + chosen_sub_move;

            chosen_move = d_moves[final_index];
            break;
        } else {
            // Normal (no forced captures) => pick any sub-move at random
            u8 random_sub_move_idx = seeds[board_idx] % count_for_candidate;

            const u64 moves_base  = (u64)board_idx * (BoardConstants::kBoardSize * move_gen::kNumMaxMovesPerPiece);
            const u64 piece_base  = (u64)candidateSquare * move_gen::kNumMaxMovesPerPiece;
            const u64 final_index = moves_base + piece_base + random_sub_move_idx;

            chosen_move = d_moves[final_index];
            break;
        }
    }
    d_best_moves[board_idx] = chosen_move;
}
__device__ __forceinline__ void SelectBestMovesForBoardIdx(
    const board_index_t board_idx, const u32* d_whites, const u32* d_blacks, const u32* d_kings, const move_t* d_moves,
    const u8* d_move_counts, const move_flags_t* d_move_capture_mask, const move_flags_t* d_per_board_flags,
    const u64 n_boards, const u8* seeds, move_t* d_best_moves
)
{
    RandomSelection(
        board_idx, d_whites, d_blacks, d_kings, d_moves, d_move_counts, d_move_capture_mask, d_per_board_flags,
        n_boards, seeds, d_best_moves
    );
}

__global__ void SelectBestMoves(
    // Board states (unused in random selection, but placeholders for expansions)
    const u32* d_whites, const u32* d_blacks, const u32* d_kings,
    // Moves
    const move_t* d_moves, const u8* d_move_counts, const move_flags_t* d_move_capture_mask,
    const move_flags_t* d_per_board_flags,
    // Number of boards
    const u64 n_boards,
    // Seeds
    const u8* seeds,
    // Output
    move_t* d_best_moves
)
{
    for (u64 board_idx = blockIdx.x * blockDim.x + threadIdx.x; board_idx < n_boards;
         board_idx += gridDim.x * blockDim.x) {
        SelectBestMovesForBoardIdx(
            board_idx, d_whites, d_blacks, d_kings, d_moves, d_move_counts, d_move_capture_mask, d_per_board_flags,
            n_boards, seeds, d_best_moves
        );
    }
}

}  // namespace checkers::gpu::move_selection
