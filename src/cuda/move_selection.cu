#include <cassert>
#include <cstdio>
#include "cuda/move_selection.cuh"

namespace checkers::gpu::move_selection
{
/**
 * @brief For a single board, picks a random piece (cyclically from a random start) that has >=1 move,
 *        then picks a random sub-move from that piece.
 */
__device__ __forceinline__ void RandomSelection(
    const u64 board_idx,
    // Moves
    const move_t* d_moves, const u8* d_move_counts,
    // Seeds
    const u8* seeds,
    // Output
    move_t* d_best_moves
)
{
    // We'll store invalid if no moves are found.
    move_t chosen_move = MoveConstants::kInvalidMove;

    const u64 board_fields_begin = board_idx * BoardConstants::kBoardSize;

    // Start offset in the piece array is chosen by the seed.
    const board_index_t initial_figure_idx = seeds[board_idx] % BoardConstants::kBoardSize;

    // First, find any piece with at least 1 move, searching in a wrap-around manner.
    board_index_t chosen_figure_idx = BoardConstants::kBoardSize;  // sentinel (means not found)
    for (board_index_t i = 0; i < BoardConstants::kBoardSize; i++) {
        board_index_t candidate = (initial_figure_idx + i) % BoardConstants::kBoardSize;
        u8 count_for_candidate  = d_move_counts[board_fields_begin + candidate];
        if (count_for_candidate > 0) {
            chosen_figure_idx = candidate;
            break;
        }
    }

    // If a piece with moves was found:
    if (chosen_figure_idx < BoardConstants::kBoardSize) {
        // Number of valid moves for that piece
        u8 num_piece_moves = d_move_counts[board_fields_begin + chosen_figure_idx];

        // Pick a random sub-move from 0..(num_piece_moves-1)
        u8 random_sub_move_idx = seeds[board_idx] % num_piece_moves;

        // Now compute the flattened index into d_moves:
        const u64 moves_base     = board_idx * (BoardConstants::kBoardSize * move_gen::kNumMaxMovesPerPiece);
        const u64 piece_base     = chosen_figure_idx * move_gen::kNumMaxMovesPerPiece;
        const u64 final_move_idx = moves_base + piece_base + random_sub_move_idx;

        chosen_move = d_moves[final_move_idx];
    }

    d_best_moves[board_idx] = chosen_move;
}

/**
 * @brief CUDA kernel: One thread per board. We gather the best (random) move for that board.
 */
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
        RandomSelection(board_idx, d_moves, d_move_counts, seeds, d_best_moves);
    }
}

}  // namespace checkers::gpu::move_selection
