#include "cuda/move_selection.cuh"

namespace checkers::gpu::move_selection
{
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
    d_best_moves[board_idx] = d_moves
        [board_idx * BoardConstants::kBoardSize * move_gen::kNumMaxMovesPerPiece +
         seeds[board_idx] % d_move_counts[board_idx]];
}

__global__ void SelectBestMoves(
    // Board States
    const u32* d_whites, const u32* d_blacks, const u32* d_kings,
    // Moves
    const move_t* d_moves, const u8* d_move_counts, const move_flags_t* d_move_capture_mask,
    const move_flags_t* d_per_board_flags,
    // Number of boards to process
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
