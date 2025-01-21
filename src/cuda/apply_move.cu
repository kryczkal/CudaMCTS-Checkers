#include "cuda/apply_move.cuh"
#include "cuda/board_helpers.cuh"
#include "cuda/capture_lookup_table.cuh"
#include "iostream"

namespace checkers::gpu::apply_move
{
__global__ void ApplyMove(
    // Board States
    board_t *d_whites, board_t *d_blacks, board_t *d_kings,
    // Moves
    const move_t *d_moves,
    // Number of boards to process
    const u64 n_boards
)
{
    u64 board_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; board_idx < n_boards; board_idx += gridDim.x * blockDim.x) {
        board_index_t from = move_gen::DecodeMove<move_gen::MovePart::From>(d_moves[board_idx]);
        board_index_t to   = move_gen::DecodeMove<move_gen::MovePart::To>(d_moves[board_idx]);

        // board[from] = board[to]
        d_whites[board_idx] |= ((d_whites[board_idx] >> from) & 1) << to;
        d_blacks[board_idx] |= ((d_blacks[board_idx] >> from) & 1) << to;
        d_kings[board_idx] |= ((d_kings[board_idx] >> from) & 1) << to;

        // board[from] = 0
        d_whites[board_idx] &= ~(1 << from);
        d_blacks[board_idx] &= ~(1 << from);
        d_kings[board_idx] &= ~(1 << from);

        // 0 out everything in between the move
        board_t kCaptureMask = d_kCaptureLookUpTable[from * BoardConstants::kBoardSize + to];
        d_blacks[board_idx] &= kCaptureMask;
        d_whites[board_idx] &= kCaptureMask;
        d_kings[board_idx] &= kCaptureMask;
    }
}
}  // namespace checkers::gpu::apply_move
