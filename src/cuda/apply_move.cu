#include <iostream>
#include "common/checkers_defines.hpp"
#include "cuda/apply_move.cuh"
#include "cuda/board_helpers.cuh"
#include "cuda/capture_lookup_table.cuh"

namespace checkers::gpu::apply_move
{

__device__ void ApplyMoveOnSingleBoard(move_t move, board_t& white_bits, board_t& black_bits, board_t& king_bits)
{
    board_index_t from = move_gen::DecodeMove<move_gen::MovePart::From>(move);
    board_index_t to   = move_gen::DecodeMove<move_gen::MovePart::To>(move);
    if (move == kInvalidMove) {
        // No move to apply
        return;
    }

    // Move the bits from "from" to "to"
    const bool from_is_white = ReadFlag(white_bits, from);
    const bool from_is_black = ReadFlag(black_bits, from);
    const bool from_is_king  = ReadFlag(king_bits, from);

    if (from_is_white) {
        white_bits |= (1ULL << to);
    }
    if (from_is_black) {
        black_bits |= (1ULL << to);
    }
    if (from_is_king) {
        king_bits |= (1ULL << to);
    }

    // Clear the original square
    white_bits &= ~(1ULL << from);
    black_bits &= ~(1ULL << from);
    king_bits &= ~(1ULL << from);

    // Eliminate captured pieces (using the precomputed capture mask)
    // d_kCaptureLookUpTable is in constant memory
    const board_t capture_mask = d_kCaptureLookUpTable[from * BoardConstants::kBoardSize + to];
    white_bits &= capture_mask;
    black_bits &= capture_mask;
    king_bits &= capture_mask;
}

__global__ void ApplyMove(
    board_t* d_whites, board_t* d_blacks, board_t* d_kings, const move_t* d_moves, const u64 n_boards
)
{
    u64 idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < n_boards) {
        ApplyMoveOnSingleBoard(d_moves[idx], d_whites[idx], d_blacks[idx], d_kings[idx]);
        idx += gridDim.x * blockDim.x;
    }
}

}  // namespace checkers::gpu::apply_move
