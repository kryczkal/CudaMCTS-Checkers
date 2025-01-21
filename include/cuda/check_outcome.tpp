#ifndef MCTS_CHECKERS_INCLUDE_CUDA_CHECK_OUTCOME_TPP_
#define MCTS_CHECKERS_INCLUDE_CUDA_CHECK_OUTCOME_TPP_

namespace checkers::gpu::check_outcome
{
template <Turn turn>
__device__ __forceinline__ void CheckGameOutcomeBoardIdx(
    const u64 board_idx, const board_t *d_whites, const board_t *d_blacks, u8 *outcomes, const u64 n_boards
)
{
    outcomes[board_idx] = d_blacks[board_idx] == 0 ? kWhiteWin : outcomes[board_idx];
    outcomes[board_idx] = d_whites[board_idx] == 0 ? kBlackWin : outcomes[board_idx];
    // Assume that outcomes is zeroed out, so this leaves unfinished games at 0 (kInProgress)
}

template <Turn turn>
__global__ void CheckGameOutcome(const board_t *d_whites, const board_t *d_blacks, u8 *outcomes, const u64 n_boards)
{
    u64 board_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; board_idx < n_boards; board_idx += gridDim.x * blockDim.x) {
        CheckGameOutcomeBoardIdx<turn>(board_idx, d_whites, d_blacks, outcomes, n_boards);
    }
}
}  // namespace checkers::gpu::check_outcome

#endif  // MCTS_CHECKERS_INCLUDE_CUDA_CHECK_OUTCOME_TPP_
