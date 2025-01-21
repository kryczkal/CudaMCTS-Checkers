#ifndef MCTS_CHECKERS_INCLUDE_CUDA_CHECK_OUTCOME_CUH_
#define MCTS_CHECKERS_INCLUDE_CUDA_CHECK_OUTCOME_CUH_

#include "types.hpp"

namespace checkers::gpu::check_outcome
{
static constexpr u8 kInProgress = 0;
static constexpr u8 kWhiteWin   = 1;
static constexpr u8 kBlackWin   = 2;

template <Turn turn>
__device__ __forceinline__ void CheckGameOutcomeBoardIdx(
    const u64 board_idx, const board_t *d_whites, const board_t *d_blacks, u8 *outcomes, const u64 n_boards
);

template <Turn turn>
__global__ void CheckGameOutcome(const board_t *d_whites, const board_t *d_blacks, u8 *outcomes, const u64 n_boards);
}  // namespace checkers::gpu::check_outcome

#include "check_outcome.tpp"

#endif  // MCTS_CHECKERS_INCLUDE_CUDA_CHECK_OUTCOME_CUH_
