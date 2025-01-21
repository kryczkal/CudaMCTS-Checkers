#ifndef MCTS_CHECKERS_INCLUDE_CUDA_CAPTURE_LOOKUP_TABLE_CUH_
#define MCTS_CHECKERS_INCLUDE_CUDA_CAPTURE_LOOKUP_TABLE_CUH_

#include "array"
#include "checkers_defines.hpp"
#include "cuda/apply_move.cuh"
#include "cuda/board_helpers.cuh"
#include "cuda_utils.cuh"

namespace checkers::gpu::apply_move
{
extern __constant__ board_t d_kCaptureLookUpTable[];
void InitializeCaptureLookupTable();
}  // namespace checkers::gpu::apply_move

#endif  // MCTS_CHECKERS_INCLUDE_CUDA_CAPTURE_LOOKUP_TABLE_CUH_
