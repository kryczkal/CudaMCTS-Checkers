#ifndef MCTS_CHECKERS_INCLUDE_CUDA_CAPTURE_LOOKUP_TABLE_CUH_
#define MCTS_CHECKERS_INCLUDE_CUDA_CAPTURE_LOOKUP_TABLE_CUH_

#include "array"
#include "common/checkers_defines.hpp"

namespace checkers::gpu::apply_move
{
extern __constant__ board_t d_kCaptureLookUpTable[];
void InitializeCaptureLookupTable();
}  // namespace checkers::gpu::apply_move

#endif  // MCTS_CHECKERS_INCLUDE_CUDA_CAPTURE_LOOKUP_TABLE_CUH_
