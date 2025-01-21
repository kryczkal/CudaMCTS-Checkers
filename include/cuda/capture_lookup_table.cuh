#include "cuda/apply_move.cuh"
#include "cuda/board_helpers.cuh"
#include "cuda/checkers_defines.hpp"

namespace checkers::gpu::apply_move
{
__constant__ board_t d_kCaptureLookUpTable[BoardConstants::kBoardSize * BoardConstants::kBoardSize];
}
