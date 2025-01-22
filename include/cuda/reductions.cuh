#ifndef MCTS_CHECKERS_INCLUDE_CUDA_REDUCTIONS_CUH_
#define MCTS_CHECKERS_INCLUDE_CUDA_REDUCTIONS_CUH_

#include <cstddef>
#include "types.hpp"

/**
 * @brief  Sums an array of n elements (each a u8) on the device, producing a u64 sum.
 *         The sum is accumulated in 64-bit integer, which supports up to 2*n range
 *         (since results can be 0,1,2).
 *
 * @param d_in  Pointer to device array of length n.
 * @param n     Number of elements to sum.
 * @return      The sum (in host memory).
 */
u64 DeviceSumU8(const u8* d_in, size_t n);

#endif  // MCTS_CHECKERS_INCLUDE_CUDA_REDUCTIONS_CUH_
