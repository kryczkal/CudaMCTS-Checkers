#ifndef MCTS_CHECKERS_INCLUDE_COMMON_PARALLEL_HPP_
#define MCTS_CHECKERS_INCLUDE_COMMON_PARALLEL_HPP_

#include <cmath>
#include <thread>
#include "types.hpp"

namespace checkers
{
const u64 kNumThreadsCPU = ceil([]() -> u64 {
    u64 threads = std::thread::hardware_concurrency();
    return (threads == 0) ? 1 : threads;
}() * 1.5);
}  // namespace checkers

#endif  // MCTS_CHECKERS_INCLUDE_COMMON_PARALLEL_HPP_
