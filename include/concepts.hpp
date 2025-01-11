#ifndef CUDA_MCTS_CHECKRS_INCLUDE_CONCEPTS_HPP_
#define CUDA_MCTS_CHECKRS_INCLUDE_CONCEPTS_HPP_

#include <concepts>

namespace CudaMctsCheckers
{

template <typename T>
concept MaxComparable = requires(T a, T b) {
    { a < b } -> std::convertible_to<bool>;
};

class MonteCarloTreeNode;
template <typename Func, typename EvalType>
concept EvalFunction =
    std::invocable<Func, MonteCarloTreeNode *> &&
    std::convertible_to<std::invoke_result_t<Func, MonteCarloTreeNode *>, EvalType>;
}  // namespace CudaMctsCheckers

#endif  // CUDA_MCTS_CHECKRS_INCLUDE_CONCEPTS_HPP_