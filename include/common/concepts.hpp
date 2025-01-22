#ifndef MCTS_CHECKERS_INCLUDE_COMMON_CONCEPTS_HPP_
#define MCTS_CHECKERS_INCLUDE_COMMON_CONCEPTS_HPP_

#include "concepts"

template <typename T>
concept MaxComparable = requires(T a, T b) {
    { a < b } -> std::convertible_to<bool>;
};

namespace checkers::mcts
{
class MonteCarloTreeNode;
template <typename Func, typename EvalType>
concept EvalFunction = std::invocable<Func, MonteCarloTreeNode *> &&
                       std::convertible_to<std::invoke_result_t<Func, MonteCarloTreeNode *>, EvalType>;
}  // namespace checkers::mcts

#endif
