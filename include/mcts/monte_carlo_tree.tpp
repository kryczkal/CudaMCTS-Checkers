#ifndef MCTS_CHECKERS_INCLUDE_MCTS_MONTE_CARLO_TREE_TPP_
#define MCTS_CHECKERS_INCLUDE_MCTS_MONTE_CARLO_TREE_TPP_

#include "cassert"
#include "iostream"

namespace checkers::mcts
{
template <MaxComparable EvalType, EvalFunction<EvalType> auto EvalFunc>
move_t MonteCarloTree::SelectBestMove(const MonteCarloTreeNode *node)
{
    assert(node != nullptr);

    move_t best_move = (node->children_.begin()->first);
    if (node->children_.empty()) {
        return best_move;
    }

    EvalType best_score = EvalFunc(node->children_.begin()->second);
    for (auto &child : node->children_) {
        EvalType score = EvalFunc(child.second);

        if (score > best_score) {
            best_score = score;
            best_move  = child.first;
        }
    }
    assert(best_move != kInvalidMove);
    return best_move;
}
}  // namespace checkers::mcts

#endif  // MCTS_CHECKERS_INCLUDE_MCTS_MONTE_CARLO_TREE_TPP_
