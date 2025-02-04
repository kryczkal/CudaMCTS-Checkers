#ifndef MCTS_CHECKERS_INCLUDE_MCTS_MONTE_CARLO_TREE_TPP_
#define MCTS_CHECKERS_INCLUDE_MCTS_MONTE_CARLO_TREE_TPP_

#include "cassert"
#include "iostream"

namespace checkers::mcts
{
template <MaxComparable EvalType, EvalFunction<EvalType> auto EvalFunc>
move_t MonteCarloTree::SelectBestMove(const MonteCarloTreeNode *node) const
{
    assert(node != nullptr);
    std::cout << "Total simulations: " << node->visits_ << std::endl;

    if (node->children_.empty()) {
        // Handle the case where there are no children.
        std::cerr << "SelectBestMove called on a node with no children." << std::endl;
        return kInvalidMove;
    }
    // Initialize best_move and best_score with the first child.
    auto first_child    = node->children_.begin();
    move_t best_move    = first_child->first;
    EvalType best_score = EvalFunc(first_child->second);

    for (auto &child : node->children_) {
        EvalType score = EvalFunc(child.second);

        if (score > best_score) {
            best_score = score;
            best_move  = child.first;
        }
    }

    assert(best_move != kInvalidMove);
    std::cout << "Best score: " << best_score << std::endl;
    return best_move;
}
}  // namespace checkers::mcts

#endif  // MCTS_CHECKERS_INCLUDE_MCTS_MONTE_CARLO_TREE_TPP_
