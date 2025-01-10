#ifndef CUDA_MCTS_CHECKRS_INCLUDE_MCT_TPP_
#define CUDA_MCTS_CHECKRS_INCLUDE_MCT_TPP_

#include <cassert>
#include <concepts.hpp>
#include <monte_carlo_tree.hpp>

namespace CudaMctsCheckers
{

template <MaxComparable EvalType, EvalFunction<EvalType> auto EvalFunc>
Move::Type MonteCarloTree::SelectBestMove()
{
    assert(root_ != nullptr);

    Move::Type best_move = 0;
    if (root_->children_.empty()) {
        return best_move;
    }

    EvalType best_score = EvalFunc(root_->children_.begin()->second);
    for (auto &child : root_->children_) {
        EvalType score = EvalFunc(child.second);

        if (score > best_score) {
            best_score = score;
            best_move  = child.first;
        }
    }
    return best_move;
}

}  // namespace CudaMctsCheckers

#endif  // CUDA_MCTS_CHECKRS_INCLUDE_MCT_TPP_