#include <MonteCarloTree.hpp>
#include <concepts.hpp>

#include <cassert>
#include <chrono>
#include <cmath>
#include <vector>

namespace CudaMctsCheckers
{

MonteCarloTree::MonteCarloTree(CudaMctsCheckers::Board board)
{
    root_ = new MonteCarloTreeNode(board);
}

MonteCarloTree::~MonteCarloTree() { delete root_; }

MonteCarloTreeNode *CudaMctsCheckers::MonteCarloTree::SelectNode()
{
    assert(root_ != nullptr);

    MonteCarloTreeNode *current = root_;

    while (!current->children_.empty()) {
        f32 best_score                 = -1.0f;
        MonteCarloTreeNode *best_child = nullptr;
        for (auto &child : current->children_) {
            f32 uct_score = child.second->UctScore();
            if (uct_score > best_score) {
                best_score = uct_score;
                best_child = child.second;
            }
        }
        current = best_child;
    }
    return current;
}

std::vector<MonteCarloTreeNode *> MonteCarloTree::ExpandNode(MonteCarloTreeNode *node)
{
    return {};  // TODO
}

std::vector<SimulationResult> MonteCarloTree::SimulateNodes(std::vector<MonteCarloTreeNode *> nodes)
{
    return {};
}

void MonteCarloTree::Backpropagate(
    std::vector<MonteCarloTreeNode *> &nodes, const std::vector<SimulationResult> &results
)
{
    for (size_t i = 0; i < nodes.size(); ++i) {
        MonteCarloTreeNode *node = nodes[i];
        while (node != nullptr) {
            node->score_ += results[i].score;
            node->visits_ += results[i].visits;
            node = node->parent_;
        }
    }
}

Move MonteCarloTree::Run(f32 time)
{
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    while (std::chrono::duration<f32>(std::chrono::system_clock::now() - start).count() < time) {
        MonteCarloTreeNode *node                         = SelectNode();
        std::vector<MonteCarloTreeNode *> expanded_nodes = ExpandNode(node);
        std::vector<SimulationResult> results            = SimulateNodes(expanded_nodes);
        Backpropagate(expanded_nodes, results);
    }
    return SelectBestMove<f32, &MonteCarloTree::WinRate>();
}

f32 MonteCarloTree::WinRate(MonteCarloTreeNode *node) { return (f32)node->score_ / node->visits_; }

MonteCarloTreeNode::MonteCarloTreeNode(Board board, MonteCarloTreeNode *parent)
    : board_(board), parent_(parent)
{
}

MonteCarloTreeNode::~MonteCarloTreeNode()
{
    for (auto &child : children_) {
        delete child.second;
    }
}

f32 MonteCarloTreeNode::UctScore() const
{
    return (f32)score_ / visits_ +
           kExplorationConstant * sqrtf(logf((f32)parent_->visits_) / visits_);
}

}  // namespace CudaMctsCheckers