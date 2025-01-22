#include "mcts/monte_carlo_tree.hpp"
#include <chrono>
#include <cmath>
#include "common/checkers_defines.hpp"

namespace checkers::mcts
{
MonteCarloTree::MonteCarloTree(Board board, Turn turn) { root_ = new MonteCarloTreeNode(board, turn); }

MonteCarloTree::~MonteCarloTree() { delete root_; }

f32 MonteCarloTree::WinRate(const MonteCarloTreeNode *node)
{
    return node->visits_ > 0 ? (f32)node->score_ / node->visits_ : 0.0f;
}

void MonteCarloTree::DescendTree(const move_t move)
{
    auto node_to_descend_to = root_->children_.find(move);
    // Delete all children of the root except the one we are descending to
    for (auto it = root_->children_.begin(); it != root_->children_.end(); ++it) {
        if (it != node_to_descend_to) {
            delete it->second;
        }
    }
    MonteCarloTreeNode *node_to_delete = root_;
    if (node_to_descend_to == root_->children_.end()) {
        root_ = new MonteCarloTreeNode(root_->board_, root_);
        return;
    }
    node_to_delete->children_.clear();  // We delete them manually
    delete node_to_delete;
    root_ = node_to_descend_to->second;
}

move_t MonteCarloTree::Run(f32 time_seconds)
{
    auto start = std::chrono::system_clock::now();

    while (true) {
        auto now     = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration<f32>(now - start).count();

        if (elapsed >= time_seconds) {
            break;
        }

        MonteCarloTreeNode *node = SelectNode();
        now                      = std::chrono::system_clock::now();
        elapsed                  = std::chrono::duration<f32>(now - start).count();

        if (elapsed >= time_seconds) {
            break;
        }
        std::vector<MonteCarloTreeNode *> expanded_nodes = ExpandNode(node);
        now                                              = std::chrono::system_clock::now();
        elapsed                                          = std::chrono::duration<f32>(now - start).count();

        if (elapsed >= time_seconds) {
            break;
        }
        std::vector<SimulationResult> results = SimulateNodes(expanded_nodes);
        now                                   = std::chrono::system_clock::now();
        elapsed                               = std::chrono::duration<f32>(now - start).count();

        if (elapsed >= time_seconds) {
            break;
        }
        BackPropagate(expanded_nodes, results);
        now     = std::chrono::system_clock::now();
        elapsed = std::chrono::duration<f32>(now - start).count();

        if (elapsed >= time_seconds) {
            break;
        }
    }
    move_t best_move = SelectBestMove<f32, &MonteCarloTree::WinRate>(root_);
    assert(kInvalidMove != best_move);
    return best_move;
}

MonteCarloTreeNode *MonteCarloTree::SelectNode()
{
    assert(root_ != nullptr);

    MonteCarloTreeNode *current = root_;

    while (!current->children_.empty()) {
        f32 best_score                 = -1.0f;
        MonteCarloTreeNode *best_child = nullptr;
        for (auto &child : current->children_) {
            //            if (child.second->visits_ == 0) {
            //                best_child = child.second;
            //                break;
            //            }
            f32 uct_score = child.second->UctScore();
            if (uct_score > best_score) {
                best_score = uct_score;
                best_child = child.second;
            }
        }
        current = best_child;
        assert(current != nullptr);
    }
    return current;
}

std::vector<MonteCarloTreeNode *> MonteCarloTree::ExpandNode(const MonteCarloTreeNode *node)
{
    assert(true && "Not implemented");
    return {};
}

std::vector<SimulationResult> MonteCarloTree::SimulateNodes(const std::vector<MonteCarloTreeNode *> nodes)
{
    assert(true && "Not implemented");
    return {};
}

void MonteCarloTree::BackPropagate(std::vector<MonteCarloTreeNode *> nodes, const std::vector<SimulationResult> results)
{
    assert(true && "Not implemented");
}

MonteCarloTreeNode::MonteCarloTreeNode(MonteCarloTreeNode::Board board, Turn turn)
{
    board_  = board;
    turn_   = turn;
    parent_ = nullptr;
}

MonteCarloTreeNode::MonteCarloTreeNode(MonteCarloTreeNode::Board board, MonteCarloTreeNode *parent)
{
    board_  = board;
    parent_ = parent;
    turn_   = parent->turn_ == Turn::kWhite ? Turn::kBlack : Turn::kWhite;
}

MonteCarloTreeNode::~MonteCarloTreeNode()
{
    for (auto child : children_) {
        delete child.second;
    }
}

f64 MonteCarloTreeNode::UctScore() const
{
    return (f64)score_ / visits_ + kExplorationConstant * sqrtf(logf((f32)parent_->visits_) / visits_);
}
}  // namespace checkers::mcts
