#include "mcts/monte_carlo_tree.hpp"
#include <chrono>
#include <cmath>
#include <limits>
#include "common/checkers_defines.hpp"
#include "cpu/board_helpers.hpp"
#include "cuda/launchers.cuh"  // for GPU-based HostSimulateCheckersGames

namespace checkers::mcts
{

MonteCarloTree::MonteCarloTree(checkers::cpu::Board board, checkers::Turn turn)
{
    root_ = new MonteCarloTreeNode(board, turn);
}

MonteCarloTree::~MonteCarloTree() { delete root_; }

void MonteCarloTree::DescendTree(const move_t move)
{
    auto it = root_->children_.find(move);
    if (it == root_->children_.end()) {
        // If the child doesn't exist in the map, we have no expansion for that move
        // or we didn't call ExpandNode. We can simply build a brand new root from that move
        assert(false && "Attempting to descend to a non-existent child!");
        CheckersEngine newEngine(root_->engine_.GetBoard(), root_->engine_.GetCurrentTurn());
        newEngine.ApplyMove(move, false);

        MonteCarloTreeNode* newRoot = new MonteCarloTreeNode(newEngine, nullptr);
        delete root_;
        root_ = newRoot;
        return;
    }

    MonteCarloTreeNode* child = it->second;

    // Remove the chosen child from the parent's map so we can safely free siblings
    root_->children_.erase(it);

    // free all siblings:
    for (auto& kv : root_->children_) {
        delete kv.second;
    }
    root_->children_.clear();

    // Now delete the old root
    delete root_;

    // The chosen child becomes new root
    child->parent_ = nullptr;
    root_          = child;
}

checkers::move_t MonteCarloTree::Run(f32 time_seconds)
{
    auto start = std::chrono::system_clock::now();

    while (true) {
        auto now      = std::chrono::system_clock::now();
        float elapsed = std::chrono::duration<float>(now - start).count();
        if (elapsed >= time_seconds) {
            break;
        }

        MonteCarloTreeNode* node = SelectNode();
        if (!node)
            break;  // Should not happen unless no moves from root

        now     = std::chrono::system_clock::now();
        elapsed = std::chrono::duration<float>(now - start).count();
        if (elapsed >= time_seconds) {
            break;
        }

        std::vector<MonteCarloTreeNode*> expanded = ExpandNode(node);
        if (expanded.empty()) {
            // The node was terminal. So no expansions
            continue;
        }

        now     = std::chrono::system_clock::now();
        elapsed = std::chrono::duration<float>(now - start).count();
        if (elapsed >= time_seconds) {
            break;
        }

        std::vector<SimulationResult> results = SimulateNodes(expanded);
        now                                   = std::chrono::system_clock::now();
        elapsed                               = std::chrono::duration<float>(now - start).count();
        if (elapsed >= time_seconds) {
            break;
        }

        BackPropagate(expanded, results);
        // loop again
    }

    // Return best move from root
    if (root_->children_.empty()) {
        // If there's no children, either terminal or no expansions. Return invalid.
        return kInvalidMove;
    }

    return SelectBestMove<f32, &MonteCarloTree::WinRate>(root_);
}

MonteCarloTreeNode* MonteCarloTree::SelectNode()
{
    MonteCarloTreeNode* current = root_;
    // If root is terminal or has no expansions possible, just return
    if (current->engine_.IsTerminal()) {
        return current;
    }

    // descend
    while (!current->children_.empty()) {
        // if current is terminal, break
        if (current->engine_.IsTerminal()) {
            break;
        }

        f64 best_score                 = -1.0;
        MonteCarloTreeNode* best_child = nullptr;
        for (auto& kv : current->children_) {
            auto* child = kv.second;
            f64 uct     = child->UctScore();
            if (uct > best_score) {
                best_score = uct;
                best_child = child;
            }
        }
        if (!best_child) {
            // No children
            break;
        }
        current = best_child;
        if (current->engine_.IsTerminal()) {
            break;
        }
    }
    return current;
}

std::vector<MonteCarloTreeNode*> MonteCarloTree::ExpandNode(MonteCarloTreeNode* node)
{
    // If node is terminal, return it and don't expand
    if (node->engine_.IsTerminal()) {
        return {node};
    }

    // Gather possible single-jump or step moves
    MoveGenResult result = node->engine_.GenerateMoves();
    assert(
        cpu::ReadFlag(result.h_per_board_flags[0], MoveFlagsConstants::kMoveFound) &&
        "ExpandNode called on a node with no moves"
    );
    assert(node->children_.empty() && "ExpandNode called on a full expanded node");

    std::vector<MonteCarloTreeNode*> expanded_nodes;
    expanded_nodes.reserve(10);

    bool is_capture = cpu::ReadFlag(result.h_per_board_flags[0], MoveFlagsConstants::kCaptureFound);
    for (u64 i = 0; i < MoveGenResult::kMaxPiecesToTrack; i++) {
        for (u8 move_idx = 0; move_idx < result.h_move_counts[i]; move_idx++) {
            if (is_capture && !cpu::ReadFlag(result.h_capture_masks[i], move_idx)) {
                continue;
            }
            move_t mv                   = result.h_moves[i * MoveGenResult::kMovesPerPiece + move_idx];
            CheckersEngine child_engine = node->engine_;  // TODO: Not sure if this is a deep copy
            child_engine.ApplyMove(mv, false);

            auto* child         = new MonteCarloTreeNode(child_engine, node);
            node->children_[mv] = child;
            expanded_nodes.push_back(child);
        }
    }
    return expanded_nodes;
}

std::vector<SimulationResult> MonteCarloTree::SimulateNodes(std::vector<MonteCarloTreeNode*> nodes)
{
    if (nodes.empty()) {
        return {};
    }

    // We'll distribute up to kMaxTotalSimulations across the children.
    // If there's more children than kMaxTotalSimulations,
    // we just do 1 simulation each.
    u64 sim_each = kMaxTotalSimulations / nodes.size();
    if (sim_each < 1)
        sim_each = 1;

    std::vector<SimulationParam> params;
    params.reserve(nodes.size());
    for (auto* node : nodes) {
        // We'll get the board from n->engine_, plus the turn
        checkers::cpu::Board board = node->engine_.GetBoard();
        Turn turn                  = node->engine_.GetCurrentTurn();
        SimulationParam sp{
            .white         = board.white,
            .black         = board.black,
            .king          = board.kings,
            .start_turn    = static_cast<u8>((turn == Turn::kWhite ? 0 : 1)),
            .n_simulations = sim_each
        };
        params.push_back(sp);
    }

    auto results = gpu::launchers::HostSimulateCheckersGames(params, kSimulationMaxDepth);
    return results;
}

void MonteCarloTree::BackPropagate(std::vector<MonteCarloTreeNode*> nodes, const std::vector<SimulationResult> results)
{
    for (size_t i = 0; i < nodes.size(); i++) {
        MonteCarloTreeNode* leaf = nodes[i];
        SimulationResult res     = results[i];

        f64 score = GetScoreFromPerspectiveOfRoot(leaf, res);

        // The leaf accumulates stats
        leaf->visits_ += res.n_simulations;
        leaf->score_ += score;

        // Now move up the tree
        MonteCarloTreeNode* parent = leaf->parent_;
        while (parent) {
            parent->visits_ += res.n_simulations;
            parent->score_ += score;
            parent = parent->parent_;
        }
    }
}

// As used in Run() to pick final move:
f32 MonteCarloTree::WinRate(const MonteCarloTreeNode* node)
{
    if (node->visits_ == 0)
        return 0.0f;
    return static_cast<f32>(node->score_ / node->visits_);
}

f64 MonteCarloTree::GetScoreFromPerspectiveOfRoot(const MonteCarloTreeNode* node, const SimulationResult& result)
{
    // If the root has the same 'current_turn' as node->engine_, we do +score
    // else we invert the score
    const Turn root_turn = root_->engine_.GetCurrentTurn();
    const Turn node_turn = node->engine_.GetCurrentTurn();

    if (root_turn != node_turn) {
        return result.n_simulations - result.score;
    } else {
        return result.score;
    }
}

// ------------------- MonteCarloTreeNode Implementation ----------------------
MonteCarloTreeNode::MonteCarloTreeNode(const cpu::Board& board, Turn turn) : engine_(board, turn), parent_(nullptr) {}

MonteCarloTreeNode::MonteCarloTreeNode(const CheckersEngine& engine, MonteCarloTreeNode* parent)
    : engine_(engine), parent_(parent)
{
}

MonteCarloTreeNode::~MonteCarloTreeNode()
{
    for (auto& kv : children_) {
        delete kv.second;
    }
    children_.clear();
}

f64 MonteCarloTreeNode::UctScore() const
{
    static constexpr f64 kEpsilon = 1e-9;
    if (visits_ == 0) {
        return std::numeric_limits<f64>::infinity();
    }
    if (!parent_) {
        // Root?
        return score_ / (visits_ + kEpsilon);
    }
    f64 parent_visits = static_cast<f64>(parent_->visits_);
    f64 exploitation  = score_ / (visits_ + kEpsilon);
    f64 exploration   = kExplorationConstant * sqrt(log(parent_visits + kEpsilon) / (visits_ + kEpsilon));
    return exploitation + exploration;
}

}  // namespace checkers::mcts
