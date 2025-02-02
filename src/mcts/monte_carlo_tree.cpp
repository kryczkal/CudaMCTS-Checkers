#include "mcts/monte_carlo_tree.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

#include "common/checkers_defines.hpp"
#include "cpu/board_helpers.hpp"
#include "cuda/launchers.cuh"

namespace checkers::mcts
{

// Global stop flag for worker threads.
static std::atomic<bool> g_stop_flag{false};

// Timer functor: sleeps for the designated time and then sets the stop flag.
class Timer
{
    public:
    explicit Timer(f32 seconds) : seconds_(seconds) {}
    void operator()() const
    {
        std::this_thread::sleep_for(std::chrono::duration<f32>(seconds_));
        g_stop_flag.store(true, std::memory_order_relaxed);
    }

    private:
    f32 seconds_;
};

// Worker thread function: repeatedly calls Iteration() until the stop flag is set.
void mcts_worker(MonteCarloTree* tree)
{
    while (!g_stop_flag.load(std::memory_order_relaxed)) {
        tree->Iteration();
    }
}

// ------------------- MonteCarloTree Implementation ----------------------

MonteCarloTree::MonteCarloTree(checkers::cpu::Board board, checkers::Turn turn, SimulationBackend backend)
    : simulation_backend_(backend)
{
    root_ = new MonteCarloTreeNode(board, turn);
}

MonteCarloTree::~MonteCarloTree() { delete root_; }

[[maybe_unused]] void MonteCarloTree::DescendTree(const move_t move)
{
    auto it = root_->children_.find(move);
    if (it == root_->children_.end()) {
        assert(false && "Attempting to descend to a non-existent child!");
    }
    MonteCarloTreeNode* child = it->second;

    // Remove the chosen child from the parent's map to free its siblings.
    root_->children_.erase(it);
    for (auto& kv : root_->children_) {
        delete kv.second;
    }
    root_->children_.clear();

    // Delete the old root and set the chosen child as the new root.
    delete root_;
    child->parent_ = nullptr;
    root_          = child;
}

u64 MonteCarloTree::GetTotalSimulations() const { return (root_ != nullptr) ? root_->visits_ : 0; }

checkers::move_t MonteCarloTree::Run(f32 time_seconds, size_t num_threads)
{
    g_stop_flag.store(false, std::memory_order_relaxed);

    if (num_threads <= 1) {
        // Sequential execution.
        auto start = std::chrono::system_clock::now();
        while (true) {
            auto now      = std::chrono::system_clock::now();
            float elapsed = std::chrono::duration<float>(now - start).count();
            if (elapsed >= time_seconds) {
                break;
            }
            Iteration();
        }
    } else {
        // Parallel execution.
        std::thread timer_thread{Timer(time_seconds)};
        std::vector<std::thread> workers;
        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back(mcts_worker, this);
        }
        timer_thread.join();
        for (auto& worker : workers) {
            worker.join();
        }
    }

    if (root_->children_.empty()) {
        return kInvalidMove;
    }
    return SelectBestMove<f32, &MonteCarloTree::WinRate>(root_);
}

// Unified iteration: performs selection, expansion, simulation, and backpropagation.
void MonteCarloTree::Iteration()
{
    // Selection.
    MonteCarloTreeNode* node = SelectNode();
    if (!node) {
        return;
    }

    // Expansion.
    std::vector<MonteCarloTreeNode*> expanded_nodes = ExpandNode(node);
    if (expanded_nodes.empty()) {
        return;
    }

    // Simulation.
    std::vector<SimulationResult> results = SimulateNodes(expanded_nodes);

    // Backpropagation.
    for (size_t i = 0; i < expanded_nodes.size(); ++i) {
        BackPropagate(expanded_nodes[i], results[i]);
    }
}

// Thread-safe selection: traverse the tree using per-node mutexes.
MonteCarloTreeNode* MonteCarloTree::SelectNode()
{
    MonteCarloTreeNode* current = root_;
    while (true) {
        std::unique_lock<std::mutex> lock(current->node_mutex);
        if (current->engine_.IsTerminal() || current->children_.empty()) {
            return current;
        }
        MonteCarloTreeNode* best_child = nullptr;
        f64 best_score                 = -std::numeric_limits<f64>::infinity();
        for (auto& kv : current->children_) {
            MonteCarloTreeNode* child = kv.second;
            f64 uct                   = child->UctScore();
            if (uct > best_score) {
                best_score = uct;
                best_child = child;
            }
        }
        lock.unlock();
        if (!best_child) {
            return current;
        }
        current = best_child;
    }
}

// Thread-safe expansion: locks the node before generating children.
std::vector<MonteCarloTreeNode*> MonteCarloTree::ExpandNode(MonteCarloTreeNode* node)
{
    std::unique_lock<std::mutex> lock(node->node_mutex);
    if (node->engine_.IsTerminal() || !node->children_.empty()) {
        return {node};
    }

    MoveGenResult result = node->engine_.GenerateMoves();
    assert(
        cpu::ReadFlag(result.h_per_board_flags[0], MoveFlagsConstants::kMoveFound) &&
        "ExpandNode called on a node with no moves"
    );

    std::vector<MonteCarloTreeNode*> expanded_nodes;
    expanded_nodes.reserve(10);
    bool is_capture = cpu::ReadFlag(result.h_per_board_flags[0], MoveFlagsConstants::kCaptureFound);
    for (u64 i = 0; i < MoveGenResult::kMaxPiecesToTrack; i++) {
        for (u8 move_idx = 0; move_idx < result.h_move_counts[i]; move_idx++) {
            if (is_capture && !cpu::ReadFlag(result.h_capture_masks[i], move_idx)) {
                continue;
            }
            move_t mv                   = result.h_moves[i * MoveGenResult::kMovesPerPiece + move_idx];
            CheckersEngine child_engine = node->engine_;
            child_engine.ApplyMove(mv, false);
            auto* child         = new MonteCarloTreeNode(child_engine, node);
            node->children_[mv] = child;
            expanded_nodes.push_back(child);
        }
    }
    if (expanded_nodes.empty()) {
        expanded_nodes.push_back(node);
    }
    return expanded_nodes;
}

// Simulation: use the selected backend (GPU by default, CPU optionally).
std::vector<SimulationResult> MonteCarloTree::SimulateNodes(const std::vector<MonteCarloTreeNode*>& nodes)
{
    if (nodes.empty()) {
        return {};
    }
    u64 sim_each = kMaxTotalSimulations / nodes.size();
    if (sim_each < 1) {
        sim_each = 1;
    }

    std::vector<SimulationParam> params;
    params.reserve(nodes.size());
    for (auto* node : nodes) {
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

    std::vector<SimulationResult> results;
    if (simulation_backend_ == SimulationBackend::GPU) {
        results = gpu::launchers::HostSimulateCheckersGames(params, kSimulationMaxDepth);
    } else {
        results = cpu::launchers::HostSimulateCheckersGames(params, kSimulationMaxDepth);
    }
    return results;
}

// Backpropagation: update statistics from the leaf up to the root.
void MonteCarloTree::BackPropagate(MonteCarloTreeNode* leaf, const SimulationResult& result)
{
    f64 score                   = GetScoreFromPerspectiveOfRoot(leaf, result);
    MonteCarloTreeNode* current = leaf;
    while (current != nullptr) {
        std::unique_lock<std::mutex> lock(current->node_mutex);
        current->visits_ += result.n_simulations;
        current->score_ += score;
        current = current->parent_;
    }
}

// Converts a simulation result from the leaf’s perspective to that of the root.
f64 MonteCarloTree::GetScoreFromPerspectiveOfRoot(const MonteCarloTreeNode* node, const SimulationResult& result)
{
    const Turn root_turn = root_->engine_.GetCurrentTurn();
    const Turn node_turn = node->engine_.GetCurrentTurn();
    if (root_turn != node_turn) {
        return result.n_simulations - result.score;
    } else {
        return result.score;
    }
}

// Computes the win rate: average score (score/visits).
f32 MonteCarloTree::WinRate(const MonteCarloTreeNode* node)
{
    if (node->visits_ == 0) {
        return 0.0f;
    }
    return static_cast<f32>(node->score_ / node->visits_);
}
MonteCarloRunInfo MonteCarloTree::GetRunInfo() const
{
    MonteCarloRunInfo info;
    info.total_simulations  = GetTotalSimulations();
    info.predicted_win_rate = WinRate(root_);
    info.best_move          = SelectBestMove<f32, &MonteCarloTree::WinRate>(root_);
    return info;
}

// ------------------- MonteCarloTreeNode Implementation ----------------------

MonteCarloTreeNode::MonteCarloTreeNode(const cpu::Board& board, Turn turn)
    : visits_(0), score_(0.0), engine_(board, turn), parent_(nullptr)
{
}

MonteCarloTreeNode::MonteCarloTreeNode(const CheckersEngine& engine, MonteCarloTreeNode* parent)
    : visits_(0), score_(0.0), engine_(engine), parent_(parent)
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
        return score_ / (visits_ + kEpsilon);
    }
    f64 parent_visits = static_cast<f64>(parent_->visits_);
    f64 exploitation  = score_ / (visits_ + kEpsilon);
    f64 exploration   = kExplorationConstant * sqrt(log(parent_visits + kEpsilon) / (visits_ + kEpsilon));
    return exploitation + exploration;
}

}  // namespace checkers::mcts
