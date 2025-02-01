#include "mcts/monte_carlo_tree.hpp"
#include <chrono>
#include <cmath>
#include <limits>
#include "common/checkers_defines.hpp"
#include "cpu/board_helpers.hpp"
#include "cuda/launchers.cuh"

#include <atomic>
#include <cassert>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

// ----------------- Global Variables for Parallelization -----------------

// Global stop flag for all worker threads.
static std::atomic<bool> g_stop_flag{false};

namespace checkers::mcts
{

// Timer functor: sleeps for the designated time and then sets the stop flag.
class Timer
{
    public:
    Timer(f32 seconds) : seconds_(seconds) {}
    void operator()()
    {
        std::this_thread::sleep_for(std::chrono::duration<f32>(seconds_));
        g_stop_flag.store(true, std::memory_order_relaxed);
    }

    private:
    f32 seconds_;
};

// Worker thread function: repeatedly call Iteration() until the stop flag is set.
static void mcts_worker(MonteCarloTree* tree)
{
    while (!g_stop_flag.load(std::memory_order_relaxed)) {
        tree->IterationParallel();
    }
}

// ------------------- MonteCarloTree Implementation ----------------------

MonteCarloTree::MonteCarloTree(checkers::cpu::Board board, checkers::Turn turn)
{
    root_ = new MonteCarloTreeNode(board, turn);
}

MonteCarloTree::~MonteCarloTree() { delete root_; }

void MonteCarloTree::DescendTree(const move_t move)
{
    auto it = root_->children_.find(move);
    if (it == root_->children_.end()) {
        assert(false && "Attempting to descend to a non-existent child!");
    }

    MonteCarloTreeNode* child = it->second;

    // Remove the chosen child from the parent's map, so we can safely free siblings.
    root_->children_.erase(it);

    // Free all siblings:
    for (auto& kv : root_->children_) {
        delete kv.second;
    }
    root_->children_.clear();

    // Delete the old root.
    delete root_;

    // The chosen child becomes the new root.
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
        assert(node != nullptr && "SelectNode returned a nullptr");

        now     = std::chrono::system_clock::now();
        elapsed = std::chrono::duration<float>(now - start).count();
        if (elapsed >= time_seconds) {
            break;
        }

        std::vector<MonteCarloTreeNode*> expanded = ExpandNode(node);
        assert(!expanded.empty() && "ExpandNode returned empty nodes");

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
        // Loop again.
    }

    // Return the best move from the root.
    if (root_->children_.empty()) {
        // If there are no children (terminal or no expansions), return invalid.
        return kInvalidMove;
    }

    return SelectBestMove<f32, &MonteCarloTree::WinRate>(root_);
}

// ------------------- Original Sequential Methods ----------------------

MonteCarloTreeNode* MonteCarloTree::SelectNode()
{
    MonteCarloTreeNode* current = root_;
    if (current->engine_.IsTerminal()) {
        return current;
    }

    // Descend the tree.
    while (!current->children_.empty()) {
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
    if (node->engine_.IsTerminal()) {
        return {node};
    }

    MoveGenResult result = node->engine_.GenerateMoves();
    assert(
        cpu::ReadFlag(result.h_per_board_flags[0], MoveFlagsConstants::kMoveFound) &&
        "ExpandNode called on a node with no moves"
    );
    assert(node->children_.empty() && "ExpandNode called on a full expanded node");

    std::vector<MonteCarloTreeNode*> expanded_nodes;
    expanded_nodes.reserve(10);

    // TODO: In the rare case of a chain-capture move, we should only expand the last move.
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
    return expanded_nodes;
}

std::vector<SimulationResult> MonteCarloTree::SimulateNodes(std::vector<MonteCarloTreeNode*> nodes)
{
    if (nodes.empty()) {
        return {};
    }

    // Distribute up to kMaxTotalSimulations across children.
    u64 sim_each = kMaxTotalSimulations / nodes.size();
    if (sim_each < 1)
        sim_each = 1;

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

    auto results = gpu::launchers::HostSimulateCheckersGames(params, kSimulationMaxDepth);
    return results;
}

void MonteCarloTree::BackPropagate(std::vector<MonteCarloTreeNode*> nodes, const std::vector<SimulationResult> results)
{
    for (size_t i = 0; i < nodes.size(); i++) {
        MonteCarloTreeNode* leaf = nodes[i];
        SimulationResult res     = results[i];

        f64 score = GetScoreFromPerspectiveOfRoot(leaf, res);

        // Update the leaf.
        leaf->visits_ += res.n_simulations;
        leaf->score_ += score;

        // Propagate upward.
        MonteCarloTreeNode* parent = leaf->parent_;
        while (parent) {
            parent->visits_ += res.n_simulations;
            parent->score_ += score;
            parent = parent->parent_;
        }
    }
}

// For final move selection.
f32 MonteCarloTree::WinRate(const MonteCarloTreeNode* node)
{
    if (node->visits_ == 0) {
        return 0.0f;
    }
    return static_cast<f32>(node->score_ / node->visits_);
}

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

// ------------------- Parallel MCTS Methods ----------------------

// One complete MCTS iteration (selection, expansion, simulation, backpropagation).
void MonteCarloTree::IterationParallel()
{
    MonteCarloTreeNode* node = SelectNodeParallel();
    if (!node)
        return;
    std::vector<MonteCarloTreeNode*> expanded_nodes = ExpandNodeParallel(node);
    if (expanded_nodes.empty()) {
        return;
    }

    std::vector<SimulationResult> results = SimulateNodes(expanded_nodes);
    BackPropagateParallel(expanded_nodes, results);
}

// Parallel selection: safely traverse the tree using per-node mutexes.
MonteCarloTreeNode* MonteCarloTree::SelectNodeParallel()
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
            // Read child statistics to compute UCT.
            f64 uct = child->UctScore();
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

// Parallel expansion: lock the node before generating children.
std::vector<MonteCarloTreeNode*> MonteCarloTree::ExpandNodeParallel(MonteCarloTreeNode* node)
{
    std::unique_lock<std::mutex> lock(node->node_mutex);
    if (node->engine_.IsTerminal() || !node->children_.empty()) {
        return {node};
    }
    std::vector<MonteCarloTreeNode*> expanded_nodes = ExpandNode(node);
    return expanded_nodes;
}

// Parallel backpropagation: update statistics with per-node locking.
void MonteCarloTree::BackPropagateParallel(
    std::vector<MonteCarloTreeNode*> nodes, const std::vector<SimulationResult>& results
)
{
    for (size_t i = 0; i < nodes.size(); i++) {
        MonteCarloTreeNode* leaf    = nodes[i];
        SimulationResult res        = results[i];
        f64 score                   = GetScoreFromPerspectiveOfRoot(leaf, res);
        MonteCarloTreeNode* current = leaf;
        while (current != nullptr) {
            std::unique_lock<std::mutex> lock(current->node_mutex);
            current->visits_ += res.n_simulations;
            current->score_ += score;
            current = current->parent_;
        }
    }
}

// RunParallel: launch a thread pool and a timer thread.
// The function returns the best move from the root after time_seconds.
checkers::move_t MonteCarloTree::RunParallel(f32 time_seconds, size_t num_threads)
{
    g_stop_flag.store(false, std::memory_order_relaxed);

    // Start the timer thread.
    std::thread timer_thread{Timer(time_seconds)};

    // Launch worker threads.
    std::vector<std::thread> workers;
    for (size_t i = 0; i < num_threads; ++i) {
        workers.emplace_back(mcts_worker, this);
    }

    // Wait for the timer to signal stop.
    timer_thread.join();

    // Join all worker threads.
    for (auto& worker : workers) {
        worker.join();
    }

    // Return the best move from the current root.
    if (root_->children_.empty()) {
        return kInvalidMove;
    }
    return SelectBestMove<f32, &MonteCarloTree::WinRate>(root_);
}

// ------------------- MonteCarloTreeNode Implementation ----------------------

MonteCarloTreeNode::MonteCarloTreeNode(const cpu::Board& board, Turn turn)
    : engine_(board, turn), parent_(nullptr), visits_(0), score_(0.0)
{
}

MonteCarloTreeNode::MonteCarloTreeNode(const CheckersEngine& engine, MonteCarloTreeNode* parent)
    : engine_(engine), parent_(parent), visits_(0), score_(0.0)
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
