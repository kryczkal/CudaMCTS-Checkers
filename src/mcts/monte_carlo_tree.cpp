#include "mcts/monte_carlo_tree.hpp"
#include <chrono>
#include <cmath>
#include <limits>
#include "common/checkers_defines.hpp"
#include "cpu/board_helpers.hpp"
#include "cpu/launchers.hpp"
#include "cuda/launchers.cuh"

#include <atomic>
#include <cassert>
#include <mutex>
#include <thread>
#include <vector>

namespace checkers::mcts
{

// Timer functor: sleeps for the designated time and then sets the stop flag.
class Timer
{
    public:
    Timer(std::atomic<bool> &stop_flag, f32 seconds) : stop_flag_(stop_flag), seconds_(seconds) {}
    void operator()()
    {
        std::this_thread::sleep_for(std::chrono::duration<f32>(seconds_));
        stop_flag_.store(true, std::memory_order_relaxed);
    }

    private:
    f32 seconds_;
    std::atomic<bool> &stop_flag_;
};

// Worker thread function: repeatedly call Iteration() until the stop flag is set.
void mcts_worker(Tree *tree)
{
    while (!tree->stop_flag_.load(std::memory_order_relaxed)) {
        tree->Iteration();
    }
}

void run_workers_independent_worker(u64 num_threads, Tree *tree)
{
    std::vector<std::thread> workers;
    for (u64 i = 0; i < num_threads; ++i) {
        workers.emplace_back(mcts_worker, tree);
    }

    for (auto &worker : workers) {
        worker.join();
    }
}

// ------------------- MonteCarloTree Implementation ----------------------

Tree::Tree(checkers::cpu::Board board, checkers::Turn turn, Backend backend)
{
    root_    = new Node(board, turn);
    backend_ = backend;
}

Tree::~Tree() { delete root_; }

[[maybe_unused]] void Tree::DescendTree(const move_t move)
{
    auto it = root_->children_.find(move);
    if (it == root_->children_.end()) {
        assert(false && "Attempting to descend to a non-existent child!");
    }

    Node *child = it->second;

    // Remove the chosen child from the parent's map, so we can safely free siblings.
    root_->children_.erase(it);

    // Free all siblings:
    for (auto &kv : root_->children_) {
        delete kv.second;
    }
    root_->children_.clear();

    // Delete the old root.
    delete root_;

    // The chosen child becomes the new root.
    child->parent_ = nullptr;
    root_          = child;
}

checkers::move_t Tree::Run(f32 time_seconds, size_t num_threads)
{
    num_threads = backend_ == Backend::kSingleThreadedCpu ? 1 : num_threads;
    stop_flag_.store(false, std::memory_order_relaxed);

    // Start the timer thread.
    std::thread timer_thread{Timer(stop_flag_, time_seconds)};

    // Launch worker threads.
    std::vector<std::thread> workers;
    for (size_t i = 0; i < num_threads; ++i) {
        workers.emplace_back(mcts_worker, this);
    }

    // Wait for the timer to signal stop.
    timer_thread.join();

    // Wait for the workers to finish after the timer has stopped them.
    for (auto &worker : workers) {
        worker.join();
    }

    // Return the best move from the current root.
    if (root_->children_.empty()) {
        return kInvalidMove;
    }
    return SelectBestMove<f32, &Tree::WinRate>(root_);
}

void Tree::Iteration()
{
    Node *node = SelectNode();
    if (!node) {
        return;
    }
    std::vector<Node *> expanded_nodes = ExpandNode(node);
    if (expanded_nodes.empty()) {
        return;
    }

    std::vector<SimulationResult> results = SimulateNodes(expanded_nodes);
    BackPropagate(expanded_nodes, results);
}

Node *Tree::SelectNode()
{
    Node *current = root_;
    while (true) {
        std::unique_lock<std::mutex> lock(current->node_mutex);
        if (current->engine_.IsTerminal() || current->children_.empty()) {
            return current;
        }

        Node *best_child = nullptr;
        f64 best_score   = -std::numeric_limits<f64>::infinity();
        for (auto &kv : current->children_) {
            Node *child = kv.second;
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
std::vector<Node *> Tree::ExpandNode(Node *node)
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

    std::vector<Node *> expanded_nodes;
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

            auto *child         = new Node(child_engine, node);
            node->children_[mv] = child;
            expanded_nodes.push_back(child);
        }
    }
    return expanded_nodes;
}

std::vector<SimulationResult> Tree::SimulateNodes(const std::vector<Node *> &nodes)
{
    if (nodes.empty()) {
        return {};
    }

    u64 sim_each = 1;
    if (backend_ == Backend::kGpu) {
        // Distribute the total number of simulations evenly among the nodes.
        sim_each = kMaxTotalSimulationsGpu / nodes.size();
        if (sim_each < 1) {
            sim_each = 1;
        }
    }

    std::vector<SimulationParam> params;
    params.reserve(nodes.size());
    for (auto *node : nodes) {
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
    if (backend_ == Backend::kCpu) {
        results = cpu::launchers::HostSimulateCheckersGames(params, kSimulationMaxDepth);
    } else {
        results = gpu::launchers::HostSimulateCheckersGames(params, kSimulationMaxDepth);
    }
    return results;
}

void Tree::BackPropagate(std::vector<Node *> nodes, const std::vector<SimulationResult> &results)
{
    for (size_t i = 0; i < nodes.size(); i++) {
        Node *leaf           = nodes[i];
        SimulationResult res = results[i];
        f64 score            = GetScoreFromPerspectiveOfRoot(leaf, res);
        Node *current        = leaf;
        while (current != nullptr) {
            std::unique_lock<std::mutex> lock(current->node_mutex);
            current->visits_ += res.n_simulations;
            current->score_ += score;
            current = current->parent_;
        }
    }
}

// For final move selection.
f32 Tree::WinRate(const Node *node)
{
    if (node->visits_ == 0) {
        return 0.0f;
    }
    return static_cast<f32>(node->score_ / node->visits_);
}

f64 Tree::GetScoreFromPerspectiveOfRoot(const Node *node, const SimulationResult &result)
{
    const Turn root_turn = root_->engine_.GetCurrentTurn();
    const Turn node_turn = node->engine_.GetCurrentTurn();

    if (root_turn != node_turn) {
        return result.n_simulations - result.score;
    } else {
        return result.score;
    }
}

u64 Tree::GetTotalSimulations() const
{
    if (!root_) {
        return 0;
    }
    return root_->visits_;
}

TreeRunInfo Tree::GetRunInfo() const
{
    TreeRunInfo info;
    info.total_simulations  = GetTotalSimulations();
    info.predicted_win_rate = WinRate(root_);
    info.best_move          = SelectBestMove<f32, &Tree::WinRate>(root_);
    info.used_backend       = backend_;
    return info;
}

// ------------------- MonteCarloTreeNode Implementation ----------------------

Node::Node(const cpu::Board &board, Turn turn) : engine_(board, turn), parent_(nullptr), visits_(0), score_(0.0) {}

Node::Node(const CheckersEngine &engine, Node *parent) : engine_(engine), parent_(parent), visits_(0), score_(0.0) {}

Node::~Node()
{
    for (auto &kv : children_) {
        delete kv.second;
    }
    children_.clear();
}

f64 Node::UctScore() const
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
