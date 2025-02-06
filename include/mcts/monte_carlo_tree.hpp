#ifndef MCTS_CHECKERS_INCLUDE_MCTS_MONTE_CARLO_TREE_HPP_
#define MCTS_CHECKERS_INCLUDE_MCTS_MONTE_CARLO_TREE_HPP_

#include <atomic>
#include <mutex>
#include <ostream>
#include <unordered_map>
#include <vector>

#include "common/checkers_defines.hpp"
#ifdef __cpp_concepts
#include "common/concepts.hpp"
#endif
#include "cpu/board.hpp"
#include "game/checkers_engine.hpp"
#include "simulation_results.hpp"

namespace checkers::mcts
{
static constexpr u64 kMaxTotalSimulationsGpu = 8e2;
static constexpr u64 kSimulationMaxDepth     = 100;
static constexpr f32 kExplorationConstant    = 1.41f;
static constexpr f64 kRunCallOverhead        = 1e-2;

/**
 * @brief The MonteCarloBackend enum is used to select the backend for the MCTS simulations.
 */
enum class Backend { kCpu, kGpu, kSingleThreadedCpu };

/**
 * @brief The MonteCarloRunInfo struct holds the results of a Monte Carlo Tree Search run.
 */
struct TreeRunInfo {
    u64 total_simulations;
    f64 predicted_win_rate;
    move_t best_move;
    Backend used_backend;

    friend std::ostream& operator<<(std::ostream& os, const TreeRunInfo& info)
    {
        os << "Total simulations: " << info.total_simulations << std::endl;
        os << "Predicted win rate: " << info.predicted_win_rate << std::endl;
        std::string backend_str;
        backend_str = info.used_backend == Backend::kCpu ? "CPU" : backend_str;
        backend_str = info.used_backend == Backend::kSingleThreadedCpu ? "Single-threaded CPU" : backend_str;
        backend_str = info.used_backend == Backend::kGpu ? "GPU" : backend_str;
        os << "Backend: " << (backend_str.empty() ? "Unknown" : backend_str) << std::endl;
        return os;
    }
};

class Node;

/**
 * @brief MonteCarloTree is a standard MCTS manager that:
 *   - holds a root node,
 *   - repeatedly expands, simulates, and backpropagates,
 *   - then selects the best move from the root.
 *
 * We embed a `CheckersEngine` in each node to keep full game logic for partial
 * single-jump moves, multi-captures, promotion, etc.
 */
class Tree
{
    public:
    // -------------------------------------------------------------------------
    // Creation / Destruction
    // -------------------------------------------------------------------------
    explicit Tree(checkers::cpu::Board board, checkers::Turn turn, Backend backend = Backend::kCpu);
    ~Tree();

    // -------------------------------------------------------------------------
    // Public methods
    // -------------------------------------------------------------------------

    /**
     * @brief Run the MCTS algorithm for a certain amount of time.
     * @param time_seconds: The time budget for the MCTS algorithm.
     * @param num_threads: The number of worker threads to use. Default is 1.
     */
    move_t Run(f32 time_seconds, size_t num_threads = 1);

    /**
     * @brief Move root to the child that was chosen. This is optional if you want to
     *        keep the tree from move to move. You can discard the old tree though.
     */
    [[maybe_unused]] void DescendTree(const move_t move);

    // -------------------------------------------------------------------------
    // Public accessors
    // -------------------------------------------------------------------------
    [[nodiscard]] u64 GetTotalSimulations() const;
    [[nodiscard]] TreeRunInfo GetRunInfo() const;

    // -------------------------------------------------------------------------
    // Private methods
    // -------------------------------------------------------------------------

    private:
    void Iteration();

    Node* SelectNode();
    /**
     * @brief Expand the node if it's not terminal. For each possible move from
     *        that node, create a child node. Return the newly created child nodes.
     * @param node: The node to expand.
     */
    static std::vector<Node*> ExpandNode(Node* node);

    /**
     * @brief Simulate each node via random GPU simulations. Return the final results
     *        from the perspective of the side to move in each node.
     * @param nodes: The nodes to simulate from.
     */
    std::vector<SimulationResult> SimulateNodes(const std::vector<Node*>& nodes);

    /**
     * @brief Backpropagate the results of the simulations to the nodes.
     * @param nodes: The nodes to start backpropagation from.
     * @param results: The results of the simulations.
     */
    void BackPropagate(std::vector<Node*> nodes, const std::vector<SimulationResult>& results);

    /**
     * @brief Utility function that yields the best move from the root according
     *        to a certain evaluation (here: highest visited-child or highest average).
     */
#ifdef __cpp_concepts
    template <MaxComparable EvalType, EvalFunction<EvalType> auto EvalFunc>
#else
    template <typename EvalType, auto Func>
#endif
    checkers::move_t SelectBestMove(const Node* node) const;

    /**
     * @brief Convert a simulation result from child’s perspective into the
     *        perspective of the node that backpropagates it. If the node’s turn
     *        is the same as root’s turn, we keep the same score. Otherwise
     *        we invert it:  score -> n_simulations - score.
     */
    f64 GetScoreFromPerspectiveOfRoot(const Node* node, const SimulationResult& result);

    /**
     * @brief Utility function to compute a node's "win rate" or average score / visits.
     */
    static f32 WinRate(const Node* node);

    // -------------------------------------------------------------------------
    // Private fields
    // -------------------------------------------------------------------------

    /**
     * @brief The root node of the tree.
     */
    Node* root_{};

    /**
     * @brief A flag to signal the worker threads to stop.
     */
    std::atomic<bool> stop_flag_{false};

    Backend backend_;

    // -------------------------------------------------------------------------
    // Friends
    // -------------------------------------------------------------------------

    friend void mcts_worker(Tree* tree);
};

class Node
{
    public:
    /**
     * @brief Creates a node from a board+turn. The engine is constructed from these.
     */
    Node(const cpu::Board& board, Turn turn);

    /**
     * @brief Creates a node from a parent’s engine-based position.
     *        The parent's engine is cloned. Then we apply one partial move if needed externally
     *        (though in ExpandNode we usually do it).
     */
    Node(const CheckersEngine& engine, Node* parent);

    ~Node();

    /**
     * @brief A standard UCT formula: (score_ / visits_) + c* sqrt( ln(parent.visits_)/ visits_ )
     */
    f64 UctScore() const;

    public:
    // MCTS accumulators
    u64 visits_{0};
    f64 score_{0};

    // The engine that fully represents this node’s position
    CheckersEngine engine_;

    // Link to parent node
    Node* parent_{nullptr};

    // Map of moves => child
    std::unordered_map<checkers::move_t, Node*> children_;
    std::mutex node_mutex;
};

}  // namespace checkers::mcts

#include "monte_carlo_tree.tpp"

#endif  // MCTS_CHECKERS_INCLUDE_MCTS_MONTE_CARLO_TREE_HPP_
