#ifndef MCTS_CHECKERS_INCLUDE_MCTS_MONTE_CARLO_TREE_HPP_
#define MCTS_CHECKERS_INCLUDE_MCTS_MONTE_CARLO_TREE_HPP_

#include <mutex>
#include <unordered_map>
#include <vector>
#include "common/checkers_defines.hpp"
#include "common/concepts.hpp"
#include "cpu/board.hpp"
#include "cpu/launchers.hpp"
#include "game/checkers_engine.hpp"  // We embed the engine in the node
#include "simulation_results.hpp"

namespace checkers::mcts
{

static constexpr u64 kMaxTotalSimulations = 12e3;
static constexpr u64 kSimulationMaxDepth  = 100;
static constexpr f32 kExplorationConstant = 1.41f;
static constexpr f64 kRunCallOverhead     = 1e-2;

struct MonteCarloRunInfo {
    u64 total_simulations;
    f64 predicted_win_rate;
    move_t best_move;
};

/**
 * @brief The SimulationBackend enum allows you to select between GPU (default)
 *        and CPU simulation backends.
 */
enum class SimulationBackend { GPU, CPU };

class MonteCarloTreeNode;

/**
 * @brief MonteCarloTree is a standard MCTS manager that:
 *   - holds a root node,
 *   - repeatedly expands, simulates, and backpropagates,
 *   - then selects the best move from the root.
 *
 * An instance may run either sequentially or in parallel (by setting the
 * number of threads in Run()). Additionally, you may select the simulation
 * backend (GPU by default, CPU optionally) via the constructor.
 */
class MonteCarloTree
{
    public:
    /**
     * @brief Constructs an MCTS tree from a board and turn, with an optional simulation backend.
     * @param board The current board state.
     * @param turn  The current turn.
     * @param backend The simulation backend to use (GPU by default).
     */
    explicit MonteCarloTree(
        checkers::cpu::Board board, checkers::Turn turn, SimulationBackend backend = SimulationBackend::GPU
    );
    ~MonteCarloTree();

    /**
     * @brief Main MCTS entrypoint: run for 'time_seconds' of expansions
     *        and returns the best move from the root.
     * @param time_seconds The allotted time for the search.
     * @param num_threads  The number of threads to use. If 1, runs sequentially.
     */
    checkers::move_t Run(f32 time_seconds, size_t num_threads = 1);

    /**
     * @brief Move the root to the child corresponding to the chosen move.
     *        This allows reusing the tree between moves.
     * @param move The move to descend into.
     */
    [[maybe_unused]] void DescendTree(const move_t move);

    MonteCarloRunInfo GetRunInfo() const;
    u64 GetTotalSimulations() const;

    private:
    // Unified iteration routine (safe for both sequential and parallel execution).
    void Iteration();

    // Thread-safe selection: traverses children using per-node locking.
    MonteCarloTreeNode* SelectNode();

    // Thread-safe expansion: locks the node before generating children.
    static std::vector<MonteCarloTreeNode*> ExpandNode(MonteCarloTreeNode* node);

    // Simulation: run simulations on the given nodes using the chosen backend.
    std::vector<SimulationResult> SimulateNodes(const std::vector<MonteCarloTreeNode*>& nodes);

    // Backpropagation: update statistics from a leaf node up to the root.
    void BackPropagate(MonteCarloTreeNode* leaf, const SimulationResult& result);

    /**
     * @brief Utility function to choose the best move from the root based on an evaluation function.
     */
    template <MaxComparable EvalType, EvalFunction<EvalType> auto EvalFunc>
    checkers::move_t SelectBestMove(const MonteCarloTreeNode* node) const;

    /**
     * @brief Converts a simulation result (from the leaf’s perspective) into the
     *        perspective of the root. If the node’s turn is different from the root’s,
     *        the score is inverted.
     */
    f64 GetScoreFromPerspectiveOfRoot(const MonteCarloTreeNode* node, const SimulationResult& result);

    /**
     * @brief Computes a node's win rate (average score divided by visits).
     */
    static f32 WinRate(const MonteCarloTreeNode* node);

    MonteCarloTreeNode* root_{nullptr};
    SimulationBackend simulation_backend_;

    friend void mcts_worker(MonteCarloTree* tree);
};

/**
 * @brief MonteCarloTreeNode represents a node in the search tree.
 *        It holds the current game state (via a CheckersEngine) and
 *        MCTS accumulators.
 */
class MonteCarloTreeNode
{
    public:
    /**
     * @brief Constructs a node from a board and turn.
     */
    MonteCarloTreeNode(const cpu::Board& board, Turn turn);

    /**
     * @brief Constructs a node by cloning a parent's engine.
     */
    MonteCarloTreeNode(const CheckersEngine& engine, MonteCarloTreeNode* parent);

    ~MonteCarloTreeNode();

    /**
     * @brief Computes the UCT score: (score_ / visits_) + c * sqrt( ln(parent.visits_) / visits_ ).
     */
    f64 UctScore() const;

    public:
    // MCTS accumulators.
    u64 visits_{0};
    f64 score_{0};

    // The game engine representing the state at this node.
    CheckersEngine engine_;

    // Pointer to the parent node.
    MonteCarloTreeNode* parent_{nullptr};

    // Map of moves to child nodes.
    std::unordered_map<checkers::move_t, MonteCarloTreeNode*> children_;
    std::mutex node_mutex;
};

}  // namespace checkers::mcts

#include "monte_carlo_tree.tpp"

#endif  // MCTS_CHECKERS_INCLUDE_MCTS_MONTE_CARLO_TREE_HPP_
