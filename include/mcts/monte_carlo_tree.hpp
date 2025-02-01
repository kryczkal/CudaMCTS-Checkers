#ifndef MCTS_CHECKERS_INCLUDE_MCTS_MONTE_CARLO_TREE_HPP_
#define MCTS_CHECKERS_INCLUDE_MCTS_MONTE_CARLO_TREE_HPP_

#include <mutex>
#include <unordered_map>
#include <vector>
#include "common/checkers_defines.hpp"
#include "common/concepts.hpp"
#include "cpu/board.hpp"
#include "game/checkers_engine.hpp"  // We embed the engine in the node
#include "simulation_results.hpp"

namespace checkers::mcts
{
static constexpr u64 kMaxTotalSimulations = 5e3;
static constexpr u64 kSimulationMaxDepth  = 100;
static constexpr f32 kExplorationConstant = 1.41f;

class MonteCarloTreeNode;

/**
 * @brief MonteCarloTree is a standard MCTS manager that:
 *   - holds a root node,
 *   - repeatedly expands, simulates, and backpropagates,
 *   - then selects the best move from the root.
 *
 * We embed a `CheckersEngine` in each node to keep full game logic for partial
 * single-jump moves, multi-captures, promotion, etc.
 */
class MonteCarloTree
{
    public:
    explicit MonteCarloTree(checkers::cpu::Board board, checkers::Turn turn);
    ~MonteCarloTree();

    /**
     * @brief Main MCTS entrypoint: run for 'time_seconds' of expansions
     *        and returns the best move from the root.
     */
    checkers::move_t Run(f32 time_seconds);
    move_t RunParallel(f32 time_seconds, size_t num_threads);

    /**
     * @brief Move root to the child that was chosen. This is optional if you want to
     *        keep the tree from move to move. You can discard the old tree though.
     */
    void DescendTree(const move_t move);

    void IterationParallel();

    private:
    /**
     * @brief Select a node by traversing children with highest UCT scores
     *        until we hit a leaf or terminal node.
     */
    MonteCarloTreeNode* SelectNode();

    /**
     * @brief Expand the node if it's not terminal. For each possible move from
     *        that node, create a child node. Return the newly created child nodes.
     */
    std::vector<MonteCarloTreeNode*> ExpandNode(MonteCarloTreeNode* node);

    /**
     * @brief Simulate each node via random GPU simulations. Return the final results
     *        from the perspective of the side to move in each node.
     */
    std::vector<SimulationResult> SimulateNodes(std::vector<MonteCarloTreeNode*> nodes);

    /**
     * @brief Propagate results up the tree (visits_++, score_+=...) from each node
     *        up to the root.
     */
    void BackPropagate(std::vector<MonteCarloTreeNode*> nodes, const std::vector<SimulationResult> results);

    /**
     * @brief Utility function that yields the best move from the root according
     *        to a certain evaluation (here: highest visited-child or highest average).
     */
    template <MaxComparable EvalType, EvalFunction<EvalType> auto EvalFunc>
    checkers::move_t SelectBestMove(const MonteCarloTreeNode* node);

    /**
     * @brief Convert a simulation result from child’s perspective into the
     *        perspective of the node that backpropagates it. If the node’s turn
     *        is the same as root’s turn, we keep the same score. Otherwise
     *        we invert it:  score -> n_simulations - score, for a symmetrical zero-sum approach.
     */
    f64 GetScoreFromPerspectiveOfRoot(const MonteCarloTreeNode* node, const SimulationResult& result);

    /**
     * @brief Utility function to compute a node's "win rate" or average score / visits.
     */
    static f32 WinRate(const MonteCarloTreeNode* node);

    private:
    MonteCarloTreeNode* root_{};
    void BackPropagateParallel(std::vector<MonteCarloTreeNode*> nodes, const std::vector<SimulationResult>& results);
    std::vector<MonteCarloTreeNode*> ExpandNodeParallel(MonteCarloTreeNode* node);
    MonteCarloTreeNode* SelectNodeParallel();
};

class MonteCarloTreeNode
{
    public:
    /**
     * @brief Creates a node from a board+turn. The engine is constructed from these.
     */
    MonteCarloTreeNode(const cpu::Board& board, Turn turn);

    /**
     * @brief Creates a node from a parent’s engine-based position.
     *        The parent's engine is cloned. Then we apply one partial move if needed externally
     *        (though in ExpandNode we usually do it).
     */
    MonteCarloTreeNode(const CheckersEngine& engine, MonteCarloTreeNode* parent);

    ~MonteCarloTreeNode();

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
    MonteCarloTreeNode* parent_{nullptr};

    // Map of moves => child
    std::unordered_map<checkers::move_t, MonteCarloTreeNode*> children_;
    std::mutex node_mutex;
};

}  // namespace checkers::mcts

#include "monte_carlo_tree.tpp"

#endif  // MCTS_CHECKERS_INCLUDE_MCTS_MONTE_CARLO_TREE_HPP_
