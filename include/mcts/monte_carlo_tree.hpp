#ifndef MCTS_CHECKERS_INCLUDE_MCTS_MONTE_CARLO_TREE_HPP_
#define MCTS_CHECKERS_INCLUDE_MCTS_MONTE_CARLO_TREE_HPP_

#include <unordered_map>
#include <vector>
#include "common/checkers_defines.hpp"
#include "common/concepts.hpp"
#include "cpu/board.hpp"
#include "simulation_results.hpp"

namespace checkers::mcts
{
static constexpr u64 kMaxTotalSimulations = 1e4;
static constexpr u64 kSimulationMaxDepth  = 200;
static constexpr f32 kExplorationConstant = 1.41f;
class MonteCarloTreeNode;

class MonteCarloTree
{
    using Board = cpu::Board;

    public:
    //------------------------------------------------------------------------------//
    //                        Class Creation and Destruction                        //
    //------------------------------------------------------------------------------//
    explicit MonteCarloTree(Board board, Turn turn);

    ~MonteCarloTree();

    //------------------------------------------------------------------------------//
    //                                Public Methods                                //
    //------------------------------------------------------------------------------//
    move_t Run(f32 time_seconds);

    //------------------------------------------------------------------------------//
    //                               Public Variables                               //
    //------------------------------------------------------------------------------//

    void DescendTree(const move_t move);

    private:
    //------------------------------------------------------------------------------//
    //                                Private Methods                               //
    //------------------------------------------------------------------------------//

    ///////////////////// Main Monte Carlo Tree Search steps /////////////////////////

    MonteCarloTreeNode *SelectNode();
    std::vector<MonteCarloTreeNode *> ExpandNode(MonteCarloTreeNode *node);
    std::vector<SimulationResult> SimulateNodes(std::vector<MonteCarloTreeNode *> nodes);
    void BackPropagate(std::vector<MonteCarloTreeNode *> nodes, const std::vector<SimulationResult> results);

    //////////////////////////////////////////////////////////////////////////////////

    template <MaxComparable EvalType, EvalFunction<EvalType> auto EvalFunc>
    move_t SelectBestMove(const MonteCarloTreeNode *node);
    f64 GetScoreFromPerspectiveOfRoot(const MonteCarloTreeNode *node, const SimulationResult &result);

    static f32 WinRate(const MonteCarloTreeNode *node);

    //------------------------------------------------------------------------------//
    //                               Private Variables                              //
    //------------------------------------------------------------------------------//
    MonteCarloTreeNode *root_{};  // Root node of the tree
};

class MonteCarloTreeNode
{
    using Board = cpu::Board;

    public:
    //------------------------------------------------------------------------------//
    //                        Class Creation and Destruction                        //
    //------------------------------------------------------------------------------//

    explicit MonteCarloTreeNode(Board board, Turn turn);
    explicit MonteCarloTreeNode(Board board, MonteCarloTreeNode *parent, Turn turn);

    ~MonteCarloTreeNode();

    //------------------------------------------------------------------------------//
    //                                Public Methods                                //
    //------------------------------------------------------------------------------//

    f64 UctScore() const;

    //------------------------------------------------------------------------------//
    //                               Public Variables                               //
    //------------------------------------------------------------------------------//

    u64 visits_ = 0;  // Number of times the node has been visited
    f64 score_  = 0;  // Score of the node
    Turn turn_;

    private:
    //------------------------------------------------------------------------------//
    //                                Private Methods                               //
    //------------------------------------------------------------------------------//

    //------------------------------------------------------------------------------//
    //                               Private Variables                              //
    //------------------------------------------------------------------------------//
    Board board_{};               // Board state of the node
    MonteCarloTreeNode *parent_;  // Parent node of the current node
    std::unordered_map<move_t,
                       MonteCarloTreeNode *> children_;  // Map of moves to child nodes

    friend MonteCarloTree;
};

}  // namespace checkers::mcts

#include "monte_carlo_tree.tpp"

#endif
