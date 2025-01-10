
#ifndef CUDA_MCTS_CHECKRS_INCLUDE_MCT_HPP_
#define CUDA_MCTS_CHECKRS_INCLUDE_MCT_HPP_

#include <Board.hpp>
#include <Move.hpp>
#include <concepts.hpp>
#include <defines.hpp>
#include <unordered_map>
#include <vector>

namespace CudaMctsCheckers
{

struct PACK SimulationResult {
    f32 score;
    u32 visits;
};

static constexpr f32 kExplorationConstant = 1.41f;

class MonteCarloTreeNode;

class MonteCarloTree
{
    public:
    //------------------------------------------------------------------------------//
    //                        Class Creation and Destruction                        //
    //------------------------------------------------------------------------------//
    explicit MonteCarloTree(Board board);

    ~MonteCarloTree();

    //------------------------------------------------------------------------------//
    //                                Public Methods                                //
    //------------------------------------------------------------------------------//
    Move Run(f32 time);

    //------------------------------------------------------------------------------//
    //                               Public Variables                               //
    //------------------------------------------------------------------------------//

    private:
    //------------------------------------------------------------------------------//
    //                                Private Methods                               //
    //------------------------------------------------------------------------------//

    ///////////////////// Main Monte Carlo Tree Search steps /////////////////////////
    MonteCarloTreeNode *SelectNode();
    std::vector<MonteCarloTreeNode *> ExpandNode(MonteCarloTreeNode *node);
    std::vector<SimulationResult> SimulateNodes(std::vector<MonteCarloTreeNode *> nodes);
    void Backpropagate(
        std::vector<MonteCarloTreeNode *> &nodes, const std::vector<SimulationResult> &results
    );

    //////////////////////////////////////////////////////////////////////////////////
    template <MaxComparable EvalType, EvalFunction<EvalType> auto EvalFunc>
    Move SelectBestMove();

    // Evaluation functions
    static f32 WinRate(MonteCarloTreeNode *node);

    //------------------------------------------------------------------------------//
    //                               Private Variables                              //
    //------------------------------------------------------------------------------//
    MonteCarloTreeNode *root_{};  // Root node of the tree
};

class MonteCarloTreeNode
{
    public:
    //------------------------------------------------------------------------------//
    //                        Class Creation and Destruction                        //
    //------------------------------------------------------------------------------//
    explicit MonteCarloTreeNode(Board board, MonteCarloTreeNode *parent = nullptr);

    ~MonteCarloTreeNode();

    //------------------------------------------------------------------------------//
    //                                Public Methods                                //
    //------------------------------------------------------------------------------//
    f32 UctScore() const;

    //------------------------------------------------------------------------------//
    //                               Public Variables                               //
    //------------------------------------------------------------------------------//
    size_t visits_ = 0;  // Number of times the node has been visited
    f32 score_     = 0;  // Score of the node

    private:
    //------------------------------------------------------------------------------//
    //                                Private Methods                               //
    //------------------------------------------------------------------------------//

    //------------------------------------------------------------------------------//
    //                               Private Variables                              //
    //------------------------------------------------------------------------------//
    Board board_{};               // Board state of the node
    MonteCarloTreeNode *parent_;  // Parent node of the current node
    std::unordered_map<Move,
                       MonteCarloTreeNode *> children_;  // Map of moves to child nodes

    friend MonteCarloTree;
};

}  // namespace CudaMctsCheckers

#include <MonteCarloTree.tpp>

#endif  // CUDA_MCTS_CHECKRS_INCLUDE_MCT_HPP_