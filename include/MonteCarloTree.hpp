
#ifndef CUDA_MCTS_CHECKRS_INCLUDE_MCT_HPP_
#define CUDA_MCTS_CHECKRS_INCLUDE_MCT_HPP_

#include <Board.hpp>
#include <Move.hpp>
#include <defines.hpp>
#include <unordered_map>

namespace CudaMctsCheckers
{

class MonteCarloTreeNode;

class MonteCarloTree
{
    public:
    //------------------------------------------------------------------------------//
    //                        Class Creation and Destruction                        //
    //------------------------------------------------------------------------------//
    MonteCarloTree();

    ~MonteCarloTree();

    //------------------------------------------------------------------------------//
    //                                Public Methods                                //
    //------------------------------------------------------------------------------//

    //------------------------------------------------------------------------------//
    //                               Public Variables                               //
    //------------------------------------------------------------------------------//

    private:
    //------------------------------------------------------------------------------//
    //                                Private Methods                               //
    //------------------------------------------------------------------------------//

    //------------------------------------------------------------------------------//
    //                               Private Variables                              //
    //------------------------------------------------------------------------------//
    MonteCarloTreeNode *root{};  // Root node of the tree
};

class MonteCarloTreeNode
{
    public:
    //------------------------------------------------------------------------------//
    //                        Class Creation and Destruction                        //
    //------------------------------------------------------------------------------//
    explicit MonteCarloTreeNode(MonteCarloTreeNode *parent = nullptr);

    ~MonteCarloTreeNode();

    //------------------------------------------------------------------------------//
    //                                Public Methods                                //
    //------------------------------------------------------------------------------//

    //------------------------------------------------------------------------------//
    //                               Public Variables                               //
    //------------------------------------------------------------------------------//

    private:
    //------------------------------------------------------------------------------//
    //                                Private Methods                               //
    //------------------------------------------------------------------------------//

    //------------------------------------------------------------------------------//
    //                               Private Variables                              //
    //------------------------------------------------------------------------------//
    Board board;                 // Board state of the node
    MonteCarloTreeNode *parent;  // Parent node of the current node
    std::unordered_map<Move,
                       MonteCarloTreeNode *> children;  // Map of moves to child nodes
};

}  // namespace CudaMctsCheckers

#endif  // CUDA_MCTS_CHECKRS_INCLUDE_MCT_HPP_