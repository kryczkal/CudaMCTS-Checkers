
#ifndef CUDA_MCTS_CHECKRS_INCLUDE_MCT_HPP_
#define CUDA_MCTS_CHECKRS_INCLUDE_MCT_HPP_

#include <array>
#include <board.hpp>
#include <concepts.hpp>
#include <cpp_defines.hpp>
#include <move.hpp>
#include <unordered_map>
#include <vector>

namespace CudaMctsCheckers
{

enum class Turn { kWhite, kBlack };

struct PACK SimulationResult {
    f32 score;
    u32 visits;
};

using TrieEncodedMove       = u16;
using TrieDecodedMoveAsPair = std::pair<Board::IndexType, Move::Type>;

static constexpr f32 kExplorationConstant = 1.41f;

class MonteCarloTreeNode;

class MonteCarloTree
{
    public:
    //------------------------------------------------------------------------------//
    //                        Class Creation and Destruction                        //
    //------------------------------------------------------------------------------//
    explicit MonteCarloTree(Board board, Turn turn);

    ~MonteCarloTree();

    //------------------------------------------------------------------------------//
    //                                Public Methods                                //
    //------------------------------------------------------------------------------//
    TrieDecodedMoveAsPair Run(f32 time_seconds);

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
    TrieEncodedMove SelectBestMove();

    // Evaluation functions
    static f32 WinRate(MonteCarloTreeNode *node);

    //------------------------------------------------------------------------------//
    //                               Private Variables                              //
    //------------------------------------------------------------------------------//
    MonteCarloTreeNode *root_{};  // Root node of the tree

    Move::MoveArrayForPlayer h_generated_moves_;
};

class MonteCarloTreeNode
{
    public:
    //------------------------------------------------------------------------------//
    //                        Class Creation and Destruction                        //
    //------------------------------------------------------------------------------//

    explicit MonteCarloTreeNode(Board board, Turn turn);
    explicit MonteCarloTreeNode(Board board, MonteCarloTreeNode *parent);

    ~MonteCarloTreeNode();

    //------------------------------------------------------------------------------//
    //                                Public Methods                                //
    //------------------------------------------------------------------------------//

    f32 UctScore() const;
    static u16 EncodeMove(Board::IndexType piece, Move::Type movement);
    static TrieDecodedMoveAsPair DecodeMove(u16 encoded_move);

    //------------------------------------------------------------------------------//
    //                               Public Variables                               //
    //------------------------------------------------------------------------------//

    size_t visits_ = 0;  // Number of times the node has been visited
    f32 score_     = 0;  // Score of the node
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
    std::unordered_map<
        TrieEncodedMove,
        MonteCarloTreeNode *>
        children_;  // Map of moves to child nodes

    friend MonteCarloTree;
};

}  // namespace CudaMctsCheckers

#include <monte_carlo_tree.tpp>

#endif  // CUDA_MCTS_CHECKRS_INCLUDE_MCT_HPP_
