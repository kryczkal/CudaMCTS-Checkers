#include <concepts.hpp>
#include <monte_carlo_tree.hpp>
#include <move_generation.hpp>

#include <cassert>
#include <chrono>
#include <cmath>
#include <vector>

namespace CudaMctsCheckers
{

MonteCarloTree::MonteCarloTree(CudaMctsCheckers::Board board)
{
    root_ = new MonteCarloTreeNode(board, Turn::kWhite);  // TODO: What if black starts?
}

MonteCarloTree::~MonteCarloTree() { delete root_; }

MonteCarloTreeNode *CudaMctsCheckers::MonteCarloTree::SelectNode()
{
    assert(root_ != nullptr);

    MonteCarloTreeNode *current = root_;

    while (!current->children_.empty()) {
        f32 best_score                 = -1.0f;
        MonteCarloTreeNode *best_child = nullptr;
        for (auto &child : current->children_) {
            f32 uct_score = child.second->UctScore();
            if (uct_score > best_score) {
                best_score = uct_score;
                best_child = child.second;
            }
        }
        current = best_child;
    }
    return current;
}

std::vector<MonteCarloTreeNode *> MonteCarloTree::ExpandNode(MonteCarloTreeNode *node)
{
    MoveGenerationOutput output;
    if (node->turn_ == Turn::kWhite) {
        output = MoveGenerator::GenerateMovesForPlayerCpu<BoardCheckType::kWhite>(node->board_);
    } else {
        output = MoveGenerator::GenerateMovesForPlayerCpu<BoardCheckType::kBlack>(node->board_);
    }

    for (u32 i = 0; i < Move::kNumMoveArrayForPlayerSize; ++i) {  // TODO: Skip invalid moves
        if (output.possible_moves[i] != Move::kInvalidMove) {
            Board new_board = node->board_;
            if (node->turn_ == Turn::kWhite) {
                new_board.MovePiece<BoardCheckType::kWhite>(
                    Move::DecodeOriginIndex(output.possible_moves, output.possible_moves[i]),
                    output.possible_moves[i]
                );
            } else {
                new_board.MovePiece<BoardCheckType::kBlack>(
                    Move::DecodeOriginIndex(output.possible_moves, output.possible_moves[i]),
                    output.possible_moves[i]
                );
            }
            node->children_[output.possible_moves[i]] = new MonteCarloTreeNode(new_board, node);
        }
    }
}

std::vector<SimulationResult> MonteCarloTree::SimulateNodes(std::vector<MonteCarloTreeNode *> nodes)
{
    return {};
}

void MonteCarloTree::Backpropagate(
    std::vector<MonteCarloTreeNode *> &nodes, const std::vector<SimulationResult> &results
)
{
    for (size_t i = 0; i < nodes.size(); ++i) {
        MonteCarloTreeNode *node = nodes[i];
        while (node != nullptr) {
            node->score_ += results[i].score;
            node->visits_ += results[i].visits;
            node = node->parent_;
        }
    }
}

Move::Type MonteCarloTree::Run(f32 time)
{
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    while (std::chrono::duration<f32>(std::chrono::system_clock::now() - start).count() < time) {
        MonteCarloTreeNode *node                         = SelectNode();
        std::vector<MonteCarloTreeNode *> expanded_nodes = ExpandNode(node);
        std::vector<SimulationResult> results            = SimulateNodes(expanded_nodes);
        Backpropagate(expanded_nodes, results);
    }
    return SelectBestMove<f32, &MonteCarloTree::WinRate>();
}

f32 MonteCarloTree::WinRate(MonteCarloTreeNode *node) { return (f32)node->score_ / node->visits_; }

MonteCarloTreeNode::MonteCarloTreeNode(Board board, Turn turn) : board_(board), turn_(turn) {}
MonteCarloTreeNode::MonteCarloTreeNode(Board board, MonteCarloTreeNode *parent)
    : board_(board),
      parent_(parent),
      turn_(parent->turn_ == Turn::kWhite ? Turn::kBlack : Turn::kWhite)
{
}

MonteCarloTreeNode::~MonteCarloTreeNode()
{
    for (auto &child : children_) {
        delete child.second;
    }
}

f32 MonteCarloTreeNode::UctScore() const
{
    return (f32)score_ / visits_ +
           kExplorationConstant * sqrtf(logf((f32)parent_->visits_) / visits_);
}

}  // namespace CudaMctsCheckers
