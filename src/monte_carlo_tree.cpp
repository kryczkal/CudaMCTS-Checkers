#include <concepts.hpp>
#include <game_simulation.hpp>
#include <monte_carlo_tree.hpp>
#include <move_generation.hpp>

#include <cassert>
#include <chrono>
#include <cmath>
#include <vector>

namespace CudaMctsCheckers
{

MonteCarloTree::MonteCarloTree(CudaMctsCheckers::Board board, Turn turn)
{
    root_ = new MonteCarloTreeNode(board, turn);
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
            //            if (child.second->visits_ == 0) {
            //                best_child = child.second;
            //                break;
            //            }
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

    std::vector<MonteCarloTreeNode *> expanded_nodes;
    for (u32 i = 0; i < Move::kNumMoveArrayForPlayerSize; ++i) {  // TODO: Skip invalid moves
        if (output.possible_moves[i] != Move::kInvalidMove) {
            if (output.capture_moves_bitmask
                    [CudaMctsCheckers::MoveGenerationOutput::CaptureFlagIndex] &&
                !output.capture_moves_bitmask[i]) {
                continue;
            }
            Board new_board = node->board_;
            if (node->turn_ == Turn::kWhite) {
                new_board.ApplyMove<BoardCheckType::kWhite>(
                    Move::DecodeOriginIndex(i), output.possible_moves[i],
                    output.capture_moves_bitmask[i]
                );
            } else {
                new_board.ApplyMove<BoardCheckType::kBlack>(
                    Move::DecodeOriginIndex(i), output.possible_moves[i],
                    output.capture_moves_bitmask[i]
                );
            }

            MonteCarloTreeNode *new_node = new MonteCarloTreeNode(new_board, node);
            new_node->turn_              = output.capture_moves_bitmask[i] ? node->turn_
                                           : node->turn_ == Turn::kWhite   ? Turn::kBlack
                                                                           : Turn::kWhite;
            expanded_nodes.push_back(new_node);
            node->children_[MonteCarloTreeNode::EncodeMove(
                Move::DecodeOriginIndex(i), output.possible_moves[i]
            )] = new_node;
        }
    }
    return expanded_nodes;
}

std::vector<SimulationResult> MonteCarloTree::SimulateNodes(std::vector<MonteCarloTreeNode *> nodes)
{
    std::vector<SimulationResult> results(nodes.size());
    for (u32 i = 0; i < nodes.size(); ++i) {
        Board board      = nodes[i]->board_;
        Turn turn        = nodes[i]->turn_;
        results[i].score = GameSimulation::RunGame(board, turn, GetWantedResult());
        results[i].visits++;
    }
    return results;
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

TrieDecodedMoveAsPair MonteCarloTree::Run(f32 time_seconds)
{
    auto start = std::chrono::system_clock::now();

    while (true) {
        auto now     = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration<f32>(now - start).count();

        if (elapsed >= time_seconds) {
            break;
        }

        MonteCarloTreeNode *node = SelectNode();
        now                      = std::chrono::system_clock::now();
        elapsed                  = std::chrono::duration<f32>(now - start).count();

        if (elapsed >= time_seconds) {
            break;
        }
        std::vector<MonteCarloTreeNode *> expanded_nodes = ExpandNode(node);
        now                                              = std::chrono::system_clock::now();
        elapsed = std::chrono::duration<f32>(now - start).count();

        if (elapsed >= time_seconds) {
            break;
        }
        std::vector<SimulationResult> results = SimulateNodes(expanded_nodes);
        now                                   = std::chrono::system_clock::now();
        elapsed                               = std::chrono::duration<f32>(now - start).count();

        if (elapsed >= time_seconds) {
            break;
        }
        Backpropagate(expanded_nodes, results);
        now     = std::chrono::system_clock::now();
        elapsed = std::chrono::duration<f32>(now - start).count();

        if (elapsed >= time_seconds) {
            break;
        }
    }
    TrieEncodedMove best_move = SelectBestMove<f32, &MonteCarloTree::WinRate>();
    assert((best_move & 0x00FF) != Board::kInvalidIndex);
    return MonteCarloTreeNode::DecodeMove(best_move);
}

f32 MonteCarloTree::WinRate(MonteCarloTreeNode *node)
{
    return node->visits_ > 0 ? (f32)node->score_ / node->visits_ : 0;
}

GameResult MonteCarloTree::GetWantedResult() const
{
    return root_->turn_ == Turn::kWhite ? GameResult::kWhiteWin : GameResult::kBlackWin;
}

void MonteCarloTree::DescendTree(const TrieEncodedMove move)
{
    auto node_to_descend_to = root_->children_.find(move);
    // Delete all children of the root except the one we are descending to
    for (auto it = root_->children_.begin(); it != root_->children_.end(); ++it) {
        if (it != node_to_descend_to) {
            delete it->second;
        }
    }
    root_->children_.clear();
    delete root_;
    root_ = node_to_descend_to->second;
}

MonteCarloTreeNode::MonteCarloTreeNode(Board board, Turn turn) : board_(board), turn_(turn)
{
    parent_ = nullptr;
}
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

u16 MonteCarloTreeNode::EncodeMove(Board::IndexType piece, Move::Type movement)
{
    return ((u16)piece << 8) | movement;
}

TrieDecodedMoveAsPair MonteCarloTreeNode::DecodeMove(u16 encoded_move)
{
    return {
        static_cast<Board::IndexType>(encoded_move >> 8),
        static_cast<Move::Type>(encoded_move & 0xFF)
    };
}

}  // namespace CudaMctsCheckers
