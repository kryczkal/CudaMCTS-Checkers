#include "mcts/monte_carlo_tree.hpp"
#include <chrono>
#include <cmath>
#include "common/checkers_defines.hpp"
#include "cpu/apply_move.hpp"
#include "cpu/board_helpers.hpp"
#include "cpu/launchers.hpp"
#include "cuda/launchers.cuh"
#include "game/cli_gui.hpp"

namespace checkers::mcts
{
MonteCarloTree::MonteCarloTree(Board board, Turn turn) { root_ = new MonteCarloTreeNode(board, turn); }

MonteCarloTree::~MonteCarloTree() { delete root_; }

f32 MonteCarloTree::WinRate(const MonteCarloTreeNode *node)
{
    return node->visits_ > 0 ? (f32)node->score_ / node->visits_ : 0.0f;
}

// void MonteCarloTree::DescendTree(const move_t move)
//{
//   auto node_to_descend_to_iter = root_->children_.find(move);
//   if (node_to_descend_to_iter == root_->children_.end()) {
//     assert(false);  // This should never happen
//     return;
//   }
//
//   MonteCarloTreeNode *node_to_descend_to = node_to_descend_to_iter->second;
//
//   // Remove the node to descend to from the children map to prevent it from being deleted
//   root_->children_.erase(node_to_descend_to_iter);
//
//   // Delete all other children
//   for (auto &child_pair : root_->children_) {
//     delete child_pair.second;
//   }
//
//   // Now delete the root itself
//   delete root_;
//
//   // Assign the new root
//   root_ = node_to_descend_to;
//   root_->parent_ = nullptr;
// }
move_t MonteCarloTree::Run(f32 time_seconds)
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
        elapsed                                          = std::chrono::duration<f32>(now - start).count();

        if (elapsed >= time_seconds) {
            break;
        }
        std::vector<SimulationResult> results = SimulateNodes(expanded_nodes);
        now                                   = std::chrono::system_clock::now();
        elapsed                               = std::chrono::duration<f32>(now - start).count();

        if (elapsed >= time_seconds) {
            break;
        }
        BackPropagate(expanded_nodes, results);
        now     = std::chrono::system_clock::now();
        elapsed = std::chrono::duration<f32>(now - start).count();

        if (elapsed >= time_seconds) {
            break;
        }
    }
    move_t best_move = SelectBestMove<f32, &MonteCarloTree::WinRate>(root_);
    assert(kInvalidMove != best_move);
    return best_move;
}

MonteCarloTreeNode *MonteCarloTree::SelectNode()
{
    assert(root_ != nullptr);

    MonteCarloTreeNode *current = root_;

    while (!current->children_.empty()) {
        f32 best_score                 = -1.0f;
        MonteCarloTreeNode *best_child = nullptr;
        for (auto &child : current->children_) {
            assert(child.second->visits_ > 0);  // Given the expansion step, this should never happen
            f32 uct_score = child.second->UctScore();
            if (uct_score > best_score) {
                best_score = uct_score;
                best_child = child.second;
            }
        }
        current = best_child;
        assert(current != nullptr);
    }
    return current;
}

std::vector<MonteCarloTreeNode *> MonteCarloTree::ExpandNode(MonteCarloTreeNode *node)
{
    using namespace checkers::cpu::launchers;
    using namespace checkers::cpu::apply_move;
    assert(node != nullptr);

    std::vector<MoveGenResult> mg = cpu::launchers::HostGenerateMoves({node->board_}, node->turn_);
    const bool capture_required =
        checkers::cpu::ReadFlag(mg[0].h_per_board_flags[0], MoveFlagsConstants::kCaptureFound);
    if (!checkers::cpu::ReadFlag(mg[0].h_per_board_flags[0], MoveFlagsConstants::kMoveFound)) {
        // TODO: Add handling of terminal nodes that just won or lost
        CliGui gui;
        std::cerr << "No moves found for the current player." << std::endl;
        gui.DisplayBoard(node->board_);
        assert(checkers::cpu::ReadFlag(mg[0].h_per_board_flags[0], MoveFlagsConstants::kMoveFound));
    }

    std::vector<MonteCarloTreeNode *> expanded_nodes;
    for (u32 i = 0; i < BoardConstants::kBoardSize; i++) {
        const u8 move_count        = mg[0].h_move_counts[i];
        const bool capture_in_move = mg[0].h_capture_masks[i] != 0;

        if (move_count == 0 || (capture_required && !capture_in_move)) {
            continue;
        }

        for (u32 j = 0; j < mg[0].h_move_counts[i]; j++) {
            bool is_capture = checkers::cpu::ReadFlag(mg[0].h_capture_masks[i], j);
            if (capture_required && !is_capture) {
                continue;
            }
            const move_t move             = mg[0].h_moves[i * kNumMaxMovesPerPiece + j];
            Board board_with_applied_move = node->board_;
            checkers::cpu::apply_move::ApplyMoveOnSingleBoard(
                move, board_with_applied_move.white, board_with_applied_move.black, board_with_applied_move.kings
            );
            const bool change_turn = !is_capture;  // TODO: This is not true for multi-captures
            // TODO: We should also check if there is promotion (assuming no multi-captures [they have promotion only at
            // the end])
            const Turn opposite_turn = node->turn_ == Turn::kWhite ? Turn::kBlack : Turn::kWhite;
            auto *new_node =
                new MonteCarloTreeNode(board_with_applied_move, node, change_turn ? opposite_turn : node->turn_);
            expanded_nodes.push_back(new_node);
            node->children_[move] = new_node;
        }
    }
    assert(expanded_nodes.size() > 0);
    assert(node->children_.size() == expanded_nodes.size());

    return expanded_nodes;
}

std::vector<SimulationResult> MonteCarloTree::SimulateNodes(std::vector<MonteCarloTreeNode *> nodes)
{
    using namespace checkers::cpu::launchers;

    const u64 kSimulationsPerNode = kMaxTotalSimulations / nodes.size();
    assert(kSimulationsPerNode > 0);
    assert(!nodes.empty());

    std::vector<SimulationParam> params;
    for (auto node : nodes) {
        params.push_back({
            .white         = node->board_.white,
            .black         = node->board_.black,
            .king          = node->board_.kings,
            .start_turn    = static_cast<u8>(node->turn_ == Turn::kWhite ? 0 : 1),
            .n_simulations = (u64)kSimulationsPerNode,
        });
    }
    return gpu::launchers::HostSimulateCheckersGames(params, kSimulationMaxDepth);
}

void MonteCarloTree::BackPropagate(std::vector<MonteCarloTreeNode *> nodes, const std::vector<SimulationResult> results)
{
    assert(nodes.size() == results.size());

    for (u32 i = 0; i < nodes.size(); i++) {
        MonteCarloTreeNode *node = nodes[i];
        SimulationResult result  = results[i];

        node->visits_ += result.n_simulations;
        node->score_ += result.score;
        while (node->parent_ != nullptr) {
            node = node->parent_;
            node->visits_ += result.n_simulations;
            node->score_ += GetScoreFromPerspectiveOfRoot(node, result);
        }
    }
}
f64 MonteCarloTree::GetScoreFromPerspectiveOfRoot(const MonteCarloTreeNode *node, const SimulationResult &result)
{
    assert(node != nullptr);
    assert(root_ != nullptr);
    if (root_->turn_ == node->turn_) {
        return result.score;
    } else {
        return result.n_simulations - result.score;
    }
}

MonteCarloTreeNode::MonteCarloTreeNode(MonteCarloTreeNode::Board board, Turn turn)
{
    board_  = board;
    turn_   = turn;
    parent_ = nullptr;
    score_  = 0;
    visits_ = 0;
}

MonteCarloTreeNode::MonteCarloTreeNode(MonteCarloTreeNode::Board board, MonteCarloTreeNode *parent, Turn turn)
{
    board_  = board;
    parent_ = parent;
    turn_   = turn;
    score_  = 0;
    visits_ = 0;
}

MonteCarloTreeNode::~MonteCarloTreeNode()
{
    for (auto child : children_) {
        delete child.second;
    }
}

f64 MonteCarloTreeNode::UctScore() const
{
    if (visits_ == 0) {
        return std::numeric_limits<f64>::infinity();
    }
    return (f64)score_ / visits_ + kExplorationConstant * sqrtf(logf((f32)parent_->visits_) / visits_);
}
}  // namespace checkers::mcts
