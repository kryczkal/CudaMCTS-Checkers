#include "cpu/launchers.hpp"
#include <ctime>
#include <random>
#include <vector>
#include "common/checkers_defines.hpp"
#include "cpu/apply_move.hpp"
#include "cpu/board.hpp"
#include "cpu/board_helpers.hpp"
#include "cpu/move_generation.hpp"
#include "cpu/move_selection.hpp"

namespace checkers::cpu::launchers
{

std::vector<MoveGenResult> HostGenerateMoves(const std::vector<Board> &boards, Turn turn)
{
    using namespace checkers::cpu::move_gen;

    size_t n_boards = boards.size();
    std::vector<MoveGenResult> results(n_boards);

    if (n_boards == 0) {
        return results;
    }

    for (size_t board_idx = 0; board_idx < n_boards; board_idx++) {
        // Extract bitmasks
        board_t white_bits = boards[board_idx].white;
        board_t black_bits = boards[board_idx].black;
        board_t king_bits  = boards[board_idx].kings;

        // We'll fill the MoveGenResult for this board
        MoveGenResult &R = results[board_idx];

        std::fill(R.h_move_counts.begin(), R.h_move_counts.end(), 0);
        std::fill(R.h_capture_masks.begin(), R.h_capture_masks.end(), 0);
        R.h_per_board_flags[0] = 0;

        for (board_index_t fig_idx = 0; fig_idx < BoardConstants::kBoardSize; fig_idx++) {
            move_t *moves_ptr      = &R.h_moves[fig_idx * kNumMaxMovesPerPiece];
            u8 &move_count         = R.h_move_counts[fig_idx];
            move_flags_t &cap_mask = R.h_capture_masks[fig_idx];
            cap_mask               = 0;

            if (turn == Turn::kWhite) {
                GenerateMovesForSinglePiece<Turn::kWhite>(
                    fig_idx, white_bits, black_bits, king_bits, moves_ptr, move_count, cap_mask, R.h_per_board_flags[0]
                );
            } else {
                GenerateMovesForSinglePiece<Turn::kBlack>(
                    fig_idx, white_bits, black_bits, king_bits, moves_ptr, move_count, cap_mask, R.h_per_board_flags[0]
                );
            }
        }
    }

    return results;
}
std::vector<Board> HostApplyMoves(const std::vector<Board> &boards, const std::vector<move_t> &moves)
{
    size_t n_boards            = boards.size();
    std::vector<Board> updated = boards;
    if (moves.size() != n_boards) {
        assert(true && "Mismatch in number of boards and moves");
        return updated;
    }

    for (size_t i = 0; i < n_boards; i++) {
        apply_move::ApplyMoveOnSingleBoard(moves[i], updated[i].white, updated[i].black, updated[i].kings);
    }

    return updated;
}
std::vector<move_t> HostSelectBestMoves(
    const std::vector<Board> &boards, const std::vector<move_t> &moves, const std::vector<u8> &move_counts,
    const std::vector<move_flags_t> &capture_masks, const std::vector<move_flags_t> &per_board_flags,
    std::vector<u32> &seeds
)
{
    size_t n_boards = boards.size();
    std::vector<move_t> best_moves(n_boards, kInvalidMove);

    if (n_boards == 0) {
        return best_moves;
    }

    size_t total_squares         = BoardConstants::kBoardSize;
    size_t moves_per_piece       = kNumMaxMovesPerPiece;
    size_t total_moves_per_board = total_squares * moves_per_piece;

    for (size_t b = 0; b < n_boards; b++) {
        // Offsets
        const move_t *board_moves          = &moves[b * total_moves_per_board];
        const u8 *board_counts             = &move_counts[b * total_squares];
        const move_flags_t *board_captures = &capture_masks[b * total_squares];
        move_flags_t flags                 = per_board_flags[b];
        u32 local_seed                     = seeds[b];

        best_moves[b] = move_selection::SelectBestMoveForSingleBoard(
            boards[b].white, boards[b].black, boards[b].kings, board_moves, board_counts, board_captures, flags,
            local_seed
        );
    }

    return best_moves;
}
std::vector<SimulationResult> HostSimulateCheckersGames(const std::vector<SimulationParam> &params, int max_iterations)
{
    using namespace checkers::cpu;

    size_t n_configs = params.size();
    std::vector<SimulationResult> results(n_configs);

    // Count total simulations
    u64 total_sims = 0;
    for (auto &p : params) {
        total_sims += p.n_simulations;
    }
    if (total_sims == 0) {
        return results;
    }

    std::vector<u8> seeds(total_sims);
    {
        std::mt19937 rng(kTrueRandom ? std::chrono::system_clock::now().time_since_epoch().count() : kSeed);
        for (u64 i = 0; i < total_sims; i++) {
            seeds[i] = static_cast<u8>(rng() & 0xFF);
        }
    }

    // We'll produce a single array "scores" of length total_sims, each in {0,1,2,3}
    // 0 => in progress, 1 => White wins, 2 => Black wins, 3 => draw
    std::vector<u8> finalScores(total_sims, 0);

    // We'll keep a running index in seeds/finalScores for each config
    u64 offset = 0;
    for (size_t idx = 0; idx < n_configs; idx++) {
        u64 n_sims                 = params[idx].n_simulations;
        results[idx].n_simulations = n_sims;
        if (n_sims == 0) {
            results[idx].score = 0.0;
            continue;
        }

        // Grab the board
        board_t w_bits   = params[idx].white;
        board_t b_bits   = params[idx].black;
        board_t k_bits   = params[idx].king;
        bool start_black = (params[idx].start_turn == 1);

        // For each simulation
        for (u64 s = 0; s < n_sims; s++) {
            u64 sim_index = offset + s;
            u32 seed_ref  = seeds[sim_index];

            // We'll copy the board so we can mutate it
            board_t white_board        = w_bits;
            board_t black_board        = b_bits;
            board_t king_board         = k_bits;
            bool current_turn_is_black = start_black;
            u8 outcome                 = 0;  // 1=WhiteWins,2=BlackWins,3=Draw

            // 40-move rule (non-reversible moves) counter
            u8 non_reversible_count = 0;

            // We'll do up to max_iterations half-moves
            for (int moveCount = 0; moveCount < max_iterations; moveCount++) {
                // Check if someone is out of pieces
                if (black_board == 0) {
                    outcome = kOutcomeWhite;  // White wins
                    break;
                } else if (white_board == 0) {
                    outcome = kOutcomeBlack;  // Black wins
                    break;
                }

                // Generate moves for current side
                MoveGenResult mg;
                mg.h_per_board_flags[0] = 0;

                // Fill with zeros
                std::fill(mg.h_move_counts.begin(), mg.h_move_counts.end(), 0);
                std::fill(mg.h_capture_masks.begin(), mg.h_capture_masks.end(), 0);

                // Each square
                for (board_index_t sq = 0; sq < BoardConstants::kBoardSize; sq++) {
                    move_t *out_moves              = &mg.h_moves[sq * kNumMaxMovesPerPiece];
                    u8 &out_count                  = mg.h_move_counts[sq];
                    move_flags_t &out_capture_mask = mg.h_capture_masks[sq];

                    if (!current_turn_is_black) {
                        move_gen::GenerateMovesForSinglePiece<Turn::kWhite>(
                            sq, white_board, black_board, king_board, out_moves, out_count, out_capture_mask,
                            mg.h_per_board_flags[0]
                        );
                    } else {
                        move_gen::GenerateMovesForSinglePiece<Turn::kBlack>(
                            sq, white_board, black_board, king_board, out_moves, out_count, out_capture_mask,
                            mg.h_per_board_flags[0]
                        );
                    }
                }

                // select a move from mg
                move_t chosen_move = move_selection::SelectBestMoveForSingleBoard(
                    white_board, black_board, king_board, mg.h_moves.data(), mg.h_move_counts.data(),
                    mg.h_capture_masks.data(), mg.h_per_board_flags[0], seed_ref
                );

                if (chosen_move == kInvalidMove) {
                    // no moves => side to move loses
                    outcome = (!current_turn_is_black ? kOutcomeBlack : kOutcomeWhite);
                    break;
                }

                // apply the chosen move
                apply_move::ApplyMoveOnSingleBoard(chosen_move, white_board, black_board, king_board);

                // check if it was a capture and try to chain
                bool was_capture    = ReadFlag(mg.h_per_board_flags[0], MoveFlagsConstants::kCaptureFound);
                board_index_t to_sq = move_gen::DecodeMove<move_gen::MovePart::To>(chosen_move);

                while (was_capture) {
                    // Clear move related variables
                    mg.h_per_board_flags[0] = 0;
                    mg.h_move_counts[0]     = 0;
                    mg.h_capture_masks[0]   = 0;

                    move_t *out_moves              = &mg.h_moves[0 * kNumMaxMovesPerPiece];
                    u8 &out_count                  = mg.h_move_counts[0];
                    move_flags_t &out_capture_mask = mg.h_capture_masks[0];

                    if (!current_turn_is_black) {
                        move_gen::GenerateMovesForSinglePiece<Turn::kWhite>(
                            to_sq, white_board, black_board, king_board, out_moves, out_count, out_capture_mask,
                            mg.h_per_board_flags[0]
                        );
                    } else {
                        move_gen::GenerateMovesForSinglePiece<Turn::kBlack>(
                            to_sq, white_board, black_board, king_board, out_moves, out_count, out_capture_mask,
                            mg.h_per_board_flags[0]
                        );
                    }

                    was_capture = ReadFlag(mg.h_per_board_flags[0], MoveFlagsConstants::kCaptureFound);
                    if (!was_capture) {
                        break;
                    }

                    move_t chain_move = move_selection::SelectBestMoveForSingleBoard(
                        white_board, black_board, king_board, mg.h_moves.data(), mg.h_move_counts.data(),
                        mg.h_capture_masks.data(), mg.h_per_board_flags[0], seed_ref
                    );

                    if (chain_move == kInvalidMove) {
                        break;
                    }

                    to_sq = move_gen::DecodeMove<move_gen::MovePart::To>(chain_move);

                    apply_move::ApplyMoveOnSingleBoard(chain_move, white_board, black_board, king_board);
                }

                bool to_is_king = ReadFlag(king_board, to_sq);

                // Promotion
                king_board |= (white_board & BoardConstants::kTopBoardEdgeMask);
                king_board |= (black_board & BoardConstants::kBottomBoardEdgeMask);

                // 40-move rule
                if (!was_capture && to_is_king) {
                    non_reversible_count++;
                } else {
                    non_reversible_count = 0;
                }
                if (non_reversible_count >= 40) {
                    outcome = kOutcomeDraw;  // draw
                    break;
                }

                // switch turn
                current_turn_is_black = !current_turn_is_black;
            }

            // If still 0, declare a draw
            if (outcome == kOutcomeInProgress) {
                outcome = kOutcomeDraw;  // draw
            }

            // Convert outcome from {1=White,2=Black,3=Draw} to perspective of the starting side
            u8 store_val = 0;  // lose
            if (outcome == kOutcomeDraw) {
                store_val = 1;  // draw
            } else if (!start_black && outcome == kOutcomeWhite) {
                // White started, White wins => storeVal=2
                store_val = 2;
            } else if (start_black && outcome == kOutcomeBlack) {
                // Black started, Black wins => storeVal=2
                store_val = 2;
            }
            results[idx].score += store_val;
        }

        results[idx].score /= 2.0;
    }

    return results;
}
}  // namespace checkers::cpu::launchers
