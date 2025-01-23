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
    const std::vector<u8> &seeds
)
{
    size_t n_boards = boards.size();
    std::vector<move_t> bestMoves(n_boards, kInvalidMove);

    if (n_boards == 0) {
        return bestMoves;
    }

    size_t totalSquares       = BoardConstants::kBoardSize;
    size_t movesPerPiece      = kNumMaxMovesPerPiece;
    size_t totalMovesPerBoard = totalSquares * movesPerPiece;

    for (size_t b = 0; b < n_boards; b++) {
        // Offsets
        const move_t *boardMoves          = &moves[b * totalMovesPerBoard];
        const u8 *boardCounts             = &move_counts[b * totalSquares];
        const move_flags_t *boardCaptures = &capture_masks[b * totalSquares];
        move_flags_t flags                = per_board_flags[b];
        u8 localSeed                      = seeds[b];

        bestMoves[b] = move_selection::SelectBestMoveForSingleBoard(
            boards[b].white, boards[b].black, boards[b].kings, boardMoves, boardCounts, boardCaptures, flags, localSeed
        );
    }

    return bestMoves;
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

    // Prepare random seeds
    std::vector<u8> seeds(total_sims);
    {
        std::mt19937 rng((unsigned)std::time(nullptr));
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
        board_t wBits   = params[idx].white;
        board_t bBits   = params[idx].black;
        board_t kBits   = params[idx].king;
        bool startBlack = (params[idx].start_turn == 1);

        // For each simulation
        for (u64 s = 0; s < n_sims; s++) {
            u64 simIndex = offset + s;
            u8 seedRef   = seeds[simIndex];

            // We'll copy the board so we can mutate it
            board_t whiteBoard      = wBits;
            board_t blackBoard      = bBits;
            board_t kingBoard       = kBits;
            bool currentTurnIsBlack = startBlack;
            u8 outcome              = 0;  // 1=WhiteWins,2=BlackWins,3=Draw

            // 40-move rule (non-reversible moves) counter
            u8 nonReversibleCount = 0;

            // We'll do up to max_iterations half-moves
            for (int moveCount = 0; moveCount < max_iterations; moveCount++) {
                // Check if someone is out of pieces
                if (blackBoard == 0) {
                    outcome = 1;  // White wins
                    break;
                } else if (whiteBoard == 0) {
                    outcome = 2;  // Black wins
                    break;
                }

                // Generate moves for current side
                MoveGenResult mg;
                mg.h_per_board_flags[0] = 0;
                // Fill with zeros
                std::fill(mg.h_moves.begin(), mg.h_moves.end(), kInvalidMove);
                std::fill(mg.h_move_counts.begin(), mg.h_move_counts.end(), 0);
                std::fill(mg.h_capture_masks.begin(), mg.h_capture_masks.end(), 0);

                // Each square
                for (board_index_t sq = 0; sq < BoardConstants::kBoardSize; sq++) {
                    move_t *outMoves             = &mg.h_moves[sq * kNumMaxMovesPerPiece];
                    u8 &outCount                 = mg.h_move_counts[sq];
                    move_flags_t &outCaptureMask = mg.h_capture_masks[sq];

                    if (!currentTurnIsBlack) {
                        move_gen::GenerateMovesForSinglePiece<Turn::kWhite>(
                            sq, whiteBoard, blackBoard, kingBoard, outMoves, outCount, outCaptureMask,
                            mg.h_per_board_flags[0]
                        );
                    } else {
                        move_gen::GenerateMovesForSinglePiece<Turn::kBlack>(
                            sq, whiteBoard, blackBoard, kingBoard, outMoves, outCount, outCaptureMask,
                            mg.h_per_board_flags[0]
                        );
                    }
                }

                // select a move from mg
                move_t chosenMove = move_selection::SelectBestMoveForSingleBoard(
                    whiteBoard, blackBoard, kingBoard, mg.h_moves.data(), mg.h_move_counts.data(),
                    mg.h_capture_masks.data(), mg.h_per_board_flags[0], seedRef
                );

                if (chosenMove == kInvalidMove) {
                    // no moves => side to move loses
                    outcome = (!currentTurnIsBlack ? 2 : 1);
                    break;
                }

                // apply the chosen move
                apply_move::ApplyMoveOnSingleBoard(chosenMove, whiteBoard, blackBoard, kingBoard);

                // check if it was a capture or a king moved
                bool was_capture      = (mg.h_per_board_flags[0] & (1 << MoveFlagsConstants::kCaptureFound));
                board_index_t from_sq = move_gen::DecodeMove<move_gen::MovePart::From>(chosenMove);
                bool from_was_king    = ((kingBoard >> from_sq) & 1U);

                // TODO: Chain captures

                // Promotion
                kingBoard |= (whiteBoard & BoardConstants::kTopBoardEdgeMask);
                kingBoard |= (blackBoard & BoardConstants::kBottomBoardEdgeMask);

                // 40-move rule
                if (!was_capture && from_was_king) {
                    nonReversibleCount++;
                } else {
                    nonReversibleCount = 0;
                }
                if (nonReversibleCount >= 40) {
                    outcome = 3;  // draw
                    break;
                }

                // switch turn
                currentTurnIsBlack = !currentTurnIsBlack;
            }

            // If still 0, declare a draw
            if (outcome == 0) {
                outcome = 3;  // draw
            }

            // Convert outcome from {1=White,2=Black,3=Draw} to perspective of 'startBlack'
            // We store finalScores[simIndex] in {0=loss,1=draw,2=win}
            // If 'startBlack == false' => White started, so outcome=1 => "win" for the starter => storeVal=2
            // If outcome=3 => draw => storeVal=1
            // If outcome=the other side => storeVal=0
            u8 storeVal = 0;  // lose
            if (outcome == 3) {
                storeVal = 1;  // draw
            } else if (!startBlack && outcome == 1) {
                // White started, White wins => storeVal=2
                storeVal = 2;
            } else if (startBlack && outcome == 2) {
                // Black started, Black wins => storeVal=2
                storeVal = 2;
            }
            finalScores[simIndex] = storeVal;
        }

        // Sum the range finalScores[offset .. offset+n_sims-1]
        u64 sum = 0;
        for (u64 s = 0; s < n_sims; s++) {
            sum += finalScores[offset + s];
        }
        offset += n_sims;

        // finalScores are in {0=lose,1=draw,2=win}
        // We convert sum_of_outcomes to a fractional score => sum/2.0
        // (2 => 1.0, 1 => 0.5, 0 => 0.0 average)
        double finalScore  = static_cast<double>(sum) / 2.0;
        results[idx].score = finalScore;
    }

    return results;
}
}  // namespace checkers::cpu::launchers
