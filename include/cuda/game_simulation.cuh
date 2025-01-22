#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "apply_move.cuh"
#include "check_outcome.cuh"
#include "checkers_defines.hpp"
#include "cuda_utils.cuh"
#include "move_generation.cuh"
#include "move_selection.cuh"

namespace checkers::gpu
{

/**
 * \brief Outcome encoding in scores[]:
 *  0 = in progress (not used at the end, but can be intermediate)
 *  1 = White wins
 *  2 = Black wins
 *  3 = Draw
 */
static constexpr u8 kOutcomeWhite = 1;
static constexpr u8 kOutcomeBlack = 2;
static constexpr u8 kOutcomeDraw  = 3;
static constexpr u8 kOutcomeNone  = 0;

/**
 * \brief Per-thread device function that simulates a single checkers game.
 * \param board_idx Index of the board to simulate.
 * \param d_whites White piece bitmasks for all boards.
 * \param d_blacks Black piece bitmasks for all boards.
 * \param d_kings  King bitmasks for all boards.
 * \param d_scores Output array for final results (1=White,2=Black,3=Draw).
 * \param n_boards Number of boards in total.
 * \param d_seeds  Array of random seeds for each board.
 * \param max_iterations A safe upper bound on the number of half-moves you allow.
 */
__device__ __forceinline__ void DeviceSimulateSingleCheckersGame(
    const u64 board_idx, board_t* d_whites, board_t* d_blacks, board_t* d_kings, u8* d_scores, const u64 n_boards,
    u8* d_seeds, const int max_iterations
)
{
    // Quick references
    board_t& whites = d_whites[board_idx];
    board_t& blacks = d_blacks[board_idx];
    board_t& kings  = d_kings[board_idx];

    // Track the final outcome in local var, then write to d_scores at the end
    u8 outcome = kOutcomeNone;

    // Alternate between Turn::kWhite and Turn::kBlack
    Turn currentTurn = Turn::kWhite;

    // 40-move rule tracking
    int num_consecutive_non_reversible_moves = 0;

    // Do up to max_iterations iterations
    for (int move_number = 0; move_number < max_iterations; ++move_number) {
        //--------------- 1) Check if either side has no pieces => game ends ---------------
        if (blacks == 0) {
            outcome = kOutcomeWhite;  // White wins
            break;
        }
        if (whites == 0) {
            outcome = kOutcomeBlack;  // Black wins
            break;
        }

        //--------------- 2) Generate all moves for the current side ---------------
        // We make local arrays (on the stack) for moves, counts, captures, etc.
        // Size = BoardConstants::kBoardSize * kNumMaxMovesPerPiece
        move_t local_moves[BoardConstants::kBoardSize * move_gen::kNumMaxMovesPerPiece];
        u8 local_move_counts[BoardConstants::kBoardSize];
        move_flags_t local_capture_masks[BoardConstants::kBoardSize];
        move_flags_t local_per_board_flags = 0;  // For forced capture detection

        // Initialize them
        for (int i = 0; i < BoardConstants::kBoardSize; i++) {
            local_move_counts[i]   = 0;
            local_capture_masks[i] = 0;
        }
        for (int i = 0; i < BoardConstants::kBoardSize * move_gen::kNumMaxMovesPerPiece; i++) {
            local_moves[i] = MoveConstants::kInvalidMove;
        }

        // We can generate moves by calling the existing per-square function:
        for (board_index_t figure_idx = 0; figure_idx < BoardConstants::kBoardSize; figure_idx++) {
            if (currentTurn == Turn::kWhite) {
                move_gen::GenerateMovesForBoardIdxFigureIdx<Turn::kWhite>(
                    board_idx, figure_idx, d_whites, d_blacks, d_kings, local_moves, local_move_counts,
                    local_capture_masks, &local_per_board_flags, n_boards
                );
            } else {
                move_gen::GenerateMovesForBoardIdxFigureIdx<Turn::kBlack>(
                    board_idx, figure_idx, d_whites, d_blacks, d_kings, local_moves, local_move_counts,
                    local_capture_masks, &local_per_board_flags, n_boards
                );
            }
        }

        //--------------- 3) Select a best move ---------------
        move_t chosen_move = MoveConstants::kInvalidMove;

        {
            // 1) detect if there's a capture overall
            bool capture_required = (move_gen::ReadFlag(local_per_board_flags, MoveFlagsConstants::kCaptureFound) != 0);
            // 2) pick a random start index
            u8 seed = d_seeds[board_idx];
            // update seed so each move changes the random selection
            d_seeds[board_idx]         = seed + 7;  // or any increment you like
            board_index_t start_square = (board_index_t)(seed % BoardConstants::kBoardSize);

            // Try up to 32 squares in wrap-around order
            for (int i = 0; i < BoardConstants::kBoardSize; i++) {
                board_index_t sq = (start_square + i) % BoardConstants::kBoardSize;
                u8 count_for_sq  = local_move_counts[sq];
                if (count_for_sq == 0) {
                    continue;
                }
                move_flags_t cmask = local_capture_masks[sq];
                if (capture_required && cmask == 0) {
                    // forced capture => skip squares that have no capture
                    continue;
                }
                // Build a list of candidate sub-moves for capturing or for normal
                u8 valid_indices[16];
                u8 count_valid = 0;
                for (u8 sub = 0; sub < count_for_sq; sub++) {
                    bool is_capture = (cmask & (1 << sub)) != 0;
                    if (capture_required && !is_capture) {
                        // skip non-captures if forced
                        continue;
                    }
                    valid_indices[count_valid++] = sub;
                }
                if (count_valid == 0) {
                    // either no capturing sub-moves in forced-capture scenario, skip
                    continue;
                }
                u8 chosen_sub = (u8)(seed % count_valid);
                // flatten offset to local_moves
                u32 offset  = sq * move_gen::kNumMaxMovesPerPiece + valid_indices[chosen_sub];
                chosen_move = local_moves[offset];
                break;
            }
        }

        //--------------- 4) If no valid move => current side loses ---------------
        if (chosen_move == MoveConstants::kInvalidMove) {
            // the current side has no moves => the other side wins
            outcome = (currentTurn == Turn::kWhite) ? kOutcomeBlack : kOutcomeWhite;
            break;
        }

        //--------------- 5) Apply the chosen move and detect if it was a capture ---------------
        // We can figure out if it was a capture by seeing if "from->to" is in the capture list.
        board_index_t from_sq = move_gen::DecodeMove<move_gen::MovePart::From>(chosen_move);
        board_index_t to_sq   = move_gen::DecodeMove<move_gen::MovePart::To>(chosen_move);

        // Check if it was a capturing move by re-checking local_capture_masks
        // We find the sub-move index in the same manner:
        bool is_capture = false;
        {
            // local approach to see if that sub move was flagged as capture
            const u8 count_for_sq    = local_move_counts[from_sq];
            const move_flags_t cmask = local_capture_masks[from_sq];
            for (u8 sub = 0; sub < count_for_sq; sub++) {
                move_t mv = local_moves[from_sq * move_gen::kNumMaxMovesPerPiece + sub];
                if (mv == chosen_move) {
                    is_capture = ((cmask & (1 << sub)) != 0);
                    break;
                }
            }
        }

        // Actually apply the move on the board
        apply_move::ApplyMoveOnBoardIdx((board_index_t)board_idx, d_whites, d_blacks, d_kings, &chosen_move, n_boards);

        //--------------- 6) Chain Capture Logic ---------------
        // If it was a capture, see if we can keep capturing with the same piece
        if (is_capture) {
            while (true) {
                // Re-generate moves, but only for the single square "to_sq":
                move_t single_moves[move_gen::kNumMaxMovesPerPiece];
                u8 single_count             = 0;
                move_flags_t single_capture = 0;
                move_flags_t single_flags   = 0;

                for (int i = 0; i < move_gen::kNumMaxMovesPerPiece; i++) single_moves[i] = MoveConstants::kInvalidMove;

                // Call the same subfunction on just that one square:
                if (currentTurn == Turn::kWhite) {
                    move_gen::GenerateMovesForBoardIdxFigureIdx<Turn::kWhite>(
                        board_idx, to_sq, d_whites, d_blacks, d_kings, single_moves, &single_count, &single_capture,
                        &single_flags, n_boards
                    );
                } else {
                    move_gen::GenerateMovesForBoardIdxFigureIdx<Turn::kBlack>(
                        board_idx, to_sq, d_whites, d_blacks, d_kings, single_moves, &single_count, &single_capture,
                        &single_flags, n_boards
                    );
                }
                // If no capturing is available in single_capture => break
                if (single_capture == 0) {
                    break;
                }
                // We do forced capture again => choose a random capturing sub-move
                // collect which sub-moves are captures
                u8 capture_indices[16];
                u8 ccount = 0;
                for (u8 sub = 0; sub < single_count; sub++) {
                    if ((single_capture & (1 << sub)) != 0) {
                        capture_indices[ccount++] = sub;
                    }
                }
                if (ccount == 0) {
                    break;  // no actual capturing sub-moves
                }
                // pick random among them
                u8 seed            = d_seeds[board_idx];
                d_seeds[board_idx] = seed + 3;
                u8 chosen_csub     = (u8)(seed % ccount);

                move_t chain_move = single_moves[capture_indices[chosen_csub]];

                // apply it
                apply_move::ApplyMoveOnBoardIdx(
                    (board_index_t)board_idx, d_whites, d_blacks, d_kings, &chain_move, n_boards
                );
                // update to_sq
                to_sq = move_gen::DecodeMove<move_gen::MovePart::To>(chain_move);
            }
        }

        //--------------- 7) Promotion ---------------
        {
            // If the piece ended in `to_sq`, check if it should become a king.

            // todo just kings =| whites & topmask
            //           kings =| blacks & topmask
            board_t endMask = (board_t{1} << to_sq);
            // If it's already a king, do nothing
            bool wasAlreadyKing = ((kings & endMask) != 0);
            if (!wasAlreadyKing) {
                if (currentTurn == Turn::kWhite) {
                    // White => top row => indexes 0..3
                    bool topRow = ((BoardConstants::kTopBoardEdgeMask & endMask) != 0);
                    if (topRow) {
                        kings |= endMask;
                    }
                } else {
                    // Black => bottom row => indexes 28..31
                    bool bottomRow = ((BoardConstants::kBottomBoardEdgeMask & endMask) != 0);
                    if (bottomRow) {
                        kings |= endMask;
                    }
                }
            }
        }

        //--------------- 8) 40-move rule update ---------------
        // By your rule: "If the previous move was a king move that wasn't a capture, then it's non_reversible => +1
        // otherwise reset to 0."

        {
            // Check if the from_sq was a king
            bool fromWasKing = ((kings & (board_t{1} << from_sq)) != 0);

            if (fromWasKing && !is_capture) {
                // non_reversible => increment
                num_consecutive_non_reversible_moves++;
            } else {
                // reversible => reset
                num_consecutive_non_reversible_moves = 0;
            }
            // if we hit 40 => draw
            if (num_consecutive_non_reversible_moves >= 40) {
                outcome = kOutcomeDraw;
                break;
            }
        }

        //--------------- 9) Switch turn for the next iteration ---------------
        currentTurn = (currentTurn == Turn::kWhite) ? Turn::kBlack : Turn::kWhite;
    }

    // If we never assigned outcome, declare a draw by iteration limit
    if (outcome == kOutcomeNone) {
        outcome = kOutcomeDraw;
    }

    // Finally, write the result
    d_scores[board_idx] = outcome;
}

/**
 * \brief Global kernel: runs a full simulation for each board in parallel, storing final results in d_scores.
 *
 * \param d_whites  White bitmasks of size n_boards.
 * \param d_blacks  Black bitmasks of size n_boards.
 * \param d_kings   King bitmasks of size n_boards.
 * \param d_scores  Outcome results (size n_boards).
 * \param n_boards  Total number of boards.
 * \param d_seeds   A random seed per board.
 * \param max_iterations  Maximum half-moves to simulate before declaring a draw.
 */
__global__ void SimulateCheckersGames(
    board_t* d_whites, board_t* d_blacks, board_t* d_kings, u8* d_scores, u64 n_boards, u8* d_seeds, int max_iterations
)
{
    for (u64 board_idx = blockIdx.x * blockDim.x + threadIdx.x; board_idx < n_boards;
         board_idx += gridDim.x * blockDim.x) {
        DeviceSimulateSingleCheckersGame(
            board_idx, d_whites, d_blacks, d_kings, d_scores, n_boards, d_seeds, max_iterations
        );
    }
}

}  // namespace checkers::gpu
