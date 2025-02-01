#include "common/checkers_defines.hpp"
#include "cuda/apply_move.cuh"
#include "cuda/board_helpers.cuh"
#include "cuda/game_simulation.cuh"
#include "cuda/move_generation.cuh"
#include "cuda/move_selection.cuh"

namespace checkers::gpu
{

__global__ void SimulateCheckersGames(
    const board_t* d_whites,         // [n_simulation_counts]
    const board_t* d_blacks,         // [n_simulation_counts]
    const board_t* d_kings,          // [n_simulation_counts]
    const u8* d_start_turns,         // [n_simulation_counts] (0=White, 1=Black)
    const u64* d_simulation_counts,  // [n_simulation_counts]
    const u64 n_simulation_counts,   // how many distinct board/turn combos
    u8* d_scores,                    // [n_total_simulations] final results
    u8* d_seeds,                     // [n_total_simulations] random seeds
    const int max_iterations,
    const u64 n_total_simulations  // sum of all d_simulation_counts[i]
)
{
    using namespace checkers::gpu;

    // -------------------------------------------------------------------------
    // Identify which global simulation thread this is.
    // Each board is processed by kThreadsPerBoardInSimulation threads.
    // -------------------------------------------------------------------------
    const u64 kGlobalSimulationIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (kGlobalSimulationIndex >= n_total_simulations * kThreadsPerBoardInSimulation) {
        return;
    }

    // -------------------------------------------------------------------------
    // Compute global board index.
    // -------------------------------------------------------------------------
    const u64 kGlobalBoardIndex = kGlobalSimulationIndex / kThreadsPerBoardInSimulation;

    // -------------------------------------------------------------------------
    // Determine simulation type index.
    // -------------------------------------------------------------------------
    u64 kSimulationTypeIndex = 0;
    u64 sum                  = d_simulation_counts[0];
    while (kGlobalBoardIndex >= sum && kSimulationTypeIndex < n_simulation_counts - 1) {
        kSimulationTypeIndex++;
        sum += d_simulation_counts[kSimulationTypeIndex];
    }
    assert(kSimulationTypeIndex < n_simulation_counts);

    // -------------------------------------------------------------------------
    // Each board is handled by a sub-block of threads defined by kThreadsPerBoardInSimulation.
    // -------------------------------------------------------------------------
    const u16 kLocalBoardIndex         = threadIdx.x / kThreadsPerBoardInSimulation;
    const u16 kLocalThreadInBoardIndex = threadIdx.x % kThreadsPerBoardInSimulation;

    // Assert that the block is not dealing with more boards than it should.
    assert(kLocalBoardIndex < kNumBoardsPerBlock);

    // ----------------------------------------------------------------------------
    // Prepare shared memory to hold board states and relevant arrays.
    // We index these with [localBoardIdx] so each board in the block has its own data.
    // ----------------------------------------------------------------------------
    __shared__ board_t s_whites[kNumBoardsPerBlock];
    __shared__ board_t s_blacks[kNumBoardsPerBlock];
    __shared__ board_t s_kings[kNumBoardsPerBlock];
    __shared__ u8 s_outcome[kNumBoardsPerBlock];
    __shared__ u32 s_seed[kNumBoardsPerBlock];
    __shared__ bool s_current_turn[kNumBoardsPerBlock];
    __shared__ u8 s_non_reversible[kNumBoardsPerBlock];

    // For move generation:
    // s_moves: up to 32 squares * kNumMaxMovesPerPiece per board
    // s_move_counts: 32 counters per board
    // s_capture_masks: 32 flags per board
    // s_per_board_flags: 1 per board
    __shared__ move_t s_moves[kNumBoardsPerBlock][BoardConstants::kBoardSize * kNumMaxMovesPerPiece];
    __shared__ u8 s_move_counts[kNumBoardsPerBlock][BoardConstants::kBoardSize];
    __shared__ move_flags_t s_capture_masks[kNumBoardsPerBlock][BoardConstants::kBoardSize];
    __shared__ move_flags_t s_per_board_flags[kNumBoardsPerBlock];
    __shared__ bool s_has_capture[kNumBoardsPerBlock];
    __shared__ move_t s_chosen_move[kNumBoardsPerBlock];
    __shared__ board_index_t s_chain_from[kNumBoardsPerBlock];

    // ----------------------------------------------------------------------------
    // Initialize shared memory from global memory for this board.
    // ----------------------------------------------------------------------------
    if (kLocalThreadInBoardIndex == 0) {
        s_whites[kLocalBoardIndex]         = d_whites[kSimulationTypeIndex];
        s_blacks[kLocalBoardIndex]         = d_blacks[kSimulationTypeIndex];
        s_kings[kLocalBoardIndex]          = d_kings[kSimulationTypeIndex];
        s_outcome[kLocalBoardIndex]        = kOutcomeInProgress;
        s_seed[kLocalBoardIndex]           = d_seeds[kGlobalBoardIndex];
        s_non_reversible[kLocalBoardIndex] = 0;
        s_current_turn[kLocalBoardIndex]   = d_start_turns[kSimulationTypeIndex];
    }
    __syncthreads();

    // ----------------------------------------------------------------------------
    // Main simulation loop.
    // ----------------------------------------------------------------------------
    for (u32 moveCount = 0; moveCount < max_iterations; moveCount++) {
        if (kLocalThreadInBoardIndex == 0) {
            board_t w = s_whites[kLocalBoardIndex];
            board_t b = s_blacks[kLocalBoardIndex];
            if (b == 0ULL) {
                s_outcome[kLocalBoardIndex] = kOutcomeWhite;  // White wins
            } else if (w == 0ULL) {
                s_outcome[kLocalBoardIndex] = kOutcomeBlack;  // Black wins
            }
        }
        __syncthreads();

        if (s_outcome[kLocalBoardIndex] != kOutcomeInProgress) {
            break;  // Game ended
        }

        // Zero the arrays used for move generation.
        for (board_index_t sq = kLocalThreadInBoardIndex; sq < BoardConstants::kBoardSize;
             sq += kThreadsPerBoardInSimulation) {
            s_move_counts[kLocalBoardIndex][sq]   = 0;
            s_capture_masks[kLocalBoardIndex][sq] = 0;
        }
        if (kLocalThreadInBoardIndex == 0) {
            s_per_board_flags[kLocalBoardIndex] = 0;
        }
        __syncthreads();

        // Generate moves for each square assigned to this thread.
        {
            const board_t w       = s_whites[kLocalBoardIndex];
            const board_t b       = s_blacks[kLocalBoardIndex];
            const board_t k       = s_kings[kLocalBoardIndex];
            const bool kTurnBlack = s_current_turn[kLocalBoardIndex];

            using move_gen::GenerateMovesForSinglePiece;
            for (board_index_t sq = kLocalThreadInBoardIndex; sq < BoardConstants::kBoardSize;
                 sq += kThreadsPerBoardInSimulation) {
                if (!kTurnBlack) {
                    GenerateMovesForSinglePiece<Turn::kWhite>(
                        sq, w, b, k, &s_moves[kLocalBoardIndex][sq * kNumMaxMovesPerPiece],
                        s_move_counts[kLocalBoardIndex][sq], s_capture_masks[kLocalBoardIndex][sq],
                        s_per_board_flags[kLocalBoardIndex]
                    );
                } else {
                    GenerateMovesForSinglePiece<Turn::kBlack>(
                        sq, w, b, k, &s_moves[kLocalBoardIndex][sq * kNumMaxMovesPerPiece],
                        s_move_counts[kLocalBoardIndex][sq], s_capture_masks[kLocalBoardIndex][sq],
                        s_per_board_flags[kLocalBoardIndex]
                    );
                }
            }
        }
        __syncthreads();
        if (kLocalThreadInBoardIndex == 0) {
            if (checkers::gpu::ReadFlag(s_per_board_flags[kLocalBoardIndex], MoveFlagsConstants::kCaptureFound)) {
                s_has_capture[kLocalBoardIndex] = true;
            }
        }
        __syncthreads();

        // Single thread picks the move.
        if (kLocalThreadInBoardIndex == 0) {
            // Flattened arrays for the board.
            move_t* board_moves          = &s_moves[kLocalBoardIndex][0];
            u8* board_move_counts        = s_move_counts[kLocalBoardIndex];
            move_flags_t* board_captures = s_capture_masks[kLocalBoardIndex];
            move_flags_t flags           = s_per_board_flags[kLocalBoardIndex];

            // The side-to-move bitmasks.
            board_t w      = s_whites[kLocalBoardIndex];
            board_t b      = s_blacks[kLocalBoardIndex];
            board_t k      = s_kings[kLocalBoardIndex];
            u32& localSeed = s_seed[kLocalBoardIndex];

            // Pick best move.
            move_t chosen = move_selection::SelectBestMoveForSingleBoard(
                w, b, k, board_moves, board_move_counts, board_captures, flags, localSeed
            );

            if (chosen == kInvalidMove) {
                // Current side cannot move => other side wins.
                s_outcome[kLocalBoardIndex] = !s_current_turn[kLocalBoardIndex] ? kOutcomeBlack : kOutcomeWhite;
            }
            s_chosen_move[kLocalBoardIndex] = chosen;
        }
        __syncthreads();

        if (s_outcome[kLocalBoardIndex] != kOutcomeInProgress) {
            break;
        }

        // Apply the chosen move.
        if (kLocalThreadInBoardIndex == 0) {
            move_t mv = s_chosen_move[kLocalBoardIndex];
            apply_move::ApplyMoveOnSingleBoard(
                mv, s_whites[kLocalBoardIndex], s_blacks[kLocalBoardIndex], s_kings[kLocalBoardIndex]
            );
            // Save the destination square for chaining.
            s_chain_from[kLocalBoardIndex] = move_gen::DecodeMove<move_gen::MovePart::To>(mv);
        }
        __syncthreads();

        // Chain capturing.
        while (s_has_capture[kLocalBoardIndex]) {
            // Clear arrays again for move generation.
            for (board_index_t sq = kLocalThreadInBoardIndex; sq < BoardConstants::kBoardSize;
                 sq += kThreadsPerBoardInSimulation) {
                s_move_counts[kLocalBoardIndex][sq]   = 0;
                s_capture_masks[kLocalBoardIndex][sq] = 0;
            }
            if (kLocalThreadInBoardIndex == 0) {
                s_has_capture[kLocalBoardIndex]     = false;
                s_per_board_flags[kLocalBoardIndex] = 0;
            }
            __syncthreads();

            // Only the piece at s_chain_from can continue capturing.
            const board_t w = s_whites[kLocalBoardIndex];
            const board_t b = s_blacks[kLocalBoardIndex];
            const board_t k = s_kings[kLocalBoardIndex];
            bool turn_black = s_current_turn[kLocalBoardIndex];

            for (board_index_t sq = kLocalThreadInBoardIndex; sq < BoardConstants::kBoardSize;
                 sq += kThreadsPerBoardInSimulation) {
                if (sq == s_chain_from[kLocalBoardIndex]) {
                    if (!turn_black) {
                        move_gen::GenerateMovesForSinglePiece<Turn::kWhite>(
                            sq, w, b, k, &s_moves[kLocalBoardIndex][sq * kNumMaxMovesPerPiece],
                            s_move_counts[kLocalBoardIndex][sq], s_capture_masks[kLocalBoardIndex][sq],
                            s_per_board_flags[kLocalBoardIndex]
                        );
                    } else {
                        move_gen::GenerateMovesForSinglePiece<Turn::kBlack>(
                            sq, w, b, k, &s_moves[kLocalBoardIndex][sq * kNumMaxMovesPerPiece],
                            s_move_counts[kLocalBoardIndex][sq], s_capture_masks[kLocalBoardIndex][sq],
                            s_per_board_flags[kLocalBoardIndex]
                        );
                    }
                }
            }
            __syncthreads();

            if (kLocalThreadInBoardIndex == 0) {
                if (checkers::gpu::ReadFlag(s_per_board_flags[kLocalBoardIndex], MoveFlagsConstants::kCaptureFound))
                    s_has_capture[kLocalBoardIndex] = true;
            }
            __syncthreads();

            if (!s_has_capture[kLocalBoardIndex]) {
                break;
            }

            // Single thread picks the chain capture move.
            if (kLocalThreadInBoardIndex == 0) {
                move_t* board_moves          = &s_moves[kLocalBoardIndex][0];
                u8* board_move_counts        = s_move_counts[kLocalBoardIndex];
                move_flags_t* board_captures = s_capture_masks[kLocalBoardIndex];
                move_flags_t flags           = s_per_board_flags[kLocalBoardIndex];
                u32& local_seed              = s_seed[kLocalBoardIndex];

                move_t chain_mv = move_selection::SelectBestMoveForSingleBoard(
                    w, b, k, board_moves, board_move_counts, board_captures, flags, local_seed
                );

                if (chain_mv == kInvalidMove) {
                    s_chosen_move[kLocalBoardIndex] = chain_mv;
                } else {
                    apply_move::ApplyMoveOnSingleBoard(
                        chain_mv, s_whites[kLocalBoardIndex], s_blacks[kLocalBoardIndex], s_kings[kLocalBoardIndex]
                    );
                    s_chain_from[kLocalBoardIndex] = move_gen::DecodeMove<move_gen::MovePart::To>(chain_mv);
                }
            }
            __syncthreads();

            if (s_chosen_move[kLocalBoardIndex] == kInvalidMove) {
                break;
            }
        }

        // Promotion.
        if (kLocalThreadInBoardIndex == 0) {
            s_kings[kLocalBoardIndex] |= (s_whites[kLocalBoardIndex] & BoardConstants::kTopBoardEdgeMask);
            s_kings[kLocalBoardIndex] |= (s_blacks[kLocalBoardIndex] & BoardConstants::kBottomBoardEdgeMask);
        }
        __syncthreads();

        // 40-move rule or non-reversible logic.
        if (kLocalThreadInBoardIndex == 0) {
            const bool was_capture =
                checkers::gpu::ReadFlag(s_per_board_flags[kLocalBoardIndex], MoveFlagsConstants::kCaptureFound);

            board_index_t from_sq    = move_gen::DecodeMove<move_gen::MovePart::From>(s_chosen_move[kLocalBoardIndex]);
            const bool from_was_king = ReadFlag(s_kings[kLocalBoardIndex], from_sq);

            if (!was_capture && from_was_king) {
                s_non_reversible[kLocalBoardIndex]++;
            } else {
                s_non_reversible[kLocalBoardIndex] = 0;
            }
            if (s_non_reversible[kLocalBoardIndex] >= 40) {
                s_outcome[kLocalBoardIndex] = kOutcomeDraw;  // draw
            }
        }
        __syncthreads();

        if (s_outcome[kLocalBoardIndex] != kOutcomeInProgress) {
            break;
        }

        // Switch turn.
        if (kLocalThreadInBoardIndex == 0) {
            s_current_turn[kLocalBoardIndex] = !s_current_turn[kLocalBoardIndex];
        }
        __syncthreads();
    }

    // If still 0, declare a draw.
    if (kLocalThreadInBoardIndex == 0 && s_outcome[kLocalBoardIndex] == kOutcomeInProgress) {
        s_outcome[kLocalBoardIndex] = kOutcomeDraw;  // draw
    }
    __syncthreads();

    // Write final results from the perspective of the starting turn.
    if (kLocalThreadInBoardIndex == 0) {
        u8 final     = s_outcome[kLocalBoardIndex];          // 1=White,2=Black,3=Draw
        u8 st        = d_start_turns[kSimulationTypeIndex];  // 0=White started, 1=Black started
        u8 store_val = 0;                                    // default: lose

        // If draw.
        if (final == kOutcomeDraw) {
            store_val = 1;
        }
        // If final is the same side as st => it's a "win" for the starter.
        else if (final == (st + 1)) {
            store_val = 2;
        }
        // else remains 0 (lose).

        d_scores[kGlobalBoardIndex] = store_val;
    }
}
}  // namespace checkers::gpu
