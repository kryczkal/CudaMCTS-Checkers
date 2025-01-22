#include "checkers_defines.hpp"
#include "cuda/apply_move.cuh"
#include "cuda/board_helpers.cuh"
#include "cuda/game_simulation.cuh"
#include "cuda/move_generation.cuh"
#include "cuda/move_selection.cuh"

namespace checkers::gpu
{

// This kernel simulates multiple checkers games. Each block processes kNumBoardsPerBlock boards.
// We store data in shared memory so each block can handle multiple boards in parallel.
// We rely on the "OnSingleBoard" device functions for generating moves, selecting a move, and applying it.
__global__ void SimulateCheckersGamesOneBoardPerBlock(
    const board_t* d_whites,         // [n_simulation_counts]
    const board_t* d_blacks,         // [n_simulation_counts]
    const board_t* d_kings,          // [n_simulation_counts]
    const u8* d_start_turns,         // [n_simulation_counts] (0=White, 1=Black)
    const u64* d_simulation_counts,  // [n_simulation_counts]
    const u64 n_simulation_counts,   // how many distinct board/turn combos
    u8* d_scores,                    // [n_total_simulations] final results
    u8* d_seeds,                     // [n_total_simulations] random seeds
    const int max_iterations,
    const u64 n_total_simulations  // sum of all d_simCounts[i])
)
{
    using namespace checkers::gpu;

    // -------------------------------------------------------------------------
    // Identify which global simulation thread this is
    // -------------------------------------------------------------------------
    const u64 kGlobalSimulationIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (kGlobalSimulationIndex >= n_total_simulations * BoardConstants::kBoardSize) {
        return;
    }

    // -------------------------------------------------------------------------
    // Figure out which "SimulationTypeIndex" (which board config) we belong to
    // -------------------------------------------------------------------------
    u64 kSimulationTypeIndex = 0;
    u64 sum                  = d_simulation_counts[0];
    while ((kGlobalSimulationIndex / BoardConstants::kBoardSize) >= sum &&
           kSimulationTypeIndex < n_simulation_counts - 1) {
        kSimulationTypeIndex++;
        sum += d_simulation_counts[kSimulationTypeIndex];
    }

    if (kSimulationTypeIndex >= n_simulation_counts) {
        return;  // Should not happen if the input data is correct
    }

    // -------------------------------------------------------------------------
    // Each "board" is handled by kNumBoardsPerBlock sub-block of threads.
    // Exactly as before, we group 32 threads (one per square).
    // -------------------------------------------------------------------------
    const int kThreadsPerBoard = BoardConstants::kBoardSize;  // 32

    // localBoardIdx: which board within the block
    const u16 kLocalBoardIndex         = threadIdx.x / kThreadsPerBoard;
    const u16 kLocalThreadInBoardIndex = threadIdx.x % kThreadsPerBoard;

    // If this block is somehow dealing with more boards that it should, we skip threads that exceed that range.
    // This shouldn't be possible with a properly configured grid.
    if (kLocalBoardIndex >= kNumBoardsPerBlock) {
        return;
    }

    const u64 kGlobalBoardIndex = kGlobalSimulationIndex / BoardConstants::kBoardSize;

    // ----------------------------------------------------------------------------
    // Prepare shared memory to hold board states and relevant arrays
    // We index these with [localBoardIdx] so each board in the block has its own data.
    // ----------------------------------------------------------------------------
    __shared__ board_t s_whites[kNumBoardsPerBlock];
    __shared__ board_t s_blacks[kNumBoardsPerBlock];
    __shared__ board_t s_kings[kNumBoardsPerBlock];
    __shared__ u8 s_outcome[kNumBoardsPerBlock];
    __shared__ u8 s_seed[kNumBoardsPerBlock];
    __shared__ bool s_current_turn[kNumBoardsPerBlock];
    __shared__ u8 s_non_reversible[kNumBoardsPerBlock];

    // For move generation:
    // s_moves: up to 32 squares * kNumMaxMovesPerPiece
    // s_moveCounts: 32 counters
    // s_captureMasks: 32 flags
    // s_perBoardFlags: 1 per board
    static constexpr int kNumMaxMovesPerPiece = checkers::gpu::move_gen::kNumMaxMovesPerPiece;
    __shared__ move_t s_moves[kNumBoardsPerBlock][BoardConstants::kBoardSize * kNumMaxMovesPerPiece];
    __shared__ u8 s_move_counts[kNumBoardsPerBlock][BoardConstants::kBoardSize];
    __shared__ move_flags_t s_capture_masks[kNumBoardsPerBlock][BoardConstants::kBoardSize];
    __shared__ move_flags_t s_per_board_flags[kNumBoardsPerBlock];
    __shared__ bool s_has_capture[kNumBoardsPerBlock];
    __shared__ move_t s_chosen_move[kNumBoardsPerBlock];
    __shared__ board_index_t s_chain_from[kNumBoardsPerBlock];

    // ----------------------------------------------------------------------------
    // Initialize shared memory from global memory, but only for the board that
    // localBoardIdx refers to.
    // ----------------------------------------------------------------------------
    if (kLocalThreadInBoardIndex == 0) {
        s_whites[kLocalBoardIndex]         = d_whites[kSimulationTypeIndex];
        s_blacks[kLocalBoardIndex]         = d_blacks[kSimulationTypeIndex];
        s_kings[kLocalBoardIndex]          = d_kings[kSimulationTypeIndex];
        s_outcome[kLocalBoardIndex]        = 0;  // 0 = in progress
        s_seed[kLocalBoardIndex]           = d_seeds[kGlobalBoardIndex];
        s_non_reversible[kLocalBoardIndex] = 0;
        s_current_turn[kLocalBoardIndex]   = (d_start_turns[kSimulationTypeIndex]);
    }
    __syncthreads();

    // ----------------------------------------------------------------------------
    // Define a helper function that returns whether a piece belongs to the current side:
    // currentTurn = 0 => White, 1 => Black.
    // ----------------------------------------------------------------------------
    auto IsCurrentSidePiece = [&](board_t w, board_t b, bool turn_black, board_index_t idx) {
        // turnBlack == false => white's turn, true => black's turn
        if (!turn_black) {
            // white turn
            return ((w >> idx) & 1ULL) != 0ULL;
        } else {
            // black turn
            return ((b >> idx) & 1ULL) != 0ULL;
        }
    };

    // ----------------------------------------------------------------------------
    // The main half-move simulation loop.
    // ----------------------------------------------------------------------------
    for (u32 moveCount = 0; moveCount < max_iterations; moveCount++) {
        if (kLocalThreadInBoardIndex == 0) {
            board_t w = s_whites[kLocalBoardIndex];
            board_t b = s_blacks[kLocalBoardIndex];
            if (b == 0ULL) {
                s_outcome[kLocalBoardIndex] = 1;  // White wins
            } else if (w == 0ULL) {
                s_outcome[kLocalBoardIndex] = 2;  // Black wins
            }
        }
        __syncthreads();

        if (s_outcome[kLocalBoardIndex] != 0) {
            break;  // Board ended
        }

        // Zero the arrays used for move generation
        {
            board_index_t sq                      = kLocalThreadInBoardIndex;
            s_move_counts[kLocalBoardIndex][sq]   = 0;
            s_capture_masks[kLocalBoardIndex][sq] = 0;
        }

        if (kLocalThreadInBoardIndex == 0) {
            s_per_board_flags[kLocalBoardIndex] = 0;
        }
        __syncthreads();

        // Generate moves for the piece corresponding to localThreadInBoard,
        {
            const board_t w      = s_whites[kLocalBoardIndex];
            const board_t b      = s_blacks[kLocalBoardIndex];
            const board_t k      = s_kings[kLocalBoardIndex];
            const bool turnBlack = s_current_turn[kLocalBoardIndex];

            if (IsCurrentSidePiece(w, b, turnBlack, kLocalThreadInBoardIndex)) {
                using checkers::gpu::move_gen::GenerateMovesForSinglePiece;
                if (!turnBlack) {
                    // White
                    GenerateMovesForSinglePiece<Turn::kWhite>(
                        (board_index_t)kLocalThreadInBoardIndex, w, b, k,
                        &s_moves[kLocalBoardIndex][kLocalThreadInBoardIndex * kNumMaxMovesPerPiece],
                        s_move_counts[kLocalBoardIndex][kLocalThreadInBoardIndex],
                        s_capture_masks[kLocalBoardIndex][kLocalThreadInBoardIndex], s_per_board_flags[kLocalBoardIndex]
                    );
                } else {
                    // Black
                    GenerateMovesForSinglePiece<Turn::kBlack>(
                        (board_index_t)kLocalThreadInBoardIndex, w, b, k,
                        &s_moves[kLocalBoardIndex][kLocalThreadInBoardIndex * kNumMaxMovesPerPiece],
                        s_move_counts[kLocalBoardIndex][kLocalThreadInBoardIndex],
                        s_capture_masks[kLocalBoardIndex][kLocalThreadInBoardIndex], s_per_board_flags[kLocalBoardIndex]
                    );
                }
            }
        }
        __syncthreads();

        // Single thread picks the move
        if (kLocalThreadInBoardIndex == 0) {
            // Flattened arrays for the board
            move_t* boardMoves          = &s_moves[kLocalBoardIndex][0];
            u8* boardMoveCounts         = s_move_counts[kLocalBoardIndex];
            move_flags_t* boardCaptures = s_capture_masks[kLocalBoardIndex];
            move_flags_t flags          = s_per_board_flags[kLocalBoardIndex];

            // The side-to-move bitmasks
            board_t w = s_whites[kLocalBoardIndex];
            board_t b = s_blacks[kLocalBoardIndex];
            board_t k = s_kings[kLocalThreadInBoardIndex];

            u8& localSeed = s_seed[kLocalBoardIndex];

            // pick best move
            using checkers::gpu::move_selection::SelectBestMoveForSingleBoard;
            move_t chosen =
                SelectBestMoveForSingleBoard(w, b, k, boardMoves, boardMoveCounts, boardCaptures, flags, localSeed);

            s_seed[kLocalBoardIndex] = (u8)(localSeed + 13);
            if (chosen == checkers::gpu::move_gen::MoveConstants::kInvalidMove) {
                // Current side cannot move => other side wins
                s_outcome[kLocalBoardIndex] = !s_current_turn[kLocalBoardIndex] ? 2 : 1;
            }
            s_chosen_move[kLocalBoardIndex] = chosen;
        }
        __syncthreads();

        if (s_outcome[kLocalBoardIndex] != 0) {
            break;
        }

        // Apply the chosen move
        if (kLocalThreadInBoardIndex == 0) {
            move_t mv = s_chosen_move[kLocalBoardIndex];
            checkers::gpu::apply_move::ApplyMoveOnSingleBoard(
                mv, s_whites[kLocalBoardIndex], s_blacks[kLocalBoardIndex], s_kings[kLocalBoardIndex]
            );
            // Save the from-square for chaining
            s_chain_from[kLocalBoardIndex] =
                checkers::gpu::move_gen::DecodeMove<checkers::gpu::move_gen::MovePart::To>(mv);
        }
        __syncthreads();

        // Chain capturing. We keep capturing if a capture was done and there's a possibility of continuing.
        // The "to" index of the chosen move might still be able to capture more.
        while (true) {
            // Clear arrays again for single-piece generation
            {
                int sq                                = kLocalThreadInBoardIndex;
                s_move_counts[kLocalBoardIndex][sq]   = 0;
                s_capture_masks[kLocalBoardIndex][sq] = 0;
                s_has_capture[kLocalBoardIndex]       = false;
            }
            if (kLocalThreadInBoardIndex == 0) {
                s_per_board_flags[kLocalBoardIndex] = 0;
            }
            __syncthreads();

            // Only the piece at s_chainFrom can keep capturing
            board_t w      = s_whites[kLocalBoardIndex];
            board_t b      = s_blacks[kLocalBoardIndex];
            board_t k      = s_kings[kLocalBoardIndex];
            bool turnBlack = s_current_turn[kLocalBoardIndex];

            if ((kLocalThreadInBoardIndex == s_chain_from[kLocalBoardIndex]) &&
                IsCurrentSidePiece(w, b, turnBlack, kLocalThreadInBoardIndex)) {
                if (!turnBlack) {
                    checkers::gpu::move_gen::GenerateMovesForSinglePiece<Turn::kWhite>(
                        (board_index_t)kLocalThreadInBoardIndex, w, b, k,
                        &s_moves[kLocalBoardIndex][kLocalThreadInBoardIndex * kNumMaxMovesPerPiece],
                        s_move_counts[kLocalBoardIndex][kLocalThreadInBoardIndex],
                        s_capture_masks[kLocalBoardIndex][kLocalThreadInBoardIndex], s_per_board_flags[kLocalBoardIndex]
                    );
                } else {
                    checkers::gpu::move_gen::GenerateMovesForSinglePiece<Turn::kBlack>(
                        (board_index_t)kLocalThreadInBoardIndex, w, b, k,
                        &s_moves[kLocalBoardIndex][kLocalThreadInBoardIndex * kNumMaxMovesPerPiece],
                        s_move_counts[kLocalBoardIndex][kLocalThreadInBoardIndex],
                        s_capture_masks[kLocalBoardIndex][kLocalThreadInBoardIndex], s_per_board_flags[kLocalBoardIndex]
                    );
                }
            }
            __syncthreads();

            // If captures exist, set the flag
            if (checkers::gpu::ReadFlag(
                    s_per_board_flags[kLocalBoardIndex], checkers::gpu::move_gen::MoveFlagsConstants::kCaptureFound
                )) {
                s_has_capture[kLocalBoardIndex] = true;
            }
            __syncthreads();

            if (!s_has_capture[kLocalBoardIndex]) {
                break;
            }

            // pick chain capture
            if (kLocalThreadInBoardIndex == 0) {
                move_t* board_moves          = &s_moves[kLocalBoardIndex][0];
                u8* board_move_counts        = s_move_counts[kLocalBoardIndex];
                move_flags_t* board_captures = s_capture_masks[kLocalBoardIndex];
                move_flags_t flags           = s_per_board_flags[kLocalBoardIndex];
                u8& local_seed               = s_seed[kLocalBoardIndex];

                move_t chainMv = checkers::gpu::move_selection::SelectBestMoveForSingleBoard(
                    w, b, k, board_moves, board_move_counts, board_captures, flags, local_seed
                );
                s_seed[kLocalBoardIndex] = (u8)(local_seed + 7);

                if (chainMv == checkers::gpu::move_gen::MoveConstants::kInvalidMove) {
                    s_chosen_move[kLocalBoardIndex] = chainMv;
                } else {
                    checkers::gpu::apply_move::ApplyMoveOnSingleBoard(
                        chainMv, s_whites[kLocalBoardIndex], s_blacks[kLocalBoardIndex], s_kings[kLocalBoardIndex]
                    );
                    s_chain_from[kLocalBoardIndex] =
                        checkers::gpu::move_gen::DecodeMove<checkers::gpu::move_gen::MovePart::To>(chainMv);
                }
            }
            __syncthreads();

            if (s_chosen_move[kLocalBoardIndex] == checkers::gpu::move_gen::MoveConstants::kInvalidMove) {
                break;
            }
        }

        // Promotion
        if (kLocalThreadInBoardIndex == 0) {
            s_kings[kLocalBoardIndex] |= (s_whites[kLocalBoardIndex] & BoardConstants::kTopBoardEdgeMask);
            s_kings[kLocalBoardIndex] |= (s_blacks[kLocalBoardIndex] & BoardConstants::kBottomBoardEdgeMask);
        }
        __syncthreads();

        // 6.h) 40-move rule or non-reversible logic
        if (kLocalThreadInBoardIndex == 0) {
            const bool was_capture = checkers::gpu::ReadFlag(
                s_per_board_flags[kLocalBoardIndex], checkers::gpu::move_gen::MoveFlagsConstants::kCaptureFound
            );

            board_index_t from_sq = checkers::gpu::move_gen::DecodeMove<checkers::gpu::move_gen::MovePart::From>(
                s_chosen_move[kLocalBoardIndex]
            );
            const bool from_was_king = ReadFlag(s_kings[kLocalBoardIndex], from_sq);

            if (!was_capture && from_was_king) {
                s_non_reversible[kLocalBoardIndex]++;
            } else {
                s_non_reversible[kLocalBoardIndex] = 0;
            }
            if (s_non_reversible[kLocalBoardIndex] >= 40) {
                s_outcome[kLocalBoardIndex] = 3;  // draw
            }
        }
        __syncthreads();

        if (s_outcome[kLocalBoardIndex] != 0) {
            break;
        }

        // Switch turn
        if (kLocalThreadInBoardIndex == 0) {
            s_current_turn[kLocalBoardIndex] = !s_current_turn[kLocalBoardIndex];
        }
        __syncthreads();
    }

    // If still 0, declare a draw
    if (kLocalThreadInBoardIndex == 0 && s_outcome[kLocalBoardIndex] == 0) {
        s_outcome[kLocalBoardIndex] = 3;  // draw
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // 7) Write final results from the perspective of the starting turn
    //    2=win, 1=draw, 0=lose
    // -------------------------------------------------------------------------
    if (kLocalThreadInBoardIndex == 0) {
        u8 final    = s_outcome[kLocalBoardIndex];          // 1=White,2=Black,3=Draw
        u8 st       = d_start_turns[kSimulationTypeIndex];  // 0=White started, 1=Black started
        u8 storeVal = 0;                                    // default: lose

        // If draw
        if (final == 3) {
            storeVal = 1;
        }
        // If final is the same side as st => it's a "win" for the starter
        else if (final == (st + 1)) {
            storeVal = 2;
        }
        // else remains 0 (lose)

        d_scores[kGlobalBoardIndex] = storeVal;
    }
}
}  // namespace checkers::gpu
