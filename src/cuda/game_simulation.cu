#include "checkers_defines.hpp"
#include "cuda/apply_move.cuh"
#include "cuda/board_helpers.cuh"
#include "cuda/game_simulation.cuh"
#include "cuda/move_generation.cuh"
#include "cuda/move_selection.cuh"

namespace checkers::gpu
{

__device__ void PrintCheckpoint(const u16 local_thread_in_board, const u64 checkpoint)
{
    //        if(local_thread_in_board == 0) {
    //            printf("[%d] Checkpoint %lu\n", blockIdx.x, checkpoint);
    //        }
}

// This kernel simulates multiple checkers games. Each block processes kNumBoardsPerBlock boards.
// We store data in shared memory so each block can handle multiple boards in parallel.
// We rely on the "OnSingleBoard" device functions for generating moves, selecting a move, and applying it.
__global__ void SimulateCheckersGamesOneBoardPerBlock(
    const board_t* d_whites, const board_t* d_blacks, const board_t* d_kings, u8* d_scores, const u8* d_seeds,
    const int max_iterations, const u64 n_boards
)
{
    // Each board uses 32 threads. So total threads in this block is (kNumBoardsPerBlock * 32).
    // Identify the localBoardIdx (which board within the block) and localThreadInBoard (0..31).
    //    const int globalThreadIdx      = blockDim.x * blockIdx.x + threadIdx.x;
    const int threadsPerBoard      = BoardConstants::kBoardSize;
    const int totalBoardsThisBlock = kNumBoardsPerBlock;

    // The localBoardIdx is which board within the block [0..kNumBoardsPerBlock-1].
    u16 localBoardIdx      = threadIdx.x / threadsPerBoard;
    u16 localThreadInBoard = threadIdx.x % threadsPerBoard;

    // The global board index is blockIdx.x * kNumBoardsPerBlock + localBoardIdx.
    // Some threads might exceed the range if kNumBoardsPerBlock * 32 > blockDim.x, so we handle that gracefully.
    u64 globalBoardIndex = blockIdx.x * kNumBoardsPerBlock + localBoardIdx;

    // If this thread is beyond the intended local boards, or if the global board index is >= n_boards,
    // there's no valid board to process. We'll skip the entire logic for that thread.
    if (localBoardIdx >= totalBoardsThisBlock || globalBoardIndex >= static_cast<int>(n_boards)) {
        return;
    }
    PrintCheckpoint(localThreadInBoard, 1);

    // ----------------------------------------------------------------------------
    // Prepare shared memory to hold board states and relevant arrays
    // We index these with [localBoardIdx] so each board in the block has its own data.
    // ----------------------------------------------------------------------------
    __shared__ board_t s_whites[kNumBoardsPerBlock];
    __shared__ board_t s_blacks[kNumBoardsPerBlock];
    __shared__ board_t s_kings[kNumBoardsPerBlock];
    __shared__ u8 s_outcome[kNumBoardsPerBlock];
    __shared__ u8 s_seed[kNumBoardsPerBlock];
    // We treat turn as 0=white, 1=black (or use an enum). Here we’ll store bool
    __shared__ bool s_currentTurn[kNumBoardsPerBlock];
    __shared__ u8 s_nonReversible[kNumBoardsPerBlock];
    PrintCheckpoint(localThreadInBoard, 2);

    // For move generation:
    // s_moves: up to 32 squares * kNumMaxMovesPerPiece
    // s_moveCounts: 32 counters
    // s_captureMasks: 32 flags
    // s_perBoardFlags: 1 per board
    static constexpr int kNumMaxMovesPerPiece = checkers::gpu::move_gen::kNumMaxMovesPerPiece;
    __shared__ move_t s_moves[kNumBoardsPerBlock][BoardConstants::kBoardSize * kNumMaxMovesPerPiece];
    __shared__ u8 s_moveCounts[kNumBoardsPerBlock][BoardConstants::kBoardSize];
    __shared__ move_flags_t s_captureMasks[kNumBoardsPerBlock][BoardConstants::kBoardSize];
    __shared__ move_flags_t s_perBoardFlags[kNumBoardsPerBlock];
    __shared__ bool s_hasCapture[kNumBoardsPerBlock];
    PrintCheckpoint(localThreadInBoard, 3);

    // ----------------------------------------------------------------------------
    // Initialize shared memory from global memory, but only for the board that
    // localBoardIdx refers to. We do that with a single warp (or single thread) approach.
    // ----------------------------------------------------------------------------
    if (localThreadInBoard == 0) {
        s_whites[localBoardIdx]        = d_whites[globalBoardIndex];
        s_blacks[localBoardIdx]        = d_blacks[globalBoardIndex];
        s_kings[localBoardIdx]         = d_kings[globalBoardIndex];
        s_outcome[localBoardIdx]       = 0;  // 0 = in progress
        s_seed[localBoardIdx]          = d_seeds[globalBoardIndex];
        s_currentTurn[localBoardIdx]   = 0;  // 0 means White’s turn
        s_nonReversible[localBoardIdx] = 0;
    }
    __syncthreads();
    PrintCheckpoint(localThreadInBoard, 4);

    // ----------------------------------------------------------------------------
    // We'll define a helper function that returns whether a piece belongs to the current side:
    // currentTurn = 0 => White, 1 => Black.
    // ----------------------------------------------------------------------------
    auto IsCurrentSidePiece = [&](board_t whiteBits, board_t blackBits, int turn, int idx) {
        if (turn == 0) {  // white
            return ((whiteBits >> idx) & 1ULL) != 0ULL;
        } else {
            return ((blackBits >> idx) & 1ULL) != 0ULL;
        }
    };
    PrintCheckpoint(localThreadInBoard, 5);

    // ----------------------------------------------------------------------------
    // The main half-move simulation loop.
    // ----------------------------------------------------------------------------
    for (int moveCount = 0; moveCount < max_iterations; moveCount++) {
        // 1) Check if the board is already ended (someone has no pieces).
        // Each thread sees shared memory, but we do final check with threadInBoard == 0.
        if (localThreadInBoard == 0) {
            board_t w = s_whites[localBoardIdx];
            board_t b = s_blacks[localBoardIdx];
            if (b == 0ULL) {
                s_outcome[localBoardIdx] = 1;  // White wins
            } else if (w == 0ULL) {
                s_outcome[localBoardIdx] = 2;  // Black wins
            }
        }
        PrintCheckpoint(localThreadInBoard, 6);
        __syncthreads();

        // If outcome != 0, we break out of the loop for that board
        if (s_outcome[localBoardIdx] != 0) {
            break;
        }

        // 2) Zero the arrays used for move generation
        // We'll do that for each square [0..31], so each thread with localThreadInBoard in [0..31] can do it
        // or we can do a condition like "if (localThreadInBoard < 32) { ... }".
        {
            int sq                            = localThreadInBoard;
            s_moveCounts[localBoardIdx][sq]   = 0;
            s_captureMasks[localBoardIdx][sq] = 0;
        }
        if (localThreadInBoard == 0) {
            s_perBoardFlags[localBoardIdx] = 0;
        }
        PrintCheckpoint(localThreadInBoard, 7);
        __syncthreads();

        // 3) Generate moves for the piece corresponding to localThreadInBoard,
        // if that square belongs to the current side. We call your tested single-board function(s).
        {
            board_t w = s_whites[localBoardIdx];
            board_t b = s_blacks[localBoardIdx];
            board_t k = s_kings[localBoardIdx];
            int t     = s_currentTurn[localBoardIdx];

            bool isMine = IsCurrentSidePiece(w, b, t, localThreadInBoard);
            if (isMine) {
                // We generate moves for this one piece index: localThreadInBoard
                // We'll store them in s_moves for that board.
                // We have a function in move_generation that can generate moves for a single piece (OnSingleBoard).
                // For example, if (t==0) => Turn::kWhite, else Turn::kBlack, etc.

                using checkers::gpu::move_gen::GenerateMovesForSinglePiece;
                // We'll do some turn-based templating. Let's create a tiny helper:
                if (t == 0) {
                    // White
                    GenerateMovesForSinglePiece<Turn::kWhite>(
                        static_cast<board_index_t>(localThreadInBoard), w, b, k,
                        &s_moves[localBoardIdx][localThreadInBoard * kNumMaxMovesPerPiece],
                        s_moveCounts[localBoardIdx][localThreadInBoard],
                        s_captureMasks[localBoardIdx][localThreadInBoard], s_perBoardFlags[localBoardIdx]
                    );
                } else {
                    // Black
                    GenerateMovesForSinglePiece<Turn::kBlack>(
                        static_cast<board_index_t>(localThreadInBoard), w, b, k,
                        &s_moves[localBoardIdx][localThreadInBoard * kNumMaxMovesPerPiece],
                        s_moveCounts[localBoardIdx][localThreadInBoard],
                        s_captureMasks[localBoardIdx][localThreadInBoard], s_perBoardFlags[localBoardIdx]
                    );
                }
            }
            PrintCheckpoint(localThreadInBoard, 8);
        }
        __syncthreads();

        // 4) A single thread (localThreadInBoard == 0) selects the move from the array s_moves
        //    for this board.
        __shared__ move_t s_chosenMove[kNumBoardsPerBlock];
        if (localThreadInBoard == 0) {
            // We'll pick either a random or best move. We'll use your tested function from move_selection.
            // The function needs to see the flattened arrays: s_moves[boardIdx], s_moveCounts[boardIdx], etc.
            // We'll gather them into pointers referencing the correct location in shared memory.
            move_t* boardMoves          = &s_moves[localBoardIdx][0];
            u8* boardMoveCounts         = s_moveCounts[localBoardIdx];
            move_flags_t* boardCaptures = s_captureMasks[localBoardIdx];
            move_flags_t boardFlags     = s_perBoardFlags[localBoardIdx];
            u8 localSeed                = s_seed[localBoardIdx];

            using checkers::gpu::move_selection::SelectBestMoveForSingleBoard;
            board_t w = s_whites[localBoardIdx];
            board_t b = s_blacks[localBoardIdx];
            board_t k = s_kings[localBoardIdx];

            move_t chosen = SelectBestMoveForSingleBoard(
                w, b, k, boardMoves, boardMoveCounts, boardCaptures, boardFlags, localSeed
            );
            PrintCheckpoint(localThreadInBoard, 9);

            // Save the updated seed back
            s_seed[localBoardIdx] = static_cast<u8>(localSeed + 13);

            // If no move => the current side loses
            if (chosen == move_gen::MoveConstants::kInvalidMove) {
                s_outcome[localBoardIdx] = (s_currentTurn[localBoardIdx] == 0) ? 2 : 1;
            }
            s_chosenMove[localBoardIdx] = chosen;
        }
        __syncthreads();

        // If outcome changed, break
        if (s_outcome[localBoardIdx] != 0) {
            break;
        }

        // 5) Apply the chosen move
        if (localThreadInBoard == 0) {
            move_t mv = s_chosenMove[localBoardIdx];
            checkers::gpu::apply_move::ApplyMoveOnSingleBoard(
                mv, s_whites[localBoardIdx], s_blacks[localBoardIdx], s_kings[localBoardIdx]
            );
        }
        PrintCheckpoint(localThreadInBoard, 10);
        __syncthreads();

        // 6) Chain capturing. We keep capturing if a capture was done and there's a possibility of continuing.
        //    The "to" index of the chosen move might still be able to capture more.
        //    We'll replicate the logic inside a while(true).
        __shared__ board_index_t s_chainFrom[kNumBoardsPerBlock];
        if (localThreadInBoard == 0) {
            move_t mv                  = s_chosenMove[localBoardIdx];
            s_chainFrom[localBoardIdx] = checkers::gpu::move_gen::DecodeMove<checkers::gpu::move_gen::MovePart::To>(mv);
        }
        PrintCheckpoint(localThreadInBoard, 11);
        __syncthreads();

        while (true) {
            // Clear arrays again
            {
                int sq                            = localThreadInBoard;
                s_moveCounts[localBoardIdx][sq]   = 0;
                s_captureMasks[localBoardIdx][sq] = 0;
                s_hasCapture[localBoardIdx]       = false;
            }
            if (localThreadInBoard == 0) {
                s_perBoardFlags[localBoardIdx] = 0;
            }
            __syncthreads();
            PrintCheckpoint(localThreadInBoard, 12);

            // Generate only for the piece at chainFrom if it’s still the side’s piece
            board_t w = s_whites[localBoardIdx];
            board_t b = s_blacks[localBoardIdx];
            board_t k = s_kings[localBoardIdx];
            int t     = s_currentTurn[localBoardIdx];

            if ((localThreadInBoard == s_chainFrom[localBoardIdx]) && IsCurrentSidePiece(w, b, t, localThreadInBoard)) {
                if (t == 0) {
                    // White
                    checkers::gpu::move_gen::GenerateMovesForSinglePiece<Turn::kWhite>(
                        static_cast<board_index_t>(localThreadInBoard), w, b, k,
                        &s_moves[localBoardIdx][localThreadInBoard * kNumMaxMovesPerPiece],
                        s_moveCounts[localBoardIdx][localThreadInBoard],
                        s_captureMasks[localBoardIdx][localThreadInBoard], s_perBoardFlags[localBoardIdx]
                    );
                } else {
                    // Black
                    checkers::gpu::move_gen::GenerateMovesForSinglePiece<Turn::kBlack>(
                        static_cast<board_index_t>(localThreadInBoard), w, b, k,
                        &s_moves[localBoardIdx][localThreadInBoard * kNumMaxMovesPerPiece],
                        s_moveCounts[localBoardIdx][localThreadInBoard],
                        s_captureMasks[localBoardIdx][localThreadInBoard], s_perBoardFlags[localBoardIdx]
                    );
                }
            }
            PrintCheckpoint(localThreadInBoard, 13);
            __syncthreads();

            // See if captures exist. We'll check the capture flag in s_perBoardFlags
            if (ReadFlag(s_perBoardFlags[localBoardIdx], move_gen::MoveFlagsConstants::kCaptureFound)) {
                s_hasCapture[localBoardIdx] = true;
            }
            __syncthreads();

            if (!s_hasCapture[localBoardIdx]) {
                // no more chain capturing
                break;
            }
            PrintCheckpoint(localThreadInBoard, 14);

            // We do the chain move selection with a single thread
            if (localThreadInBoard == 0) {
                // pick chain capture from s_moves
                move_t* boardMoves          = &s_moves[localBoardIdx][0];
                u8* boardMoveCounts         = s_moveCounts[localBoardIdx];
                move_flags_t* boardCaptures = s_captureMasks[localBoardIdx];
                move_flags_t boardFlags     = s_perBoardFlags[localBoardIdx];
                u8 localSeed                = s_seed[localBoardIdx];

                move_t chainMv = checkers::gpu::move_selection::SelectBestMoveForSingleBoard(
                    w, b, k, boardMoves, boardMoveCounts, boardCaptures, boardFlags, localSeed
                );
                s_seed[localBoardIdx] = static_cast<u8>(localSeed + 7);  // update seed

                if (chainMv == move_gen::MoveConstants::kInvalidMove) {
                    // can't continue capturing
                    s_chosenMove[localBoardIdx] = chainMv;
                } else {
                    // apply move
                    checkers::gpu::apply_move::ApplyMoveOnSingleBoard(
                        chainMv, s_whites[localBoardIdx], s_blacks[localBoardIdx], s_kings[localBoardIdx]
                    );
                    s_chainFrom[localBoardIdx] =
                        checkers::gpu::move_gen::DecodeMove<checkers::gpu::move_gen::MovePart::To>(chainMv);
                }
            }
            PrintCheckpoint(localThreadInBoard, 15);
            __syncthreads();

            // If chosen move is invalid => break
            if (s_chosenMove[localBoardIdx] == move_gen::MoveConstants::kInvalidMove) {
                break;
            }
        }  // end while (chain capturing)

        // 7) Check promotion
        if (localThreadInBoard == 0) {
            s_kings[localBoardIdx] |= (s_whites[localBoardIdx] & BoardConstants::kTopBoardEdgeMask);
            s_kings[localBoardIdx] |= (s_blacks[localBoardIdx] & BoardConstants::kBottomBoardEdgeMask);
        }
        PrintCheckpoint(localThreadInBoard, 16);
        __syncthreads();

        // 8) 40-move rule or “non-reversible” logic. We'll do it if your game wants that.
        if (localThreadInBoard == 0) {
            bool wasCapture =
                ((s_perBoardFlags[localBoardIdx] >> move_gen::MoveFlagsConstants::kCaptureFound) & 1U) != 0U;

            // from square of s_chosenMove:
            board_index_t fromSq =
                checkers::gpu::move_gen::DecodeMove<checkers::gpu::move_gen::MovePart::From>(s_chosenMove[localBoardIdx]
                );
            bool fromWasKing = ((s_kings[localBoardIdx] >> fromSq) & 1ULL) != 0ULL;

            if (!wasCapture && fromWasKing) {
                s_nonReversible[localBoardIdx]++;
            } else {
                s_nonReversible[localBoardIdx] = 0;
            }
            if (s_nonReversible[localBoardIdx] >= 40) {
                s_outcome[localBoardIdx] = 3;  // draw
            }
        }
        __syncthreads();
        if (s_outcome[localBoardIdx] != 0) {
            break;
        }
        PrintCheckpoint(localThreadInBoard, 17);

        // 9) Switch turn
        if (localThreadInBoard == 0) {
            s_currentTurn[localBoardIdx] = 1 - s_currentTurn[localBoardIdx];
        }
        __syncthreads();
        PrintCheckpoint(localThreadInBoard, 18);
    }  // end of for(moveCount..)

    // If no outcome, declare a draw
    if (localThreadInBoard == 0 && s_outcome[localBoardIdx] == 0) {
        s_outcome[localBoardIdx] = 3;  // 3 = draw
    }
    PrintCheckpoint(localThreadInBoard, 19);
    __syncthreads();

    // ----------------------------------------------------------------------------
    // Write final results back to global memory
    // ----------------------------------------------------------------------------
    if (localThreadInBoard == 0) {
        d_scores[globalBoardIndex] = s_outcome[localBoardIdx];
    }
    PrintCheckpoint(localThreadInBoard, 20);
}

}  // namespace checkers::gpu
