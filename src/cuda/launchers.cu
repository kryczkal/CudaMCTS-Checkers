#include "cuda/apply_move.cuh"
#include "cuda/board_helpers.cuh"
#include "cuda/game_simulation.cuh"
#include "cuda/launchers.cuh"
#include "cuda/move_generation.cuh"
#include "cuda/move_selection.cuh"

namespace checkers::gpu::launchers
{

/**
 * @brief This function allocates device memory for a vector of boards,
 *        copies data to device, launches the GPU kernel, and retrieves results.
 *
 * @tparam turn Whether to generate moves for White or Black.
 * @param boards Vector of host GpuBoard objects (white/black/kings bitmasks).
 * @return Vector of MoveGenResult objects, one per board.
 */
std::vector<MoveGenResult> HostGenerateMoves(const std::vector<GpuBoard>& boards, Turn turn)
{
    using namespace checkers;
    using namespace checkers::gpu::move_gen;

    const size_t n_boards = boards.size();
    std::vector<MoveGenResult> results(n_boards);

    // Early exit if nothing to process
    if (n_boards == 0) {
        return results;
    }

    //--------------------------------------------------------------------------
    // 1) Prepare host-side arrays for white, black, kings
    //--------------------------------------------------------------------------
    std::vector<board_t> host_whites(n_boards), host_blacks(n_boards), host_kings(n_boards);

    for (size_t i = 0; i < n_boards; ++i) {
        host_whites[i] = boards[i].white;
        host_blacks[i] = boards[i].black;
        host_kings[i]  = boards[i].kings;
    }

    //--------------------------------------------------------------------------
    // 2) Allocate device memory
    //--------------------------------------------------------------------------
    board_t* d_whites = nullptr;
    board_t* d_blacks = nullptr;
    board_t* d_kings  = nullptr;

    CHECK_CUDA_ERROR(cudaMalloc(&d_whites, n_boards * sizeof(board_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_blacks, n_boards * sizeof(board_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_kings, n_boards * sizeof(board_t)));

    //--------------------------------------------------------------------------
    // 3) Copy host boards to device
    //--------------------------------------------------------------------------
    CHECK_CUDA_ERROR(cudaMemcpy(d_whites, host_whites.data(), n_boards * sizeof(board_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_blacks, host_blacks.data(), n_boards * sizeof(board_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_kings, host_kings.data(), n_boards * sizeof(board_t), cudaMemcpyHostToDevice));

    //--------------------------------------------------------------------------
    // 4) Allocate device memory for results
    //--------------------------------------------------------------------------
    const size_t kTotalSquares       = MoveGenResult::kTotalSquares;
    const size_t kMovesPerPiece      = MoveGenResult::kMovesPerPiece;
    const size_t kTotalMovesPerBoard = kTotalSquares * kMovesPerPiece;
    const size_t kTotalMoves         = n_boards * kTotalMovesPerBoard;

    move_t* d_moves                 = nullptr;
    u8* d_move_counts               = nullptr;
    move_flags_t* d_capture_masks   = nullptr;
    move_flags_t* d_per_board_flags = nullptr;

    CHECK_CUDA_ERROR(cudaMalloc(&d_moves, kTotalMoves * sizeof(move_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_move_counts, n_boards * kTotalSquares * sizeof(u8)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_capture_masks, n_boards * kTotalSquares * sizeof(move_flags_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_per_board_flags, n_boards * sizeof(move_flags_t)));

    //--------------------------------------------------------------------------
    // 5) Initialize device buffers
    //--------------------------------------------------------------------------
    {
        // Moves to invalid
        std::vector<move_t> initMoves(kTotalMoves, MoveConstants::kInvalidMove);
        CHECK_CUDA_ERROR(cudaMemcpy(d_moves, initMoves.data(), kTotalMoves * sizeof(move_t), cudaMemcpyHostToDevice));

        // Move counts to zero
        std::vector<u8> initCounts(n_boards * kTotalSquares, 0);
        CHECK_CUDA_ERROR(
            cudaMemcpy(d_move_counts, initCounts.data(), n_boards * kTotalSquares * sizeof(u8), cudaMemcpyHostToDevice)
        );

        // Capture masks to zero
        std::vector<move_flags_t> initCapture(n_boards * kTotalSquares, 0);
        CHECK_CUDA_ERROR(cudaMemcpy(
            d_capture_masks, initCapture.data(), n_boards * kTotalSquares * sizeof(move_flags_t), cudaMemcpyHostToDevice
        ));

        // Per-board flags to zero
        std::vector<move_flags_t> zeroBoardFlags(n_boards, 0);
        CHECK_CUDA_ERROR(cudaMemcpy(
            d_per_board_flags, zeroBoardFlags.data(), n_boards * sizeof(move_flags_t), cudaMemcpyHostToDevice
        ));
    }

    //--------------------------------------------------------------------------
    // 6) Launch the device kernel
    //--------------------------------------------------------------------------
    // Each board needs 32 threads (1 thread per board square).
    // We'll pick a block size and number of blocks accordingly.
    const int kThreadsPerBlock = 256;
    const size_t kTotalThreads = n_boards * 32ULL;
    const int kBlocks          = static_cast<int>((kTotalThreads + kThreadsPerBlock - 1) / kThreadsPerBlock);

    switch (turn) {
        case Turn::kWhite:
            GenerateMoves<Turn::kWhite><<<kBlocks, kThreadsPerBlock>>>(
                d_whites, d_blacks, d_kings, d_moves, d_move_counts, d_capture_masks, d_per_board_flags,
                static_cast<u64>(n_boards)
            );
            break;
        case Turn::kBlack:
            GenerateMoves<Turn::kBlack><<<kBlocks, kThreadsPerBlock>>>(
                d_whites, d_blacks, d_kings, d_moves, d_move_counts, d_capture_masks, d_per_board_flags,
                static_cast<u64>(n_boards)
            );
            break;
    }
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    //--------------------------------------------------------------------------
    // 7) Copy results back to host
    //--------------------------------------------------------------------------
    std::vector<move_t> host_moves(kTotalMoves);
    std::vector<u8> host_move_counts(n_boards * kTotalSquares);
    std::vector<move_flags_t> host_capture_masks(n_boards * kTotalSquares);
    std::vector<move_flags_t> host_board_flags(n_boards);

    CHECK_CUDA_ERROR(cudaMemcpy(host_moves.data(), d_moves, kTotalMoves * sizeof(move_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(
        host_move_counts.data(), d_move_counts, n_boards * kTotalSquares * sizeof(u8), cudaMemcpyDeviceToHost
    ));
    CHECK_CUDA_ERROR(cudaMemcpy(
        host_capture_masks.data(), d_capture_masks, n_boards * kTotalSquares * sizeof(move_flags_t),
        cudaMemcpyDeviceToHost
    ));
    CHECK_CUDA_ERROR(
        cudaMemcpy(host_board_flags.data(), d_per_board_flags, n_boards * sizeof(move_flags_t), cudaMemcpyDeviceToHost)
    );

    //--------------------------------------------------------------------------
    // 8) Populate results
    //--------------------------------------------------------------------------
    for (size_t i = 0; i < n_boards; ++i) {
        // Each board's chunk of moves
        MoveGenResult& r = results[i];

        // Moves
        const size_t offset = i * kTotalMovesPerBoard;
        std::copy(host_moves.begin() + offset, host_moves.begin() + offset + kTotalMovesPerBoard, r.h_moves.begin());

        // Move counts & capture masks
        const size_t offsetBoard = i * kTotalSquares;
        std::copy(
            host_move_counts.begin() + offsetBoard, host_move_counts.begin() + offsetBoard + kTotalSquares,
            r.h_move_counts.begin()
        );

        std::copy(
            host_capture_masks.begin() + offsetBoard, host_capture_masks.begin() + offsetBoard + kTotalSquares,
            r.h_capture_masks.begin()
        );

        // Per-board flags
        r.h_per_board_flags[0] = host_board_flags[i];
    }

    //--------------------------------------------------------------------------
    // 9) Free device resources
    //--------------------------------------------------------------------------
    CHECK_CUDA_ERROR(cudaFree(d_whites));
    CHECK_CUDA_ERROR(cudaFree(d_blacks));
    CHECK_CUDA_ERROR(cudaFree(d_kings));
    CHECK_CUDA_ERROR(cudaFree(d_moves));
    CHECK_CUDA_ERROR(cudaFree(d_move_counts));
    CHECK_CUDA_ERROR(cudaFree(d_capture_masks));
    CHECK_CUDA_ERROR(cudaFree(d_per_board_flags));

    return results;
}

std::vector<GpuBoard> HostApplyMoves(const std::vector<GpuBoard>& boards, const std::vector<move_t>& moves)
{
    using namespace checkers;
    using namespace checkers::gpu::apply_move;

    const size_t n_boards = boards.size();

    // Copy of the original boards to be updated
    std::vector<GpuBoard> updatedBoards = boards;
    if (n_boards == 0) {
        return updatedBoards;
    }

    //--------------------------------------------------------------------------
    // 1) Prepare host arrays for white, black, and king bitmasks
    //--------------------------------------------------------------------------
    std::vector<board_t> host_whites(n_boards), host_blacks(n_boards), host_kings(n_boards);
    for (size_t i = 0; i < n_boards; ++i) {
        host_whites[i] = boards[i].white;
        host_blacks[i] = boards[i].black;
        host_kings[i]  = boards[i].kings;
    }

    //--------------------------------------------------------------------------
    // 2) Allocate device memory
    //--------------------------------------------------------------------------
    board_t* d_whites = nullptr;
    board_t* d_blacks = nullptr;
    board_t* d_kings  = nullptr;
    move_t* d_moves   = nullptr;

    CHECK_CUDA_ERROR(cudaMalloc(&d_whites, n_boards * sizeof(board_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_blacks, n_boards * sizeof(board_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_kings, n_boards * sizeof(board_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_moves, n_boards * sizeof(move_t)));

    //--------------------------------------------------------------------------
    // 3) Copy host data to device
    //--------------------------------------------------------------------------
    CHECK_CUDA_ERROR(cudaMemcpy(d_whites, host_whites.data(), n_boards * sizeof(board_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_blacks, host_blacks.data(), n_boards * sizeof(board_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_kings, host_kings.data(), n_boards * sizeof(board_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_moves, moves.data(), n_boards * sizeof(move_t), cudaMemcpyHostToDevice));

    //--------------------------------------------------------------------------
    // 4) Launch the ApplyMove kernel
    //--------------------------------------------------------------------------
    const int threadsPerBlock = 256;
    const int blocks          = static_cast<int>((n_boards + threadsPerBlock - 1) / threadsPerBlock);

    ApplyMove<<<blocks, threadsPerBlock>>>(d_whites, d_blacks, d_kings, d_moves, static_cast<u64>(n_boards));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    //--------------------------------------------------------------------------
    // 5) Copy results back to host
    //--------------------------------------------------------------------------
    CHECK_CUDA_ERROR(cudaMemcpy(host_whites.data(), d_whites, n_boards * sizeof(board_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(host_blacks.data(), d_blacks, n_boards * sizeof(board_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(host_kings.data(), d_kings, n_boards * sizeof(board_t), cudaMemcpyDeviceToHost));

    //--------------------------------------------------------------------------
    // 6) Update our returned board states
    //--------------------------------------------------------------------------
    for (size_t i = 0; i < n_boards; ++i) {
        updatedBoards[i].white = host_whites[i];
        updatedBoards[i].black = host_blacks[i];
        updatedBoards[i].kings = host_kings[i];
    }

    //--------------------------------------------------------------------------
    // 7) Cleanup
    //--------------------------------------------------------------------------
    CHECK_CUDA_ERROR(cudaFree(d_whites));
    CHECK_CUDA_ERROR(cudaFree(d_blacks));
    CHECK_CUDA_ERROR(cudaFree(d_kings));
    CHECK_CUDA_ERROR(cudaFree(d_moves));

    return updatedBoards;
}

std::vector<move_t> HostSelectBestMoves(
    const std::vector<GpuBoard>& boards, const std::vector<move_t>& moves, const std::vector<u8>& move_counts,
    const std::vector<move_flags_t>& capture_masks, const std::vector<move_flags_t>& per_board_flags,
    const std::vector<u8>& seeds
)
{
    using namespace checkers;
    using namespace checkers::gpu::move_selection;

    const size_t n_boards = boards.size();
    std::vector<move_t> bestMoves(n_boards, MoveConstants::kInvalidMove);

    if (n_boards == 0) {
        return bestMoves;
    }

    //--------------------------------------------------------------------------
    // 1) Prepare host arrays for white, black, and king bitmasks
    //--------------------------------------------------------------------------
    std::vector<board_t> h_whites(n_boards), h_blacks(n_boards), h_kings(n_boards);
    for (size_t i = 0; i < n_boards; ++i) {
        h_whites[i] = boards[i].white;
        h_blacks[i] = boards[i].black;
        h_kings[i]  = boards[i].kings;
    }

    // Basic size checks (could be expanded with error handling)
    const size_t totalSquares       = BoardConstants::kBoardSize;
    const size_t movesPerPiece      = gpu::move_gen::kNumMaxMovesPerPiece;
    const size_t totalMovesPerBoard = totalSquares * movesPerPiece;
    if (moves.size() != n_boards * totalMovesPerBoard) {
        // Potentially handle or throw an error. For now, assume correct input.
    }

    //--------------------------------------------------------------------------
    // 2) Allocate device memory
    //--------------------------------------------------------------------------
    u32* d_whites                   = nullptr;
    u32* d_blacks                   = nullptr;
    u32* d_kings                    = nullptr;
    move_t* d_moves                 = nullptr;
    u8* d_move_counts               = nullptr;
    move_flags_t* d_capture_masks   = nullptr;
    move_flags_t* d_per_board_flags = nullptr;
    u8* d_seeds                     = nullptr;
    move_t* d_best_moves            = nullptr;

    CHECK_CUDA_ERROR(cudaMalloc(&d_whites, n_boards * sizeof(u32)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_blacks, n_boards * sizeof(u32)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_kings, n_boards * sizeof(u32)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_moves, moves.size() * sizeof(move_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_move_counts, n_boards * totalSquares * sizeof(u8)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_capture_masks, n_boards * totalSquares * sizeof(move_flags_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_per_board_flags, n_boards * sizeof(move_flags_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_seeds, n_boards * sizeof(u8)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_best_moves, n_boards * sizeof(move_t)));

    //--------------------------------------------------------------------------
    // 3) Copy host data to device
    //--------------------------------------------------------------------------
    CHECK_CUDA_ERROR(cudaMemcpy(d_whites, h_whites.data(), n_boards * sizeof(u32), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_blacks, h_blacks.data(), n_boards * sizeof(u32), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_kings, h_kings.data(), n_boards * sizeof(u32), cudaMemcpyHostToDevice));

    CHECK_CUDA_ERROR(cudaMemcpy(d_moves, moves.data(), moves.size() * sizeof(move_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(
        cudaMemcpy(d_move_counts, move_counts.data(), n_boards * totalSquares * sizeof(u8), cudaMemcpyHostToDevice)
    );
    CHECK_CUDA_ERROR(cudaMemcpy(
        d_capture_masks, capture_masks.data(), n_boards * totalSquares * sizeof(move_flags_t), cudaMemcpyHostToDevice
    ));
    CHECK_CUDA_ERROR(
        cudaMemcpy(d_per_board_flags, per_board_flags.data(), n_boards * sizeof(move_flags_t), cudaMemcpyHostToDevice)
    );
    CHECK_CUDA_ERROR(cudaMemcpy(d_seeds, seeds.data(), n_boards * sizeof(u8), cudaMemcpyHostToDevice));

    // Initialize d_best_moves to invalid
    std::vector<move_t> initBest(n_boards, MoveConstants::kInvalidMove);
    CHECK_CUDA_ERROR(cudaMemcpy(d_best_moves, initBest.data(), n_boards * sizeof(move_t), cudaMemcpyHostToDevice));

    //--------------------------------------------------------------------------
    // 4) Launch the SelectBestMoves kernel
    //--------------------------------------------------------------------------
    const int threadsPerBlock = 256;
    const size_t totalThreads = n_boards;
    const int blocks          = static_cast<int>((totalThreads + threadsPerBlock - 1) / threadsPerBlock);

    SelectBestMoves<<<blocks, threadsPerBlock>>>(
        d_whites, d_blacks, d_kings, d_moves, d_move_counts, d_capture_masks, d_per_board_flags,
        static_cast<u64>(n_boards), d_seeds, d_best_moves
    );
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    //--------------------------------------------------------------------------
    // 5) Copy results back to host
    //--------------------------------------------------------------------------
    CHECK_CUDA_ERROR(cudaMemcpy(bestMoves.data(), d_best_moves, n_boards * sizeof(move_t), cudaMemcpyDeviceToHost));

    //--------------------------------------------------------------------------
    // 6) Cleanup
    //--------------------------------------------------------------------------
    CHECK_CUDA_ERROR(cudaFree(d_whites));
    CHECK_CUDA_ERROR(cudaFree(d_blacks));
    CHECK_CUDA_ERROR(cudaFree(d_kings));
    CHECK_CUDA_ERROR(cudaFree(d_moves));
    CHECK_CUDA_ERROR(cudaFree(d_move_counts));
    CHECK_CUDA_ERROR(cudaFree(d_capture_masks));
    CHECK_CUDA_ERROR(cudaFree(d_per_board_flags));
    CHECK_CUDA_ERROR(cudaFree(d_seeds));
    CHECK_CUDA_ERROR(cudaFree(d_best_moves));

    return bestMoves;
}

std::vector<u8> HostSimulateCheckersGames(
    const std::vector<board_t>& h_whites, const std::vector<board_t>& h_blacks, const std::vector<board_t>& h_kings,
    const std::vector<u8>& h_seeds, int max_iterations
)
{
    //--------------------------------------------------------------------------
    // 1) Basic checks
    //--------------------------------------------------------------------------
    const size_t n_boards = h_whites.size();
    std::vector<u8> results(n_boards, 0);

    if (n_boards == 0) {
        return results;
    }

    //--------------------------------------------------------------------------
    // 2) Allocate device memory
    //--------------------------------------------------------------------------
    board_t* d_whites = nullptr;
    board_t* d_blacks = nullptr;
    board_t* d_kings  = nullptr;
    u8* d_scores      = nullptr;
    u8* d_seeds       = nullptr;

    CHECK_CUDA_ERROR(cudaMalloc(&d_whites, n_boards * sizeof(board_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_blacks, n_boards * sizeof(board_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_kings, n_boards * sizeof(board_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_scores, n_boards * sizeof(u8)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_seeds, n_boards * sizeof(u8)));

    //--------------------------------------------------------------------------
    // 3) Copy data from host to device
    //--------------------------------------------------------------------------
    CHECK_CUDA_ERROR(cudaMemcpy(d_whites, h_whites.data(), n_boards * sizeof(board_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_blacks, h_blacks.data(), n_boards * sizeof(board_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_kings, h_kings.data(), n_boards * sizeof(board_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_seeds, h_seeds.data(), n_boards * sizeof(u8), cudaMemcpyHostToDevice));

    // Initialize scores to 0 (in-progress)
    std::vector<u8> initScores(n_boards, 0);
    CHECK_CUDA_ERROR(cudaMemcpy(d_scores, initScores.data(), n_boards * sizeof(u8), cudaMemcpyHostToDevice));

    //--------------------------------------------------------------------------
    // 4) Determine kernel launch configuration
    //--------------------------------------------------------------------------
    const int threadsPerBlock = 256;
    const int blocks          = static_cast<int>((n_boards + threadsPerBlock - 1) / threadsPerBlock);

    //--------------------------------------------------------------------------
    // 5) Launch the simulation kernel
    //--------------------------------------------------------------------------
    SimulateCheckersGames<<<blocks, threadsPerBlock>>>(
        d_whites, d_blacks, d_kings, d_scores, static_cast<u64>(n_boards), d_seeds, max_iterations
    );
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    //--------------------------------------------------------------------------
    // 6) Copy final results from device to host
    //--------------------------------------------------------------------------
    CHECK_CUDA_ERROR(cudaMemcpy(results.data(), d_scores, n_boards * sizeof(u8), cudaMemcpyDeviceToHost));

    //--------------------------------------------------------------------------
    // 7) Free device memory
    //--------------------------------------------------------------------------
    CHECK_CUDA_ERROR(cudaFree(d_whites));
    CHECK_CUDA_ERROR(cudaFree(d_blacks));
    CHECK_CUDA_ERROR(cudaFree(d_kings));
    CHECK_CUDA_ERROR(cudaFree(d_scores));
    CHECK_CUDA_ERROR(cudaFree(d_seeds));

    //--------------------------------------------------------------------------
    // 8) Return the outcomes
    //--------------------------------------------------------------------------
    return results;
}

}  // namespace checkers::gpu::launchers
