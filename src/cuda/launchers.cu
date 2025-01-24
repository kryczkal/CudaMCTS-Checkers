#include <random>
#include "common/checkers_defines.hpp"
#include "cuda/apply_move.cuh"
#include "cuda/board_helpers.cuh"
#include "cuda/capture_lookup_table.cuh"
#include "cuda/game_simulation.cuh"
#include "cuda/launchers.cuh"
#include "cuda/move_generation.cuh"
#include "cuda/move_selection.cuh"
#include "cuda/reductions.cuh"

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
    using namespace move_gen;

    const size_t n_boards = boards.size();
    std::vector<MoveGenResult> results(n_boards);

    // Early exit if nothing to process
    if (n_boards == 0) {
        return results;
    }

    //--------------------------------------------------------------------------
    // Prepare host-side arrays for white, black, kings
    //--------------------------------------------------------------------------
    std::vector<board_t> host_whites(n_boards), host_blacks(n_boards), host_kings(n_boards);

    for (size_t i = 0; i < n_boards; ++i) {
        host_whites[i] = boards[i].white;
        host_blacks[i] = boards[i].black;
        host_kings[i]  = boards[i].kings;
    }

    //--------------------------------------------------------------------------
    // Allocate device memory
    //--------------------------------------------------------------------------
    board_t* d_whites = nullptr;
    board_t* d_blacks = nullptr;
    board_t* d_kings  = nullptr;

    CHECK_CUDA_ERROR(cudaMalloc(&d_whites, n_boards * sizeof(board_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_blacks, n_boards * sizeof(board_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_kings, n_boards * sizeof(board_t)));

    //--------------------------------------------------------------------------
    // Copy host boards to device
    //--------------------------------------------------------------------------
    CHECK_CUDA_ERROR(cudaMemcpy(d_whites, host_whites.data(), n_boards * sizeof(board_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_blacks, host_blacks.data(), n_boards * sizeof(board_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_kings, host_kings.data(), n_boards * sizeof(board_t), cudaMemcpyHostToDevice));

    //--------------------------------------------------------------------------
    // Allocate device memory for results
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
    // Initialize device buffers
    //--------------------------------------------------------------------------
    {
        // Moves to invalid
        std::vector<move_t> initMoves(kTotalMoves, kInvalidMove);
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
    // Launch the device kernel
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
    // Copy results back to host
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
    // Populate results
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
    // Free device resources
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
    // Prepare host arrays for white, black, and king bitmasks
    //--------------------------------------------------------------------------
    std::vector<board_t> host_whites(n_boards), host_blacks(n_boards), host_kings(n_boards);
    for (size_t i = 0; i < n_boards; ++i) {
        host_whites[i] = boards[i].white;
        host_blacks[i] = boards[i].black;
        host_kings[i]  = boards[i].kings;
    }

    //--------------------------------------------------------------------------
    // Allocate device memory
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
    // Copy host data to device
    //--------------------------------------------------------------------------
    CHECK_CUDA_ERROR(cudaMemcpy(d_whites, host_whites.data(), n_boards * sizeof(board_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_blacks, host_blacks.data(), n_boards * sizeof(board_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_kings, host_kings.data(), n_boards * sizeof(board_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_moves, moves.data(), n_boards * sizeof(move_t), cudaMemcpyHostToDevice));

    //--------------------------------------------------------------------------
    // Launch the kernel
    //--------------------------------------------------------------------------
    const int threadsPerBlock = 256;
    const int blocks          = static_cast<int>((n_boards + threadsPerBlock - 1) / threadsPerBlock);

    ApplyMove<<<blocks, threadsPerBlock>>>(d_whites, d_blacks, d_kings, d_moves, static_cast<u64>(n_boards));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    //--------------------------------------------------------------------------
    // Copy results back to host
    //--------------------------------------------------------------------------
    CHECK_CUDA_ERROR(cudaMemcpy(host_whites.data(), d_whites, n_boards * sizeof(board_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(host_blacks.data(), d_blacks, n_boards * sizeof(board_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(host_kings.data(), d_kings, n_boards * sizeof(board_t), cudaMemcpyDeviceToHost));

    //--------------------------------------------------------------------------
    // Update our returned board states
    //--------------------------------------------------------------------------
    for (size_t i = 0; i < n_boards; ++i) {
        updatedBoards[i].white = host_whites[i];
        updatedBoards[i].black = host_blacks[i];
        updatedBoards[i].kings = host_kings[i];
    }

    //--------------------------------------------------------------------------
    // Cleanup
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
    std::vector<move_t> bestMoves(n_boards, kInvalidMove);

    if (n_boards == 0) {
        return bestMoves;
    }

    //--------------------------------------------------------------------------
    // Prepare host arrays
    //--------------------------------------------------------------------------
    std::vector<board_t> h_whites(n_boards), h_blacks(n_boards), h_kings(n_boards);
    for (size_t i = 0; i < n_boards; ++i) {
        h_whites[i] = boards[i].white;
        h_blacks[i] = boards[i].black;
        h_kings[i]  = boards[i].kings;
    }

    // Basic size checks (could be expanded with error handling)
    const size_t totalSquares       = BoardConstants::kBoardSize;
    const size_t movesPerPiece      = kNumMaxMovesPerPiece;
    const size_t totalMovesPerBoard = totalSquares * movesPerPiece;
    if (moves.size() != n_boards * totalMovesPerBoard) {
        // Potentially handle or throw an error. For now, assume correct input.
    }

    //--------------------------------------------------------------------------
    // Allocate device memory
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
    // Copy host data to device
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
    std::vector<move_t> initBest(n_boards, kInvalidMove);
    CHECK_CUDA_ERROR(cudaMemcpy(d_best_moves, initBest.data(), n_boards * sizeof(move_t), cudaMemcpyHostToDevice));

    //--------------------------------------------------------------------------
    // Launch the kernel
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
    // Copy results back to host
    //--------------------------------------------------------------------------
    CHECK_CUDA_ERROR(cudaMemcpy(bestMoves.data(), d_best_moves, n_boards * sizeof(move_t), cudaMemcpyDeviceToHost));

    //--------------------------------------------------------------------------
    // Cleanup
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

std::vector<SimulationResult> HostSimulateCheckersGames(const std::vector<SimulationParam>& params, int max_iterations)
{
    const u64 n_simulation_counts = static_cast<u64>(params.size());

    u64 n_total_simulations = 0;
    for (auto& p : params) {
        n_total_simulations += p.n_simulations;
    }

    if (n_total_simulations == 0) {
        return {};
    }

    //--------------------------------------------------------------------------
    // Prepare host arrays for white, black, and king bitmasks
    //--------------------------------------------------------------------------
    std::vector<board_t> h_whites(n_simulation_counts);
    std::vector<board_t> h_blacks(n_simulation_counts);
    std::vector<board_t> h_kings(n_simulation_counts);
    std::vector<u8> h_startTurns(n_simulation_counts);
    std::vector<u64> h_simCounts(n_simulation_counts);

    for (size_t i = 0; i < n_simulation_counts; i++) {
        h_whites[i]     = params[i].white;
        h_blacks[i]     = params[i].black;
        h_kings[i]      = params[i].king;
        h_startTurns[i] = params[i].start_turn;
        h_simCounts[i]  = params[i].n_simulations;
    }

    // Generate random seeds for each of the total simulations
    std::vector<u8> h_seeds(n_total_simulations);
    {
        std::mt19937 rng(kTrueRandom ? std::random_device{}() : kSeed);
        for (u64 i = 0; i < n_total_simulations; i++) {
            h_seeds[i] = static_cast<u8>(rng() & 0xFF);
        }
    }

    //--------------------------------------------------------------------------
    // Allocate device memory
    //--------------------------------------------------------------------------
    board_t* d_whites = nullptr;
    board_t* d_blacks = nullptr;
    board_t* d_kings  = nullptr;
    u8* d_startTurns  = nullptr;
    u64* d_simCounts  = nullptr;
    u8* d_scores      = nullptr;  // size = n_total_simulations
    u8* d_seeds       = nullptr;

    CHECK_CUDA_ERROR(cudaMalloc(&d_whites, n_simulation_counts * sizeof(board_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_blacks, n_simulation_counts * sizeof(board_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_kings, n_simulation_counts * sizeof(board_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_startTurns, n_simulation_counts * sizeof(u8)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_simCounts, n_simulation_counts * sizeof(u64)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_scores, n_total_simulations * sizeof(u8)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_seeds, n_total_simulations * sizeof(u8)));

    //--------------------------------------------------------------------------
    // Copy host data to device
    //--------------------------------------------------------------------------
    CHECK_CUDA_ERROR(
        cudaMemcpy(d_whites, h_whites.data(), n_simulation_counts * sizeof(board_t), cudaMemcpyHostToDevice)
    );
    CHECK_CUDA_ERROR(
        cudaMemcpy(d_blacks, h_blacks.data(), n_simulation_counts * sizeof(board_t), cudaMemcpyHostToDevice)
    );
    CHECK_CUDA_ERROR(cudaMemcpy(d_kings, h_kings.data(), n_simulation_counts * sizeof(board_t), cudaMemcpyHostToDevice)
    );
    CHECK_CUDA_ERROR(
        cudaMemcpy(d_startTurns, h_startTurns.data(), n_simulation_counts * sizeof(u8), cudaMemcpyHostToDevice)
    );
    CHECK_CUDA_ERROR(
        cudaMemcpy(d_simCounts, h_simCounts.data(), n_simulation_counts * sizeof(u64), cudaMemcpyHostToDevice)
    );

    // Scores set to 0 initially
    CHECK_CUDA_ERROR(cudaMemset(d_scores, 0, n_total_simulations * sizeof(u8)));

    // Seeds
    CHECK_CUDA_ERROR(cudaMemcpy(d_seeds, h_seeds.data(), n_total_simulations * sizeof(u8), cudaMemcpyHostToDevice));

    //--------------------------------------------------------------------------
    // Launch the simulation kernel
    //--------------------------------------------------------------------------
    //   This kernel populates d_scores with {0,1,2} for each simulation.
    //   Implementation is the same as shown in your code, except that it
    //   places final results in d_scores[] for each simulation.
    //--------------------------------------------------------------------------
    const int threadsPerBlock = kNumBoardsPerBlock * BoardConstants::kBoardSize;
    const u64 totalThreads    = n_total_simulations * BoardConstants::kBoardSize;
    const int blocks          = static_cast<int>((totalThreads + threadsPerBlock - 1) / threadsPerBlock);

    SimulateCheckersGames<<<blocks, threadsPerBlock>>>(
        d_whites, d_blacks, d_kings, d_startTurns, d_simCounts, n_simulation_counts,
        d_scores,  // each simulation’s 0/1/2 outcome
        d_seeds, max_iterations, n_total_simulations
    );
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    //--------------------------------------------------------------------------
    // For each batch, sum the portion of d_scores that belongs to that batch.
    //--------------------------------------------------------------------------
    std::vector<SimulationResult> results(n_simulation_counts);

    u64 offset = 0;
    for (size_t i = 0; i < n_simulation_counts; i++) {
        u64 simCount = h_simCounts[i];
        if (simCount == 0) {
            // no simulations in this batch
            results[i].score         = 0.0;
            results[i].n_simulations = 0;
            continue;
        }

        // sum the subarray d_scores[offset .. offset+simCount-1]
        const u8* d_subarray = d_scores + offset;
        u64 sum              = DeviceSumU8(d_subarray, simCount);

        // Each result can be 0=lose,1=draw,2=win.
        // Conversion: final_score = sum_of_all_outcomes / 2.0
        // so that 2 => 1.0, 1 => 0.5, 0 => 0.0
        double finalScore = static_cast<double>(sum) / 2.0;

        results[i].score         = finalScore;
        results[i].n_simulations = simCount;

        offset += simCount;
    }

    //--------------------------------------------------------------------------
    // Cleanup device memory
    //--------------------------------------------------------------------------
    CHECK_CUDA_ERROR(cudaFree(d_whites));
    CHECK_CUDA_ERROR(cudaFree(d_blacks));
    CHECK_CUDA_ERROR(cudaFree(d_kings));
    CHECK_CUDA_ERROR(cudaFree(d_startTurns));
    CHECK_CUDA_ERROR(cudaFree(d_simCounts));
    CHECK_CUDA_ERROR(cudaFree(d_scores));
    CHECK_CUDA_ERROR(cudaFree(d_seeds));

    return results;
}
}  // namespace checkers::gpu::launchers
