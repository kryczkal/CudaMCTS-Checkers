#ifndef MCTS_CHECKERS_INCLUDE_CUDA_LAUNCHERS_MOVE_GENERATION_LAUNCHER_CUH_
#define MCTS_CHECKERS_INCLUDE_CUDA_LAUNCHERS_MOVE_GENERATION_LAUNCHER_CUH_

#include <algorithm>
#include <cstring>
#include <vector>

#include "checkers_defines.hpp"
#include "cuda/cuda_utils.cuh"
#include "cuda/move_generation.cuh"

namespace checkers::gpu::launchers
{

/**
 * @brief Holds a simple board definition for host usage. We store
 *        bitmasks for white/black pieces, plus king flags.
 */
struct GpuBoard {
    board_t white = 0;
    board_t black = 0;
    board_t kings = 0;

    /**
     * @brief Helper to set a piece.
     *        'W' -> white, 'B' -> black, 'K' -> king flag.
     */
    void setPieceAt(board_index_t idx, char pieceType)
    {
        switch (pieceType) {
            case 'W':
                white |= (static_cast<board_t>(1) << idx);
                break;
            case 'B':
                black |= (static_cast<board_t>(1) << idx);
                break;
            case 'K':
                kings |= (static_cast<board_t>(1) << idx);
                break;
            default:
                break;
        }
    }
};

/**
 * @brief Holds the result of calling the GPU-based GenerateMoves kernel
 *        for exactly one board.
 */
struct MoveGenResult {
    // We track 32 squares, with up to kNumMaxMovesPerPiece = 13 possible moves per piece
    static constexpr size_t kTotalSquares  = checkers::BoardConstants::kBoardSize;
    static constexpr size_t kMovesPerPiece = checkers::gpu::move_gen::kNumMaxMovesPerPiece;

    // Flattened array of moves: size 32*kMovesPerPiece
    std::vector<move_t> h_moves;
    // Number of generated moves per square
    std::vector<u8> h_move_counts;
    // For each square, a mask indicating which sub-moves are captures
    std::vector<move_flags_t> h_capture_masks;
    // Additional per-board flags (bitwise MoveFlagsConstants)
    std::vector<move_flags_t> h_per_board_flags;

    MoveGenResult()
        : h_moves(kTotalSquares * kMovesPerPiece, MoveConstants::kInvalidMove),
          h_move_counts(kTotalSquares, 0),
          h_capture_masks(kTotalSquares, 0),
          h_per_board_flags(1, 0)
    {
    }
};

/**
 * @brief This function allocates device memory for a vector of boards,
 *        copies data to device, launches the GPU kernel, and retrieves results.
 *
 * @tparam turn Whether to generate moves for White or Black.
 * @param boards Vector of host GpuBoard objects (white/black/kings bitmasks).
 * @return Vector of MoveGenResult objects, one per board.
 */
template <Turn turn>
std::vector<MoveGenResult> HostGenerateMoves(const std::vector<GpuBoard>& boards)
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

    GenerateMoves<turn><<<kBlocks, kThreadsPerBlock>>>(
        d_whites, d_blacks, d_kings, d_moves, d_move_counts, d_capture_masks, d_per_board_flags,
        static_cast<u64>(n_boards)
    );
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

}  // namespace checkers::gpu::launchers
#endif  // MCTS_CHECKERS_INCLUDE_CUDA_LAUNCHERS_MOVE_GENERATION_LAUNCHER_CUH_
