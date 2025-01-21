#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "checkers_defines.hpp"
#include "cuda/cuda_utils.cuh"
#include "cuda/move_generation.cuh"

static constexpr u8 kNumMaxMovesPerPiece = checkers::gpu::move_gen::kNumMaxMovesPerPiece;

namespace checkers
{
__host__ __device__ inline move_t EncodeMove(board_index_t from, board_index_t to)
{
    return static_cast<move_t>((static_cast<u16>(to) << 8) | from);
}

inline std::pair<board_index_t, board_index_t> DecodeMove(move_t move)
{
    board_index_t from = static_cast<board_index_t>(move & 0xFF);
    board_index_t to   = static_cast<board_index_t>((move >> 8) & 0xFF);
    return {from, to};
}

struct GpuBoard {
    board_t white = 0;
    board_t black = 0;
    board_t kings = 0;

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
                break;  // ignore any unrecognized char
        }
    }
};

struct MoveGenResult {
    static constexpr size_t kTotalSquares  = checkers::BoardConstants::kBoardSize;
    static constexpr size_t kMovesPerPiece = checkers::gpu::move_gen::kNumMaxMovesPerPiece;

    std::vector<move_t> h_moves;
    std::vector<u8> h_move_counts;
    std::vector<move_flags_t> h_capture_masks;
    std::vector<move_flags_t> h_per_board_flags;

    MoveGenResult()
        : h_moves(kTotalSquares * kMovesPerPiece, MoveConstants::kInvalidMove),
          h_move_counts(kTotalSquares, 0),
          h_capture_masks(kTotalSquares, 0),
          h_per_board_flags(1, 0)
    {
    }
};

template <Turn turn>
std::vector<MoveGenResult> LaunchGpuMoveGen(const std::vector<GpuBoard> &boards)
{
    using namespace checkers::gpu::move_gen;

    size_t n_boards = boards.size();
    std::vector<MoveGenResult> results(n_boards, MoveGenResult());

    if (n_boards == 0) {
        return results;  // No boards to process
    }

    // 1. Prepare host-side data arrays
    std::vector<board_t> host_whites(n_boards);
    std::vector<board_t> host_blacks(n_boards);
    std::vector<board_t> host_kings(n_boards);

    for (size_t i = 0; i < n_boards; ++i) {
        host_whites[i] = boards[i].white;
        host_blacks[i] = boards[i].black;
        host_kings[i]  = boards[i].kings;
    }

    // 2. Allocate device memory for all boards
    board_t *d_whites = nullptr;
    board_t *d_blacks = nullptr;
    board_t *d_kings  = nullptr;

    CHECK_CUDA_ERROR(cudaMalloc(&d_whites, n_boards * sizeof(board_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_blacks, n_boards * sizeof(board_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_kings, n_boards * sizeof(board_t)));

    // 3. Copy all boards' data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_whites, host_whites.data(), n_boards * sizeof(board_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_blacks, host_blacks.data(), n_boards * sizeof(board_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_kings, host_kings.data(), n_boards * sizeof(board_t), cudaMemcpyHostToDevice));

    // 4. Allocate device memory for results
    move_t *d_moves                 = nullptr;
    u8 *d_move_counts               = nullptr;
    move_flags_t *d_capture_masks   = nullptr;
    move_flags_t *d_per_board_flags = nullptr;

    size_t totalMovesPerBoard = MoveGenResult::kTotalSquares * MoveGenResult::kMovesPerPiece;
    size_t totalMoves         = n_boards * totalMovesPerBoard;

    CHECK_CUDA_ERROR(cudaMalloc(&d_moves, totalMoves * sizeof(move_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_move_counts, n_boards * MoveGenResult::kTotalSquares * sizeof(u8)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_capture_masks, n_boards * MoveGenResult::kTotalSquares * sizeof(move_flags_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_per_board_flags, n_boards * sizeof(move_flags_t)));

    // 5. Initialize device buffers with invalid moves / zero
    {
        // Initialize moves to invalid
        std::vector<move_t> initMoves(totalMoves, MoveConstants::kInvalidMove);
        CHECK_CUDA_ERROR(cudaMemcpy(d_moves, initMoves.data(), totalMoves * sizeof(move_t), cudaMemcpyHostToDevice));

        // Initialize move counts to zero
        std::vector<u8> initCounts(n_boards * MoveGenResult::kTotalSquares, 0);
        CHECK_CUDA_ERROR(cudaMemcpy(
            d_move_counts, initCounts.data(), n_boards * MoveGenResult::kTotalSquares * sizeof(u8),
            cudaMemcpyHostToDevice
        ));

        // Initialize capture masks to zero
        std::vector<move_flags_t> initMask(n_boards * MoveGenResult::kTotalSquares, 0);
        CHECK_CUDA_ERROR(cudaMemcpy(
            d_capture_masks, initMask.data(), n_boards * MoveGenResult::kTotalSquares * sizeof(move_flags_t),
            cudaMemcpyHostToDevice
        ));

        // Initialize per-board flags to zero
        std::vector<move_flags_t> zeroGlobal(n_boards, 0);
        CHECK_CUDA_ERROR(
            cudaMemcpy(d_per_board_flags, zeroGlobal.data(), n_boards * sizeof(move_flags_t), cudaMemcpyHostToDevice)
        );
    }

    // 6. Launch kernel with appropriate configuration
    const int threadsPerBlock = 256;  // Choose a multiple of 32 for efficiency
    // Each board requires 32 threads, so calculate the total number of threads needed
    size_t totalThreads = n_boards * 32;
    int blocks          = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    GenerateMoves<turn><<<blocks, threadsPerBlock>>>(
        d_whites, d_blacks, d_kings, d_moves, d_move_counts, d_capture_masks, d_per_board_flags,
        /* n_boards = */ n_boards
    );

    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // 7. Copy results back to host
    // Prepare host-side result arrays
    std::vector<move_t> host_moves(totalMoves);
    std::vector<u8> host_move_counts(n_boards * MoveGenResult::kTotalSquares);
    std::vector<move_flags_t> host_capture_masks(n_boards * MoveGenResult::kTotalSquares);
    std::vector<move_flags_t> host_per_board_flags(n_boards);

    CHECK_CUDA_ERROR(cudaMemcpy(host_moves.data(), d_moves, totalMoves * sizeof(move_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(
        host_move_counts.data(), d_move_counts, n_boards * MoveGenResult::kTotalSquares * sizeof(u8),
        cudaMemcpyDeviceToHost
    ));
    CHECK_CUDA_ERROR(cudaMemcpy(
        host_capture_masks.data(), d_capture_masks, n_boards * MoveGenResult::kTotalSquares * sizeof(move_flags_t),
        cudaMemcpyDeviceToHost
    ));
    CHECK_CUDA_ERROR(cudaMemcpy(
        host_per_board_flags.data(), d_per_board_flags, n_boards * sizeof(move_flags_t), cudaMemcpyDeviceToHost
    ));

    // 8. Populate the results vector
    for (size_t i = 0; i < n_boards; ++i) {
        MoveGenResult &result = results[i];
        // Assuming MoveGenResult has appropriately sized arrays
        std::copy(
            host_moves.begin() + i * totalMovesPerBoard, host_moves.begin() + (i + 1) * totalMovesPerBoard,
            result.h_moves.begin()
        );
        std::copy(
            host_move_counts.begin() + i * MoveGenResult::kTotalSquares,
            host_move_counts.begin() + (i + 1) * MoveGenResult::kTotalSquares, result.h_move_counts.begin()
        );
        std::copy(
            host_capture_masks.begin() + i * MoveGenResult::kTotalSquares,
            host_capture_masks.begin() + (i + 1) * MoveGenResult::kTotalSquares, result.h_capture_masks.begin()
        );

        std::copy(
            host_per_board_flags.begin() + i, host_per_board_flags.begin() + i + 1, result.h_per_board_flags.begin()
        );
    }

    // 9. Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_whites));
    CHECK_CUDA_ERROR(cudaFree(d_blacks));
    CHECK_CUDA_ERROR(cudaFree(d_kings));
    CHECK_CUDA_ERROR(cudaFree(d_moves));
    CHECK_CUDA_ERROR(cudaFree(d_move_counts));
    CHECK_CUDA_ERROR(cudaFree(d_capture_masks));
    CHECK_CUDA_ERROR(cudaFree(d_per_board_flags));

    return results;
}
}  // namespace checkers

class MoveGenerationKernelTest : public ::testing::Test
{
    protected:
    static bool GlobalCaptureFound(const checkers::MoveGenResult &result, u64 board_index)
    {
        return ((result.h_per_board_flags[board_index] >> checkers::MoveFlagsConstants::kCaptureFound) & 1) ==
               (checkers::move_flags_t)1;
    }

    static bool GlobalMoveFound(const checkers::MoveGenResult &result, u64 board_index)
    {
        return ((result.h_per_board_flags[board_index] >> checkers::MoveFlagsConstants::kMoveFound) & 1) ==
               (checkers::move_flags_t)1;
    }

    static bool FoundAllExpectedMoves(
        const checkers::MoveGenResult &result, std::unordered_map<checkers::move_t, bool> &expected_moves,
        u64 move_index
    )
    {
        u64 current_index = move_index * kNumMaxMovesPerPiece;
        printf("Expected moves: %lu\n", expected_moves.size());
        printf("Result moves: %hhu\n", result.h_move_counts[move_index]);
        if (result.h_move_counts[move_index] != expected_moves.size()) {
            return false;
        }
        for (u64 i = 0; i < result.h_move_counts[move_index]; i++) {
            if (expected_moves.find(result.h_moves[current_index]) == expected_moves.end()) {
                return false;
            }
            if ((result.h_capture_masks[move_index] >> i & 1) != expected_moves[result.h_moves[current_index]]) {
                return false;
            }
            current_index++;
        }
        return true;
    }
};

//-------------------------------------------------------------------------------------
// TEST CASES
//-------------------------------------------------------------------------------------

TEST_F(MoveGenerationKernelTest, NoPiecesShouldGenerateNoMoves)
{
    checkers::GpuBoard board_;
    auto whiteResult = checkers::LaunchGpuMoveGen<checkers::Turn::kWhite>(std::vector<checkers::GpuBoard>(1, board_));
    EXPECT_FALSE(GlobalMoveFound(whiteResult[0], 0));
    EXPECT_FALSE(GlobalCaptureFound(whiteResult[0], 0));

    auto blackResult = checkers::LaunchGpuMoveGen<checkers::Turn::kBlack>(std::vector<checkers::GpuBoard>(1, board_));
    EXPECT_FALSE(GlobalMoveFound(blackResult[0], 0));
    EXPECT_FALSE(GlobalCaptureFound(blackResult[0], 0));
}

TEST_F(MoveGenerationKernelTest, SingleWhitePieceMoves)
{
    checkers::GpuBoard board_;
    board_.setPieceAt(12, 'W');

    auto result = checkers::LaunchGpuMoveGen<checkers::Turn::kWhite>(std::vector<checkers::GpuBoard>(1, board_));

    checkers::move_t move1 = checkers::EncodeMove(12, 8);
    checkers::move_t move2 = checkers::EncodeMove(12, 9);

    std::unordered_map<checkers::move_t, bool> expected = {
        {move1, false},
        {move2, false},
    };
    EXPECT_TRUE(FoundAllExpectedMoves(result[0], expected, 12));

    EXPECT_TRUE(GlobalMoveFound(result[0], 0));
    EXPECT_FALSE(GlobalCaptureFound(result[0], 0));
}

TEST_F(MoveGenerationKernelTest, SingleBlackPieceMoves)
{
    checkers::GpuBoard board_;
    board_.setPieceAt(5, 'B');

    auto result = checkers::LaunchGpuMoveGen<checkers::Turn::kBlack>(std::vector<checkers::GpuBoard>(1, board_));

    checkers::move_t move1 = checkers::EncodeMove(5, 9);
    checkers::move_t move2 = checkers::EncodeMove(5, 10);

    std::unordered_map<checkers::move_t, bool> expected = {
        {move1, false},
        {move2, false},
    };
    EXPECT_TRUE(FoundAllExpectedMoves(result[0], expected, 5));

    EXPECT_TRUE(GlobalMoveFound(result[0], 0));
    EXPECT_FALSE(GlobalCaptureFound(result[0], 0));
}

TEST_F(MoveGenerationKernelTest, WhitePieceCanCaptureBlackPiece)
{
    checkers::GpuBoard board_;
    board_.setPieceAt(13, 'W');
    board_.setPieceAt(9, 'B');

    auto result = checkers::LaunchGpuMoveGen<checkers::Turn::kWhite>(std::vector<checkers::GpuBoard>(1, board_));

    checkers::move_t captureMove = checkers::EncodeMove(13, 4);
    checkers::move_t normalMove  = checkers::EncodeMove(13, 10);

    std::unordered_map<checkers::move_t, bool> expected = {
        {captureMove,  true},
        { normalMove, false},
    };
    EXPECT_TRUE(FoundAllExpectedMoves(result[0], expected, 13));

    EXPECT_TRUE(GlobalMoveFound(result[0], 0));
    EXPECT_TRUE(GlobalCaptureFound(result[0], 0));
}

TEST_F(MoveGenerationKernelTest, KingPieceGeneratesDiagonalMoves)
{
    checkers::GpuBoard board_;
    board_.setPieceAt(12, 'W');
    board_.setPieceAt(12, 'K');

    auto result = checkers::LaunchGpuMoveGen<checkers::Turn::kWhite>(std::vector<checkers::GpuBoard>(1, board_));

    std::unordered_map<checkers::move_t, bool> expected;
    expected.emplace(checkers::EncodeMove(12, 8), false);
    expected.emplace(checkers::EncodeMove(12, 9), false);
    expected.emplace(checkers::EncodeMove(12, 5), false);
    expected.emplace(checkers::EncodeMove(12, 2), false);
    expected.emplace(checkers::EncodeMove(12, 16), false);
    expected.emplace(checkers::EncodeMove(12, 17), false);
    expected.emplace(checkers::EncodeMove(12, 21), false);
    expected.emplace(checkers::EncodeMove(12, 26), false);
    expected.emplace(checkers::EncodeMove(12, 30), false);

    EXPECT_TRUE(FoundAllExpectedMoves(result[0], expected, 12));

    EXPECT_TRUE(GlobalMoveFound(result[0], 0));
    EXPECT_FALSE(GlobalCaptureFound(result[0], 0));
}

TEST_F(MoveGenerationKernelTest, KingPieceMoveWithCapture)
{
    checkers::GpuBoard board_;
    board_.setPieceAt(12, 'W');
    board_.setPieceAt(12, 'K');
    board_.setPieceAt(9, 'B');

    auto result = checkers::LaunchGpuMoveGen<checkers::Turn::kWhite>(std::vector<checkers::GpuBoard>(1, board_));

    std::unordered_map<checkers::move_t, bool> expected;
    expected.emplace(checkers::EncodeMove(12, 8), false);
    expected.emplace(checkers::EncodeMove(12, 5), true);
    expected.emplace(checkers::EncodeMove(12, 2), true);
    expected.emplace(checkers::EncodeMove(12, 16), false);
    expected.emplace(checkers::EncodeMove(12, 17), false);
    expected.emplace(checkers::EncodeMove(12, 21), false);
    expected.emplace(checkers::EncodeMove(12, 26), false);
    expected.emplace(checkers::EncodeMove(12, 30), false);

    EXPECT_TRUE(FoundAllExpectedMoves(result[0], expected, 12));

    EXPECT_TRUE(GlobalMoveFound(result[0], 0));
    EXPECT_TRUE(GlobalCaptureFound(result[0], 0));
}

TEST_F(MoveGenerationKernelTest, KingPieceMoveBlockedByDifferentColor)
{
    checkers::GpuBoard board_;
    board_.setPieceAt(12, 'W');
    board_.setPieceAt(12, 'K');
    board_.setPieceAt(9, 'B');
    board_.setPieceAt(5, 'B');

    auto result = checkers::LaunchGpuMoveGen<checkers::Turn::kWhite>(std::vector<checkers::GpuBoard>(1, board_));

    std::unordered_map<checkers::move_t, bool> expected;
    expected.emplace(checkers::EncodeMove(12, 8), false);
    expected.emplace(checkers::EncodeMove(12, 16), false);
    expected.emplace(checkers::EncodeMove(12, 17), false);
    expected.emplace(checkers::EncodeMove(12, 21), false);
    expected.emplace(checkers::EncodeMove(12, 26), false);
    expected.emplace(checkers::EncodeMove(12, 30), false);
    EXPECT_TRUE(FoundAllExpectedMoves(result[0], expected, 12));

    EXPECT_TRUE(GlobalMoveFound(result[0], 0));
    EXPECT_FALSE(GlobalCaptureFound(result[0], 0));
}

TEST_F(MoveGenerationKernelTest, KingPieceMoveBlockedBySameColor)
{
    checkers::GpuBoard board_;
    board_.setPieceAt(12, 'W');
    board_.setPieceAt(12, 'K');
    board_.setPieceAt(9, 'W');

    auto result = checkers::LaunchGpuMoveGen<checkers::Turn::kWhite>(std::vector<checkers::GpuBoard>(1, board_));

    std::unordered_map<checkers::move_t, bool> expected_from_king;
    expected_from_king.emplace(checkers::EncodeMove(12, 8), false);
    expected_from_king.emplace(checkers::EncodeMove(12, 16), false);
    expected_from_king.emplace(checkers::EncodeMove(12, 17), false);
    expected_from_king.emplace(checkers::EncodeMove(12, 21), false);
    expected_from_king.emplace(checkers::EncodeMove(12, 26), false);
    expected_from_king.emplace(checkers::EncodeMove(12, 30), false);

    EXPECT_TRUE(FoundAllExpectedMoves(result[0], expected_from_king, 12));

    EXPECT_TRUE(GlobalMoveFound(result[0], 0));
    EXPECT_FALSE(GlobalCaptureFound(result[0], 0));
}

TEST_F(MoveGenerationKernelTest, WhitePieceBlockedBySameColorAdjacent)
{
    checkers::GpuBoard board_;
    board_.setPieceAt(12, 'W');
    board_.setPieceAt(8, 'W');

    auto result = checkers::LaunchGpuMoveGen<checkers::Turn::kWhite>(std::vector<checkers::GpuBoard>(1, board_));

    checkers::move_t move1 = checkers::EncodeMove(12, 9);
    checkers::move_t move2 = checkers::EncodeMove(8, 4);

    std::unordered_map<checkers::move_t, bool> expected_1 = {
        {move1, false},
    };
    std::unordered_map<checkers::move_t, bool> expected_2 = {
        {move2, false},
    };
    EXPECT_TRUE(FoundAllExpectedMoves(result[0], expected_1, 12));
    EXPECT_TRUE(FoundAllExpectedMoves(result[0], expected_2, 8));

    EXPECT_TRUE(GlobalMoveFound(result[0], 0));
    EXPECT_FALSE(GlobalCaptureFound(result[0], 0));
}

TEST_F(MoveGenerationKernelTest, BlackPieceMultipleCaptureScenario)
{
    checkers::GpuBoard board_;
    board_.setPieceAt(13, 'B');
    board_.setPieceAt(17, 'W');
    board_.setPieceAt(21, 'W');

    auto result = checkers::LaunchGpuMoveGen<checkers::Turn::kBlack>(std::vector<checkers::GpuBoard>(1, board_));

    checkers::move_t captureMove1 = checkers::EncodeMove(13, 20);
    checkers::move_t normalMove   = checkers::EncodeMove(13, 18);

    std::unordered_map<checkers::move_t, bool> expected = {
        {captureMove1,  true},
        {  normalMove, false},
    };
    EXPECT_TRUE(FoundAllExpectedMoves(result[0], expected, 13));

    EXPECT_TRUE(GlobalMoveFound(result[0], 0));
    EXPECT_TRUE(GlobalCaptureFound(result[0], 0));
}

TEST_F(MoveGenerationKernelTest, KingPieceBlockedBySameColorInAlmostAllDirections)
{
    checkers::GpuBoard board_;
    board_.setPieceAt(12, 'W');
    board_.setPieceAt(12, 'K');

    board_.setPieceAt(8, 'W');
    board_.setPieceAt(5, 'W');
    board_.setPieceAt(16, 'W');
    board_.setPieceAt(17, 'W');

    auto result = checkers::LaunchGpuMoveGen<checkers::Turn::kWhite>(std::vector<checkers::GpuBoard>(1, board_));

    std::unordered_map<checkers::move_t, bool> expected = {
        {checkers::EncodeMove(12, 9), false},
    };
    EXPECT_TRUE(FoundAllExpectedMoves(result[0], expected, 12));

    EXPECT_TRUE(GlobalMoveFound(result[0], 0));
    EXPECT_FALSE(GlobalCaptureFound(result[0], 0));
}

TEST_F(MoveGenerationKernelTest, DifficultBoard1)
{
    checkers::GpuBoard board_;
    board_.setPieceAt(31, 'W');
    board_.setPieceAt(21, 'W');
    board_.setPieceAt(16, 'W');

    board_.setPieceAt(12, 'B');
    board_.setPieceAt(9, 'B');
    board_.setPieceAt(10, 'B');

    board_.setPieceAt(9, 'K');

    auto result = checkers::LaunchGpuMoveGen<checkers::Turn::kBlack>(std::vector<checkers::GpuBoard>(1, board_));

    EXPECT_FALSE(GlobalCaptureFound(result[0], 0));
}
