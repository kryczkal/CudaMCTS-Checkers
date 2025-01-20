#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "checkers_defines.hpp"
#include "cuda_utils.cuh"
#include "move_generation.cuh"

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
    move_flags_t h_global_flags;

    MoveGenResult()
        : h_moves(kTotalSquares * kMovesPerPiece, MoveConstants::kInvalidMove),
          h_move_counts(kTotalSquares, 0),
          h_capture_masks(kTotalSquares, 0),
          h_global_flags(0)
    {
    }
};

template <Turn turn>
MoveGenResult LaunchGpuMoveGen(const GpuBoard &board)
{
    using namespace checkers::gpu::move_gen;

    MoveGenResult result;

    // 1. Allocate device memory for 1 board:
    board_t *d_whites = nullptr;
    board_t *d_blacks = nullptr;
    board_t *d_kings  = nullptr;

    // We store 1 board, so we have space for 1 value each.
    CHECK_CUDA_ERROR(cudaMalloc(&d_whites, sizeof(board_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_blacks, sizeof(board_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_kings, sizeof(board_t)));

    // 2. Copy data to device:
    CHECK_CUDA_ERROR(cudaMemcpy(d_whites, &board.white, sizeof(board_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_blacks, &board.black, sizeof(board_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_kings, &board.kings, sizeof(board_t), cudaMemcpyHostToDevice));

    // 3. Allocate arrays for results:
    move_t *d_moves               = nullptr;
    u8 *d_move_counts             = nullptr;
    move_flags_t *d_capture_masks = nullptr;
    move_flags_t *d_global_flags  = nullptr;  // single global flag memory

    size_t totalMoves = MoveGenResult::kTotalSquares * MoveGenResult::kMovesPerPiece;
    CHECK_CUDA_ERROR(cudaMalloc(&d_moves, totalMoves * sizeof(move_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_move_counts, MoveGenResult::kTotalSquares * sizeof(u8)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_capture_masks, MoveGenResult::kTotalSquares * sizeof(move_flags_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_global_flags, sizeof(move_flags_t)));
    CHECK_CUDA_ERROR(cudaMemset(d_global_flags, 0, sizeof(move_flags_t)));

    // Initialize device buffers with invalid moves / zero
    {
        std::vector<move_t> initMoves(totalMoves, MoveConstants::kInvalidMove);
        cudaMemcpy(d_moves, initMoves.data(), totalMoves * sizeof(move_t), cudaMemcpyHostToDevice);
        CHECK_LAST_CUDA_ERROR();

        std::vector<u8> initCounts(MoveGenResult::kTotalSquares, 0);
        cudaMemcpy(d_move_counts, initCounts.data(), MoveGenResult::kTotalSquares * sizeof(u8), cudaMemcpyHostToDevice);
        CHECK_LAST_CUDA_ERROR();

        std::vector<move_flags_t> initMask(MoveGenResult::kTotalSquares, 0);
        cudaMemcpy(
            d_capture_masks, initMask.data(), MoveGenResult::kTotalSquares * sizeof(move_flags_t),
            cudaMemcpyHostToDevice
        );
        CHECK_LAST_CUDA_ERROR();

        move_flags_t zeroGlobal = 0;
        cudaMemcpy(d_global_flags, &zeroGlobal, sizeof(move_flags_t), cudaMemcpyHostToDevice);
        CHECK_LAST_CUDA_ERROR();
    }

    // 4. Launch kernel with 32 threads for 1 board:
    const int threadsPerBlock = 32;
    const int blocks          = 1;  // since we have only 1 board
    GenerateMoves<turn><<<blocks, threadsPerBlock>>>(
        d_whites, d_blacks, d_kings, d_moves, d_move_counts, d_capture_masks, d_global_flags,
        /* n_boards = */ 1
    );
    CHECK_LAST_CUDA_ERROR();
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    // 5. Copy results back to host:
    CHECK_CUDA_ERROR(cudaMemcpy(result.h_moves.data(), d_moves, totalMoves * sizeof(move_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(
        result.h_move_counts.data(), d_move_counts, MoveGenResult::kTotalSquares * sizeof(u8), cudaMemcpyDeviceToHost
    ));
    CHECK_CUDA_ERROR(cudaMemcpy(
        result.h_capture_masks.data(), d_capture_masks, MoveGenResult::kTotalSquares * sizeof(move_flags_t),
        cudaMemcpyDeviceToHost
    ));
    CHECK_CUDA_ERROR(cudaMemcpy(&result.h_global_flags, d_global_flags, sizeof(move_flags_t), cudaMemcpyDeviceToHost));
    // 6. Free device memory:
    CHECK_CUDA_ERROR(cudaFree(d_whites));
    CHECK_CUDA_ERROR(cudaFree(d_blacks));
    CHECK_CUDA_ERROR(cudaFree(d_kings));
    CHECK_CUDA_ERROR(cudaFree(d_moves));
    CHECK_CUDA_ERROR(cudaFree(d_move_counts));
    CHECK_CUDA_ERROR(cudaFree(d_capture_masks));
    CHECK_CUDA_ERROR(cudaFree(d_global_flags));
    return result;
}

}  // namespace checkers

class MoveGenerationKernelTest : public ::testing::Test
{
    protected:
    static bool GlobalCaptureFound(const checkers::MoveGenResult &result)
    {
        return ((result.h_global_flags >> checkers::MoveFlagsConstants::kCaptureFound) & 1) ==
               (checkers::move_flags_t)1;
    }

    static bool GlobalMoveFound(const checkers::MoveGenResult &result)
    {
        return ((result.h_global_flags >> checkers::MoveFlagsConstants::kMoveFound) & 1) == (checkers::move_flags_t)1;
    }

    static bool FoundAllExpectedMoves(
        const checkers::MoveGenResult &result, std::unordered_map<checkers::move_t, bool> &expected_moves,
        u64 move_index
    )
    {
        u64 current_index = move_index * kNumMaxMovesPerPiece;
        printf("Expected moves: %lu\n", expected_moves.size());
        printf("Result moves: %lu\n", result.h_move_counts[move_index]);
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
    auto whiteResult = checkers::LaunchGpuMoveGen<checkers::Turn::kWhite>(board_);
    EXPECT_FALSE(GlobalMoveFound(whiteResult));
    EXPECT_FALSE(GlobalCaptureFound(whiteResult));

    auto blackResult = checkers::LaunchGpuMoveGen<checkers::Turn::kBlack>(board_);
    EXPECT_FALSE(GlobalMoveFound(blackResult));
    EXPECT_FALSE(GlobalCaptureFound(blackResult));
}

TEST_F(MoveGenerationKernelTest, SingleWhitePieceMoves)
{
    checkers::GpuBoard board_;
    board_.setPieceAt(12, 'W');

    auto result = checkers::LaunchGpuMoveGen<checkers::Turn::kWhite>(board_);

    checkers::move_t move1 = checkers::EncodeMove(12, 8);
    checkers::move_t move2 = checkers::EncodeMove(12, 9);

    std::unordered_map<checkers::move_t, bool> expected = {
        {move1, false},
        {move2, false},
    };
    EXPECT_TRUE(FoundAllExpectedMoves(result, expected, 12));

    EXPECT_TRUE(GlobalMoveFound(result));
    EXPECT_FALSE(GlobalCaptureFound(result));
}

TEST_F(MoveGenerationKernelTest, SingleBlackPieceMoves)
{
    checkers::GpuBoard board_;
    board_.setPieceAt(5, 'B');

    auto result = checkers::LaunchGpuMoveGen<checkers::Turn::kBlack>(board_);

    checkers::move_t move1 = checkers::EncodeMove(5, 9);
    checkers::move_t move2 = checkers::EncodeMove(5, 10);

    std::unordered_map<checkers::move_t, bool> expected = {
        {move1, false},
        {move2, false},
    };
    EXPECT_TRUE(FoundAllExpectedMoves(result, expected, 5));

    EXPECT_TRUE(GlobalMoveFound(result));
    EXPECT_FALSE(GlobalCaptureFound(result));
}

TEST_F(MoveGenerationKernelTest, WhitePieceCanCaptureBlackPiece)
{
    checkers::GpuBoard board_;
    board_.setPieceAt(13, 'W');
    board_.setPieceAt(9, 'B');

    auto result = checkers::LaunchGpuMoveGen<checkers::Turn::kWhite>(board_);

    checkers::move_t captureMove = checkers::EncodeMove(13, 4);
    checkers::move_t normalMove  = checkers::EncodeMove(13, 10);

    std::unordered_map<checkers::move_t, bool> expected = {
        {captureMove,  true},
        { normalMove, false},
    };
    EXPECT_TRUE(FoundAllExpectedMoves(result, expected, 13));

    EXPECT_TRUE(GlobalMoveFound(result));
    EXPECT_TRUE(GlobalCaptureFound(result));
}

TEST_F(MoveGenerationKernelTest, KingPieceGeneratesDiagonalMoves)
{
    checkers::GpuBoard board_;
    board_.setPieceAt(12, 'W');
    board_.setPieceAt(12, 'K');

    auto result = checkers::LaunchGpuMoveGen<checkers::Turn::kWhite>(board_);

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

    EXPECT_TRUE(FoundAllExpectedMoves(result, expected, 12));

    EXPECT_TRUE(GlobalMoveFound(result));
    EXPECT_FALSE(GlobalCaptureFound(result));
}

TEST_F(MoveGenerationKernelTest, KingPieceMoveWithCapture)
{
    checkers::GpuBoard board_;
    board_.setPieceAt(12, 'W');
    board_.setPieceAt(12, 'K');
    board_.setPieceAt(9, 'B');

    auto result = checkers::LaunchGpuMoveGen<checkers::Turn::kWhite>(board_);

    std::unordered_map<checkers::move_t, bool> expected;
    expected.emplace(checkers::EncodeMove(12, 8), false);
    expected.emplace(checkers::EncodeMove(12, 5), true);
    expected.emplace(checkers::EncodeMove(12, 2), true);
    expected.emplace(checkers::EncodeMove(12, 16), false);
    expected.emplace(checkers::EncodeMove(12, 17), false);
    expected.emplace(checkers::EncodeMove(12, 21), false);
    expected.emplace(checkers::EncodeMove(12, 26), false);
    expected.emplace(checkers::EncodeMove(12, 30), false);

    EXPECT_TRUE(FoundAllExpectedMoves(result, expected, 12));

    EXPECT_TRUE(GlobalMoveFound(result));
    EXPECT_TRUE(GlobalCaptureFound(result));
}

TEST_F(MoveGenerationKernelTest, KingPieceMoveBlockedByDifferentColor)
{
    checkers::GpuBoard board_;
    board_.setPieceAt(12, 'W');
    board_.setPieceAt(12, 'K');
    board_.setPieceAt(9, 'B');
    board_.setPieceAt(5, 'B');

    auto result = checkers::LaunchGpuMoveGen<checkers::Turn::kWhite>(board_);

    std::unordered_map<checkers::move_t, bool> expected;
    expected.emplace(checkers::EncodeMove(12, 8), false);
    expected.emplace(checkers::EncodeMove(12, 16), false);
    expected.emplace(checkers::EncodeMove(12, 17), false);
    expected.emplace(checkers::EncodeMove(12, 21), false);
    expected.emplace(checkers::EncodeMove(12, 26), false);
    expected.emplace(checkers::EncodeMove(12, 30), false);
    EXPECT_TRUE(FoundAllExpectedMoves(result, expected, 12));

    EXPECT_TRUE(GlobalMoveFound(result));
    EXPECT_FALSE(GlobalCaptureFound(result));
}

TEST_F(MoveGenerationKernelTest, KingPieceMoveBlockedBySameColor)
{
    checkers::GpuBoard board_;
    board_.setPieceAt(12, 'W');
    board_.setPieceAt(12, 'K');
    board_.setPieceAt(9, 'W');

    auto result = checkers::LaunchGpuMoveGen<checkers::Turn::kWhite>(board_);

    std::unordered_map<checkers::move_t, bool> expected_from_king;
    expected_from_king.emplace(checkers::EncodeMove(12, 8), false);
    expected_from_king.emplace(checkers::EncodeMove(12, 16), false);
    expected_from_king.emplace(checkers::EncodeMove(12, 17), false);
    expected_from_king.emplace(checkers::EncodeMove(12, 21), false);
    expected_from_king.emplace(checkers::EncodeMove(12, 26), false);
    expected_from_king.emplace(checkers::EncodeMove(12, 30), false);

    EXPECT_TRUE(FoundAllExpectedMoves(result, expected_from_king, 12));

    EXPECT_TRUE(GlobalMoveFound(result));
    EXPECT_FALSE(GlobalCaptureFound(result));
}

TEST_F(MoveGenerationKernelTest, WhitePieceBlockedBySameColorAdjacent)
{
    checkers::GpuBoard board_;
    board_.setPieceAt(12, 'W');
    board_.setPieceAt(8, 'W');

    auto result = checkers::LaunchGpuMoveGen<checkers::Turn::kWhite>(board_);

    checkers::move_t move1 = checkers::EncodeMove(12, 9);
    checkers::move_t move2 = checkers::EncodeMove(8, 4);

    std::unordered_map<checkers::move_t, bool> expected_1 = {
        {move1, false},
    };
    std::unordered_map<checkers::move_t, bool> expected_2 = {
        {move2, false},
    };
    EXPECT_TRUE(FoundAllExpectedMoves(result, expected_1, 12));
    EXPECT_TRUE(FoundAllExpectedMoves(result, expected_2, 8));

    EXPECT_TRUE(GlobalMoveFound(result));
    EXPECT_FALSE(GlobalCaptureFound(result));
}

TEST_F(MoveGenerationKernelTest, BlackPieceMultipleCaptureScenario)
{
    checkers::GpuBoard board_;
    board_.setPieceAt(13, 'B');
    board_.setPieceAt(17, 'W');
    board_.setPieceAt(21, 'W');

    auto result = checkers::LaunchGpuMoveGen<checkers::Turn::kBlack>(board_);

    checkers::move_t captureMove1 = checkers::EncodeMove(13, 20);
    checkers::move_t normalMove   = checkers::EncodeMove(13, 18);

    std::unordered_map<checkers::move_t, bool> expected = {
        {captureMove1,  true},
        {  normalMove, false},
    };
    EXPECT_TRUE(FoundAllExpectedMoves(result, expected, 13));

    EXPECT_TRUE(GlobalMoveFound(result));
    EXPECT_TRUE(GlobalCaptureFound(result));
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

    auto result = checkers::LaunchGpuMoveGen<checkers::Turn::kWhite>(board_);

    std::unordered_map<checkers::move_t, bool> expected = {
        {checkers::EncodeMove(12, 9), false},
    };
    EXPECT_TRUE(FoundAllExpectedMoves(result, expected, 12));

    EXPECT_TRUE(GlobalMoveFound(result));
    EXPECT_FALSE(GlobalCaptureFound(result));
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

    auto result = checkers::LaunchGpuMoveGen<checkers::Turn::kBlack>(board_);

    EXPECT_FALSE(GlobalCaptureFound(result));
}
