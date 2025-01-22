#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>

#include "checkers_defines.hpp"
#include "cpu/board_helpers.hpp"
#include "cuda/launchers.cuh"

namespace checkers::gpu::launchers
{

// A helper to see if a global capture/move was flagged
static bool GlobalCaptureFound(const MoveGenResult &result)
{
    return ((result.h_per_board_flags[0] >> move_gen::MoveFlagsConstants::kCaptureFound) & 1) == 1;
}

static bool GlobalMoveFound(const MoveGenResult &result)
{
    return ((result.h_per_board_flags[0] >> move_gen::MoveFlagsConstants::kMoveFound) & 1) == 1;
}

/**
 * @brief Verifies that the moves we expect were generated for a specific piece (squareIndex).
 *        The 'expected' map has move_t as keys and bool as the "isCapture" flag.
 */
static bool FoundAllExpectedMoves(
    const checkers::gpu::launchers::MoveGenResult &r, const std::unordered_map<checkers::move_t, bool> &expected_moves,
    size_t square_index
)
{
    const size_t offset   = square_index * MoveGenResult::kMovesPerPiece;
    const u8 actual_count = r.h_move_counts[square_index];

    if (actual_count != expected_moves.size())
        return false;

    for (u8 i = 0; i < actual_count; ++i) {
        checkers::move_t mv = r.h_moves[offset + i];
        auto it             = expected_moves.find(mv);
        if (it == expected_moves.end())
            return false;

        // Check if capture bit matches
        bool was_capture = ((r.h_capture_masks[square_index] >> i) & 1) == 1;
        if (was_capture != it->second)
            return false;
    }
    return true;
}

using namespace checkers;
using GpuBoard = checkers::gpu::launchers::GpuBoard;

TEST(GpuMoveGenerationTest, NoPiecesShouldGenerateNoMoves)
{
    GpuBoard board;
    std::vector<GpuBoard> boards{board};

    auto whiteRes = HostGenerateMoves(boards, Turn::kWhite);
    EXPECT_FALSE(GlobalMoveFound(whiteRes[0]));
    EXPECT_FALSE(GlobalCaptureFound(whiteRes[0]));

    auto blackRes = HostGenerateMoves(boards, Turn::kBlack);
    EXPECT_FALSE(GlobalMoveFound(blackRes[0]));
    EXPECT_FALSE(GlobalCaptureFound(blackRes[0]));
}

TEST(GpuMoveGenerationTest, SingleWhitePieceMoves)
{
    GpuBoard board;
    board.setPieceAt(12, 'W');
    std::vector<GpuBoard> boards{board};

    auto results           = HostGenerateMoves(boards, Turn::kWhite);
    const MoveGenResult &r = results[0];

    checkers::move_t m1                       = checkers::cpu::move_gen::EncodeMove(12, 8);
    checkers::move_t m2                       = checkers::cpu::move_gen::EncodeMove(12, 9);
    std::unordered_map<move_t, bool> expected = {
        {m1, false},
        {m2, false}
    };

    EXPECT_TRUE(FoundAllExpectedMoves(r, expected, 12));
    EXPECT_TRUE(GlobalMoveFound(r));
    EXPECT_FALSE(GlobalCaptureFound(r));
}

TEST(GpuMoveGenerationTest, SingleBlackPieceMoves)
{
    GpuBoard board;
    board.setPieceAt(5, 'B');
    std::vector<GpuBoard> boards{board};

    auto results           = HostGenerateMoves(boards, Turn::kBlack);
    const MoveGenResult &r = results[0];

    checkers::move_t m1                       = cpu::move_gen::EncodeMove(5, 9);
    checkers::move_t m2                       = cpu::move_gen::EncodeMove(5, 10);
    std::unordered_map<move_t, bool> expected = {
        {m1, false},
        {m2, false}
    };

    EXPECT_TRUE(FoundAllExpectedMoves(r, expected, 5));
    EXPECT_TRUE(GlobalMoveFound(r));
    EXPECT_FALSE(GlobalCaptureFound(r));
}

TEST(GpuMoveGenerationTest, WhitePieceCanCaptureBlackPiece)
{
    GpuBoard board;
    board.setPieceAt(13, 'W');
    board.setPieceAt(9, 'B');

    std::vector<GpuBoard> boards{board};
    auto results           = HostGenerateMoves(boards, Turn::kWhite);
    const MoveGenResult &r = results[0];

    checkers::move_t capMove                  = cpu::move_gen::EncodeMove(13, 4);
    checkers::move_t normMove                 = cpu::move_gen::EncodeMove(13, 10);
    std::unordered_map<move_t, bool> expected = {
        { capMove,  true},
        {normMove, false}
    };

    EXPECT_TRUE(FoundAllExpectedMoves(r, expected, 13));
    EXPECT_TRUE(GlobalMoveFound(r));
    EXPECT_TRUE(GlobalCaptureFound(r));
}

TEST(GpuMoveGenerationTest, KingPieceGeneratesDiagonalMoves)
{
    GpuBoard board;
    board.setPieceAt(12, 'W');
    board.setPieceAt(12, 'K');
    std::vector<GpuBoard> boards{board};

    auto results           = HostGenerateMoves(boards, Turn::kWhite);
    const MoveGenResult &r = results[0];

    std::unordered_map<move_t, bool> expected;
    expected.emplace(cpu::move_gen::EncodeMove(12, 8), false);
    expected.emplace(cpu::move_gen::EncodeMove(12, 9), false);
    expected.emplace(cpu::move_gen::EncodeMove(12, 5), false);
    expected.emplace(cpu::move_gen::EncodeMove(12, 2), false);
    expected.emplace(cpu::move_gen::EncodeMove(12, 16), false);
    expected.emplace(cpu::move_gen::EncodeMove(12, 17), false);
    expected.emplace(cpu::move_gen::EncodeMove(12, 21), false);
    expected.emplace(cpu::move_gen::EncodeMove(12, 26), false);
    expected.emplace(cpu::move_gen::EncodeMove(12, 30), false);

    EXPECT_TRUE(FoundAllExpectedMoves(r, expected, 12));
    EXPECT_TRUE(GlobalMoveFound(r));
    EXPECT_FALSE(GlobalCaptureFound(r));
}

TEST(GpuMoveGenerationTest, KingPieceMoveWithCapture)
{
    GpuBoard board;
    board.setPieceAt(12, 'W');
    board.setPieceAt(12, 'K');
    board.setPieceAt(9, 'B');

    std::vector<GpuBoard> boards{board};
    auto results           = HostGenerateMoves(boards, Turn::kWhite);
    const MoveGenResult &r = results[0];

    std::unordered_map<move_t, bool> expected;
    expected.emplace(cpu::move_gen::EncodeMove(12, 8), false);
    expected.emplace(cpu::move_gen::EncodeMove(12, 5), true);
    expected.emplace(cpu::move_gen::EncodeMove(12, 2), true);
    expected.emplace(cpu::move_gen::EncodeMove(12, 16), false);
    expected.emplace(cpu::move_gen::EncodeMove(12, 17), false);
    expected.emplace(cpu::move_gen::EncodeMove(12, 21), false);
    expected.emplace(cpu::move_gen::EncodeMove(12, 26), false);
    expected.emplace(cpu::move_gen::EncodeMove(12, 30), false);

    EXPECT_TRUE(FoundAllExpectedMoves(r, expected, 12));
    EXPECT_TRUE(GlobalMoveFound(r));
    EXPECT_TRUE(GlobalCaptureFound(r));
}

TEST(GpuMoveGenerationTest, KingPieceMoveBlockedByDifferentColor)
{
    GpuBoard board;
    board.setPieceAt(12, 'W');
    board.setPieceAt(12, 'K');
    board.setPieceAt(9, 'B');
    board.setPieceAt(5, 'B');

    std::vector<GpuBoard> boards{board};
    auto results           = HostGenerateMoves(boards, Turn::kWhite);
    const MoveGenResult &r = results[0];

    std::unordered_map<move_t, bool> expected;
    expected.emplace(cpu::move_gen::EncodeMove(12, 8), false);
    expected.emplace(cpu::move_gen::EncodeMove(12, 16), false);
    expected.emplace(cpu::move_gen::EncodeMove(12, 17), false);
    expected.emplace(cpu::move_gen::EncodeMove(12, 21), false);
    expected.emplace(cpu::move_gen::EncodeMove(12, 26), false);
    expected.emplace(cpu::move_gen::EncodeMove(12, 30), false);

    EXPECT_TRUE(FoundAllExpectedMoves(r, expected, 12));
    EXPECT_TRUE(GlobalMoveFound(r));
    EXPECT_FALSE(GlobalCaptureFound(r));
}

TEST(GpuMoveGenerationTest, KingPieceMoveBlockedBySameColor)
{
    GpuBoard board;
    board.setPieceAt(12, 'W');
    board.setPieceAt(12, 'K');
    board.setPieceAt(9, 'W');

    std::vector<GpuBoard> boards{board};
    auto results           = HostGenerateMoves(boards, Turn::kWhite);
    const MoveGenResult &r = results[0];

    std::unordered_map<move_t, bool> expected;
    expected.emplace(cpu::move_gen::EncodeMove(12, 8), false);
    expected.emplace(cpu::move_gen::EncodeMove(12, 16), false);
    expected.emplace(cpu::move_gen::EncodeMove(12, 17), false);
    expected.emplace(cpu::move_gen::EncodeMove(12, 21), false);
    expected.emplace(cpu::move_gen::EncodeMove(12, 26), false);
    expected.emplace(cpu::move_gen::EncodeMove(12, 30), false);

    EXPECT_TRUE(FoundAllExpectedMoves(r, expected, 12));
    EXPECT_TRUE(GlobalMoveFound(r));
    EXPECT_FALSE(GlobalCaptureFound(r));
}

TEST(GpuMoveGenerationTest, WhitePieceBlockedBySameColorAdjacent)
{
    GpuBoard board;
    board.setPieceAt(12, 'W');
    board.setPieceAt(8, 'W');
    std::vector<GpuBoard> boards{board};

    auto results           = HostGenerateMoves(boards, Turn::kWhite);
    const MoveGenResult &r = results[0];

    move_t m1                                 = cpu::move_gen::EncodeMove(12, 9);
    std::unordered_map<move_t, bool> expected = {
        {m1, false},
    };

    EXPECT_TRUE(FoundAllExpectedMoves(r, expected, 12));
    EXPECT_TRUE(GlobalMoveFound(r));
    EXPECT_FALSE(GlobalCaptureFound(r));
}

TEST(GpuMoveGenerationTest, BlackPieceMultipleCaptureScenario)
{
    GpuBoard board;
    board.setPieceAt(13, 'B');
    board.setPieceAt(17, 'W');
    board.setPieceAt(21, 'W');

    std::vector<GpuBoard> boards{board};
    auto results           = HostGenerateMoves(boards, Turn::kBlack);
    const MoveGenResult &r = results[0];

    move_t capMove1                           = cpu::move_gen::EncodeMove(13, 20);
    move_t normMove                           = cpu::move_gen::EncodeMove(13, 18);
    std::unordered_map<move_t, bool> expected = {
        {capMove1,  true},
        {normMove, false},
    };

    EXPECT_TRUE(FoundAllExpectedMoves(r, expected, 13));
    EXPECT_TRUE(GlobalMoveFound(r));
    EXPECT_TRUE(GlobalCaptureFound(r));
}

TEST(GpuMoveGenerationTest, KingPieceBlockedBySameColorInAlmostAllDirections)
{
    GpuBoard board;
    board.setPieceAt(12, 'W');
    board.setPieceAt(12, 'K');

    board.setPieceAt(8, 'W');
    board.setPieceAt(5, 'W');
    board.setPieceAt(16, 'W');
    board.setPieceAt(17, 'W');

    std::vector<GpuBoard> boards{board};
    auto results           = HostGenerateMoves(boards, Turn::kWhite);
    const MoveGenResult &r = results[0];

    std::unordered_map<move_t, bool> expected = {
        {cpu::move_gen::EncodeMove(12, 9), false},
    };

    EXPECT_TRUE(FoundAllExpectedMoves(r, expected, 12));
    EXPECT_TRUE(GlobalMoveFound(r));
    EXPECT_FALSE(GlobalCaptureFound(r));
}

TEST(GpuMoveGenerationTest, DifficultBoard1)
{
    GpuBoard board;
    board.setPieceAt(31, 'W');
    board.setPieceAt(21, 'W');
    board.setPieceAt(16, 'W');

    board.setPieceAt(12, 'B');
    board.setPieceAt(9, 'B');
    board.setPieceAt(10, 'B');

    board.setPieceAt(9, 'K');

    std::vector<GpuBoard> boards{board};
    auto results           = HostGenerateMoves(boards, Turn::kBlack);
    const MoveGenResult &r = results[0];

    EXPECT_FALSE(GlobalCaptureFound(r));
}

TEST(GpuMoveGenerationTest, ExceedThreadsAndMemory)
{
    // Steps:
    // 1. Query GPU properties to get maximum number of threads.
    // 2. Compute board_count so that board_count * board_size > max_threads.
    // 3. Create board_count copies of the same board.
    // 4. Run HostGenerateMoves.
    // 5. Sample some boards (left, right, middle).
    // 6. Verify the moves are generated correctly.

    // Get device properties
    cudaDeviceProp deviceProp;
    int device;
    cudaGetDevice(&device);
    CHECK_LAST_CUDA_ERROR();
    cudaGetDeviceProperties(&deviceProp, device);

    size_t maxTotalThreads = deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount;

    // Define board_count as (maxTotalThreads / board_size) +
    size_t exceed_factor = 30;
    size_t board_size    = BoardConstants::kBoardSize;
    size_t board_count   = (maxTotalThreads / board_size) + exceed_factor;

    // Create a sample board
    GpuBoard sampleBoard;
    sampleBoard.setPieceAt(12, 'W');
    sampleBoard.setPieceAt(17, 'B');
    sampleBoard.setPieceAt(20, 'B');
    sampleBoard.setPieceAt(22, 'B');

    // Create board_count copies of sampleBoard
    std::vector<GpuBoard> boards(board_count, sampleBoard);

    // Run HostGenerateMoves for White's turn
    auto results = HostGenerateMoves(boards, Turn::kWhite);

    // Now, sample some boards: first, middle, and last
    size_t indices_to_test[] = {0, board_count / 2, board_count - 1};

    for (size_t idx : indices_to_test) {
        const MoveGenResult &r = results[idx];
        // Define expected moves
        std::unordered_map<move_t, bool> expected_moves;
        expected_moves.emplace(cpu::move_gen::EncodeMove(12, 8), false);
        expected_moves.emplace(cpu::move_gen::EncodeMove(12, 9), false);
        expected_moves.emplace(cpu::move_gen::EncodeMove(12, 21), true);

        // Check if all expected moves are present
        bool all_found = FoundAllExpectedMoves(r, expected_moves, 12);
        EXPECT_TRUE(all_found) << "Failed on board index " << idx;

        // Verify global flags
        bool moveFound    = GlobalMoveFound(r);
        bool captureFound = GlobalCaptureFound(r);
        EXPECT_TRUE(moveFound) << "Move flag not set for board index " << idx;
        EXPECT_TRUE(captureFound) << "Capture flag not set for board index " << idx;
    }
}

}  // namespace checkers::gpu::launchers
