#include <gtest/gtest.h>
#include <cmath>
#include "cpu/launchers.hpp"
#include "cuda/game_simulation.cuh"
#include "cuda/launchers.cuh"

namespace checkers::gpu::launchers
{

class GpuBoardTest : public ::testing::Test
{
    protected:
    GpuBoard board1;
    GpuBoard board2;

    void SetUp() override
    {
        board1.white = 0;
        board1.black = 0;
        board1.kings = 0;

        board2.white = 0;
        board2.black = 0;
        board2.kings = 0;
    }
};

// Test that two identical boards are equal
TEST_F(GpuBoardTest, BoardsAreEqual) { EXPECT_EQ(board1, board2); }

// Test that boards with different white pieces are not equal
TEST_F(GpuBoardTest, BoardsWithDifferentWhitePiecesAreNotEqual)
{
    board1.setPieceAt(5, 'W');
    EXPECT_NE(board1, board2);
}

// Test that boards with different black pieces are not equal
TEST_F(GpuBoardTest, BoardsWithDifferentBlackPiecesAreNotEqual)
{
    board2.setPieceAt(10, 'B');
    EXPECT_NE(board1, board2);
}

// Test that boards with different kings are not equal
TEST_F(GpuBoardTest, BoardsWithDifferentKingsAreNotEqual)
{
    board1.setPieceAt(15, 'K');
    EXPECT_NE(board1, board2);
}

// Test that boards with same pieces but different king statuses are not equal
TEST_F(GpuBoardTest, BoardsSamePiecesDifferentKingsAreNotEqual)
{
    board1.setPieceAt(20, 'W');
    board2.setPieceAt(20, 'W');
    board1.setPieceAt(20, 'K');
    EXPECT_NE(board1, board2);
}

class SimulationTest : public ::testing::Test
{
    protected:
    std::vector<SimulationParam> params;

    void SetUp() override
    {
        // Initialize a single board where White can immediately win
        GpuBoard board;
        // White has a king at position 5
        board.setPieceAt(5, 'W');
        board.setPieceAt(5, 'K');
        board.setPieceAt(9, 'B');

        // Add a SimulationParam for this board
        SimulationParam param;
        param.white         = board.white;
        param.black         = board.black;
        param.king          = board.kings;
        param.start_turn    = 0;  // White to move
        param.n_simulations = 100;

        params.push_back(param);
    }
};

// Test immediate win for White
TEST_F(SimulationTest, ImmediateWinWhite)
{
    const int max_iterations = 10;

    auto results = checkers::gpu::launchers::HostSimulateCheckersGames(params, max_iterations);

    // We expect a single result because we have only one SimulationParam
    ASSERT_EQ(results.size(), 1);
    // We expect a win ratio close to 1.0 (White should win almost every time)
    EXPECT_NEAR(results[0].score / results[0].n_simulations, 1.0, 0.1);
}

class ImmediateLossTest : public ::testing::Test
{
    protected:
    std::vector<SimulationParam> params;

    void SetUp() override
    {
        // Initialize a single board where Black has no pieces
        GpuBoard board;
        // White has a piece at position 0
        board.setPieceAt(0, 'W');
        // Black has no pieces

        // Add a SimulationParam for this board
        SimulationParam param;
        param.white         = board.white;
        param.black         = board.black;
        param.king          = board.kings;
        param.start_turn    = 1;  // Black to move (will immediately lose)
        param.n_simulations = 100;

        params.push_back(param);
    }
};

// Test immediate loss for Black (since Black has no pieces)
TEST_F(ImmediateLossTest, ImmediateLossBlack)
{
    const int max_iterations = 10;

    auto results = checkers::gpu::launchers::HostSimulateCheckersGames(params, max_iterations);

    // We expect a single result because we have only one SimulationParam
    ASSERT_EQ(results.size(), 1);
    // We expect a win ratio close to 0.0 (Black should lose every time)
    EXPECT_NEAR(results[0].score / results[0].n_simulations, 0.0, 0.1);
}

class DrawTest : public ::testing::Test
{
    protected:
    std::vector<SimulationParam> params;

    void SetUp() override
    {
        // Initialize a single board with alternating kings that cannot capture each other
        GpuBoard board;
        // White kings at positions 0, 2, 4
        board.setPieceAt(0, 'W');
        board.setPieceAt(0, 'K');
        board.setPieceAt(2, 'W');
        board.setPieceAt(2, 'K');
        board.setPieceAt(4, 'W');
        board.setPieceAt(4, 'K');
        // Black kings at positions 1, 3, 5
        board.setPieceAt(1, 'B');
        board.setPieceAt(1, 'K');
        board.setPieceAt(3, 'B');
        board.setPieceAt(3, 'K');
        board.setPieceAt(5, 'B');
        board.setPieceAt(5, 'K');

        // Add a SimulationParam for this board
        SimulationParam param;
        param.white         = board.white;
        param.black         = board.black;
        param.king          = board.kings;
        param.start_turn    = 0;  // White to move
        param.n_simulations = 100;

        params.push_back(param);
    }
};

// Test for Draw Outcome
TEST_F(DrawTest, DrawOutcome)
{
    const u8 max_iterations = 5;  // Low number to force a draw

    auto results = checkers::gpu::launchers::HostSimulateCheckersGames(params, max_iterations);

    // We expect a single result because we have only one SimulationParam
    ASSERT_EQ(results.size(), 1);
    // We expect a win ratio close to 0.5 (a draw)
    EXPECT_NEAR(results[0].score / results[0].n_simulations, 0.5, 0.1);
}

class WinRatioTest : public ::testing::Test

{
    protected:
    std::vector<SimulationParam> params;

    void SetUp() override
    {
        // Initialize multiple boards with the standard starting position
        // Standard Checkers initial setup: White pieces on the bottom three rows, Black pieces on the top three rows
        // Assuming positions 0-31 are arranged row-wise from top-left to bottom-right
        u8 turn = 0;
        for (size_t game = 0; game < 100; ++game) {  // Reduced to 100 for test speed
            GpuBoard board;
            // Place White pieces on rows 3, 4, 5 (indices 24-31)
            for (board_index_t i = 24; i < 32; ++i) {
                board.setPieceAt(i, 'W');
            }
            // Place Black pieces on rows 0, 1, 2 (indices 0-7)
            for (board_index_t i = 0; i < 8; ++i) {
                board.setPieceAt(i, 'B');
            }
            // No kings initially
            // Add a SimulationParam for this board
            SimulationParam param;
            param.white         = board.white;
            param.black         = board.black;
            param.king          = board.kings;
            param.start_turn    = turn;
            param.n_simulations = 10;  // 10 simulations per board
            params.push_back(param);

            // Alternate turns
            turn = 1 - turn;
        }
    }
};

// Statistical Test for Win Ratios

TEST_F(WinRatioTest, WinRatioWithinExpectedBounds)

{
    const u8 max_iterations = 150;

    auto outcomes     = checkers::gpu::launchers::HostSimulateCheckersGames(params, max_iterations);
    auto cpu_outcomes = checkers::cpu::launchers::HostSimulateCheckersGames(params, max_iterations);

    ASSERT_EQ(outcomes.size(), 100);      // 100 boards
    ASSERT_EQ(cpu_outcomes.size(), 100);  // 100 boards

    f64 total_score = 0;
    u64 total_games = 0;

    for (auto outcome : outcomes) {
        total_score += outcome.score;
        total_games += outcome.n_simulations;
    }

    const f64 win_ratio = (f64)total_score / total_games;
    std::cout << "GPU Win ratio: " << win_ratio << std::endl;

    f64 cpu_total_score = 0;
    u64 cpu_total_games = 0;

    for (auto outcome : cpu_outcomes) {
        cpu_total_score += outcome.score;
        cpu_total_games += outcome.n_simulations;
    }

    const f64 cpu_win_ratio = (f64)cpu_total_score / cpu_total_games;
    std::cout << "CPU Win ratio: " << cpu_win_ratio << std::endl;

    EXPECT_NEAR(win_ratio, 0.47f, 0.05f);      // Around 45% win ratio, accounting for draws
    EXPECT_NEAR(cpu_win_ratio, 0.47f, 0.05f);  // Around 45% win ratio, accounting for draws
}

}  // namespace checkers::gpu::launchers
