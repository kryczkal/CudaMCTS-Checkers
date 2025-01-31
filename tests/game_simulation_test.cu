#include <gtest/gtest.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "common/checkers_defines.hpp"
#include "cpu/launchers.hpp"
#include "cuda/game_simulation.cuh"
#include "cuda/launchers.cuh"

namespace
{

/**
 * CPU Implementation for simulation.
 */
struct CPUSimImpl {
    // For the "BoardEqualityTest"
    using BoardType = checkers::cpu::Board;

    static BoardType MakeBoard() { return BoardType{}; }
    static void SetPiece(BoardType& board, checkers::board_index_t idx, char pieceType)
    {
        board.SetPieceAt(idx, pieceType);
    }

    // For the "SimulationTest"
    static std::vector<SimulationResult> SimulateCheckersGames(
        const std::vector<checkers::SimulationParam>& params, int maxIterations
    )
    {
        return checkers::cpu::launchers::HostSimulateCheckersGames(params, maxIterations);
    }
};

/**
 * GPU Implementation for simulation.
 */
struct GPUSimImpl {
    // For the "BoardEqualityTest"
    using BoardType = checkers::gpu::launchers::GpuBoard;

    static BoardType MakeBoard() { return BoardType{}; }
    static void SetPiece(BoardType& board, checkers::board_index_t idx, char pieceType)
    {
        board.SetPieceAt(idx, pieceType);
    }

    // For the "SimulationTest"
    static std::vector<SimulationResult> SimulateCheckersGames(
        const std::vector<checkers::SimulationParam>& params, int maxIterations
    )
    {
        return checkers::gpu::launchers::HostSimulateCheckersGames(params, maxIterations);
    }
};

}  // namespace

//////////////////////////////////////////////////////////////////////////////////
//                        1) Board Equality Typed Test                          //
//////////////////////////////////////////////////////////////////////////////////

template <typename Impl>
class BoardEqualityTest : public ::testing::Test
{
};

using BoardEqualityImplementations = ::testing::Types<CPUSimImpl, GPUSimImpl>;
TYPED_TEST_SUITE(BoardEqualityTest, BoardEqualityImplementations);

TYPED_TEST(BoardEqualityTest, BoardsAreEqual)
{
    using BoardType = typename TypeParam::BoardType;

    BoardType board1 = TypeParam::MakeBoard();
    BoardType board2 = TypeParam::MakeBoard();
    EXPECT_EQ(board1, board2);
}

TYPED_TEST(BoardEqualityTest, BoardsWithDifferentWhitePiecesAreNotEqual)
{
    using BoardType = typename TypeParam::BoardType;

    BoardType board1 = TypeParam::MakeBoard();
    BoardType board2 = TypeParam::MakeBoard();
    TypeParam::SetPiece(board1, 5, 'W');  // different white piece
    EXPECT_NE(board1, board2);
}

TYPED_TEST(BoardEqualityTest, BoardsWithDifferentBlackPiecesAreNotEqual)
{
    using BoardType = typename TypeParam::BoardType;

    BoardType board1 = TypeParam::MakeBoard();
    BoardType board2 = TypeParam::MakeBoard();
    TypeParam::SetPiece(board2, 10, 'B');
    EXPECT_NE(board1, board2);
}

TYPED_TEST(BoardEqualityTest, BoardsWithDifferentKingsAreNotEqual)
{
    using BoardType = typename TypeParam::BoardType;

    BoardType board1 = TypeParam::MakeBoard();
    BoardType board2 = TypeParam::MakeBoard();
    TypeParam::SetPiece(board1, 15, 'K');
    EXPECT_NE(board1, board2);
}

TYPED_TEST(BoardEqualityTest, BoardsSamePiecesDifferentKingsAreNotEqual)
{
    using BoardType = typename TypeParam::BoardType;

    BoardType board1 = TypeParam::MakeBoard();
    BoardType board2 = TypeParam::MakeBoard();

    // same piece
    TypeParam::SetPiece(board1, 20, 'W');
    TypeParam::SetPiece(board2, 20, 'W');

    // but board1 has king at 20, board2 doesn't
    TypeParam::SetPiece(board1, 20, 'K');

    EXPECT_NE(board1, board2);
}

//////////////////////////////////////////////////////////////////////////////////
//                      2) Simulation Outcome Typed Test                        //
//////////////////////////////////////////////////////////////////////////////////

template <typename Impl>
class SimulationTest : public ::testing::Test
{
    protected:
    std::vector<checkers::SimulationParam> params;

    // We'll populate "params" in different tests manually,
    // or right in the tests as needed. This fixture
    // simply provides an optional container.
};

using SimulationImplementations = ::testing::Types<CPUSimImpl, GPUSimImpl>;
TYPED_TEST_SUITE(SimulationTest, SimulationImplementations);

/**
 * Recreates the immediate-white-win scenario test:
 *   White has a king that can immediately capture Black.
 */
TYPED_TEST(SimulationTest, ImmediateWinWhite)
{
    // White king at 5, black piece at 9 => immediate capture
    using ImplType = TypeParam;

    checkers::gpu::launchers::GpuBoard dummy;  // Not used, but ensures compile
    // Build the param
    checkers::SimulationParam param{};
    param.white = 0;
    param.black = 0;
    param.king  = 0;

    // We place white king at 5, black piece at 9
    // The function SetPiece is from TypeParam's BoardType,
    // but SimulationParam is direct bitmasks. We'll do it directly:
    param.white |= (1u << 5);
    param.king |= (1u << 5);
    param.black |= (1u << 9);

    // White to move
    param.start_turn    = 0;
    param.n_simulations = 100;

    std::vector<checkers::SimulationParam> singleParam{param};
    int max_iterations = 10;

    auto results = ImplType::SimulateCheckersGames(singleParam, max_iterations);

    ASSERT_EQ(results.size(), 1u);
    // White should almost always succeed in capturing quickly
    double fractionWhiteWins = results[0].score / results[0].n_simulations;
    EXPECT_NEAR(fractionWhiteWins, 1.0, 0.1);
}

/**
 * Recreates the immediate-black-loss scenario:
 *   Black to move, but has no pieces => White automatically wins.
 */
TYPED_TEST(SimulationTest, ImmediateWinBlack)
{
    using ImplType = TypeParam;

    checkers::SimulationParam param{};
    // White piece at 0
    param.black |= (1u << 5);
    param.black |= (1u << 5);
    param.white |= (1u << 9);
    // black has no pieces
    param.start_turn    = 1;  // black to move
    param.n_simulations = 100;

    std::vector<checkers::SimulationParam> singleParam{param};
    int max_iterations = 10;

    auto results = ImplType::SimulateCheckersGames(singleParam, max_iterations);
    ASSERT_EQ(results.size(), 1u);

    double fraction_black_wins = results[0].score / results[0].n_simulations;
    EXPECT_NEAR(fraction_black_wins, 1.0, 0.1);
}

/**
 * Recreates a draw scenario:
 *   Positions that can't capture each other, forced to draw by iteration limit.
 */
TYPED_TEST(SimulationTest, DrawOutcome)
{
    using ImplType = TypeParam;

    checkers::SimulationParam param{};
    // White kings at 0,2,4
    param.white |= (1u << 0);
    param.white |= (1u << 2);
    param.white |= (1u << 4);
    param.king |= (1u << 0);
    param.king |= (1u << 2);
    param.king |= (1u << 4);

    // Black kings at 1,3,5
    param.black |= (1u << 1);
    param.black |= (1u << 3);
    param.black |= (1u << 5);
    param.king |= (1u << 1);
    param.king |= (1u << 3);
    param.king |= (1u << 5);

    // White to move
    param.start_turn    = 0;
    param.n_simulations = 100;

    std::vector<checkers::SimulationParam> singleParam{param};

    // Force low iteration => likely draw
    int max_iterations = 5;
    auto results       = ImplType::SimulateCheckersGames(singleParam, max_iterations);
    ASSERT_EQ(results.size(), 1u);

    // 0.5 => means (whiteScore / total) ~ 0.5 => draws
    double fractionWhiteWins = results[0].score / results[0].n_simulations;
    EXPECT_NEAR(fractionWhiteWins, 0.5, 0.1);
}

/**
 * Recreates a scenario with multiple standard boards to check overall ratio.
 */
TYPED_TEST(SimulationTest, WinRatioWithinExpectedBounds)
{
    using ImplType = TypeParam;

    std::vector<checkers::SimulationParam> params;
    params.reserve(100);

    // We'll do 100 boards with standard initial setup
    // plus random turn assignments
    for (size_t i = 0; i < 100; ++i) {
        checkers::SimulationParam sp{};
        // White on last 3 rows, black on first 3 rows in 4x8 layout
        for (checkers::board_index_t idx = 24; idx < 32; ++idx) {
            sp.white |= (1u << idx);
        }
        for (checkers::board_index_t idx = 0; idx < 8; ++idx) {
            sp.black |= (1u << idx);
        }
        // Randomly pick start turn
        sp.start_turn    = (i % 2 == 0) ? 0 : 1;
        sp.n_simulations = 100;

        params.push_back(sp);
    }

    int max_iterations = 150;
    auto outcomes      = ImplType::SimulateCheckersGames(params, max_iterations);

    ASSERT_EQ(outcomes.size(), 100u);

    // Summation
    double totalScore = 0;
    double totalGames = 0;
    for (auto& out : outcomes) {
        totalScore += out.score;
        totalGames += out.n_simulations;
    }

    double fractionWhiteWins = totalScore / totalGames;
    std::cout << "[SimulationTest] fractionWhiteWins = " << fractionWhiteWins << std::endl;

    // Typically near ~0.45-0.55 in random playouts
    EXPECT_NEAR(fractionWhiteWins, 0.50, 0.05);
}

TYPED_TEST(SimulationTest, NonZeroScore1)
{
    using ImplType = TypeParam;
    std::vector<checkers::SimulationParam> params;
    params.reserve(1);
    params.push_back(checkers::SimulationParam{});
    params[0].white         = 4292874240;
    params[0].black         = 19455;
    params[0].king          = 0;
    params[0].start_turn    = 1;
    params[0].n_simulations = 12500;
    int max_iterations      = 150;
    auto outcomes           = ImplType::SimulateCheckersGames(params, max_iterations);

    ASSERT_EQ(outcomes.size(), 1u);
    ASSERT_NE(outcomes[0].score, 0);

    std::cout << "[SimulationTest:NonZeroScore1] win ratio = " << outcomes[0].score / outcomes[0].n_simulations
              << std::endl;
}

//////////////////////////////////////////////////////////////////////////////////
//        3) Decreasing One Side's Pieces => Increasing Opponent's Win Ratio     //
//////////////////////////////////////////////////////////////////////////////////

TYPED_TEST(SimulationTest, DecreasingPieceCountIncreasesWinRatio)
{
    using ImplType = TypeParam;

    // We'll create three scenarios:
    //   Scenario A: White has 8 pieces, Black has 8 pieces
    //   Scenario B: White has 8 pieces, Black has 4 pieces
    //   Scenario C: White has 8 pieces, Black has 2 pieces
    //
    // In all scenarios, White moves first. With more simulations, we expect
    // White's win rate to increase as Black's piece count decreases.

    std::vector<checkers::SimulationParam> params;
    params.reserve(3);

    auto MakeParam = [&](int white_count, int black_count) {
        checkers::SimulationParam p{};
        // Place "whiteCount" pieces for White in the top half of the 8 indexes [24..31]
        // Place "blackCount" pieces for Black in the bottom half of the 8 indexes [0..7]
        // We use simple loops to set bits for the correct number of squares.

        // White squares [24..31]
        for (checkers::board_index_t idx = 24; idx < 24 + white_count; ++idx) {
            p.white |= (1u << idx);
        }
        // Black squares [0..(blackCount-1)]
        for (checkers::board_index_t idx = 0; idx < static_cast<checkers::board_index_t>(black_count); ++idx) {
            p.black |= (1u << idx);
        }

        // White to move
        p.start_turn = 0;
        // Use a larger simulation count for stability
        p.n_simulations = 1000;
        return p;
    };

    // Build scenarios
    params.push_back(MakeParam(8, 8));
    params.push_back(MakeParam(8, 7));
    params.push_back(MakeParam(8, 6));
    params.push_back(MakeParam(8, 5));
    params.push_back(MakeParam(8, 4));
    params.push_back(MakeParam(8, 3));
    params.push_back(MakeParam(8, 2));

    // We'll allow a moderate iteration limit so the game can progress.
    int max_iterations = 200;
    auto results       = ImplType::SimulateCheckersGames(params, max_iterations);
    ASSERT_EQ(results.size(), 7u);

    // Fraction of wins for White in each scenario
    std::vector<double> fractions;
    fractions.reserve(7);

    for (auto& out : results) {
        double fraction_white_wins = out.score / out.n_simulations;
        fractions.push_back(fraction_white_wins);
        std::cout << "WhiteWinFraction = " << fraction_white_wins << std::endl;
    }

    // Expect that scenario B > scenario A, and scenario C > scenario B
    // (i.e., decreasing Black's pieces should raise White's win fraction).
    for (int i = 1; i < fractions.size(); i++) {
        EXPECT_GT(fractions[i], fractions[i - 1]);
    }
}

//////////////////////////////////////////////////////////////////////////////////
//               4) Three Different Advantage Setups => Clear Favorite          //
//////////////////////////////////////////////////////////////////////////////////

TYPED_TEST(SimulationTest, AdvantageousSetups)
{
    using ImplType = TypeParam;

    // We'll build three scenarios:
    // 1) White has 8 pieces vs Black has 2 pieces. White to move => White favored.
    // 2) White has 2 pieces vs Black has 8 pieces. Black to move => Black favored.
    // 3) White has 4 kings vs Black has 4 normal pieces. White to move => White favored.
    //
    // We'll run each scenario with enough simulations and ensure
    // the favored side's winning fraction is clearly above/below a threshold.

    std::vector<checkers::SimulationParam> params;
    params.reserve(3);

    {
        // Scenario 1: White favored
        checkers::SimulationParam s1{};
        // White squares [24..31], but we only place 8 from [24..31].
        for (checkers::board_index_t idx = 24; idx < 32; ++idx) {
            s1.white |= (1u << idx);
        }
        // Black squares [0..1]
        s1.black |= (1u << 0);
        s1.black |= (1u << 1);

        // White moves first
        s1.start_turn = 0;
        // More simulations to reduce randomness
        s1.n_simulations = 1000;
        params.push_back(s1);
    }

    {
        // Scenario 2: Black favored
        checkers::SimulationParam s2{};
        // White squares [24..25], only 2 white pieces
        s2.white |= (1u << 24);
        s2.white |= (1u << 25);

        // Black squares [0..7]
        for (checkers::board_index_t idx = 0; idx < 8; ++idx) {
            s2.black |= (1u << idx);
        }
        // Black moves first
        s2.start_turn    = 1;
        s2.n_simulations = 1000;
        params.push_back(s2);
    }

    {
        // Scenario 3: White has 4 kings vs Black has 4 normal pieces.
        checkers::SimulationParam s3{};
        // White squares [24..27], set them as kings as well
        for (checkers::board_index_t idx = 28; idx < 32; ++idx) {
            s3.white |= (1u << idx);
            s3.king |= (1u << idx);  // Make them kings
        }
        // Black squares [0..3], normal pieces
        for (checkers::board_index_t idx = 0; idx < 3; ++idx) {
            s3.black |= (1u << idx);
        }

        // White moves first
        s3.start_turn    = 0;
        s3.n_simulations = 1000;
        params.push_back(s3);
    }

    int max_iterations = 200;
    auto results       = ImplType::SimulateCheckersGames(params, max_iterations);
    ASSERT_EQ(results.size(), 3u);

    // Calculate White's fraction of wins in each scenario
    double fractionScenario1 = results[0].score / results[0].n_simulations;
    double fractionScenario2 = results[1].score / results[1].n_simulations;
    double fractionScenario3 = results[2].score / results[2].n_simulations;

    std::cout << "[AdvantageousSetups] Scenario1(White favored) WhiteWinFrac  = " << fractionScenario1 << std::endl;
    std::cout << "[AdvantageousSetups] Scenario2(Black favored) WhiteWinFrac  = " << fractionScenario2 << std::endl;
    std::cout << "[AdvantageousSetups] Scenario3(White favored Kings) WhiteWinFrac = " << fractionScenario3
              << std::endl;

    EXPECT_GT(fractionScenario1, 0.7) << "White should be strongly favored with 8 vs 2 pieces";
    EXPECT_GT(fractionScenario2, 0.7) << "Black should dominate with 8 vs 2 pieces and the first move";
    EXPECT_GT(fractionScenario3, 0.7) << "White with 4 kings vs 4 normal black pieces should have a strong edge";
}
