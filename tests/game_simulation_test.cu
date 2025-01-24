#include <gtest/gtest.h>
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
        board.setPieceAt(idx, pieceType);
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
        board.setPieceAt(idx, pieceType);
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
