#include <gtest/gtest.h>
#include <vector>

#include "cuda/capture_lookup_table.cuh"
#include "cuda/launchers.cuh"

namespace checkers::gpu::launchers
{

// Outcome encoding in scores[]:
// 1 = White wins
// 2 = Black wins
// 3 = Draw

/**
 * @brief Helper function to create a GpuBoard with specified pieces.
 *
 * @param white_positions Vector of board indices for white pieces.
 * @param black_positions Vector of board indices for black pieces.
 * @param king_positions Vector of board indices that should be kings.
 * @return GpuBoard with the specified configuration.
 */
GpuBoard CreateBoard(
    const std::vector<board_index_t>& white_positions, const std::vector<board_index_t>& black_positions,
    const std::vector<board_index_t>& king_positions = {}
)
{
    GpuBoard board;
    for (auto pos : white_positions) {
        board.white |= (1u << pos);
    }
    for (auto pos : black_positions) {
        board.black |= (1u << pos);
    }
    for (auto pos : king_positions) {
        board.kings |= (1u << pos);
    }
    return board;
}

/**
 * @brief Helper function to run simulation and verify outcome.
 *
 * @param boards Vector of GpuBoard configurations.
 * @param seeds Vector of seeds for random selection.
 * @param max_iterations Maximum half-moves to simulate.
 * @param expected_outcomes Vector of expected outcomes corresponding to each board.
 */
void VerifyOutcome(
    const std::vector<GpuBoard>& boards, const std::vector<u8>& seeds, int max_iterations,
    const std::vector<u8>& expected_outcomes
)
{
    // Convert GpuBoard to separate vectors
    std::vector<board_t> h_whites, h_blacks, h_kings;
    for (const auto& board : boards) {
        h_whites.push_back(board.white);
        h_blacks.push_back(board.black);
        h_kings.push_back(board.kings);
    }

    // Ensure seeds vector matches board count
    ASSERT_EQ(seeds.size(), boards.size());

    // Run simulation
    std::vector<u8> results = HostSimulateCheckersGames(h_whites, h_blacks, h_kings, seeds, max_iterations);

    // Verify results
    ASSERT_EQ(results.size(), expected_outcomes.size());
    for (size_t i = 0; i < results.size(); ++i) {
        EXPECT_EQ(results[i], expected_outcomes[i]) << "Mismatch at board index " << i;
    }
}

// Initialize the Capture Lookup Table before running tests
class GameSimulationTest : public ::testing::Test
{
    protected:
    static void SetUpTestSuite() { checkers::gpu::apply_move::InitializeCaptureLookupTable(); }
};

// Test Case 1: White Can Force a Win by Capturing All Black Pieces
TEST_F(GameSimulationTest, WhiteWinsByCapturingAllBlacks)
{
    // Setup board: White has two pieces that can capture Black's single piece
    // White positions: 12, 16
    // Black positions: 21
    // No kings
    GpuBoard board               = CreateBoard({12, 16}, {21});
    std::vector<GpuBoard> boards = {board};

    // Seed chosen to pick specific moves if multiple options exist
    std::vector<u8> seeds = {42};  // Arbitrary seed

    // Expected outcome: White wins
    std::vector<u8> expected = {1};

    // Run simulation with sufficient iterations
    VerifyOutcome(boards, seeds, 100, expected);
}

// Test Case 2: Black Can Force a Win by Capturing All White Pieces
TEST_F(GameSimulationTest, BlackWinsByCapturingAllWhites)
{
    // Setup board: Black has two pieces that can capture White's single piece
    // Black positions: 9, 13
    // White positions: 4
    // No kings
    GpuBoard board               = CreateBoard({4}, {9, 13});
    std::vector<GpuBoard> boards = {board};

    // Seed chosen to pick specific moves
    std::vector<u8> seeds = {24};  // Arbitrary seed

    // Expected outcome: Black wins
    std::vector<u8> expected = {2};

    // Run simulation with sufficient iterations
    VerifyOutcome(boards, seeds, 100, expected);
}

// Test Case 3: Draw by 40-Move Non-Reversible Rule
TEST_F(GameSimulationTest, DrawByFortyNonReversibleMoves)
{
    // Setup board: Both players have pieces that cannot force a capture or promotion
    // This setup should reach 40 non-reversible moves and result in a draw
    // White positions: 12
    // Black positions: 13
    // No kings
    GpuBoard board               = CreateBoard({12}, {13});
    std::vector<GpuBoard> boards = {board};

    // Seed chosen to alternate moves without captures or promotions
    std::vector<u8> seeds = {0};  // Seed=0

    // Expected outcome: Draw
    std::vector<u8> expected = {3};

    // Run simulation with 40 non-reversible moves
    VerifyOutcome(boards, seeds, 80, expected);  // 80 half-moves = 40 full moves
}

// Test Case 4: Chain Capture by a Single Piece
TEST_F(GameSimulationTest, ChainCaptureSinglePiece)
{
    // Setup board: White can perform multiple captures in a chain
    // White positions: 12
    // Black positions: 17, 21, 25
    // No kings
    GpuBoard board               = CreateBoard({12}, {17, 21, 25});
    std::vector<GpuBoard> boards = {board};

    // Seed chosen to ensure captures are taken in sequence
    std::vector<u8> seeds = {10};  // Arbitrary seed

    // Expected outcome: White captures all Blacks and wins
    std::vector<u8> expected = {1};

    // Run simulation with sufficient iterations
    VerifyOutcome(boards, seeds, 100, expected);
}

// Test Case 5: Promotion to King and Subsequent Moves
TEST_F(GameSimulationTest, PromotionToKing)
{
    // Setup board: White piece at position 28 can be promoted to a king
    // After promotion, it can move backward
    // Black positions: 23
    // No initial kings
    GpuBoard board               = CreateBoard({28}, {23});
    std::vector<GpuBoard> boards = {board};

    // Seed chosen to promote and then capture
    std::vector<u8> seeds = {100};  // Arbitrary seed

    // Expected outcome: White promotes to king and captures Black, winning the game
    std::vector<u8> expected = {1};

    // Run simulation with sufficient iterations
    VerifyOutcome(boards, seeds, 100, expected);
}

// Test Case 6: No Available Moves for White (Black Wins)
TEST_F(GameSimulationTest, WhiteHasNoAvailableMoves)
{
    // Setup board: White has a single piece blocked by Black
    // White positions: 12
    // Black positions: 8, 16
    // No kings
    GpuBoard board               = CreateBoard({12}, {8, 16});
    std::vector<GpuBoard> boards = {board};

    // Seed chosen to have White attempt to move but has no legal moves
    std::vector<u8> seeds = {55};  // Arbitrary seed

    // Expected outcome: Black wins
    std::vector<u8> expected = {2};

    // Run simulation with sufficient iterations
    VerifyOutcome(boards, seeds, 100, expected);
}

// Test Case 7: Multiple Boards Simulated in Parallel
TEST_F(GameSimulationTest, MultipleBoardsParallelSimulation)
{
    // Setup multiple boards with varying outcomes
    std::vector<GpuBoard> boards;
    std::vector<u8> seeds;
    std::vector<u8> expected;

    // Board 0: White wins by capturing all Blacks
    boards.push_back(CreateBoard({12, 16}, {21}));
    seeds.push_back(42);
    expected.push_back(1);

    // Board 1: Black wins by capturing all Whites
    boards.push_back(CreateBoard({4}, {9, 13}));
    seeds.push_back(24);
    expected.push_back(2);

    // Board 2: Draw by 40-move rule
    boards.push_back(CreateBoard({12}, {13}));
    seeds.push_back(0);
    expected.push_back(3);

    // Board 3: White performs chain captures
    boards.push_back(CreateBoard({12}, {17, 21, 25}));
    seeds.push_back(10);
    expected.push_back(1);

    // Board 4: Promotion to King and capture
    boards.push_back(CreateBoard({28}, {23}));
    seeds.push_back(100);
    expected.push_back(1);

    // Board 5: White has no available moves
    boards.push_back(CreateBoard({12}, {8, 16}));
    seeds.push_back(55);
    expected.push_back(2);

    // Run simulation with sufficient iterations
    VerifyOutcome(boards, seeds, 100, expected);
}

}  // namespace checkers::gpu::launchers
