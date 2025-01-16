// tests/checkers_engine_memory_test.cpp

#include <gtest/gtest.h>
#include <checkers_engine.hpp>
#include <iostream>
#include <sstream>

namespace CudaMctsCheckers
{

class CheckersEngineMemoryTest : public ::testing::Test
{
    protected:
    CheckersEngine engine_;

    void SetUp() override
    {
        // Initialize the engine with the standard starting position
        Board initial_board;
        // Initialize standard checkers layout
        for (Board::IndexType i = 0; i < 12; ++i) {
            initial_board.SetPieceAt<BoardCheckType::kBlack>(i);
        }
        for (Board::IndexType i = 20; i < 32; ++i) {
            initial_board.SetPieceAt<BoardCheckType::kWhite>(i);
        }
        engine_ = CheckersEngine(initial_board, Turn::kWhite);
    }
};

/**
 * @brief Test restoring game state from an in-memory move list.
 */
TEST_F(CheckersEngineMemoryTest, RestoreFromMoveList_Success)
{
    std::string error_message;
    bool success =
        engine_.RestoreFromHistoryFile("./game_histories/crash-game-1.txt", error_message);
    EXPECT_TRUE(success) << "RestoreFromMoveList should succeed, but failed with error: "
                         << error_message;

    // Verify the final board state
    const Board &board = engine_.GetBoard();
    std::cout << board << std::endl;

    EXPECT_TRUE(board.IsPieceAt<BoardCheckType::kWhite>(9));
    EXPECT_FALSE(board.IsPieceAt<BoardCheckType::kBlack>(13));
    EXPECT_FALSE(board.IsPieceAt<BoardCheckType::kWhite>(21));
    EXPECT_TRUE(board.IsPieceAt<BoardCheckType::kWhite>(16));
    EXPECT_FALSE(board.IsPieceAt<BoardCheckType::kBlack>(10));
    EXPECT_FALSE(board.IsPieceAt<BoardCheckType::kBlack>(13));
    EXPECT_FALSE(board.IsPieceAt<BoardCheckType::kWhite>(17));
    EXPECT_TRUE(board.IsPieceAt<BoardCheckType::kBlack>(6));
    EXPECT_TRUE(board.IsPieceAt<BoardCheckType::kBlack>(14));

    // Verify the current turn is Black (since the last move was a capture, but in our engine
    // implementation, captures do not allow for multi-captures, and the turn is switched
    // after each move, so it should now be Black's turn)
    EXPECT_EQ(engine_.GetCurrentTurn(), Turn::kBlack);
}

}  // namespace CudaMctsCheckers
