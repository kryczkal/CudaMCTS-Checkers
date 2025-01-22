#include <gtest/gtest.h>
#include <vector>

#include "checkers_defines.hpp"
#include "cpu/board_helpers.hpp"
#include "cuda/capture_lookup_table.cuh"
#include "cuda/launchers.cuh"

namespace checkers::gpu::launchers
{

TEST(GpuApplyMoveTest, NoBoardsNoMoves)
{
    // Expect an empty result when there are no boards
    std::vector<GpuBoard> boards;
    std::vector<move_t> moves;

    auto updated = HostApplyMoves(boards, moves);
    EXPECT_EQ(updated.size(), 0u);
}

TEST(GpuApplyMoveTest, SingleBoardNoMove)
{
    // One board, but we pass in an invalid move => no effect
    GpuBoard board;
    board.white = 0;
    board.black = 0;
    board.kings = 0;

    std::vector<GpuBoard> boards{board};
    std::vector<move_t> moves{move_gen::MoveConstants::kInvalidMove};

    auto updated = HostApplyMoves(boards, moves);
    ASSERT_EQ(updated.size(), 1u);

    EXPECT_EQ(updated[0].white, 0u);
    EXPECT_EQ(updated[0].black, 0u);
    EXPECT_EQ(updated[0].kings, 0u);
}

TEST(GpuApplyMoveTest, SinglePieceSimpleMove)
{
    // White piece at index 12 moves to index 8
    GpuBoard board;
    board.white = (1u << 12);
    board.black = 0u;
    board.kings = 0u;

    std::vector<GpuBoard> boards{board};
    move_t move = static_cast<move_t>(12 | (8u << 8));  // from=12, to=8
    std::vector<move_t> moves{move};

    auto updated = HostApplyMoves(boards, moves);
    ASSERT_EQ(updated.size(), 1u);

    // After applying the move, the piece should be at index 8
    EXPECT_EQ(updated[0].white, (1u << 8));
    EXPECT_EQ(updated[0].black, 0u);
    EXPECT_EQ(updated[0].kings, 0u);
}

TEST(GpuApplyMoveTest, CaptureRemovesEnemyPiece)
{
    // White piece at 13, black piece at 9, capturing from 13 -> 4
    GpuBoard board;
    board.white = (1u << 13);
    board.black = (1u << 9);
    board.kings = 0u;

    std::vector<GpuBoard> boards{board};
    move_t captureMove = static_cast<move_t>(13 | (4u << 8));
    std::vector<move_t> moves{captureMove};

    auto updated = HostApplyMoves(boards, moves);
    ASSERT_EQ(updated.size(), 1u);

    // The white piece ends at 4, black piece at 9 is removed
    EXPECT_EQ(updated[0].white, (1u << 4));
    EXPECT_EQ(updated[0].black, 0u);
    EXPECT_EQ(updated[0].kings, 0u);
}

TEST(GpuApplyMoveTest, MoveKingPreservesKingFlag)
{
    // White piece at 20 is also a king
    GpuBoard board;
    board.white = (1u << 20);
    board.black = 0u;
    board.kings = (1u << 20);

    std::vector<GpuBoard> boards{board};
    move_t move = static_cast<move_t>(20 | (16u << 8));
    std::vector<move_t> moves{move};

    auto updated = HostApplyMoves(boards, moves);
    ASSERT_EQ(updated.size(), 1u);

    EXPECT_EQ(updated[0].white, (1u << 16));
    EXPECT_EQ(updated[0].black, 0u);
    EXPECT_EQ(updated[0].kings, (1u << 16));
}

/**
 * @brief Test that a king can move far diagonally
 */
TEST(GpuApplyMoveTest, KingFarMove)
{
    // Initialize a king at position 12, no blocking pieces
    GpuBoard board;
    board.setPieceAt(12, 'W');
    board.setPieceAt(12, 'K');  // Make it a king

    std::vector<GpuBoard> boards{board};
    move_t move = cpu::move_gen::EncodeMove(12, 30);  // Move from 12 to 30 (far move)
    std::vector<move_t> moves{move};

    auto updated = HostApplyMoves(boards, moves);
    ASSERT_EQ(updated.size(), 1u);

    // After applying the move, the king should be at 30
    EXPECT_EQ(updated[0].white, (1u << 30));
    EXPECT_EQ(updated[0].black, 0u);
    EXPECT_EQ(updated[0].kings, (1u << 30));
}

/**
 * @brief Test that a king can perform a far capture over an enemy piece.
 */
TEST(GpuApplyMoveTest, KingFarCapture)
{
    GpuBoard board;
    board.setPieceAt(12, 'W');
    board.setPieceAt(12, 'K');
    board.setPieceAt(26, 'B');

    std::vector<GpuBoard> boards{board};
    move_t captureMove = cpu::move_gen::EncodeMove(12, 30);
    std::vector<move_t> moves{captureMove};

    auto updated = HostApplyMoves(boards, moves);
    ASSERT_EQ(updated.size(), 1u);

    EXPECT_EQ(updated[0].white, (1u << 30));
    EXPECT_EQ(updated[0].black, 0u);
    EXPECT_EQ(updated[0].kings, (1u << 30));
}

/**
 * @brief Test that capturing a piece does not remove other enemy pieces.
 */
TEST(GpuApplyMoveTest, CaptureDoesNotRemoveUncapturedEnemyPieces1)
{
    GpuBoard board;
    board.setPieceAt(12, 'W');
    board.setPieceAt(12, 'K');
    board.setPieceAt(21, 'B');
    board.setPieceAt(23, 'B');

    std::vector<GpuBoard> boards{board};
    move_t captureMove = cpu::move_gen::EncodeMove(12, 26);
    std::vector<move_t> moves{captureMove};

    auto updated = HostApplyMoves(boards, moves);
    ASSERT_EQ(updated.size(), 1u);

    EXPECT_EQ(updated[0].white, (1u << 26));
    EXPECT_EQ(updated[0].black, (1u << 23));
    EXPECT_EQ(updated[0].kings, (1u << 26));
}

/**
 * @brief Another test to ensure that capturing a piece does not remove other enemy pieces.
 */
TEST(GpuApplyMoveTest, CaptureDoesNotRemoveUncapturedEnemyPieces2)
{
    GpuBoard board;
    board.setPieceAt(14, 'W');
    board.setPieceAt(14, 'K');
    board.setPieceAt(25, 'B');
    board.setPieceAt(4, 'B');
    board.setPieceAt(10, 'B');

    std::vector<GpuBoard> boards{board};
    move_t captureMove = cpu::move_gen::EncodeMove(14, 28);
    std::vector<move_t> moves{captureMove};

    auto updated = HostApplyMoves(boards, moves);
    ASSERT_EQ(updated.size(), 1u);

    EXPECT_EQ(updated[0].white, (1u << 28));
    EXPECT_EQ(updated[0].black, (1u << 4) | (1u << 10));
    EXPECT_EQ(updated[0].kings, (1u << 28));
}

}  // namespace checkers::gpu::launchers
