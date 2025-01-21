#include <gtest/gtest.h>
#include <vector>

#include "cpu/board_helpers.hpp"
#include "cuda/launchers.cuh"

using namespace checkers;
using namespace checkers::gpu::launchers;

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
    std::vector<move_t> moves{MoveConstants::kInvalidMove};

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
