#include <gtest/gtest.h>
#include <vector>

#include "common/checkers_defines.hpp"
#include "cpu/board_helpers.hpp"
#include "cpu/launchers.hpp"
#include "cuda/capture_lookup_table.cuh"
#include "cuda/launchers.cuh"

namespace
{

/**
 * CPU implementation wrapper
 */
struct CPUImpl {
    using BoardType = checkers::cpu::Board;

    static std::vector<BoardType> ApplyMoves(
        const std::vector<BoardType>& boards, const std::vector<checkers::move_t>& moves
    )
    {
        return checkers::cpu::launchers::HostApplyMoves(boards, moves);
    }

    static BoardType MakeBoard() { return BoardType{}; }

    static void SetPiece(BoardType& board, checkers::board_index_t idx, char pieceType)
    {
        board.setPieceAt(idx, pieceType);
    }
};

/**
 * GPU implementation wrapper
 */
struct GPUImpl {
    using BoardType = checkers::gpu::launchers::GpuBoard;

    static std::vector<BoardType> ApplyMoves(
        const std::vector<BoardType>& boards, const std::vector<checkers::move_t>& moves
    )
    {
        return checkers::gpu::launchers::HostApplyMoves(boards, moves);
    }

    static BoardType MakeBoard() { return BoardType{}; }

    static void SetPiece(BoardType& board, checkers::board_index_t idx, char pieceType)
    {
        board.setPieceAt(idx, pieceType);
    }
};

}  // namespace

//////////////////////////////////////////////////////////////////////////////////
//                           Typed Test Fixture                                 //
//////////////////////////////////////////////////////////////////////////////////

template <typename Impl>
class ApplyMoveTest : public ::testing::Test
{
};

using ApplyMoveImplementations = ::testing::Types<CPUImpl, GPUImpl>;
TYPED_TEST_SUITE(ApplyMoveTest, ApplyMoveImplementations);

TYPED_TEST(ApplyMoveTest, NoBoardsNoMoves)
{
    using BoardType = typename TypeParam::BoardType;

    std::vector<BoardType> boards;
    std::vector<checkers::move_t> moves;

    auto updated = TypeParam::ApplyMoves(boards, moves);
    EXPECT_EQ(updated.size(), 0u);
}

TYPED_TEST(ApplyMoveTest, SingleBoardNoMove)
{
    using BoardType = typename TypeParam::BoardType;

    // One board, but we pass in an invalid move => no effect
    BoardType board = TypeParam::MakeBoard();

    std::vector<BoardType> boards{board};
    std::vector<checkers::move_t> moves{checkers::kInvalidMove};

    auto updated = TypeParam::ApplyMoves(boards, moves);
    ASSERT_EQ(updated.size(), 1u);

    EXPECT_EQ(updated[0].white, 0u);
    EXPECT_EQ(updated[0].black, 0u);
    EXPECT_EQ(updated[0].kings, 0u);
}

TYPED_TEST(ApplyMoveTest, SinglePieceSimpleMove)
{
    using BoardType = typename TypeParam::BoardType;

    // White piece at index 12 moves to index 8
    BoardType board = TypeParam::MakeBoard();
    TypeParam::SetPiece(board, 12, 'W');

    std::vector<BoardType> boards{board};
    checkers::move_t move = static_cast<checkers::move_t>(12 | (8u << 8));
    std::vector<checkers::move_t> moves{move};

    auto updated = TypeParam::ApplyMoves(boards, moves);
    ASSERT_EQ(updated.size(), 1u);

    // After applying the move, the piece should be at index 8
    EXPECT_EQ(updated[0].white, (1u << 8));
    EXPECT_EQ(updated[0].black, 0u);
    EXPECT_EQ(updated[0].kings, 0u);
}

TYPED_TEST(ApplyMoveTest, CaptureRemovesEnemyPiece)
{
    using BoardType = typename TypeParam::BoardType;

    // White piece at 13, black piece at 9, capturing from 13 -> 4
    BoardType board = TypeParam::MakeBoard();
    TypeParam::SetPiece(board, 13, 'W');
    TypeParam::SetPiece(board, 9, 'B');

    std::vector<BoardType> boards{board};
    checkers::move_t captureMove = static_cast<checkers::move_t>(13 | (4u << 8));
    std::vector<checkers::move_t> moves{captureMove};

    auto updated = TypeParam::ApplyMoves(boards, moves);
    ASSERT_EQ(updated.size(), 1u);

    // The white piece ends at 4, black piece at 9 is removed
    EXPECT_EQ(updated[0].white, (1u << 4));
    EXPECT_EQ(updated[0].black, 0u);
    EXPECT_EQ(updated[0].kings, 0u);
}

TYPED_TEST(ApplyMoveTest, MoveKingPreservesKingFlag)
{
    using BoardType = typename TypeParam::BoardType;

    // White piece at 20 is also a king
    BoardType board = TypeParam::MakeBoard();
    TypeParam::SetPiece(board, 20, 'W');
    TypeParam::SetPiece(board, 20, 'K');

    std::vector<BoardType> boards{board};
    checkers::move_t move = static_cast<checkers::move_t>(20 | (16u << 8));
    std::vector<checkers::move_t> moves{move};

    auto updated = TypeParam::ApplyMoves(boards, moves);
    ASSERT_EQ(updated.size(), 1u);

    EXPECT_EQ(updated[0].white, (1u << 16));
    EXPECT_EQ(updated[0].black, 0u);
    EXPECT_EQ(updated[0].kings, (1u << 16));
}

TYPED_TEST(ApplyMoveTest, KingFarMove)
{
    using BoardType = typename TypeParam::BoardType;

    // Initialize a king at position 12, no blocking pieces
    BoardType board = TypeParam::MakeBoard();
    TypeParam::SetPiece(board, 12, 'W');
    TypeParam::SetPiece(board, 12, 'K');

    std::vector<BoardType> boards{board};
    checkers::move_t move = checkers::cpu::move_gen::EncodeMove(12, 30);
    std::vector<checkers::move_t> moves{move};

    auto updated = TypeParam::ApplyMoves(boards, moves);
    ASSERT_EQ(updated.size(), 1u);

    // After applying the move, the king should be at 30
    EXPECT_EQ(updated[0].white, (1u << 30));
    EXPECT_EQ(updated[0].black, 0u);
    EXPECT_EQ(updated[0].kings, (1u << 30));
}

TYPED_TEST(ApplyMoveTest, KingFarCapture)
{
    using BoardType = typename TypeParam::BoardType;

    BoardType board = TypeParam::MakeBoard();
    TypeParam::SetPiece(board, 12, 'W');
    TypeParam::SetPiece(board, 12, 'K');
    TypeParam::SetPiece(board, 26, 'B');

    std::vector<BoardType> boards{board};
    checkers::move_t captureMove = checkers::cpu::move_gen::EncodeMove(12, 30);
    std::vector<checkers::move_t> moves{captureMove};

    auto updated = TypeParam::ApplyMoves(boards, moves);
    ASSERT_EQ(updated.size(), 1u);

    EXPECT_EQ(updated[0].white, (1u << 30));
    EXPECT_EQ(updated[0].black, 0u);
    EXPECT_EQ(updated[0].kings, (1u << 30));
}

TYPED_TEST(ApplyMoveTest, CaptureDoesNotRemoveUncapturedEnemyPieces1)
{
    using BoardType = typename TypeParam::BoardType;

    BoardType board = TypeParam::MakeBoard();
    TypeParam::SetPiece(board, 12, 'W');
    TypeParam::SetPiece(board, 12, 'K');
    TypeParam::SetPiece(board, 21, 'B');
    TypeParam::SetPiece(board, 23, 'B');

    std::vector<BoardType> boards{board};
    checkers::move_t captureMove = checkers::cpu::move_gen::EncodeMove(12, 26);
    std::vector<checkers::move_t> moves{captureMove};

    auto updated = TypeParam::ApplyMoves(boards, moves);
    ASSERT_EQ(updated.size(), 1u);

    EXPECT_EQ(updated[0].white, (1u << 26));
    EXPECT_EQ(updated[0].black, (1u << 23));
    EXPECT_EQ(updated[0].kings, (1u << 26));
}

TYPED_TEST(ApplyMoveTest, CaptureDoesNotRemoveUncapturedEnemyPieces2)
{
    using BoardType = typename TypeParam::BoardType;

    BoardType board = TypeParam::MakeBoard();
    TypeParam::SetPiece(board, 14, 'W');
    TypeParam::SetPiece(board, 14, 'K');
    TypeParam::SetPiece(board, 25, 'B');
    TypeParam::SetPiece(board, 4, 'B');
    TypeParam::SetPiece(board, 10, 'B');

    std::vector<BoardType> boards{board};
    checkers::move_t captureMove = checkers::cpu::move_gen::EncodeMove(14, 28);
    std::vector<checkers::move_t> moves{captureMove};

    auto updated = TypeParam::ApplyMoves(boards, moves);
    ASSERT_EQ(updated.size(), 1u);

    EXPECT_EQ(updated[0].white, (1u << 28));
    EXPECT_EQ(updated[0].black, ((1u << 4) | (1u << 10)));
    EXPECT_EQ(updated[0].kings, (1u << 28));
}
