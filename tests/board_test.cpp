// tests/board_test.cpp

#include <gtest/gtest.h>
#include <board.hpp>
#include <move_direction.hpp>

namespace CudaMctsCheckers
{

//------------------------------------------------------------------------------//
//                                Test Fixture                                  //
//------------------------------------------------------------------------------//

    class BoardTest : public ::testing::Test
    {
    protected:
        Board board_;

        void SetUp() override
        {
            // Initialize the board to empty:
            // (All bits off for white_pieces, black_pieces, and kings)
            board_.white_pieces = 0;
            board_.black_pieces = 0;
            board_.kings        = 0;
        }
    };

//------------------------------------------------------------------------------//
//                                Utility Tests                                 //
//------------------------------------------------------------------------------//

    TEST_F(BoardTest, ParityOffsetEven)
    {
        i8 offset = Board::ParityOffset(RowParity::kEven);
        EXPECT_EQ(offset, -1)
                            << "ParityOffset(RowParity::kEven) should return -1.";
    }

    TEST_F(BoardTest, ParityOffsetOdd)
    {
        Board::IndexType offset = Board::ParityOffset(RowParity::kOdd);
        EXPECT_EQ(offset, 0)
                            << "ParityOffset(RowParity::kOdd) should return 0.";
    }

    TEST_F(BoardTest, GetOppositeTypeWhite)
    {
        constexpr auto opposite = Board::GetOppositeType<BoardCheckType::kWhite>();
        EXPECT_EQ(opposite, BoardCheckType::kBlack)
                            << "Opposite of White should be Black.";
    }

    TEST_F(BoardTest, GetOppositeTypeBlack)
    {
        constexpr auto opposite = Board::GetOppositeType<BoardCheckType::kBlack>();
        EXPECT_EQ(opposite, BoardCheckType::kWhite)
                            << "Opposite of Black should be White.";
    }

//------------------------------------------------------------------------------//
//                         Piece Setting, Checking, Unsetting                   //
//------------------------------------------------------------------------------//

    TEST_F(BoardTest, IsPieceAt_WhenEmpty_ShouldBeFalse)
    {
        // Initially, board is empty (no bits set).
        EXPECT_FALSE(board_.IsPieceAt<BoardCheckType::kWhite>(0));
        EXPECT_FALSE(board_.IsPieceAt<BoardCheckType::kBlack>(0));
        EXPECT_FALSE(board_.IsPieceAt<BoardCheckType::kKings>(0));
        EXPECT_FALSE(board_.IsPieceAt<BoardCheckType::kAll>(0));
    }

    TEST_F(BoardTest, SetPieceAt_White)
    {
        // Set a white piece at index 0
        board_.SetPieceAt<BoardCheckType::kWhite>(0);

        // Check if the bit is set
        EXPECT_TRUE(board_.IsPieceAt<BoardCheckType::kWhite>(0));
        // Should not affect black or kings
        EXPECT_FALSE(board_.IsPieceAt<BoardCheckType::kBlack>(0));
        EXPECT_FALSE(board_.IsPieceAt<BoardCheckType::kKings>(0));
        // "All" should see it
        EXPECT_TRUE(board_.IsPieceAt<BoardCheckType::kAll>(0));
    }

    TEST_F(BoardTest, UnsetPieceAt_White)
    {
        // First set the piece
        board_.SetPieceAt<BoardCheckType::kWhite>(5);
        EXPECT_TRUE(board_.IsPieceAt<BoardCheckType::kWhite>(5));

        // Now unset the piece
        board_.UnsetPieceAt<BoardCheckType::kWhite>(5);
        EXPECT_FALSE(board_.IsPieceAt<BoardCheckType::kWhite>(5));
        EXPECT_FALSE(board_.IsPieceAt<BoardCheckType::kAll>(5));
    }

//------------------------------------------------------------------------------//
//                             Moving Pieces                                    //
//------------------------------------------------------------------------------//

    TEST_F(BoardTest, MovePiece_White)
    {
        // Place a White piece at index 2, move it to 7
        board_.SetPieceAt<BoardCheckType::kWhite>(2);
        EXPECT_TRUE(board_.IsPieceAt<BoardCheckType::kWhite>(2));

        board_.MovePiece<BoardCheckType::kWhite>(2, 7);

        // old location should be unset, new location should be set
        EXPECT_FALSE(board_.IsPieceAt<BoardCheckType::kWhite>(2));
        EXPECT_TRUE(board_.IsPieceAt<BoardCheckType::kWhite>(7));
    }

//------------------------------------------------------------------------------//
//                            Edge Checking                                     //
//------------------------------------------------------------------------------//

    TEST_F(BoardTest, IsAtLeftEdge)
    {
        // Indices that are multiples of Board::kEdgeLength = 8
        // 0, 8, 16, 24, 32, etc., should be the left edge
        EXPECT_TRUE(Board::IsAtLeftEdge(0));
        EXPECT_TRUE(Board::IsAtLeftEdge(8));
        EXPECT_TRUE(Board::IsAtLeftEdge(16));
        EXPECT_TRUE(Board::IsAtLeftEdge(24));

        // 1, 9, 15... should NOT be left edges
        EXPECT_FALSE(Board::IsAtLeftEdge(1));
        EXPECT_FALSE(Board::IsAtLeftEdge(9));
    }

    TEST_F(BoardTest, IsAtRightEdge)
    {
        // Indices that are (kEdgeLength - 1), 7, 15, 23, 31... are right edges
        EXPECT_TRUE(Board::IsAtRightEdge(7));
        EXPECT_TRUE(Board::IsAtRightEdge(15));
        EXPECT_TRUE(Board::IsAtRightEdge(23));
        EXPECT_TRUE(Board::IsAtRightEdge(31));

        // 6, 8, 16 are not the right edge
        EXPECT_FALSE(Board::IsAtRightEdge(6));
        EXPECT_FALSE(Board::IsAtRightEdge(8));
    }

    TEST_F(BoardTest, InvalidateOutBoundsIndex)
    {
        // Indices >= kHalfBoardSize (32) become kInvalidIndex
        EXPECT_EQ(Board::InvalidateOutBoundsIndex(32), Board::kInvalidIndex);
        EXPECT_EQ(Board::InvalidateOutBoundsIndex(33), Board::kInvalidIndex);

        // In-range index stays the same
        EXPECT_EQ(Board::InvalidateOutBoundsIndex(31), 31);
        EXPECT_EQ(Board::InvalidateOutBoundsIndex(0), 0);
    }

//------------------------------------------------------------------------------//
//                              Row Parity                                      //
//------------------------------------------------------------------------------//

    TEST_F(BoardTest, GetRowParity_Examples)
    {
        EXPECT_EQ(Board::GetRowParity(0), RowParity::kEven);  // row=0, col=0
        EXPECT_EQ(Board::GetRowParity(1), RowParity::kEven);  // row=0, col=2
        EXPECT_EQ(Board::GetRowParity(2), RowParity::kEven);  // row=0, col=4
        EXPECT_EQ(Board::GetRowParity(3), RowParity::kEven);  // row=0, col=6

        EXPECT_EQ(Board::GetRowParity(4), RowParity::kOdd);   // row=1, col=1
        EXPECT_EQ(Board::GetRowParity(5), RowParity::kOdd);   // row=1, col=3
        EXPECT_EQ(Board::GetRowParity(6), RowParity::kOdd);   // row=1, col=5
        EXPECT_EQ(Board::GetRowParity(7), RowParity::kOdd);   // row=1, col=7

        EXPECT_EQ(Board::GetRowParity(8), RowParity::kEven);  // row=2, col=0
        EXPECT_EQ(Board::GetRowParity(9), RowParity::kEven);  // row=2, col=2
        EXPECT_EQ(Board::GetRowParity(10), RowParity::kEven); // row=2, col=4
        EXPECT_EQ(Board::GetRowParity(11), RowParity::kEven); // row=2, col=6

        EXPECT_EQ(Board::GetRowParity(12), RowParity::kOdd);  // row=3, col=1
        EXPECT_EQ(Board::GetRowParity(13), RowParity::kOdd);  // row=3, col=3
        EXPECT_EQ(Board::GetRowParity(14), RowParity::kOdd);  // row=3, col=5
        EXPECT_EQ(Board::GetRowParity(15), RowParity::kOdd);  // row=3, col=7

    }

//------------------------------------------------------------------------------//
//                           PieceReachedEnd                                    //
//------------------------------------------------------------------------------//

    TEST_F(BoardTest, PieceReachedEnd_White)
    {
        // Indices < 4 => top row in checkers sense => true
        for (Board::IndexType i = 0; i < 4; ++i) {
            EXPECT_TRUE(board_.PieceReachedEnd<BoardCheckType::kWhite>(i));
        }

        // Otherwise => false
        EXPECT_FALSE(board_.PieceReachedEnd<BoardCheckType::kWhite>(4));
        EXPECT_FALSE(board_.PieceReachedEnd<BoardCheckType::kWhite>(31));
    }

    TEST_F(BoardTest, PieceReachedEnd_Black)
    {
        // >= 32 - 4 = 28 => "top" region for black
        for (Board::IndexType i = 28; i < 32; ++i) {
            EXPECT_TRUE(board_.PieceReachedEnd<BoardCheckType::kBlack>(i));
        }

        EXPECT_FALSE(board_.PieceReachedEnd<BoardCheckType::kBlack>(27));
        EXPECT_FALSE(board_.PieceReachedEnd<BoardCheckType::kBlack>(0));
    }

//------------------------------------------------------------------------------//
//                          Relative Move Index                                 //
//------------------------------------------------------------------------------//

    TEST_F(BoardTest, GetRelativeMoveIndex_UpLeft)
    {
        Board::IndexType from = 9;
        Board::IndexType up_left = board_.GetRelativeMoveIndex<MoveDirection::kUpLeft>(from);
        EXPECT_EQ(up_left, 4);
        EXPECT_NE(up_left, Board::kInvalidIndex);
    }

    TEST_F(BoardTest, GetRelativeMoveIndex_UpLeft_WhenAtLeftEdge)
    {
        // from is at left edge => we expect invalid index
        Board::IndexType from = 8; // left edge
        Board::IndexType up_left = board_.GetRelativeMoveIndex<MoveDirection::kUpLeft>(from);

        EXPECT_EQ(up_left, Board::kInvalidIndex);
    }

    TEST_F(BoardTest, GetRelativeMoveIndex_UpRight_WhenAtRightEdge)
    {
        // from is at right edge => we expect invalid index
        Board::IndexType from = 15; // right edge
        Board::IndexType up_right = board_.GetRelativeMoveIndex<MoveDirection::kUpRight>(from);

        EXPECT_EQ(up_right, Board::kInvalidIndex);
    }

    TEST_F(BoardTest, GetRelativeMoveIndex_DownLeft)
    {
        Board::IndexType from = 9;
        Board::IndexType down_left = board_.GetRelativeMoveIndex<MoveDirection::kDownLeft>(from);
        EXPECT_EQ(down_left, 12);
        EXPECT_NE(down_left, Board::kInvalidIndex);
    }

    TEST_F(BoardTest, GetRelativeMoveIndex_DownRight)
    {
        Board::IndexType from = 9;
        Board::IndexType down_right = board_.GetRelativeMoveIndex<MoveDirection::kDownRight>(from);
        EXPECT_EQ(down_right, 13);
        EXPECT_NE(down_right, Board::kInvalidIndex);
    }

    TEST_F(BoardTest, GetRelativeMoveIndex_UpRight)
    {
        Board::IndexType from = 9;
        Board::IndexType up_right = board_.GetRelativeMoveIndex<MoveDirection::kUpRight>(from);
        EXPECT_EQ(up_right, 5);
        EXPECT_NE(up_right, Board::kInvalidIndex);
    }

//------------------------------------------------------------------------------//
//                          GetPieceLeftMoveIndex / GetPieceRightMoveIndex      //
//------------------------------------------------------------------------------//

    TEST_F(BoardTest, GetPieceLeftMoveIndex_White)
    {
        // White attempts "up-left" internally
        Board::IndexType from = 10;
        Board::IndexType left_move =
                board_.GetPieceLeftMoveIndex<BoardCheckType::kWhite>(from);
        EXPECT_EQ(left_move, 5);
        EXPECT_NE(left_move, Board::kInvalidIndex);
    }

    TEST_F(BoardTest, GetPieceRightMoveIndex_White)
    {
        // White attempts "up-right" internally
        Board::IndexType from = 10;
        Board::IndexType right_move =
                board_.GetPieceRightMoveIndex<BoardCheckType::kWhite>(from);
        EXPECT_EQ(right_move, 6);
        EXPECT_NE(right_move, Board::kInvalidIndex);
    }

    TEST_F(BoardTest, GetPieceLeftMoveIndex_Black)
    {
        // Black attempts "down-left" internally
        Board::IndexType from = 5;
        Board::IndexType left_move =
                board_.GetPieceLeftMoveIndex<BoardCheckType::kBlack>(from);
        EXPECT_EQ(left_move, 9);
        EXPECT_NE(left_move, Board::kInvalidIndex);
    }

    TEST_F(BoardTest, GetPieceRightMoveIndex_Black)
    {
        // Black attempts "down-right" internally
        Board::IndexType from = 5;
        Board::IndexType right_move =
                board_.GetPieceRightMoveIndex<BoardCheckType::kBlack>(from);
        EXPECT_EQ(right_move, 10);
        EXPECT_NE(right_move, Board::kInvalidIndex);
    }

}  // namespace CudaMctsCheckers
