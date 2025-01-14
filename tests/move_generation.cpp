
#include <gtest/gtest.h>
#include <move_generation.hpp>
#include <board.hpp>
#include <move.hpp>
#include <iostream>
#include <unordered_map>

namespace CudaMctsCheckers
{

    class MoveGenerationTest : public ::testing::Test
    {
    protected:
        Board board_;

        void SetUp() override
        {
            // By default, start with an empty board
            // (All bits off for white_pieces, black_pieces, and kings)
            board_.white_pieces = 0;
            board_.black_pieces = 0;
            board_.kings        = 0;
        }

        /**
         * @brief Counts how many moves in the MoveGenerationOutput::possible_moves
         *        array are valid (i.e., not Move::kInvalidMove).
         */
        static size_t CountValidMoves(const MoveGenerationOutput& output)
        {
            size_t count = 0;
            for (auto move : output.possible_moves) {
                if (move != Move::kInvalidMove) {
                    ++count;
                }
            }
            return count;
        }

        /**
         * @brief Compares the generated moves and captures with the expected moves and captures.
         *
         * @param output The MoveGenerationOutput containing generated moves and capture flags.
         * @param base_idx The base index for the piece being tested.
         * @param expected_moves_and_capture A map where the key is the expected move type and the value is the expected capture flag.
         */
        void CompareMoves(const MoveGenerationOutput& output, size_t base_idx, const std::unordered_map<Move::Type, bool>& expected_moves_and_capture)
        {
            size_t valid_moves = CountValidMoves(output);
            EXPECT_EQ(valid_moves, expected_moves_and_capture.size());

            for (const auto& [move, should_capture] : expected_moves_and_capture) {
                // Find the move in the possible_moves
                bool found = false;
                for (size_t i = 0; i < Move::kNumMoveArrayForPlayerSize; ++i) {
                    if (output.possible_moves[i] == move) {
                        found = true;
                        EXPECT_EQ(output.capture_moves[i], should_capture) << "Move " << move << " capture flag mismatch.";
                        break;
                    }
                }
                EXPECT_TRUE(found) << "Expected move " << move << " not found.";
            }
        }

        void TearDown() override
        {
            if (::testing::Test::HasFailure()) {
                std::cout << "Board state after test failure:\n";
                std::cout << board_;
            }
        }
    };

    /**
     * @brief Test that no pieces on the board yields no moves.
     */
    TEST_F(MoveGenerationTest, NoPiecesShouldGenerateNoMoves)
    {
        // No white pieces
        {
            auto output_white =
                    MoveGenerator::GenerateMovesForPlayerCpu<BoardCheckType::kWhite>(board_);
            EXPECT_EQ(CountValidMoves(output_white), 0);
            EXPECT_FALSE(output_white.capture_moves[MoveGenerationOutput::CaptureFlagIndex]);
        }
        // No black pieces
        {
            auto output_black =
                    MoveGenerator::GenerateMovesForPlayerCpu<BoardCheckType::kBlack>(board_);
            EXPECT_EQ(CountValidMoves(output_black), 0);
            EXPECT_FALSE(output_black.capture_moves[MoveGenerationOutput::CaptureFlagIndex]);
        }
    }

    /**
     * @brief Test that a single white piece near the top of the half-board
     *        can generate up-left/up-right moves when they are in range.
     *
     *        We'll place one White piece at index 12 as an example.
     */
    TEST_F(MoveGenerationTest, SingleWhitePieceMoves)
    {
        // Place a White piece at index 12
        board_.SetPieceAt<BoardCheckType::kWhite>(12);

        auto output = MoveGenerator::GenerateMovesForPlayerCpu<BoardCheckType::kWhite>(board_);

        std::unordered_map<Move::Type, bool> expected_moves_and_capture = {
                {8, false},
                {9, false}
        };

        size_t base_idx = 12 * Move::kNumMaxPossibleMovesPerPiece;
        CompareMoves(output, base_idx, expected_moves_and_capture);

        EXPECT_FALSE(output.capture_moves[MoveGenerationOutput::CaptureFlagIndex]);
    }

    /**
     * @brief Test that a single black piece can move "down-left" or "down-right."
     *
     *        We'll place one Black piece at index 5 and see if we get valid moves.
     */
    TEST_F(MoveGenerationTest, SingleBlackPieceMoves)
    {
        // Place a Black piece at index 5
        board_.SetPieceAt<BoardCheckType::kBlack>(5);

        auto output = MoveGenerator::GenerateMovesForPlayerCpu<BoardCheckType::kBlack>(board_);

        std::unordered_map<Move::Type, bool> expected_moves_and_capture = {
                {9, false},
                {10, false}
        };

        size_t base_idx = 5 * Move::kNumMaxPossibleMovesPerPiece;
        CompareMoves(output, base_idx, expected_moves_and_capture);

        EXPECT_FALSE(output.capture_moves[MoveGenerationOutput::CaptureFlagIndex]);
    }

    /**
     * @brief Test that captures are detected. We set up a white piece with
     *        an adjacent black piece that can be captured.
     */
    TEST_F(MoveGenerationTest, WhitePieceCanCaptureBlackPiece)
    {
        board_.SetPieceAt<BoardCheckType::kWhite>(13);
        board_.SetPieceAt<BoardCheckType::kBlack>(9);

        auto output = MoveGenerator::GenerateMovesForPlayerCpu<BoardCheckType::kWhite>(board_);

        std::unordered_map<Move::Type, bool> expected_moves_and_capture = {
                {4, true},
                {10, false}
        };

        size_t base_idx = 13 * Move::kNumMaxPossibleMovesPerPiece;
        CompareMoves(output, base_idx, expected_moves_and_capture);

        EXPECT_TRUE(output.capture_moves[MoveGenerationOutput::CaptureFlagIndex]);
    }

    /**
     * @brief Test that a king can move in all diagonal directions. We'll set a piece
     *        as both White and King, then check that it generates diagonal moves
     *        in every direction. This tests the "GenerateMovesDiagonalCpu" logic.
     */
    TEST_F(MoveGenerationTest, KingPieceGeneratesDiagonalMoves)
    {
        board_.SetPieceAt<BoardCheckType::kWhite>(12);
        board_.SetPieceAt<BoardCheckType::kKings>(12);

        auto output = MoveGenerator::GenerateMovesForPlayerCpu<BoardCheckType::kWhite>(board_);

        // Define expected moves without captures
        std::unordered_map<Move::Type, bool> expected_moves_and_capture = {
                {8, false}, {9, false},
                {5, false}, {2, false},
                {16, false}, {17, false},
                {21, false}, {26, false}, {30, false}
        };

        size_t base_idx = 12 * Move::kNumMaxPossibleMovesPerPiece;
        CompareMoves(output, base_idx, expected_moves_and_capture);

        EXPECT_FALSE(output.capture_moves[MoveGenerationOutput::CaptureFlagIndex]);
    }

    TEST_F(MoveGenerationTest, KingPieceMoveWithCapture) {
        board_.SetPieceAt<BoardCheckType::kWhite>(12);
        board_.SetPieceAt<BoardCheckType::kKings>(12);
        board_.SetPieceAt<BoardCheckType::kBlack>(9);

        auto output = MoveGenerator::GenerateMovesForPlayerCpu<BoardCheckType::kWhite>(board_);

        std::unordered_map<Move::Type, bool> expected_moves_and_capture = {
                {8, false},
                {5, true},
                {2, true},
                {16, false},
                {17, false},
                {21, false},
                {26, false},
                {30, false}
        };

        size_t base_idx = 12 * Move::kNumMaxPossibleMovesPerPiece;
        CompareMoves(output, base_idx, expected_moves_and_capture);

        EXPECT_TRUE(output.capture_moves[MoveGenerationOutput::CaptureFlagIndex]);
    }

    TEST_F(MoveGenerationTest, KingPieceMoveBlockedByDifferentColor) {
        board_.SetPieceAt<BoardCheckType::kWhite>(12);
        board_.SetPieceAt<BoardCheckType::kKings>(12);
        board_.SetPieceAt<BoardCheckType::kBlack>(9);
        board_.SetPieceAt<BoardCheckType::kBlack>(5);

        auto output = MoveGenerator::GenerateMovesForPlayerCpu<BoardCheckType::kWhite>(board_);

        std::unordered_map<Move::Type, bool> expected_moves_and_capture = {
                {8, false},
                {16, false},
                {17, false},
                {21, false},
                {26, false},
                {30, false}
        };

        size_t base_idx = 12 * Move::kNumMaxPossibleMovesPerPiece;
        CompareMoves(output, base_idx, expected_moves_and_capture);

        EXPECT_FALSE(output.capture_moves[MoveGenerationOutput::CaptureFlagIndex]);
    }

    TEST_F(MoveGenerationTest, KingPieceMoveBlockedBySameColor) {
        board_.SetPieceAt<BoardCheckType::kWhite>(12);
        board_.SetPieceAt<BoardCheckType::kKings>(12);
        board_.SetPieceAt<BoardCheckType::kWhite>(9);

        auto output = MoveGenerator::GenerateMovesForPlayerCpu<BoardCheckType::kWhite>(board_);

        std::unordered_map<Move::Type, bool> expected_moves_and_capture = {
                {8, false},
                {4, false},
                {5, false},
                {16, false},
                {17, false},
                {21, false},
                {26, false},
                {30, false}
        };

        size_t base_idx = 12 * Move::kNumMaxPossibleMovesPerPiece;
        CompareMoves(output, base_idx, expected_moves_and_capture);

        EXPECT_FALSE(output.capture_moves[MoveGenerationOutput::CaptureFlagIndex]);
    }

    TEST_F(MoveGenerationTest, WhitePieceBlockedBySameColorAdjacent)
    {
        board_.SetPieceAt<BoardCheckType::kWhite>(12);
        board_.SetPieceAt<BoardCheckType::kWhite>(8);

        auto output = MoveGenerator::GenerateMovesForPlayerCpu<BoardCheckType::kWhite>(board_);
        EXPECT_FALSE(output.capture_moves[MoveGenerationOutput::CaptureFlagIndex]);

        std::unordered_map<Move::Type, bool> expected_moves_and_capture = {
                {9, false}, {4, false}
        };

        size_t base_idx       = 12 * Move::kNumMaxPossibleMovesPerPiece;
        CompareMoves(output, base_idx, expected_moves_and_capture);
    }

    /**
     *   We set up a scenario (rough example):
     *   - Black piece at index 13
     *   - Two White pieces at indices 17 and 21, each can be captured if the board logic supported multi-jump.
     */
    TEST_F(MoveGenerationTest, BlackPieceMultipleCaptureScenario)
    {
        board_.SetPieceAt<BoardCheckType::kBlack>(13);
        board_.SetPieceAt<BoardCheckType::kWhite>(17);
        board_.SetPieceAt<BoardCheckType::kWhite>(21);

        auto output = MoveGenerator::GenerateMovesForPlayerCpu<BoardCheckType::kBlack>(board_);

        std::unordered_map<Move::Type, bool> expected_moves_and_capture = {
                {20, true},
                {18, false}
        };

        size_t base_idx = 13 * Move::kNumMaxPossibleMovesPerPiece;
        CompareMoves(output, base_idx, expected_moves_and_capture);

        EXPECT_TRUE(output.capture_moves[MoveGenerationOutput::CaptureFlagIndex]);
    }

    /**
     *   King (white) at index 12
     *   White pieces blocking almost all diagonal directions except one move for king.
     */
    TEST_F(MoveGenerationTest, KingPieceBlockedBySameColorInAlmostAllDirections)
    {
        board_.SetPieceAt<BoardCheckType::kWhite>(12);
        board_.SetPieceAt<BoardCheckType::kKings>(12);

        board_.SetPieceAt<BoardCheckType::kWhite>(8);
        board_.SetPieceAt<BoardCheckType::kWhite>(5);
        board_.SetPieceAt<BoardCheckType::kWhite>(16);
        board_.SetPieceAt<BoardCheckType::kWhite>(17);

        auto output = MoveGenerator::GenerateMovesForPlayerCpu<BoardCheckType::kWhite>(board_);

        std::unordered_map<Move::Type, bool> expected_moves_and_capture = {
                {4, false}, {9, false}, {13, false}, {1, false}, {2, false}
        };

        size_t base_idx = 12 * Move::kNumMaxPossibleMovesPerPiece;
        CompareMoves(output, base_idx, expected_moves_and_capture);

        EXPECT_FALSE(output.capture_moves[MoveGenerationOutput::CaptureFlagIndex]);
    }

}  // namespace CudaMctsCheckers
