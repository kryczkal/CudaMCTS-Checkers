// File: checkers_engine_test.cpp

#include <gtest/gtest.h>
#include <checkers_engine.hpp>
#include <iostream>
#include <move_generation.hpp>
#include <string>
#include <vector>

namespace CudaMctsCheckers
{

class CheckersEngineTest : public ::testing::Test
{
    protected:
    CheckersEngine *engine_ = nullptr;

    void SetUp() override
    {
        // Create a standard board with White to move by default
        Board initial_board;
        // Standard checkers layout:
        for (Board::IndexType i = 0; i < 12; ++i) {
            initial_board.SetPieceAt<BoardCheckType::kBlack>(i);
        }
        for (Board::IndexType i = 20; i < 32; ++i) {
            initial_board.SetPieceAt<BoardCheckType::kWhite>(i);
        }
        engine_ = new CheckersEngine(initial_board, Turn::kWhite);
    }

    void TearDown() override
    {
        delete engine_;
        engine_ = nullptr;
    }
};

/**
 * @brief Utility function to parse "e.g. d2-e3" moves directly from within the test,
 *        applying them to the engine in sequence. If a move can't apply, returns false.
 */
static bool ApplyNotationMovesSequence(
    CheckersEngine &engine, const std::vector<std::string> &moves
)
{
    for (auto &m : moves) {
        // Split "d2-e3" or "d2:e4:d6" by delimiter
        char delim = (m.find(':') != std::string::npos) ? ':' : '-';

        std::vector<std::string> fields;
        {
            std::stringstream ss(m);
            std::string part;
            while (std::getline(ss, part, delim)) {
                if (!part.empty()) {
                    fields.push_back(part);
                }
            }
        }
        // If multi-capture
        if (fields.size() > 2) {
            // consecutive segments
            for (size_t i = 0; i + 1 < fields.size(); ++i) {
                Board::IndexType from_idx = engine.ConvertNotationToIndex(fields[i]);
                Board::IndexType to_idx   = engine.ConvertNotationToIndex(fields[i + 1]);
                if (from_idx == Board::kInvalidIndex || to_idx == Board::kInvalidIndex) {
                    return false;
                }
                bool success = engine.ApplyMove<ApplyMoveType::kValidate>(from_idx, to_idx);
                if (!success) {
                    return false;
                }
            }
        } else {
            // Single move
            if (fields.size() < 2) {
                return false;
            }
            Board::IndexType from_idx = engine.ConvertNotationToIndex(fields[0]);
            Board::IndexType to_idx   = engine.ConvertNotationToIndex(fields[1]);
            if (from_idx == Board::kInvalidIndex || to_idx == Board::kInvalidIndex) {
                return false;
            }
            bool success = engine.ApplyMove<ApplyMoveType::kValidate>(from_idx, to_idx);
            if (!success) {
                return false;
            }
        }
    }
    return true;
}

/**
 * Basic test: ensure engine initializes with correct default turn (White).
 */
TEST_F(CheckersEngineTest, InitialTurnShouldBeWhite)
{
    EXPECT_EQ(engine_->GetCurrentTurn(), Turn::kWhite)
        << "Initial turn should be White for a new engine instance.";
}

/**
 * Generate moves for White from the standard start. We expect each piece
 * to have 2 possible forward moves if no blocking. Usually the second rank (row=5)
 * blocks some moves, but let's see if it matches the standard checkers layout expectations.
 */
TEST_F(CheckersEngineTest, GenerateMovesFromStandardLayout_White)
{
    MoveGenerationOutput moves = engine_->GenerateCurrentPlayerMoves();
    // Count valid moves
    size_t valid = 0;
    for (auto move : moves.possible_moves) {
        if (move != Move::kInvalidMove) {
            valid++;
        }
    }
    // Typically from the standard layout, White in row 5 is partially blocked,
    // but row 6 pieces can move. The exact number can vary if all pieces are in place.
    // For a US 8x8 checkers opening, we often see at least 7 valid single-step moves.
    EXPECT_GE(valid, 7u) << "Expected at least 7 possible single-step moves from the White side in "
                            "the default opening.";
    EXPECT_FALSE(moves.capture_moves_bitmask[MoveGenerationOutput::CaptureFlagIndex])
        << "No captures should be possible in the default initial position.";
}

/**
 * Make a simple White move: e.g. "c3-b4" or "c3-d4" in standard notation.
 * We'll attempt "c3-b4" => from index 20 (?), to index ~ 16, depends on the half-board indexing.
 */
TEST_F(CheckersEngineTest, ApplySimpleMove_White)
{
    // For convenience, let's try a known valid move from standard layout:
    // Usually "c3" is row=3 from bottom => but let's do a direct approach:
    // We'll just choose a piece we know is in row=5 in half-board indexing:
    // White pieces occupy indices [20..31]. Let’s pick 20 => top left White piece.
    // We'll attempt a typical upward-left or upward-right move if it exists.
    auto moves = engine_->GenerateCurrentPlayerMoves();
    // Find the first valid move from piece index=20
    bool found_move = false;
    for (u32 i = 20 * Move::kNumMaxPossibleMovesPerPiece;
         i < (21 * Move::kNumMaxPossibleMovesPerPiece); ++i) {
        if (i >= Move::kNumMoveArrayForPlayerSize)
            break;
        if (moves.possible_moves[i] != Move::kInvalidMove) {
            // Attempt applying it
            Board::IndexType to_idx = moves.possible_moves[i];
            bool success =
                engine_->ApplyMove<ApplyMoveType::kValidate>(Move::DecodeOriginIndex(i), to_idx);
            EXPECT_TRUE(success);
            found_move = true;
            break;
        }
    }
    EXPECT_TRUE(found_move) << "Should find at least one valid move for the piece at index=20.";
    // Turn should now be Black
    EXPECT_EQ(engine_->GetCurrentTurn(), Turn::kBlack);
}

/**
 * Test forced capture scenario: If there's a capture available, a normal non-capture move
 * must fail if the ruleset enforces forced captures. Our engine code has a
 * `bool force_capture = false` parameter we can pass. We can simulate forced capture
 * by passing `true` and seeing if non-capturing moves are invalid.
 */
TEST_F(CheckersEngineTest, ForcedCaptureScenario_White)
{
    // Clear board
    Board board;
    board.white_pieces = 0;
    board.black_pieces = 0;
    board.kings        = 0;

    board.SetPieceAt<BoardCheckType::kWhite>(12);
    board.SetPieceAt<BoardCheckType::kBlack>(9);

    CheckersEngine engine(board, Turn::kWhite);

    bool success_non_capture = engine.ApplyMove<ApplyMoveType::kValidate>(12, 8);
    EXPECT_FALSE(success_non_capture)
        << "Should fail if we try a non-capturing move when a capture exists";

    bool success_capture = engine.ApplyMove<ApplyMoveType::kValidate>(12, 5);
    EXPECT_TRUE(success_capture) << "Should succeed to capture the Black piece";
}

/**
 * Test a multi-capture scenario for White. We put black pieces so that White can jump
 * multiple times in a single turn.
 */
TEST_F(CheckersEngineTest, MultiCapture_White)
{
    Board board;
    board.white_pieces = 0;
    board.black_pieces = 0;
    board.kings        = 0;
    board.SetPieceAt<BoardCheckType::kWhite>(21);
    board.SetPieceAt<BoardCheckType::kBlack>(17);
    board.SetPieceAt<BoardCheckType::kBlack>(9);
    board.SetPieceAt<BoardCheckType::kBlack>(4);

    // White to move
    CheckersEngine engine(board, Turn::kWhite);

    bool success_first = engine.ApplyMove<ApplyMoveType::kValidate>(21, 12);
    EXPECT_TRUE(success_first);
    EXPECT_FALSE(engine.GetBoard().IsPieceAt<BoardCheckType::kBlack>(17));

    EXPECT_EQ(engine.GetCurrentTurn(), Turn::kWhite)
        << "By default, the turn should remain with White after a successful capture";

    bool success_second = engine.ApplyMove<ApplyMoveType::kValidate>(12, 5);
    EXPECT_TRUE(engine.GetCurrentTurn() == Turn::kBlack)
        << "After a successful multi-capture, the turn should switch to Black";
    EXPECT_TRUE(success_second);
    EXPECT_FALSE(engine.GetBoard().IsPieceAt<BoardCheckType::kBlack>(17));
}

/**
 * Test that a piece that reaches the opposite side gets promoted to king
 * and can then move backward in subsequent moves.
 */
TEST_F(CheckersEngineTest, PromotionToKingAndBackwardsMove)
{
    Board board;
    board.white_pieces = 0;
    board.black_pieces = 0;
    board.kings        = 0;
    board.SetPieceAt<BoardCheckType::kWhite>(5);
    board.SetPieceAt<BoardCheckType::kBlack>(25);
    board.SetPieceAt<BoardCheckType::kBlack>(24);

    CheckersEngine engine(board, Turn::kWhite);

    bool success = engine.ApplyMove<ApplyMoveType::kValidate>(5, 1);
    EXPECT_TRUE(success);
    EXPECT_TRUE(engine.GetBoard().IsPieceAt<BoardCheckType::kWhite>(1));
    EXPECT_TRUE(engine.GetCurrentTurn() == Turn::kBlack);

    EXPECT_TRUE(engine.GetBoard().IsPieceAt<BoardCheckType::kKings>(1));

    bool black_move_success = engine.ApplyRandomMove();
    EXPECT_TRUE(engine.GetCurrentTurn() == Turn::kWhite);
    EXPECT_TRUE(black_move_success);

    success = engine.ApplyMove<ApplyMoveType::kValidate>(1, 23);
    EXPECT_TRUE(success);
    EXPECT_TRUE(engine.GetBoard().IsPieceAt<BoardCheckType::kWhite>(23));
    EXPECT_TRUE(engine.GetBoard().IsPieceAt<BoardCheckType::kKings>(23));
}

/**
 * Test that if one side has no moves, the engine returns false for ApplyRandomMove and eventually
 * that side is considered losing if it's forced to skip.
 */
TEST_F(CheckersEngineTest, NoMovesRemaining)
{
    // Create a board where White has no moves from the start:
    // For example, place White in corners blocked by its own pieces or off the board.
    Board board;
    board.white_pieces = 0;
    board.black_pieces = 0;
    // Put White piece in row=0, but also block it somehow. Actually, simplest is to have no White
    // pieces at all:
    board.SetPieceAt<BoardCheckType::kWhite>(0);
    board.SetPieceAt<BoardCheckType::kBlack>(4);
    board.SetPieceAt<BoardCheckType::kBlack>(9);
    CheckersEngine engine(board, Turn::kWhite);

    // If we try to apply a random move for White, it should fail:
    bool success = engine.ApplyRandomMove();
    EXPECT_FALSE(success) << "White has no pieces => no moves => ApplyRandomMove should fail.";

    // The game result should now show that White is effectively lost => Black wins:
    GameResult r = engine.CheckGameResult();
    EXPECT_EQ(r, GameResult::kBlackWin)
        << "If White has no pieces, the result is immediate black win.";
}

/**
 * Test partial game flow: we apply a sequence of moves from a text-based notation,
 * then check if the final result matches an expected state.
 */
TEST_F(CheckersEngineTest, PartialGameFlow)
{
    CheckersEngine engine{};

    std::vector<std::string> moves = {"b3-a4", "c6-b5", "a4:c6"};
    bool success                   = ApplyNotationMovesSequence(engine, moves);
    EXPECT_TRUE(success) << "All partial moves should succeed in sequence.";

    const Board &b = engine.GetBoard();
    EXPECT_TRUE(b.IsPieceAt<BoardCheckType::kWhite>(9));
    EXPECT_FALSE(b.IsPieceAt<BoardCheckType::kBlack>(12));
    EXPECT_FALSE(b.IsPieceAt<BoardCheckType::kWhite>(16));
    EXPECT_EQ(engine.GetCurrentTurn(), Turn::kBlack);
}

/**
 * Random moves test: the engine should apply random moves for each side until game finishes
 * or we reach a large iteration count. This ensures the engine doesn't crash in random-play.
 * (We won't run this for very many moves, just enough to test stability.)
 */
TEST_F(CheckersEngineTest, RandomPlayStability)
{
    const int max_turns = 200;  // limit random moves
    for (int i = 0; i < max_turns; ++i) {
        if (!engine_->ApplyRandomMove()) {
            // That means current player had no moves => other player wins
            break;
        }
        GameResult r = engine_->CheckGameResult();
        if (r != GameResult::kInProgress) {
            break;  // someone won or it's a draw
        }
    }
    // At this point we either used up all moves or the game ended.
    // We can do a final result check to ensure no weird states occurred:
    GameResult final = engine_->CheckGameResult();
    // Could be InProgress if we didn't do enough turns, or it ended.
    // We won't do a strict check here, just verifying it doesn't crash or produce nonsense.
    EXPECT_TRUE(
        final == GameResult::kInProgress || final == GameResult::kWhiteWin ||
        final == GameResult::kBlackWin || final == GameResult::kDraw
    );
}

}  // namespace CudaMctsCheckers
