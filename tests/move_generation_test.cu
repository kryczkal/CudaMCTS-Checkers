#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>

#include "common/checkers_defines.hpp"
#include "cpu/board_helpers.hpp"
#include "cpu/launchers.hpp"
#include "cuda/launchers.cuh"

/**
 * We unify CPU vs GPU "GenerateMoves" tests via typed tests.
 * We'll define a small helper to check expected moves,
 * plus a typed test fixture that calls HostGenerateMoves
 * for CPU or GPU.
 */

namespace
{

/**
 * CPU Implementation
 */
struct CPUMoveGenImpl {
    using BoardType = checkers::cpu::Board;

    static std::vector<checkers::MoveGenResult> GenerateMoves(const std::vector<BoardType> &boards, checkers::Turn turn)
    {
        return checkers::cpu::launchers::HostGenerateMoves(boards, turn);
    }

    static BoardType MakeBoard() { return BoardType{}; }

    static void SetPiece(BoardType &board, checkers::board_index_t idx, char pieceType)
    {
        board.setPieceAt(idx, pieceType);
    }

    static void SetBoardFromU32(BoardType &board, u32 white, u32 black, u32 kings)
    {
        board.white = white;
        board.black = black;
        board.kings = kings;
    }
};

/**
 * GPU Implementation
 */
struct GPUMoveGenImpl {
    using BoardType = checkers::gpu::launchers::GpuBoard;

    static std::vector<checkers::MoveGenResult> GenerateMoves(const std::vector<BoardType> &boards, checkers::Turn turn)
    {
        return checkers::gpu::launchers::HostGenerateMoves(boards, turn);
    }

    static BoardType MakeBoard() { return BoardType{}; }

    static void SetPiece(BoardType &board, checkers::board_index_t idx, char pieceType)
    {
        board.setPieceAt(idx, pieceType);
    }

    static void SetBoardFromU32(BoardType &board, u32 white, u32 black, u32 kings)
    {
        board.white = white;
        board.black = black;
        board.kings = kings;
    }
};

/**
 * Helper function to check if the entire board has a global capture/move
 * from the MoveGenResult.
 */
bool GlobalCaptureFound(const checkers::MoveGenResult &result)
{
    return ((result.h_per_board_flags[0] >> checkers::MoveFlagsConstants::kCaptureFound) & 1) == 1;
}
bool GlobalMoveFound(const checkers::MoveGenResult &result)
{
    return ((result.h_per_board_flags[0] >> checkers::MoveFlagsConstants::kMoveFound) & 1) == 1;
}

/**
 * @brief Confirms that all expected moves for a given squareIndex appear in the MoveGenResult,
 *        with correct capture bits. Outputs detailed error messages to std::cerr if discrepancies are found.
 *
 * @param r           The MoveGenResult containing generated moves and flags.
 * @param expectedMoves An unordered_map where keys are expected move_t and values indicate if the move is a capture.
 * @param squareIndex The index of the square to verify moves for.
 * @return True if all expected moves are found with correct capture flags, False otherwise.
 */
bool FoundAllExpectedMoves(
    const checkers::MoveGenResult &r, const std::unordered_map<checkers::move_t, bool> &expectedMoves,
    size_t squareIndex
)
{
    const size_t offset         = squareIndex * checkers::MoveGenResult::kMovesPerPiece;
    const uint8_t actual_count  = r.h_move_counts[squareIndex];
    const size_t expected_count = expectedMoves.size();

    // Create a copy of expected moves to track which ones have been found
    std::unordered_map<checkers::move_t, bool> expectedCopy = expectedMoves;

    // Iterate through each generated move and verify it against expected moves
    for (uint8_t i = 0; i < actual_count; ++i) {
        checkers::move_t mv = r.h_moves[offset + i];
        auto it             = expectedCopy.find(mv);

        // Check if the move is expected
        if (it == expectedCopy.end()) {
            std::cout
                << "Unexpected move found for square " << squareIndex << ": "
                << "from "
                << static_cast<u32>(checkers::cpu::move_gen::DecodeMove<checkers::cpu::move_gen::MovePart::From>(mv))
                << " to "
                << static_cast<u32>(checkers::cpu::move_gen::DecodeMove<checkers::cpu::move_gen::MovePart::To>(mv))
                << ".\n";
            return false;
        }

        bool wasCapture = ((r.h_capture_masks[squareIndex] >> i) & 1) == 1;

        // Check if the capture flag matches the expectation
        if (wasCapture != it->second) {
            std::cout << "Capture flag mismatch for move " << mv << " on square " << squareIndex << ": "
                      << "expected " << (it->second ? "capture" : "non-capture") << ", but found "
                      << (wasCapture ? "capture" : "non-capture") << ".\n";
            return false;
        }

        // Remove the move from expectedCopy to track found moves
        expectedCopy.erase(it);
    }

    // After processing all generated moves, check if any expected moves were not found
    if (!expectedCopy.empty()) {
        std::cout << "Missing expected moves for square " << squareIndex << ":\n";
        for (const auto &pair : expectedCopy) {
            checkers::move_t mv = pair.first;
            bool isCapture      = pair.second;
            std::cout
                << "  Move from "
                << static_cast<u32>(checkers::cpu::move_gen::DecodeMove<checkers::cpu::move_gen::MovePart::From>(mv))
                << " to "
                << static_cast<u32>(checkers::cpu::move_gen::DecodeMove<checkers::cpu::move_gen::MovePart::To>(mv))
                << " (" << (isCapture ? "capture" : "non-capture") << ")\n";
        }
        return false;
    }

    // All expected moves were found with correct capture flags
    return true;
}
}  // namespace

//////////////////////////////////////////////////////////////////////////////////
//                      Typed Test Fixture: MoveGeneration                      //
//////////////////////////////////////////////////////////////////////////////////

template <typename Impl>
class MoveGenerationTest : public ::testing::Test
{
};

using MoveGenImplementations = ::testing::Types<CPUMoveGenImpl, GPUMoveGenImpl>;
TYPED_TEST_SUITE(MoveGenerationTest, MoveGenImplementations);

TYPED_TEST(MoveGenerationTest, NoPiecesShouldGenerateNoMoves)
{
    using BoardType = typename TypeParam::BoardType;

    BoardType board = TypeParam::MakeBoard();
    std::vector<BoardType> boards{board};

    auto whiteRes = TypeParam::GenerateMoves(boards, checkers::Turn::kWhite);
    EXPECT_FALSE(GlobalMoveFound(whiteRes[0]));
    EXPECT_FALSE(GlobalCaptureFound(whiteRes[0]));

    auto blackRes = TypeParam::GenerateMoves(boards, checkers::Turn::kBlack);
    EXPECT_FALSE(GlobalMoveFound(blackRes[0]));
    EXPECT_FALSE(GlobalCaptureFound(blackRes[0]));
}

TYPED_TEST(MoveGenerationTest, SingleWhitePieceMoves)
{
    using BoardType = typename TypeParam::BoardType;

    BoardType board = TypeParam::MakeBoard();
    TypeParam::SetPiece(board, 12, 'W');

    std::vector<BoardType> boards{board};
    auto results  = TypeParam::GenerateMoves(boards, checkers::Turn::kWhite);
    const auto &r = results[0];

    checkers::move_t m1                                 = checkers::cpu::move_gen::EncodeMove(12, 8);
    checkers::move_t m2                                 = checkers::cpu::move_gen::EncodeMove(12, 9);
    std::unordered_map<checkers::move_t, bool> expected = {
        {m1, false},
        {m2, false},
    };

    EXPECT_TRUE(FoundAllExpectedMoves(r, expected, 12));
    EXPECT_TRUE(GlobalMoveFound(r));
    EXPECT_FALSE(GlobalCaptureFound(r));
}

TYPED_TEST(MoveGenerationTest, SingleBlackPieceMoves)
{
    using BoardType = typename TypeParam::BoardType;

    BoardType board = TypeParam::MakeBoard();
    TypeParam::SetPiece(board, 5, 'B');

    std::vector<BoardType> boards{board};
    auto results  = TypeParam::GenerateMoves(boards, checkers::Turn::kBlack);
    const auto &r = results[0];

    checkers::move_t m1 = checkers::cpu::move_gen::EncodeMove(5, 9);
    checkers::move_t m2 = checkers::cpu::move_gen::EncodeMove(5, 10);

    std::unordered_map<checkers::move_t, bool> expected = {
        {m1, false},
        {m2, false},
    };

    EXPECT_TRUE(FoundAllExpectedMoves(r, expected, 5));
    EXPECT_TRUE(GlobalMoveFound(r));
    EXPECT_FALSE(GlobalCaptureFound(r));
}

TYPED_TEST(MoveGenerationTest, WhitePieceCanCaptureBlackPiece)
{
    using BoardType = typename TypeParam::BoardType;

    BoardType board = TypeParam::MakeBoard();
    TypeParam::SetPiece(board, 13, 'W');
    TypeParam::SetPiece(board, 9, 'B');

    std::vector<BoardType> boards{board};
    auto results  = TypeParam::GenerateMoves(boards, checkers::Turn::kWhite);
    const auto &r = results[0];

    checkers::move_t capMove  = checkers::cpu::move_gen::EncodeMove(13, 4);
    checkers::move_t normMove = checkers::cpu::move_gen::EncodeMove(13, 10);

    std::unordered_map<checkers::move_t, bool> expected = {
        { capMove,  true},
        {normMove, false},
    };

    EXPECT_TRUE(FoundAllExpectedMoves(r, expected, 13));
    EXPECT_TRUE(GlobalMoveFound(r));
    EXPECT_TRUE(GlobalCaptureFound(r));
}

TYPED_TEST(MoveGenerationTest, KingPieceGeneratesDiagonalMoves)
{
    using BoardType = typename TypeParam::BoardType;

    BoardType board = TypeParam::MakeBoard();
    TypeParam::SetPiece(board, 12, 'W');
    TypeParam::SetPiece(board, 12, 'K');  // make it a king

    std::vector<BoardType> boards{board};
    auto results  = TypeParam::GenerateMoves(boards, checkers::Turn::kWhite);
    const auto &r = results[0];

    std::unordered_map<checkers::move_t, bool> expected;
    expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 8), false);
    expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 9), false);
    expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 5), false);
    expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 2), false);
    expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 16), false);
    expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 17), false);
    expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 21), false);
    expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 26), false);
    expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 30), false);

    EXPECT_TRUE(FoundAllExpectedMoves(r, expected, 12));
    EXPECT_TRUE(GlobalMoveFound(r));
    EXPECT_FALSE(GlobalCaptureFound(r));
}

TYPED_TEST(MoveGenerationTest, KingPieceMoveWithCapture)
{
    using BoardType = typename TypeParam::BoardType;

    BoardType board = TypeParam::MakeBoard();
    TypeParam::SetPiece(board, 12, 'W');
    TypeParam::SetPiece(board, 12, 'K');
    TypeParam::SetPiece(board, 9, 'B');

    std::vector<BoardType> boards{board};
    auto results  = TypeParam::GenerateMoves(boards, checkers::Turn::kWhite);
    const auto &r = results[0];

    std::unordered_map<checkers::move_t, bool> expected;
    // normal diagonal: 12->8 = not capture
    expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 8), false);
    // The next squares 12->5,12->2 might be captures if there's a piece at 9
    // Indeed it sets capture bits
    expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 5), true);
    expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 2), true);
    // continuing in other directions is normal moves
    expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 16), false);
    expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 17), false);
    expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 21), false);
    expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 26), false);
    expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 30), false);

    EXPECT_TRUE(FoundAllExpectedMoves(r, expected, 12));
    EXPECT_TRUE(GlobalMoveFound(r));
    EXPECT_TRUE(GlobalCaptureFound(r));
}

TYPED_TEST(MoveGenerationTest, KingPieceMoveBlockedByDifferentColor)
{
    using BoardType = typename TypeParam::BoardType;

    BoardType board = TypeParam::MakeBoard();
    TypeParam::SetPiece(board, 12, 'W');
    TypeParam::SetPiece(board, 12, 'K');
    TypeParam::SetPiece(board, 9, 'B');
    TypeParam::SetPiece(board, 5, 'B');

    std::vector<BoardType> boards{board};
    auto results  = TypeParam::GenerateMoves(boards, checkers::Turn::kWhite);
    const auto &r = results[0];

    std::unordered_map<checkers::move_t, bool> expected;
    expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 8), false);
    expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 16), false);
    expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 17), false);
    expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 21), false);
    expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 26), false);
    expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 30), false);

    EXPECT_TRUE(FoundAllExpectedMoves(r, expected, 12));
    EXPECT_TRUE(GlobalMoveFound(r));
    EXPECT_FALSE(GlobalCaptureFound(r));
}

TYPED_TEST(MoveGenerationTest, KingPieceMoveBlockedBySameColor)
{
    using BoardType = typename TypeParam::BoardType;

    BoardType board = TypeParam::MakeBoard();
    TypeParam::SetPiece(board, 12, 'W');
    TypeParam::SetPiece(board, 12, 'K');
    TypeParam::SetPiece(board, 9, 'W');

    std::vector<BoardType> boards{board};
    auto results  = TypeParam::GenerateMoves(boards, checkers::Turn::kWhite);
    const auto &r = results[0];

    std::unordered_map<checkers::move_t, bool> expected;
    expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 8), false);
    expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 16), false);
    expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 17), false);
    expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 21), false);
    expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 26), false);
    expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 30), false);

    EXPECT_TRUE(FoundAllExpectedMoves(r, expected, 12));
    EXPECT_TRUE(GlobalMoveFound(r));
    EXPECT_FALSE(GlobalCaptureFound(r));
}

TYPED_TEST(MoveGenerationTest, WhitePieceBlockedBySameColorAdjacent)
{
    using BoardType = typename TypeParam::BoardType;

    BoardType board = TypeParam::MakeBoard();
    TypeParam::SetPiece(board, 12, 'W');
    TypeParam::SetPiece(board, 8, 'W');

    std::vector<BoardType> boards{board};
    auto results  = TypeParam::GenerateMoves(boards, checkers::Turn::kWhite);
    const auto &r = results[0];

    checkers::move_t m1                                 = checkers::cpu::move_gen::EncodeMove(12, 9);
    std::unordered_map<checkers::move_t, bool> expected = {
        {m1, false},
    };

    EXPECT_TRUE(FoundAllExpectedMoves(r, expected, 12));
    EXPECT_TRUE(GlobalMoveFound(r));
    EXPECT_FALSE(GlobalCaptureFound(r));
}

TYPED_TEST(MoveGenerationTest, BlackPieceMultipleCaptureScenario)
{
    using BoardType = typename TypeParam::BoardType;

    BoardType board = TypeParam::MakeBoard();
    TypeParam::SetPiece(board, 13, 'B');
    TypeParam::SetPiece(board, 17, 'W');
    TypeParam::SetPiece(board, 21, 'W');

    std::vector<BoardType> boards{board};
    auto results  = TypeParam::GenerateMoves(boards, checkers::Turn::kBlack);
    const auto &r = results[0];

    checkers::move_t capMove1 = checkers::cpu::move_gen::EncodeMove(13, 20);
    checkers::move_t normMove = checkers::cpu::move_gen::EncodeMove(13, 18);

    std::unordered_map<checkers::move_t, bool> expected = {
        {capMove1,  true},
        {normMove, false},
    };

    EXPECT_TRUE(FoundAllExpectedMoves(r, expected, 13));
    EXPECT_TRUE(GlobalMoveFound(r));
    EXPECT_TRUE(GlobalCaptureFound(r));
}

TYPED_TEST(MoveGenerationTest, KingPieceBlockedBySameColorInAlmostAllDirections)
{
    using BoardType = typename TypeParam::BoardType;

    BoardType board = TypeParam::MakeBoard();
    TypeParam::SetPiece(board, 12, 'W');
    TypeParam::SetPiece(board, 12, 'K');

    TypeParam::SetPiece(board, 8, 'W');
    TypeParam::SetPiece(board, 5, 'W');
    TypeParam::SetPiece(board, 16, 'W');
    TypeParam::SetPiece(board, 17, 'W');

    std::vector<BoardType> boards{board};
    auto results  = TypeParam::GenerateMoves(boards, checkers::Turn::kWhite);
    const auto &r = results[0];

    std::unordered_map<checkers::move_t, bool> expected = {
        {checkers::cpu::move_gen::EncodeMove(12, 9), false},
    };

    EXPECT_TRUE(FoundAllExpectedMoves(r, expected, 12));
    EXPECT_TRUE(GlobalMoveFound(r));
    EXPECT_FALSE(GlobalCaptureFound(r));
}

TYPED_TEST(MoveGenerationTest, DifficultBoard1)
{
    using BoardType = typename TypeParam::BoardType;

    BoardType board = TypeParam::MakeBoard();
    // White pieces
    TypeParam::SetPiece(board, 31, 'W');
    TypeParam::SetPiece(board, 21, 'W');
    TypeParam::SetPiece(board, 16, 'W');
    // Black pieces
    TypeParam::SetPiece(board, 12, 'B');
    TypeParam::SetPiece(board, 9, 'B');
    TypeParam::SetPiece(board, 10, 'B');
    // King
    TypeParam::SetPiece(board, 9, 'K');

    std::vector<BoardType> boards{board};
    auto results  = TypeParam::GenerateMoves(boards, checkers::Turn::kBlack);
    const auto &r = results[0];

    EXPECT_FALSE(GlobalCaptureFound(r));
}

TYPED_TEST(MoveGenerationTest, ExceedThreadsAndMemory)
{
    // This is a stress test for GPU to exceed possible max threads, but we'll
    // also run it on CPU in the same manner. It's just a big data test.
    using BoardType = typename TypeParam::BoardType;

    // We'll pick a large number of boards. For GPU, we want to exceed the maximum
    // threads in a block or so. We'll guess a large number.
    size_t bigCount = 4096;  // for example

    // Build a sample board
    BoardType sample = TypeParam::MakeBoard();
    TypeParam::SetPiece(sample, 12, 'W');
    TypeParam::SetPiece(sample, 17, 'B');
    TypeParam::SetPiece(sample, 20, 'B');
    TypeParam::SetPiece(sample, 22, 'B');

    std::vector<BoardType> boards(bigCount, sample);

    // Generate moves for White
    auto results = TypeParam::GenerateMoves(boards, checkers::Turn::kWhite);

    // We'll sample some boards
    size_t indices[] = {0, bigCount / 2, bigCount - 1};
    for (size_t idx : indices) {
        const auto &r = results[idx];
        // We expect normal forward moves and a possible capture
        // White piece at 12 can move 12->8, 12->9, 12->21 (capture?), etc.

        std::unordered_map<checkers::move_t, bool> expected;
        // normal moves
        expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 8), false);
        expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 9), false);
        // capture
        expected.emplace(checkers::cpu::move_gen::EncodeMove(12, 21), true);

        bool allFound = FoundAllExpectedMoves(r, expected, 12);
        EXPECT_TRUE(allFound) << "Failed at board " << idx;

        bool moveFound    = GlobalMoveFound(r);
        bool captureFound = GlobalCaptureFound(r);
        EXPECT_TRUE(moveFound) << "No moves found at board " << idx;
        EXPECT_TRUE(captureFound) << "No captures found at board " << idx;
    }
}

TYPED_TEST(MoveGenerationTest, NoInvalidEdgeMove)
{
    using BoardType = typename TypeParam::BoardType;

    // Create a new board and place a black piece at h7 (index 7)
    BoardType board = TypeParam::MakeBoard();
    TypeParam::SetPiece(board, 7, 'B');  // h7

    std::vector<BoardType> boards{board};

    // Generate moves for Black
    auto results  = TypeParam::GenerateMoves(boards, checkers::Turn::kBlack);
    const auto &r = results[0];

    // Define the invalid move h7-b5 (from 7 to 12)
    checkers::move_t invalidMove = checkers::cpu::move_gen::EncodeMove(7, 12);

    // Calculate the offset for square 7 in the flattened moves array
    size_t offset = 7 * checkers::MoveGenResult::kMovesPerPiece;

    // Flag to check if the invalid move is found
    bool found = false;

    for (uint8_t i = 0; i < r.h_move_counts[7]; ++i) {
        if (r.h_moves[offset + i] == invalidMove) {
            found = true;
            break;
        }
    }

    EXPECT_FALSE(found) << "Invalid move h7-b5 was generated.";
}

TYPED_TEST(MoveGenerationTest, FORCE_CAPTURE)
{
    using BoardType = typename TypeParam::BoardType;

    BoardType board = TypeParam::MakeBoard();
    TypeParam::SetBoardFromU32(board, 4290969600, 14847, 0);

    std::vector<BoardType> boards{board};
    auto results  = TypeParam::GenerateMoves(boards, checkers::Turn::kBlack);
    const auto &r = results[0];

    EXPECT_TRUE(GlobalCaptureFound(r));
}
