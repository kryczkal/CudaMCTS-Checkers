#include <gtest/gtest.h>
#include <vector>

#include "common/checkers_defines.hpp"
#include "cpu/board_helpers.hpp"
#include "cpu/launchers.hpp"
#include "cuda/launchers.cuh"
#include "types.hpp"

namespace
{

/**
 * CPU Implementation
 */
struct CPUMoveSelImpl {
    using BoardType = checkers::cpu::Board;

    static std::vector<checkers::move_t> SelectBestMoves(
        const std::vector<BoardType>& boards, const std::vector<checkers::move_t>& moves,
        const std::vector<u8>& move_counts, const std::vector<checkers::move_flags_t>& capture_masks,
        const std::vector<checkers::move_flags_t>& per_board_flags, const std::vector<u8>& seeds
    )
    {
        return checkers::cpu::launchers::HostSelectBestMoves(
            boards, moves, move_counts, capture_masks, per_board_flags, seeds
        );
    }

    static BoardType MakeBoard() { return BoardType{}; }

    static void SetPiece(BoardType& board, checkers::board_index_t idx, char pieceType)
    {
        board.setPieceAt(idx, pieceType);
    }
};

/**
 * GPU Implementation
 */
struct GPUMoveSelImpl {
    using BoardType = checkers::gpu::launchers::GpuBoard;

    static std::vector<checkers::move_t> SelectBestMoves(
        const std::vector<BoardType>& boards, const std::vector<checkers::move_t>& moves,
        const std::vector<u8>& move_counts, const std::vector<checkers::move_flags_t>& capture_masks,
        const std::vector<checkers::move_flags_t>& per_board_flags, const std::vector<u8>& seeds
    )
    {
        return checkers::gpu::launchers::HostSelectBestMoves(
            boards, moves, move_counts, capture_masks, per_board_flags, seeds
        );
    }

    static BoardType MakeBoard() { return BoardType{}; }

    static void SetPiece(BoardType& board, checkers::board_index_t idx, char pieceType)
    {
        board.setPieceAt(idx, pieceType);
    }
};

}  // namespace

//////////////////////////////////////////////////////////////////////////////////
//                  Typed Test Fixture for Move Selection                       //
//////////////////////////////////////////////////////////////////////////////////

template <typename Impl>
class MoveSelectionTest : public ::testing::Test
{
};

using MoveSelImplementations = ::testing::Types<CPUMoveSelImpl, GPUMoveSelImpl>;
TYPED_TEST_SUITE(MoveSelectionTest, MoveSelImplementations);

TYPED_TEST(MoveSelectionTest, NoBoards)
{
    using BoardType = typename TypeParam::BoardType;

    std::vector<BoardType> boards;
    std::vector<checkers::move_t> moves;
    std::vector<u8> move_counts;
    std::vector<checkers::move_flags_t> capture_masks;
    std::vector<checkers::move_flags_t> per_board_flags;
    std::vector<u8> seeds;

    auto best = TypeParam::SelectBestMoves(boards, moves, move_counts, capture_masks, per_board_flags, seeds);
    EXPECT_TRUE(best.empty());
}

TYPED_TEST(MoveSelectionTest, SingleBoardSingleMove)
{
    using BoardType = typename TypeParam::BoardType;

    // If there's only one available move, that should always be selected
    const size_t kTotalSquares  = checkers::BoardConstants::kBoardSize;
    const size_t kMovesPerPiece = checkers::kNumMaxMovesPerPiece;

    BoardType board = TypeParam::MakeBoard();
    // White piece at 12, for instance
    TypeParam::SetPiece(board, 12, 'W');

    std::vector<BoardType> boards{board};

    // Flattened moves array: 1 board * 32 squares * 13 possible moves
    std::vector<checkers::move_t> allMoves(kTotalSquares * kMovesPerPiece, checkers::kInvalidMove);

    // We'll place a single valid move for square 12 at subindex 0
    checkers::move_t validMove = static_cast<checkers::move_t>(12 | (8u << 8));  // from=12, to=8
    size_t pieceOffset         = 12 * kMovesPerPiece;
    allMoves[pieceOffset + 0]  = validMove;

    // For that single board, we have a move_counts array of size 32
    std::vector<u8> moveCounts(kTotalSquares, 0);
    moveCounts[12] = 1;  // exactly 1 valid move for piece 12

    // capture_masks also has size 32
    std::vector<checkers::move_flags_t> captureMasks(kTotalSquares, 0);

    // One per_board_flags entry
    std::vector<checkers::move_flags_t> perBoardFlags(1, 0);

    // Seeds vector has exactly 1 element
    std::vector<u8> seeds{(u8)293};

    auto best = TypeParam::SelectBestMoves(boards, allMoves, moveCounts, captureMasks, perBoardFlags, seeds);

    ASSERT_EQ(best.size(), 1u);
    EXPECT_EQ(best[0], validMove);
}

TYPED_TEST(MoveSelectionTest, SingleBoardMultipleMoves)
{
    using BoardType = typename TypeParam::BoardType;

    const size_t totalSquares  = checkers::BoardConstants::kBoardSize;
    const size_t movesPerPiece = checkers::kNumMaxMovesPerPiece;

    // One board with a single piece
    BoardType board = TypeParam::MakeBoard();
    // White piece at 12
    TypeParam::SetPiece(board, 12, 'W');

    std::vector<BoardType> boards{board};

    // Build flattened moves array
    std::vector<checkers::move_t> allMoves(totalSquares * movesPerPiece, checkers::kInvalidMove);

    // Suppose piece 12 has 3 valid moves
    checkers::move_t m0 = static_cast<checkers::move_t>(12 | (8u << 8));
    checkers::move_t m1 = static_cast<checkers::move_t>(12 | (9u << 8));
    checkers::move_t m2 = static_cast<checkers::move_t>(12 | (5u << 8));

    size_t offset        = 12 * movesPerPiece;
    allMoves[offset + 0] = m0;
    allMoves[offset + 1] = m1;
    allMoves[offset + 2] = m2;

    std::vector<u8> moveCounts(totalSquares, 0);
    moveCounts[12] = 3;

    std::vector<checkers::move_flags_t> captureMasks(totalSquares, 0);
    std::vector<checkers::move_flags_t> perBoardFlags(1, 0);

    std::vector<u8> seeds(1, (u8)0);

    static constexpr u8 kFirstSeed = (u8)93;

    // We'll check multiple seeds quickly
    std::vector<checkers::move_t> bests;
    seeds[0]  = kFirstSeed;
    auto best = TypeParam::SelectBestMoves(boards, allMoves, moveCounts, captureMasks, perBoardFlags, seeds);
    ASSERT_EQ(best.size(), 1u);
    bests.push_back(best[0]);

    seeds[0] = (u8)(kFirstSeed + 1);
    best     = TypeParam::SelectBestMoves(boards, allMoves, moveCounts, captureMasks, perBoardFlags, seeds);
    ASSERT_EQ(best.size(), 1u);
    bests.push_back(best[0]);

    seeds[0] = (u8)(kFirstSeed + 2);
    best     = TypeParam::SelectBestMoves(boards, allMoves, moveCounts, captureMasks, perBoardFlags, seeds);
    ASSERT_EQ(best.size(), 1u);
    bests.push_back(best[0]);

    // We expect random different moves for consecutive seeds
    EXPECT_NE(bests[0], bests[1]);
    EXPECT_NE(bests[1], bests[2]);
    EXPECT_NE(bests[0], bests[2]);
}

TYPED_TEST(MoveSelectionTest, CaptureMoveIsSelectedOverNonCaptureMove)
{
    using BoardType = typename TypeParam::BoardType;

    const size_t kTotalSquares  = checkers::BoardConstants::kBoardSize;
    const size_t kMovesPerPiece = checkers::kNumMaxMovesPerPiece;

    // Setup board
    BoardType board = TypeParam::MakeBoard();
    TypeParam::SetPiece(board, 12, 'W');  // White piece

    std::vector<BoardType> boards{board};

    std::vector<checkers::move_t> allMoves(kTotalSquares * kMovesPerPiece, checkers::kInvalidMove);

    checkers::move_t normalMove  = checkers::cpu::move_gen::EncodeMove(12, 8);
    checkers::move_t captureMove = checkers::cpu::move_gen::EncodeMove(12, 4);

    size_t pieceOffset        = 12 * kMovesPerPiece;
    allMoves[pieceOffset + 0] = normalMove;
    allMoves[pieceOffset + 1] = captureMove;

    std::vector<u8> moveCounts(kTotalSquares, 0);
    moveCounts[12] = 2;

    std::vector<checkers::move_flags_t> captureMasks(kTotalSquares, 0);
    captureMasks[12] = 0b10;  // second sub-move is capture

    std::vector<checkers::move_flags_t> perBoardFlags(1, 0);
    perBoardFlags[0] |= (1 << checkers::MoveFlagsConstants::kCaptureFound);

    std::vector<u8> seeds(1, (u8)255);

    auto best = TypeParam::SelectBestMoves(boards, allMoves, moveCounts, captureMasks, perBoardFlags, seeds);

    ASSERT_EQ(best.size(), 1u);
    EXPECT_EQ(best[0], captureMove);
}

TYPED_TEST(MoveSelectionTest, OnlyCaptureMovesAreSelectedWhenMultipleCapturesAvailable)
{
    using BoardType = typename TypeParam::BoardType;

    const size_t kTotalSquares  = checkers::BoardConstants::kBoardSize;
    const size_t kMovesPerPiece = checkers::kNumMaxMovesPerPiece;

    // Setup board: two squares can capture
    BoardType board = TypeParam::MakeBoard();
    TypeParam::SetPiece(board, 12, 'W');
    TypeParam::SetPiece(board, 20, 'W');

    std::vector<BoardType> boards{board};

    std::vector<checkers::move_t> allMoves(kTotalSquares * kMovesPerPiece, checkers::kInvalidMove);

    // piece at 12
    checkers::move_t normalMove12  = checkers::cpu::move_gen::EncodeMove(12, 8);
    checkers::move_t captureMove12 = checkers::cpu::move_gen::EncodeMove(12, 4);

    size_t offset12        = 12 * kMovesPerPiece;
    allMoves[offset12 + 0] = normalMove12;
    allMoves[offset12 + 1] = captureMove12;

    // piece at 20
    checkers::move_t normalMove20  = checkers::cpu::move_gen::EncodeMove(20, 24);
    checkers::move_t captureMove20 = checkers::cpu::move_gen::EncodeMove(20, 28);

    size_t offset20        = 20 * kMovesPerPiece;
    allMoves[offset20 + 0] = normalMove20;
    allMoves[offset20 + 1] = captureMove20;

    std::vector<u8> moveCounts(kTotalSquares, 0);
    moveCounts[12] = 2;
    moveCounts[20] = 2;

    std::vector<checkers::move_flags_t> captureMasks(kTotalSquares, 0);
    captureMasks[12] = 0b10;  // second sub-move is capture
    captureMasks[20] = 0b10;  // second sub-move is capture

    std::vector<checkers::move_flags_t> perBoardFlags(1, 0);
    perBoardFlags[0] |= (1 << checkers::MoveFlagsConstants::kCaptureFound);

    std::vector<u8> seeds(1, (u8)255);

    auto best = TypeParam::SelectBestMoves(boards, allMoves, moveCounts, captureMasks, perBoardFlags, seeds);

    ASSERT_EQ(best.size(), 1u);
    EXPECT_TRUE((best[0] == captureMove12) || (best[0] == captureMove20));
}
