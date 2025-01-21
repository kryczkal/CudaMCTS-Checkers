#include <gtest/gtest.h>
#include <vector>

#include "cpu/board_helpers.hpp"
#include "cuda/launchers.cuh"

using namespace checkers;
using namespace checkers::gpu::launchers;

TEST(GpuMoveSelectionTest, NoBoards)
{
    std::vector<GpuBoard> boards;
    std::vector<move_t> moves;
    std::vector<u8> move_counts;
    std::vector<move_flags_t> capture_masks;
    std::vector<move_flags_t> per_board_flags;
    std::vector<u8> seeds;

    auto best = HostSelectBestMoves(boards, moves, move_counts, capture_masks, per_board_flags, seeds);
    EXPECT_TRUE(best.empty());
}

TEST(GpuMoveSelectionTest, SingleBoardSingleMove)
{
    // If there's only one available move, that should always be selected
    const size_t kTotalSquares  = BoardConstants::kBoardSize;
    const size_t kMovesPerPiece = gpu::move_gen::kNumMaxMovesPerPiece;

    // Board with a single piece
    GpuBoard board;
    board.white = (1u << 12);
    board.black = 0u;
    board.kings = 0u;
    std::vector<GpuBoard> boards{board};

    // Flattened moves array: 1 board * 32 squares * 13 possible moves
    std::vector<move_t> allMoves(kTotalSquares * kMovesPerPiece, MoveConstants::kInvalidMove);

    // We'll place a single valid move for square 12 at subindex 0
    move_t validMove          = static_cast<move_t>(12 | (8u << 8));
    size_t pieceOffset        = 12 * kMovesPerPiece;
    allMoves[pieceOffset + 0] = validMove;

    // For that single board, we have a move_counts array of size 32
    std::vector<u8> moveCounts(kTotalSquares, 0);
    moveCounts[12] = 1;  // exactly 1 valid move for piece 12

    // capture_masks also has size 32 for each board
    std::vector<move_flags_t> captureMasks(kTotalSquares, 0);

    // One flags entry
    std::vector<move_flags_t> perBoardFlags(1, 0);

    // Seeds vector has exactly 1 element
    std::vector<u8> seeds{0};
    seeds[0] = (u8)293409258;

    auto best = HostSelectBestMoves(boards, allMoves, moveCounts, captureMasks, perBoardFlags, seeds);
    ASSERT_EQ(best.size(), 1u);
    EXPECT_EQ(best[0], validMove);
}

TEST(GpuMoveSelectionTest, SingleBoardMultipleMoves)
{
    // We test random selection logic by using different seeds
    // that index into the same array of possible moves.
    const size_t totalSquares  = BoardConstants::kBoardSize;
    const size_t movesPerPiece = gpu::move_gen::kNumMaxMovesPerPiece;

    GpuBoard board;
    board.white = (1u << 12);
    board.black = 0u;
    board.kings = 0u;
    std::vector<GpuBoard> boards{board};

    // Build flattened moves array
    std::vector<move_t> allMoves(totalSquares * movesPerPiece, MoveConstants::kInvalidMove);

    // Suppose piece 12 has 3 valid moves
    move_t m0 = static_cast<move_t>(12 | (8u << 8));
    move_t m1 = static_cast<move_t>(12 | (9u << 8));
    move_t m2 = static_cast<move_t>(12 | (5u << 8));

    size_t offset        = 12 * movesPerPiece;
    allMoves[offset + 0] = m0;
    allMoves[offset + 1] = m1;
    allMoves[offset + 2] = m2;

    // move_counts
    std::vector<u8> moveCounts(totalSquares, 0);
    moveCounts[12] = 3;  // 3 valid moves for that piece

    // captureMasks
    std::vector<move_flags_t> captureMasks(totalSquares, 0);

    // perBoardFlags
    std::vector<move_flags_t> perBoardFlags(1, 0);

    // seeds
    std::vector<u8> seeds(1, (u8)0);

    std::vector<move_t> bests;
    static constexpr u8 kFirstSeed = (u8)93425834;

    seeds[0]  = kFirstSeed;
    auto best = HostSelectBestMoves(boards, allMoves, moveCounts, captureMasks, perBoardFlags, seeds);
    ASSERT_EQ(best.size(), 1u);
    bests.push_back(best[0]);

    seeds[0] = (u8)kFirstSeed + 1;
    best     = HostSelectBestMoves(boards, allMoves, moveCounts, captureMasks, perBoardFlags, seeds);
    ASSERT_EQ(best.size(), 1u);
    bests.push_back(best[0]);

    seeds[0] = (u8)kFirstSeed + 2;
    best     = HostSelectBestMoves(boards, allMoves, moveCounts, captureMasks, perBoardFlags, seeds);
    ASSERT_EQ(best.size(), 1u);
    bests.push_back(best[0]);

    // Given seeds are +1 apart, we expect different moves to be selected
    EXPECT_NE(bests[0], bests[1]);
    EXPECT_NE(bests[1], bests[2]);
    EXPECT_NE(bests[0], bests[2]);
}
