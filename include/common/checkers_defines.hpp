#ifndef MCTS_CHECKERS_INCLUDE_COMMON_CHECKERS_DEFINES_HPP_
#define MCTS_CHECKERS_INCLUDE_COMMON_CHECKERS_DEFINES_HPP_

#include "types.hpp"
#include "vector"

#define UNUSED(x) (void)(x)

namespace checkers
{
static constexpr bool kTrueRandom = true;
static constexpr u32 kSeed        = 0x12345678;

/////////////////////////////////// Types ////////////////////////////////////

enum class Turn { kWhite, kBlack };

using board_t       = u32;
using move_t        = u16;
using board_index_t = u8;
using move_flags_t  = u16;

static constexpr move_t kInvalidMove = 0xFFFF;

///////////////////////////////// Constants //////////////////////////////////
/**
 * \brief Outcome encoding in scores[]:
 *  0 = in progress (not used at the end, but can be intermediate)
 *  1 = White wins
 *  2 = Black wins
 *  3 = Draw
 */
static constexpr u8 kOutcomeWhite      = 1;
static constexpr u8 kOutcomeBlack      = 2;
static constexpr u8 kOutcomeDraw       = 3;
static constexpr u8 kOutcomeInProgress = 0;

class BoardConstants
{
    public:
    static constexpr u8 kBoardEdgeLength = 4;
    static constexpr u8 kBoardSize       = 32;

    static constexpr board_t kLeftBoardEdgeMask = []() constexpr {
        board_t mask = 0;
        for (u8 i = 0; i < kBoardEdgeLength; ++i) {
            mask |= 1 << (i * kBoardEdgeLength * 2);
        }
        return mask;
    }();

    static constexpr board_t kRightBoardEdgeMask = []() constexpr {
        board_t mask = 0;
        for (u8 i = 0; i < kBoardEdgeLength; ++i) {
            mask |= 1 << (i * kBoardEdgeLength * 2 + 2 * kBoardEdgeLength - 1);
        }
        return mask;
    }();

    static constexpr board_t kTopBoardEdgeMask = []() constexpr {
        board_t mask = 0;
        for (u8 i = 0; i < kBoardEdgeLength; ++i) {
            mask |= 1 << i;
        }
        return mask;
    }();

    static constexpr board_t kBottomBoardEdgeMask = []() constexpr {
        board_t mask = 0;
        for (u8 i = 0; i < kBoardEdgeLength; ++i) {
            mask |= 1 << (kBoardSize - kBoardEdgeLength + i);
        }
        return mask;
    }();

    static constexpr board_t kEdgeMask =
        kLeftBoardEdgeMask | kRightBoardEdgeMask | kTopBoardEdgeMask | kBottomBoardEdgeMask;
};

static constexpr u8 kNumMaxMovesPerPiece = 13;

class MoveFlagsConstants
{
    public:
    static constexpr u8 kMoveFound    = 0;
    static constexpr u8 kCaptureFound = 1;
};

/**
 * @brief Holds simulation parameters for a single board configuration.
 *        We'll do 'n_simulations' random rollouts from that position.
 */
struct SimulationParam {
    board_t white;
    board_t black;
    board_t king;
    u8 start_turn;      // 0=White starts, 1=Black starts
    u64 n_simulations;  // how many times to simulate from this config
};

/**
 * @brief Holds the result of calling the GPU-based GenerateMoves kernel
 *        for exactly one board.
 */
struct MoveGenResult {
    // We track 32 squares, with up to kNumMaxMovesPerPiece = 13 possible moves per piece
    static constexpr u64 kMaxPiecesToTrack = BoardConstants::kBoardSize;
    static constexpr u64 kMovesPerPiece    = kNumMaxMovesPerPiece;
    static constexpr u64 kMaxTotalMoves    = kMaxPiecesToTrack * kMovesPerPiece;

    // Flattened array of moves: size 32*kMovesPerPiece
    std::vector<move_t> h_moves;
    // Number of generated moves per square
    std::vector<u8> h_move_counts;
    // For each square, a mask indicating which sub-moves are captures
    std::vector<move_flags_t> h_capture_masks;
    // Additional per-board flags (bitwise MoveFlagsConstants)
    std::vector<move_flags_t> h_per_board_flags;

    MoveGenResult()
        : h_moves(kMaxPiecesToTrack * kMovesPerPiece, kInvalidMove),
          h_move_counts(kMaxPiecesToTrack, 0),
          h_capture_masks(kMaxPiecesToTrack, 0),
          h_per_board_flags(1, 0)
    {
    }
};

}  // namespace checkers

#endif  // MCTS_CHECKERS_INCLUDE_COMMON_CHECKERS_DEFINES_HPP_
