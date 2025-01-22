#ifndef MCTS_CHECKERS_INCLUDE_CUDA_LAUNCHERS_CUH_
#define MCTS_CHECKERS_INCLUDE_CUDA_LAUNCHERS_CUH_

#include <algorithm>
#include <cstring>
#include <vector>

#include "common/checkers_defines.hpp"
#include "cuda/cuda_utils.cuh"
#include "cuda/move_generation.cuh"
#include "mcts/simulation_results.hpp"

namespace checkers::gpu::launchers
{

/**
 * @brief Holds a simple board definition for host usage. We store
 *        bitmasks for white/black pieces, plus king flags.
 */
struct GpuBoard {
    board_t white = 0;
    board_t black = 0;
    board_t kings = 0;

    bool operator==(const GpuBoard& other) const
    {
        return (white == other.white) && (black == other.black) && (kings == other.kings);
    }

    bool operator!=(const GpuBoard& other) const { return !(*this == other); }

    /**
     * @brief Helper to set a piece.
     *        'W' -> white, 'B' -> black, 'K' -> king flag.
     */
    void setPieceAt(board_index_t idx, char pieceType)
    {
        switch (pieceType) {
            case 'W':
                white |= (static_cast<board_t>(1) << idx);
                break;
            case 'B':
                black |= (static_cast<board_t>(1) << idx);
                break;
            case 'K':
                kings |= (static_cast<board_t>(1) << idx);
                break;
            default:
                break;
        }
    }
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
    static constexpr size_t kTotalSquares  = BoardConstants::kBoardSize;
    static constexpr size_t kMovesPerPiece = checkers::gpu::move_gen::kNumMaxMovesPerPiece;

    // Flattened array of moves: size 32*kMovesPerPiece
    std::vector<move_t> h_moves;
    // Number of generated moves per square
    std::vector<u8> h_move_counts;
    // For each square, a mask indicating which sub-moves are captures
    std::vector<move_flags_t> h_capture_masks;
    // Additional per-board flags (bitwise MoveFlagsConstants)
    std::vector<move_flags_t> h_per_board_flags;

    MoveGenResult()
        : h_moves(kTotalSquares * kMovesPerPiece, kInvalidMove),
          h_move_counts(kTotalSquares, 0),
          h_capture_masks(kTotalSquares, 0),
          h_per_board_flags(1, 0)
    {
    }
};

/**
 * @brief This function allocates device memory for a vector of boards,
 *        copies data to device, launches the GPU kernel, and retrieves results.
 *
 * @tparam turn Whether to generate moves for White or Black.
 * @param boards Vector of host GpuBoard objects (white/black/kings bitmasks).
 * @param turn Whether to generate moves for White or Black.
 * @return Vector of MoveGenResult objects, one per board.
 */
std::vector<MoveGenResult> HostGenerateMoves(const std::vector<GpuBoard>& boards, Turn turn);

/**
 * @brief Host function to apply a single move per board on the GPU.
 *        The size of @p moves must match the size of @p boards (one move per board).
 *        Returns updated board states after each move is applied.
 */
std::vector<GpuBoard> HostApplyMoves(const std::vector<GpuBoard>& boards, const std::vector<move_t>& moves);

/**
 * @brief Host function to select a "best" move (or randomly chosen move) per board.
 *
 * @param boards           Board states for each position.
 * @param moves            Flattened moves for all boards; size = n_boards * 32 * kNumMaxMovesPerPiece.
 * @param move_counts      Number of sub-moves per square for each board; size = n_boards * 32.
 * @param capture_masks    For each square, bitmask indicating which sub-moves are captures; size = n_boards * 32.
 * @param per_board_flags  Additional flags (bitwise MoveFlagsConstants) per board; size = n_boards.
 * @param seeds            One random byte per board (used for random selection).
 *
 * @return A vector of size n_boards, each element is the chosen move_t for that board.
 */
std::vector<move_t> HostSelectBestMoves(
    const std::vector<GpuBoard>& boards, const std::vector<move_t>& moves, const std::vector<u8>& move_counts,
    const std::vector<move_flags_t>& capture_masks, const std::vector<move_flags_t>& per_board_flags,
    const std::vector<u8>& seeds
);

/**
 * @brief Updated function that:
 *   - Accepts a vector of SimulationParam.
 *   - Allocates GPU memory for the board definitions, calls the kernel that
 *     simulates all the games in one go, storing partial results in d_scores.
 *   - For each batch, calls an optimized GPU reduction kernel to sum up
 *     the outcomes in d_scores for that batch.
 *   - Returns a vector of SimulationResult, containing final .score
 *     (sum/2.0) and the number of simulations for that batch.
 *
 * @param params          Vector of SimulationParam structures.
 * @param max_iterations  If we reach that many half-moves, declare a draw.
 * @return A vector of size `params.size()`, each entry is SimulationResult.
 */
std::vector<SimulationResult> HostSimulateCheckersGames(const std::vector<SimulationParam>& params, int max_iterations);
}  // namespace checkers::gpu::launchers

#endif  // MCTS_CHECKERS_INCLUDE_CUDA_LAUNCHERS_CUH_
