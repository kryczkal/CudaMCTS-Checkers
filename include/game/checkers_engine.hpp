#ifndef MCTS_CHECKERS_INCLUDE_GAME_CHECKERS_ENGINE_HPP_
#define MCTS_CHECKERS_INCLUDE_GAME_CHECKERS_ENGINE_HPP_

#include <string>
#include <vector>
#include "common/checkers_defines.hpp"
#include "cpu/board.hpp"
#include "mcts/monte_carlo_tree.hpp"

namespace checkers
{

/**
 * @brief Represents the outcome of a game from the engine's perspective.
 */
enum class GameResult { kInProgress = 0, kWhiteWin, kBlackWin, kDraw };

/**
 * @brief The CheckersEngine manages a single board position, current turn,
 *        applying moves, generating moves, and checking for terminal conditions.
 */
class CheckersEngine
{
    public:
    /**
     * @brief Constructs an engine from a given board and side to move.
     */
    CheckersEngine(const checkers::cpu::Board &board, checkers::Turn turn);

    /**
     * @brief Returns the current board state.
     */
    checkers::cpu::Board GetBoard() const;

    /**
     * @brief Returns whose turn it currently is.
     */
    checkers::Turn GetCurrentTurn() const;

    /**
     * @brief Generates all moves for the current side (CPU-based by default).
     *        If no moves are found, sets an internal flag indicating "no moves".
     */
    void GenerateMovesCPU();

    /**
     * @brief Generates all moves for the current side (GPU-based).
     */
    void GenerateMovesGPU();

    /**
     * @brief Returns a reference to the aggregated move generation results.
     */
    const checkers::MoveGenResult &GetLastMoveGenResult() const;

    /**
     * @brief Checks if the internal flag indicates "no moves".
     */
    bool HasNoMoves() const;

    /**
     * @brief Applies a move for the current side. Optionally validates if the move is legal.
     *
     * @param move       The move to be applied (encoded from->to).
     * @param do_validate If true, checks if the move is valid among the previously generated moves.
     * @return True if successfully applied, false if invalid or no moves left.
     */
    bool ApplyMove(checkers::move_t move, bool do_validate = false);

    /**
     * @brief Switches to the next side to move.
     *        Contains logic to see if there's still a capturing chain, etc.
     */
    void SwitchTurnIfNeeded(checkers::move_t last_move);

    /**
     * @brief Checks if the game is over, returning the final outcome if so.
     */
    GameResult CheckGameResult() const;

    /**
     * @brief Sets a "non-reversible" count or increments it to detect draws by 40 moves without captures/king moves.
     */
    void UpdateNonReversibleCount(bool was_capture, checkers::board_index_t from_sq);

    /**
     * @brief Resets the non-reversible move counter to 0.
     */
    void ResetNonReversibleCount();

    private:
    checkers::cpu::Board board_;
    checkers::Turn current_turn_;
    checkers::MoveGenResult last_moves_;
    bool has_no_moves_{false};
    u8 non_reversible_count_{0};

    /**
     * @brief Validates if a move is in the last-generated list (and, if captures exist, is actually a capture).
     */
    bool IsMoveValid(checkers::move_t mv) const;

    /**
     * @brief Finds if the move is a capture by checking the capture mask.
     */
    bool IsCaptureMove(checkers::move_t mv) const;
};

}  // namespace checkers

#endif  // MCTS_CHECKERS_INCLUDE_GAME_CHECKERS_ENGINE_HPP_
