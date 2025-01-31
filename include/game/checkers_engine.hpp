#ifndef MCTS_CHECKERS_INCLUDE_GAME_CHECKERS_ENGINE_HPP_
#define MCTS_CHECKERS_INCLUDE_GAME_CHECKERS_ENGINE_HPP_

#include "common/checkers_defines.hpp"
#include "cpu/board.hpp"

namespace checkers
{

/**
 * @brief Represents the outcome of a game from the engine's perspective.
 */
enum class GameResult { kInProgress = 0, kWhiteWin, kBlackWin, kDraw };

/**
 * @brief The CheckersEngine manages a single board position and the current turn,
 *        applying moves (including multi-captures), generating moves, and detecting
 *        terminal outcomes such as a forced win, loss, or draw.
 *
 * This version removes any fields like `last_moves_` and `has_no_moves_`. Instead,
 * the engine queries moves on-demand, applies partial or single-jump moves,
 * and handles multi-capturing by not switching the side if a piece can still capture.
 * We also track a 'non-reversible' counter for the 40-move draw rule (some variants use 50).
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
     * @brief Checks if no more moves are possible OR some immediate winning/draw condition
     *        is triggered. If so, we call it a terminal position for MCTS or any game logic.
     */
    bool IsTerminal();

    /**
     * @brief Returns the full game result (InProgress, WhiteWin, BlackWin, or Draw).
     *        If in progress, returns kInProgress.
     */
    GameResult CheckGameResult();

    /**
     * @brief Generates all single-step or single-jump moves for the current side.
     *        This does NOT do multi-capture sequences in one go. Instead, each jump
     *        is treated as a separate move. If there's a mandatory capture, we filter out
     *        non-capturing moves from the generation.
     *
     * @return A vector of possible single-move expansions (encoded from->to in 16 bits).
     */
    MoveGenResult GenerateMoves();

    /**
     * @brief Applies a move for the current side. If it is a capture, checks if the same piece
     *        can continue capturing. Promotion is applied after the chain completes (or after
     *        each jump, depending on your rules). The turn is switched only if the chain is done.
     *
     * @param mv The single jump or step to be applied.
     * @return True if the move was valid and got applied, false if invalid or no piece belongs to side, etc.
     */
    bool ApplyMove(move_t mv, bool validate);

    private:
    checkers::cpu::Board board_;
    checkers::Turn current_turn_;
    GameResult game_result_ = GameResult::kInProgress;

    /**
     * @brief Keeps track of non-reversible moves for the 40-move draw rule.
     *        Increment only if move is by a king and is not capturing.
     *        Reset to 0 if a capture or non-king moves.
     */
    u8 non_reversible_count_{0};

    /**
     * @brief Internal function to check if we can continue capturing from the piece
     *        that just jumped to 'to_sq'. If so, do NOT switch turn. Otherwise, switch it.
     */
    bool CheckAndMaybeContinueCapture(checkers::move_t last_move);

    /**
     * @brief Called after each move or chain to do king promotions.
     */
    void HandlePromotions();

    /**
     * @brief Called after each single jump. If the jump was capturing a piece,
     *        we reset the non-reversible counter. Otherwise, if it was a king move
     *        that didn't capture, we increment the counter.
     *
     * @param was_capture True if the last jump was capturing.
     * @param from_sq The 'from' part of the move to see if it was a king piece.
     */
    void UpdateNonReversibleCount(bool was_capture, checkers::board_index_t from_sq);
};

}  // namespace checkers
#endif  // MCTS_CHECKERS_INCLUDE_GAME_CHECKERS_ENGINE_HPP_
