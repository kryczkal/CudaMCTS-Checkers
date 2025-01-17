#ifndef CUDA_MCTS_CHECKRS_INCLUDE_CHECKERS_ENGINE_HPP_
#define CUDA_MCTS_CHECKRS_INCLUDE_CHECKERS_ENGINE_HPP_

#include <board.hpp>
#include <move_generation.hpp>

namespace CudaMctsCheckers
{
enum class Turn { kWhite, kBlack };
enum class GameResult { kWhiteWin, kBlackWin, kDraw, kInProgress };

enum class ApplyMoveType {
    kNoValidate,
    kValidate,
};

/**
 * @brief Core engine to handle checkers logic: applying moves, generating random moves,
 *        switching turns, tracking the board state, etc.
 *
 * This engine can be shared by both human/AI-driven games (CheckersGame) and
 * purely simulated random-play games (GameSimulation).
 */
class CheckersEngine
{
    public:
    /**
     * @brief Default constructor creates an empty engine (empty board, White to move).
     */
    CheckersEngine();

    /**
     * @brief Create an engine with a custom initial board and the current turn.
     */
    CheckersEngine(const Board &board, Turn turn);

    /**
     * @brief Retrieve the current board state.
     */
    [[nodiscard]] const Board &GetBoard() const;

    /**
     * @brief Retrieve the current player's turn.
     */
    Turn GetCurrentTurn() const;

    /**
     * @brief Generate all possible moves for the current player.
     */
    MoveGenerationOutput GenerateCurrentPlayerMoves();

    /**
     * @brief Apply a move (with optional force_capture) from a piece’s origin to the destination.
     * @param from_idx   Index of the piece in half-board representation.
     * @param to_idx     Index to move the piece to.
     * @param force_capture If true, the move must be a capture to be considered valid.
     * @return True if the move was found and applied, otherwise false.
     */
    template <ApplyMoveType type>
    bool ApplyMove(Board::IndexType from_idx, Board::IndexType to_idx);

    /**
     * @brief Applies a random valid move (if any). Returns false if no moves available.
     */
    bool ApplyRandomMove();

    /**
     * @brief Switch turn if no multi-capture is forced. Typically called automatically.
     */
    void SwitchTurnIfNoChainCapture(bool capture_performed);

    /**
     * @brief PromoteAll any pieces that have reached the end rows.
     *        Also increments or resets the non-reversible move counter as needed.
     * @param was_non_reversible True if the move was a normal (non-capturing) pawn move.
     */
    void PromoteAndUpdateReversibleCount(bool was_non_reversible);

    /**
     * @brief Check the game result based on the current board state (win/loss/draw/in progress).
     */
    GameResult CheckGameResult() const;

    /**
     * @brief Restore game state from a history file containing move notations.
     *        Each line in the file should contain a move in the format "b3-a4" or "e4:c6".
     *
     * @param history_file  Path to the history file.
     * @param error_message If a move fails to apply, set the error message here.
     * @return True if all moves were applied successfully, false otherwise.
     */
    bool RestoreFromHistoryFile(const std::string &history_file, std::string &error_message);

    /**
     * @brief Convert a move notation string (e.g., "b3") to an internal board index.
     *
     * @param field The move notation string.
     * @return Board::IndexType The corresponding board index, or Board::kInvalidIndex if invalid.
     */
    Board::IndexType ConvertNotationToIndex(const std::string &field) const;

    private:
    Board board_{};
    Turn current_turn_{Turn::kWhite};

    bool has_precomputed_moves_{false};
    MoveGenerationOutput last_moves_output_{};

    MoveGenerationOutput GetPrecomputedMovesOrGenerate();

    void PromoteAll();

    void UpdateTimeFromNonReversibleMove(Move::Type played_move, bool was_capture);

    void PlayMove(unsigned char from_idx, unsigned char to_idx, bool is_capture);

    static bool IsMoveValid(
        Board::IndexType from_idx, Move::Type to_idx, const MoveGenerationOutput &possible_moves
    );
};

}  // namespace CudaMctsCheckers

#include <checkers_engine.tpp>

#endif  // CUDA_MCTS_CHECKRS_INCLUDE_CHECKERS_ENGINE_HPP_
