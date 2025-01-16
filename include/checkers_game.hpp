#ifndef CUDA_MCTS_CHECKERS_CHECKERS_GAME_HPP
#define CUDA_MCTS_CHECKERS_CHECKERS_GAME_HPP

#include <board.hpp>
#include <checkers_engine.hpp>  // <-- Use the new engine
#include <memory>
#include <monte_carlo_tree.hpp>
#include <move_generation.hpp>
#include <string>
#include <utility>
#include <vector>

namespace CudaMctsCheckers
{

/**
 * @brief Interface for a GUI/CLI adapter that can display the board, prompt for moves, etc.
 */
class ICheckersGui
{
    public:
    virtual ~ICheckersGui() = default;

    /**
     * @brief Display the current board state to the user.
     */
    virtual void DisplayBoard(const Board &board) = 0;

    /**
     * @brief Display a text message.
     */
    virtual void DisplayMessage(const std::string &msg) = 0;

    /**
     * @brief Prompt the user to enter a move in notation. Return the string typed.
     */
    virtual std::string PromptForMove() = 0;
};

/**
 * @brief Main class to run a checkers game between a human and an AI (or two humans).
 *
 * The default constructor sets up a normal game with White as the human. Another constructor
 * allows specifying the human’s color. Yet another takes a board to start from.
 */
class CheckersGame
{
    public:
    /**
     * @brief Default constructor creates a standard start position with White as the human.
     */
    CheckersGame();

    /**
     * @brief Create a game with standard start position, specifying which side is human.
     */
    explicit CheckersGame(Turn humanTurn);

    /**
     * @brief Create a game from an existing board, specifying which side is human (white by
     * default).
     */
    CheckersGame(const Board &board, Turn humanTurn = Turn::kWhite);

    /**
     * @brief Set the time limit per move (in seconds) for the human.
     */
    void SetTimeLimit(f32 seconds);

    /**
     * @brief Set time limit per move for AI (in seconds).
     */
    void SetTimeLimitAi(f32 seconds);

    /**
     * @brief Attach a GUI object for display/input. If not set, runs “headless”.
     */
    void SetGui(std::shared_ptr<ICheckersGui> gui);

    /**
     * @brief Start the game loop. If output_file is nonempty, store the final notation record.
     */
    void Play(const std::string &output_file = "");

    private:
    /**
     * @brief Determines if it's the human player's turn.
     */
    [[nodiscard]] bool IsHumanTurn() const;

    /**
     * @brief Fill the board with a standard checkers initial setup.
     */
    void SetupStandardBoard(Board &board);

    /**
     * @brief Attempt to parse and apply a move from user notation.
     * @return {true, ""} if success, else {false, errorMessage}
     */
    std::pair<bool, std::string> AttemptMoveViaEngine(const std::string &move_str);

    /**
     * @brief If the user typed a multi-capture (e.g. d2:f4:d6), handle that.
     */
    bool ApplyMultiCaptureMoveViaEngine(const std::vector<std::string> &fields);

    /**
     * @brief Convert e.g. "d2" to internal half-board index. Return kInvalidIndex if invalid.
     */
    [[nodiscard]] Board::IndexType ConvertNotationToIndex(const std::string &field) const;

    /**
     * @brief Write move_history_ to a file or print to console if no file given.
     */
    void SaveGameRecord(const std::string &output_file) const;

    private:
    // We no longer store Board & Turn directly; we store a CheckersEngine
    std::unique_ptr<CheckersEngine> engine_{};

    Turn human_turn_{Turn::kWhite};
    f32 time_limit_per_move_ai_ = 5.f;  // default 5 seconds for AI
    f32 time_limit_per_move_    = 5.f;  // default 5 seconds for human
    std::vector<std::string> move_history_;

    std::shared_ptr<ICheckersGui> gui_{};
};

}  // namespace CudaMctsCheckers

#endif  // CUDA_MCTS_CHECKERS_CHECKERS_GAME_HPP
