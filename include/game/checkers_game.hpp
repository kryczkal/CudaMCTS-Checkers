#ifndef MCTS_CHECKERS_INCLUDE_GAME_CHECKERS_GAME_HPP_
#define MCTS_CHECKERS_INCLUDE_GAME_CHECKERS_GAME_HPP_

#ifndef CHECKERS_GAME_HPP
#define CHECKERS_GAME_HPP

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include "cpu/board.hpp"
#include "game/checkers_engine.hpp"
#include "mcts/monte_carlo_tree.hpp"

namespace checkers
{

/**
 * @brief Interface for a GUI (or CLI).
 *        You can implement a class that shows the board, messages,
 *        and collects user input in whichever way you like.
 */
class ICheckersGui
{
    public:
    virtual ~ICheckersGui()                                                                   = default;
    virtual void DisplayBoard(const checkers::cpu::Board& board, const checkers::move_t move) = 0;
    virtual void DisplayMessage(const std::string& msg)                                       = 0;
    virtual std::string PromptForMove()                                                       = 0;
};

enum class PlayerType { kHuman, kAi };

struct GameTypeInfo {
    checkers::Turn start_side    = checkers::Turn::kWhite;
    PlayerType white_player_type = PlayerType::kHuman;
    PlayerType black_player_type = PlayerType::kAi;
    f64 white_time_limit         = 60.0;
    f64 black_time_limit         = 60.0;
    std::optional<mcts::Backend> white_backend;
    std::optional<mcts::Backend> black_backend;
    std::shared_ptr<ICheckersGui> gui = nullptr;
};

/**
 * @brief The CheckersGame manages an interactive game flow,
 *        orchestrating the CheckersEngine and (optionally) a GUI or CLI for user input.
 */
class Game
{
    public:
    /**
     * @brief Constructor: sets up a new game from a certain board with an engine and a "human turn" side.
     */
    Game(const checkers::cpu::Board& initial_board, const GameTypeInfo& game_type_info);

    /**
     * @brief Runs the game loop until completion, optionally saving the move record to a file.
     */
    GameResult Play(const std::string& record_file = "");

    /**
     * @brief Loads a move record from a text file with lines like: "d2-e3" or "d2:f4:d6".
     *        Applies them one by one to the engine state.
     */
    bool LoadGameRecord(const std::string& inputFile);

    /**
     * @brief Converts a move_t to a human-readable string like "d2-e3".
     */
    static std::string GetMoveString(move_t move);

    private:
    /**
     * @brief Helper to parse a single user input line ("d2-e3" or "d2:f4:d6")
     *        and apply it to the engine. Returns success or failure with a message.
     */
    std::tuple<bool, std::string, move_t> AttemptMoveFromNotation(const std::string& move_line);

    /**
     * @brief Splits a multi-capture "d2:f4:d6" into partial moves and applies them in sequence.
     */
    bool ApplyMultiCapture(const std::vector<std::string>& fields);

    /**
     * @brief Converts notation like 'd2' or 'f4' to a 0..31 index, matching your 32-square layout.
     */
    [[nodiscard]] checkers::board_index_t NotationToIndex(const std::string& cell) const;

    /**
     * @brief Saves the entire move history to a text file or prints to stdout if empty file string.
     */
    void SaveRecord(const std::string& recordFile) const;

    private:
    GameTypeInfo game_type_info_;

    std::unique_ptr<checkers::CheckersEngine> engine_;
    std::shared_ptr<ICheckersGui> gui_;

    // For storing all moves in notation form.
    std::vector<std::string> move_history_;
    static std::basic_string<char, std::char_traits<char>, std::allocator<char>> SquareToNotation(board_index_t sq);
    static GameResult GetOppositeSideWin(const Turn& side_to_move);
    [[nodiscard]] bool IsAI(const Turn& side_to_move) const;
    [[nodiscard]] f64 GetSideTimeLimit(const Turn& side_to_move) const;
};

}  // namespace checkers

#endif  // CHECKERS_GAME_HPP

#endif  // MCTS_CHECKERS_INCLUDE_GAME_CHECKERS_GAME_HPP_
