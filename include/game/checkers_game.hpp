#ifndef MCTS_CHECKERS_INCLUDE_GAME_CHECKERS_GAME_HPP_
#define MCTS_CHECKERS_INCLUDE_GAME_CHECKERS_GAME_HPP_

#ifndef CHECKERS_GAME_HPP
#define CHECKERS_GAME_HPP

#include <memory>
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
    virtual ~ICheckersGui() {}
    virtual void DisplayBoard(const checkers::cpu::Board& board) = 0;
    virtual void DisplayMessage(const std::string& msg)          = 0;
    virtual std::string PromptForMove()                          = 0;
};

/**
 * @brief The CheckersGame manages an interactive game flow,
 *        orchestrating the CheckersEngine and (optionally) a GUI or CLI for user input.
 */
class CheckersGame
{
    public:
    /**
     * @brief Constructor: sets up a new game from a certain board with an engine and a "human turn" side.
     */
    CheckersGame(const checkers::cpu::Board& initialBoard, checkers::Turn startTurn, checkers::Turn humanTurn);

    /**
     * @brief Sets how many seconds the human has to input a move.
     */
    void SetHumanTimeLimit(float seconds);

    /**
     * @brief Sets how many seconds the AI is allowed to think per move.
     */
    void SetAiTimeLimit(float seconds);

    /**
     * @brief Sets the GUI/CLI instance for user interaction.
     */
    void SetGui(std::shared_ptr<ICheckersGui> gui);

    /**
     * @brief Runs the game loop until completion, optionally saving the move record to a file.
     */
    void Play(const std::string& recordFile = "");

    /**
     * @brief Loads a move record from a text file with lines like: "d2-e3" or "d2:f4:d6".
     *        Applies them one by one to the engine state.
     */
    bool LoadGameRecord(const std::string& inputFile);

    private:
    /**
     * @brief Helper to parse a single user input line ("d2-e3" or "d2:f4:d6")
     *        and apply it to the engine. Returns success or failure with a message.
     */
    std::pair<bool, std::string> AttemptMoveFromNotation(const std::string& move_line);

    /**
     * @brief Splits a multi-capture "d2:f4:d6" into partial moves and applies them in sequence.
     */
    bool ApplyMultiCapture(const std::vector<std::string>& fields);

    /**
     * @brief Converts notation like 'd2' or 'f4' to a 0..31 index, matching your 32-square layout.
     */
    checkers::board_index_t NotationToIndex(const std::string& cell) const;

    /**
     * @brief Saves the entire move history to a text file or prints to stdout if empty file string.
     */
    void SaveRecord(const std::string& recordFile) const;

    private:
    std::unique_ptr<checkers::CheckersEngine> engine_;
    checkers::Turn human_turn_;
    float human_time_limit_{60.0f};
    float ai_time_limit_{3.0f};
    std::shared_ptr<ICheckersGui> gui_;

    // For storing all moves in notation form.
    std::vector<std::string> move_history_;
};

}  // namespace checkers

#endif  // CHECKERS_GAME_HPP

#endif  // MCTS_CHECKERS_INCLUDE_GAME_CHECKERS_GAME_HPP_
