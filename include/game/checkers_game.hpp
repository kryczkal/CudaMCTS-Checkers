#ifndef MCTS_CHECKERS_INCLUDE_GAME_CHECKERS_GAME_HPP_
#define MCTS_CHECKERS_INCLUDE_GAME_CHECKERS_GAME_HPP_

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "cpu/board.hpp"
#include "game/checkers_engine.hpp"
#include "mcts/monte_carlo_tree.hpp"

namespace checkers
{

// Defines the available game modes.
enum class GameMode { HumanVsHuman, HumanVsAi, AiVsAi };

/**
 * @brief Interface for a GUI/CLI to interact with the Checkers game.
 */
class ICheckersGui
{
    public:
    virtual ~ICheckersGui() {}
    virtual void DisplayBoard(const checkers::cpu::Board& board, const checkers::move_t move) = 0;
    virtual void DisplayMessage(const std::string& msg)                                       = 0;
    virtual std::string PromptForMove()                                                       = 0;
};

/**
 * @brief The CheckersGame class orchestrates game flow for Checkers.
 *        It supports Human vs Human, Human vs AI, and AI vs AI game modes.
 *        Additionally, it provides a simple command parser for special commands.
 */
class CheckersGame
{
    public:
    /**
     * @brief Constructor.
     *
     * @param initialBoard The starting board configuration.
     * @param startTurn The side to move first.
     * @param mode The game mode (HumanVsHuman, HumanVsAi, AiVsAi).
     * @param humanTurn For modes involving humans, which turn the human controls.
     */
    CheckersGame(
        const checkers::cpu::Board& initialBoard, checkers::Turn startTurn, GameMode mode,
        checkers::Turn humanTurn = checkers::Turn::kWhite
    );

    void SetHumanTimeLimit(float seconds);
    void SetAiTimeLimit(float seconds);
    void SetGui(std::shared_ptr<ICheckersGui> gui);
    void SetSimulationBackend(mcts::SimulationBackend backend);

    /**
     * @brief Runs the game loop until completion or until a quit command is issued.
     *
     * @param recordFile Optional filename in which to save the move record.
     */
    void Play(const std::string& recordFile = "");

    /**
     * @brief Loads a game record from file and applies the moves.
     */
    bool LoadGameRecord(const std::string& inputFile);

    private:
    // Helper for parsing move notation from human input.
    std::tuple<bool, std::string, move_t> AttemptMoveFromNotation(const std::string& move_line);
    bool ApplyMultiCapture(const std::vector<std::string>& fields);
    checkers::board_index_t NotationToIndex(const std::string& cell) const;
    std::string SquareToNotation(checkers::board_index_t sq) const;
    void SaveRecord(const std::string& recordFile) const;

    // Command parser related methods.
    void InitializeCommandParser();
    bool ProcessCommand(const std::string& input);
    void CommandHelp();
    void CommandDumpBoard();
    void CommandSave();
    void CommandQuit();

    std::unique_ptr<checkers::CheckersEngine> engine_;
    checkers::Turn human_turn_;
    GameMode game_mode_;
    float human_time_limit_{60.0f};
    float ai_time_limit_{3.0f};
    std::shared_ptr<ICheckersGui> gui_;
    std::vector<std::string> move_history_;

    // AI simulation backend (CPU or GPU).
    mcts::SimulationBackend simulation_backend_{mcts::SimulationBackend::GPU};

    // Flag used by the command parser to quit the game loop.
    bool quit_{false};

    // Command map: maps command strings to their handler functions.
    std::unordered_map<std::string, std::function<void()>> command_map_;
};

}  // namespace checkers

#endif  // MCTS_CHECKERS_INCLUDE_GAME_CHECKERS_GAME_HPP_
