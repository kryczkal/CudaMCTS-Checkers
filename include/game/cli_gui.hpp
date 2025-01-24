#ifndef MCTS_CHECKERS_INCLUDE_GAME_CLI_GUI_HPP_
#define MCTS_CHECKERS_INCLUDE_GAME_CLI_GUI_HPP_

#include <iostream>
#include "game/checkers_game.hpp"

namespace checkers
{

/**
 * @brief A simple CLI-based GUI that prints the board to std::cout and reads user moves from std::cin.
 */
class CliGui : public ICheckersGui
{
    public:
    virtual ~CliGui() {}

    void DisplayBoard(const checkers::cpu::Board &board) override;

    void DisplayMessage(const std::string &msg) override;

    std::string PromptForMove() override;
};

}  // namespace checkers

#endif  // MCTS_CHECKERS_INCLUDE_GAME_CLI_GUI_HPP_
