#ifndef CUDA_MCTS_CHECKERS_CLI_GUI_HPP
#define CUDA_MCTS_CHECKERS_CLI_GUI_HPP

#include <iostream>
#include <string>
#include "checkers_game.hpp"  // for ICheckersGui (or forward-declare it if needed)

namespace CudaMctsCheckers
{

/**
 * @brief A simple CLI-based GUI adapter as an example.
 */
class CliCheckersGui : public ICheckersGui
{
    public:
    CliCheckersGui()           = default;
    ~CliCheckersGui() override = default;

    /**
     * @brief Display the current board state to the user.
     */
    void DisplayBoard(const Board &board) override;

    /**
     * @brief Display a text message (log, prompt, etc.).
     */
    void DisplayMessage(const std::string &msg) override;

    /**
     * @brief Prompt the user (e.g., from console input) to enter a move in the requested notation.
     * @return The move string in the checkers notation (e.g. "d2-e3" or "d2:f4:d6").
     */
    std::string PromptForMove() override;
};

}  // namespace CudaMctsCheckers

#endif  // CUDA_MCTS_CHECKERS_CLI_GUI_HPP
