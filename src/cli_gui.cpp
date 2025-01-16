#include "cli_gui.hpp"
#include <iostream>

namespace CudaMctsCheckers
{

void CliCheckersGui::DisplayBoard(const Board &board) { std::cout << board << std::endl; }

void CliCheckersGui::DisplayMessage(const std::string &msg) { std::cout << msg << std::endl; }

std::string CliCheckersGui::PromptForMove()
{
    std::cout << "Enter your move (example: d2-e3 or d2:f4:d6): ";
    std::string move_str;
    std::getline(std::cin, move_str);
    return move_str;
}

}  // namespace CudaMctsCheckers
