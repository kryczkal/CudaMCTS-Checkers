#include <iostream>
#include "cpu/board.hpp"
#include "game/checkers_game.hpp"
#include "game/cli_gui.hpp"

int main()
{
    checkers::cpu::Board board;
    board.CreateStandard();
    checkers::CheckersGame game(board, checkers::Turn::kWhite, checkers::Turn::kWhite);

    game.SetHumanTimeLimit(1000.0f);  // 60 seconds for human
    game.SetAiTimeLimit(10.0f);

    // Attach CLI GUI
    auto gui = std::make_shared<checkers::CliGui>();
    game.SetGui(gui);

    // Optional: Load partial game from a record file
    //     game.LoadGameRecord("game_histories/test_game_16.txt");

    // Play until completion, record into "game_record.txt"
    game.PlayAiAi("game_record.txt");

    return 0;
}
