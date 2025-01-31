#include "game/checkers_engine.hpp"
#include "game/checkers_game.hpp"
#include "game/cli_gui.hpp"
#include "gtest/gtest.h"

#include "iostream"

namespace checkers
{

static CheckersGame SetupGame(std::string game_file)
{
    cpu::Board board;
    board.CreateStandard();
    CheckersGame game(board, Turn::kWhite, Turn::kWhite);
    game.LoadGameRecord(game_file);
    game.SetAiTimeLimit(4.0f);
    game.SetHumanTimeLimit(120.0f);

    auto gui = std::make_shared<CliGui>();
    game.SetGui(gui);
    return game;
}

/**
 * This is a type of manual test. It is used to load a game (that caused some problems)
 * from a file and play it.
 */
TEST(GameHistoriesTest, TestGame1)
{
    //    auto game = SetupGame("game_histories/test_game_13.txt");

    //    game.Play();
}
}  // namespace checkers
