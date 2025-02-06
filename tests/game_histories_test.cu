#include "game/checkers_engine.hpp"
#include "game/checkers_game.hpp"
#include "game/cli_gui.hpp"
#include "gtest/gtest.h"

#include "iostream"

namespace checkers
{

static constexpr bool kRunGameHistoriesTest = false;

[[maybe_unused]] static Game SetupGame(std::string game_file)
{
    cpu::Board board;
    board.CreateStandard();
    checkers::GameTypeInfo game_type_info;
    game_type_info.black_player_type = checkers::PlayerType::kAi;
    game_type_info.black_backend     = checkers::mcts::Backend::kCpu;
    game_type_info.white_player_type = checkers::PlayerType::kHuman;
    game_type_info.white_backend     = checkers::mcts::Backend::kGpu;
    game_type_info.start_side        = checkers::Turn::kWhite;
    game_type_info.gui               = std::make_shared<checkers::CliGui>();
    game_type_info.black_time_limit  = 2.0f;
    game_type_info.white_time_limit  = 2.0f;

    checkers::Game game(board, game_type_info);
    return game;
}

/**
 * This is a type of manual test. It is used to load a game (that caused some problems)
 * from a file and play it.
 */
TEST(GameHistoriesTest, TestGame1)
{
    if constexpr (!kRunGameHistoriesTest) {
        return;
    }
    auto game = SetupGame("game_histories/test_game_13.txt");
    game.Play();
}
}  // namespace checkers
