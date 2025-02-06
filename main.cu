#include <iostream>
#include "cpu/board.hpp"
#include "game/checkers_game.hpp"
#include "game/cli_gui.hpp"

checkers::mcts::Backend QueryBackend()
{
    std::cout << "Pick AI backend (0: CPU, 1: Single-threaded CPU, 2: GPU): ";
    u32 backend_num;
    std::cin >> backend_num;

    switch (backend_num) {
        case 0:
            return checkers::mcts::Backend::kCpu;
        case 1:
            return checkers::mcts::Backend::kSingleThreadedCpu;
        case 2:
            return checkers::mcts::Backend::kGpu;
        default:
            return checkers::mcts::Backend::kCpu;
    }
    return checkers::mcts::Backend::kCpu;
}

checkers::PlayerType QueryPlayerType()
{
    std::cout << "Pick player type (0: Human, 1: AI): ";
    u32 player_num;
    std::cin >> player_num;

    switch (player_num) {
        case 0:
            return checkers::PlayerType::kHuman;
        case 1:
            return checkers::PlayerType::kAi;
        default:
            return checkers::PlayerType::kHuman;
    }
    return checkers::PlayerType::kHuman;
}

f64 QueryTimeLimit()
{
    std::cout << "Enter time per move limit (in seconds): ";
    f64 time_limit;
    std::cin >> time_limit;
    return time_limit;
}
std::string QueryGameRecordPath()
{
    std::cout << "Enter path relative to binary to a game record file: ";
    std::string path;
    std::cin >> path;
    if (path.empty()) {
        path = "game_record.txt";
    }
    return path;
}

checkers::GameTypeInfo& QueryInfoForOnePlayer(checkers::GameTypeInfo& game_type_info, checkers::Turn side)
{
    std::cout << "Settings for " << (side == checkers::Turn::kBlack ? "black" : "white") << " player:" << std::endl;
    if (side == checkers::Turn::kBlack) {
        game_type_info.black_player_type = QueryPlayerType();
        if (game_type_info.black_player_type == checkers::PlayerType::kAi) {
            game_type_info.black_backend = QueryBackend();
        }
        game_type_info.black_time_limit = QueryTimeLimit();
    } else {
        game_type_info.white_player_type = QueryPlayerType();
        if (game_type_info.white_player_type == checkers::PlayerType::kAi) {
            game_type_info.white_backend = QueryBackend();
        }
        game_type_info.white_time_limit = QueryTimeLimit();
    }
    return game_type_info;
}

checkers::GameTypeInfo QueryGameTypeInfo()
{
    checkers::GameTypeInfo game_type_info;
    QueryInfoForOnePlayer(game_type_info, checkers::Turn::kBlack);
    QueryInfoForOnePlayer(game_type_info, checkers::Turn::kWhite);
    game_type_info.gui = std::make_shared<checkers::CliGui>();
    return game_type_info;
}

void CustomGame()
{
    checkers::cpu::Board board;
    board.CreateStandard();

    checkers::GameTypeInfo game_type_info = QueryGameTypeInfo();
    checkers::Game game(board, game_type_info);

    game.Play(QueryGameRecordPath());
}

void DefaultHumanAiGame()
{
    checkers::cpu::Board board;
    board.CreateStandard();

    checkers::GameTypeInfo game_type_info;

    game_type_info.black_player_type = checkers::PlayerType::kAi;
    game_type_info.black_backend     = QueryBackend();
    game_type_info.white_player_type = checkers::PlayerType::kHuman;
    game_type_info.start_side        = checkers::Turn::kWhite;
    game_type_info.gui               = std::make_shared<checkers::CliGui>();
    game_type_info.black_time_limit  = 1.5f;
    game_type_info.white_time_limit  = 1000.0f;

    checkers::Game game(board, game_type_info);
    game.Play(QueryGameRecordPath());
}

void DefaultHumanHumanGame()
{
    checkers::cpu::Board board;
    board.CreateStandard();

    checkers::GameTypeInfo game_type_info;
    game_type_info.black_player_type = checkers::PlayerType::kHuman;
    game_type_info.black_backend     = checkers::mcts::Backend::kSingleThreadedCpu;
    game_type_info.white_player_type = checkers::PlayerType::kHuman;
    game_type_info.white_backend     = checkers::mcts::Backend::kSingleThreadedCpu;
    game_type_info.start_side        = checkers::Turn::kWhite;
    game_type_info.gui               = std::make_shared<checkers::CliGui>();
    game_type_info.black_time_limit  = 60.0f;
    game_type_info.white_time_limit  = 60.0f;

    checkers::Game game(board, game_type_info);
    game.Play(QueryGameRecordPath());
}

void DefaultAiAiGame()
{
    checkers::cpu::Board board;
    board.CreateStandard();

    checkers::GameTypeInfo game_type_info;
    game_type_info.black_player_type = checkers::PlayerType::kAi;
    game_type_info.black_backend     = checkers::mcts::Backend::kSingleThreadedCpu;
    game_type_info.white_player_type = checkers::PlayerType::kAi;
    game_type_info.white_backend     = checkers::mcts::Backend::kGpu;
    game_type_info.start_side        = checkers::Turn::kWhite;
    game_type_info.gui               = std::make_shared<checkers::CliGui>();
    game_type_info.black_time_limit  = 2.5f;
    game_type_info.white_time_limit  = 2.5f;

    checkers::Game game(board, game_type_info);
    game.Play(QueryGameRecordPath());
}

int main()
{
    std::cout << "Welcome to Checkers!" << std::endl;
    std::cout << "Quick Start Options:" << std::endl;
    std::cout << "0: Human vs AI" << std::endl;
    std::cout << "1: Human vs Human" << std::endl;
    std::cout << "2: AI (Single-threaded CPU) vs AI (GPU)" << std::endl;
    std::cout << "3: Custom Game" << std::endl;
    std::cout << "4: Exit" << std::endl;
    std::cout << "Enter choice: ";
    i32 choice;
    std::cin >> choice;

    while (choice < 0 || choice > 4) {
        std::cout << "Invalid choice. Please enter a number between 0 and 4: ";
        std::cin >> choice;
    }

    switch (choice) {
        case 0:
            DefaultHumanAiGame();
            break;
        case 1:
            DefaultHumanHumanGame();
            break;
        case 2:
            DefaultAiAiGame();
            break;
        case 3:
            CustomGame();
            break;
        case 4:
            std::cout << "Goodbye!" << std::endl;
            break;
        default:
            std::cout << "Invalid choice." << std::endl;
            break;
    }

    return EXIT_SUCCESS;
}
