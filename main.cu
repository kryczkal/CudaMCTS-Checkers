#include <iostream>
#include <limits>
#include <memory>
#include <string>

// Include the board, game, and CLI GUI headers.
#include "cpu/board.hpp"
#include "game/checkers_game.hpp"
#include "game/cli_gui.hpp"

int main()
{
    using namespace checkers;

    // Display welcome message.
    std::cout << "Welcome to Checkers!\n\n";

    // Explain available quick-match options.
    std::cout << "Select a match type:\n"
              << "  1) Human vs AI (default options)\n"
              << "  2) Human vs Human (default options)\n"
              << "  3) AI vs AI (default options)\n"
              << "  4) Custom match (enter all parameters)\n"
              << "Your choice: ";

    u32 match_choice = 0;
    std::cin >> match_choice;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    // Default values.
    checkers::GameMode game_mode    = checkers::GameMode::HumanVsAi;
    checkers::Turn human_turn       = checkers::Turn::kWhite;
    f32 human_time_limit            = 60.0f;
    f32 ai_time_limit               = 3.0f;
    mcts::SimulationBackend backend = mcts::SimulationBackend::GPU;

    if (match_choice == 1 || match_choice == 2 || match_choice == 3) {
        // Quick-start options with default parameters.
        if (match_choice == 1) {
            game_mode = checkers::GameMode::HumanVsAi;
            std::cout << "\nQuick Match: Human vs AI using default options.\n";
        } else if (match_choice == 2) {
            game_mode = checkers::GameMode::HumanVsHuman;
            std::cout << "\nQuick Match: Human vs Human using default options.\n";
        } else if (match_choice == 3) {
            game_mode = checkers::GameMode::AiVsAi;
            std::cout << "\nQuick Match: AI vs AI using default options.\n";
        }
        // Defaults already set.
    } else if (match_choice == 4) {
        // Custom match: query every parameter.
        u32 custom_mode = 0;
        std::cout << "\nCustom Match Configuration:\n";
        std::cout << "Select game mode:\n"
                  << "  1) Human vs AI\n"
                  << "  2) Human vs Human\n"
                  << "  3) AI vs AI\n"
                  << "Your choice: ";
        std::cin >> custom_mode;
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        switch (custom_mode) {
            case 1:
                game_mode = checkers::GameMode::HumanVsAi;
                break;
            case 2:
                game_mode = checkers::GameMode::HumanVsHuman;
                break;
            case 3:
                game_mode = checkers::GameMode::AiVsAi;
                break;
            default:
                std::cout << "Invalid selection. Defaulting to Human vs AI.\n";
                game_mode = checkers::GameMode::HumanVsAi;
        }

        // For modes with an AI component, query the simulation backend.
        if (game_mode == checkers::GameMode::HumanVsAi || game_mode == checkers::GameMode::AiVsAi) {
            u32 backend_choice = 0;
            std::cout << "\nSelect AI simulation backend:\n"
                      << "  1) CPU\n"
                      << "  2) GPU\n"
                      << "Your choice: ";
            std::cin >> backend_choice;
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            if (backend_choice == 1) {
                backend = mcts::SimulationBackend::CPU;
            } else if (backend_choice == 2) {
                backend = mcts::SimulationBackend::GPU;
            } else {
                std::cout << "Invalid selection. Defaulting to GPU.\n";
                backend = mcts::SimulationBackend::GPU;
            }
        }

        // For modes with human input, query the human time limit.
        if (game_mode == checkers::GameMode::HumanVsAi || game_mode == checkers::GameMode::HumanVsHuman) {
            std::cout << "\nEnter time limit (in seconds) for human moves: ";
            std::cin >> human_time_limit;
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }

        // For modes with AI moves, query the AI time limit.
        if (game_mode == checkers::GameMode::HumanVsAi || game_mode == checkers::GameMode::AiVsAi) {
            std::cout << "\nEnter time limit (in seconds) for AI moves: ";
            std::cin >> ai_time_limit;
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }

        // If human is involved in a Human vs AI match, query which side the human plays.
        if (game_mode == checkers::GameMode::HumanVsAi) {
            u32 side_choice = 0;
            std::cout << "\nFor Human vs AI, which side should the human play?\n"
                      << "  1) White\n"
                      << "  2) Black\n"
                      << "Your choice: ";
            std::cin >> side_choice;
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            human_turn = (side_choice == 2) ? checkers::Turn::kBlack : checkers::Turn::kWhite;
        }
    } else {
        std::cout << "Invalid match type. Exiting.\n";
        return EXIT_FAILURE;
    }

    // Only CLI is implemented at present.
    auto gui = std::make_shared<checkers::CliGui>();

    // Create a standard board.
    checkers::cpu::Board board;
    board.CreateStandard();

    // Construct the game object.
    // Note: The CheckersGame constructor takes the board, the starting turn (always white at game start),
    // the game mode, and, for modes involving humans, which turn is controlled by the human.
    checkers::CheckersGame game(board, checkers::Turn::kWhite, game_mode, human_turn);

    // Set the collected parameters.
    game.SetGui(gui);
    game.SetHumanTimeLimit(human_time_limit);
    game.SetAiTimeLimit(ai_time_limit);
    game.SetSimulationBackend(backend);

    // Start the game loop.
    game.Play();

    return EXIT_SUCCESS;
}
