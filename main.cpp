#include <iostream>
#include "checkers_game.hpp"
#include "cli_gui.hpp"

using namespace CudaMctsCheckers;

int main(int argc, char** argv)
{
    // TODO: Bug: Kings stop being kings
    // TODO: Bug: Not working multi-capture
    // TODO: Bug: Overlapping tiles possible ???
    // TODO: Bug: Possibly counting wins in reverse

    std::string output_file;
    if (argc > 1) {
        output_file = argv[1];
    }

    // Create a CLI-based GUI
    auto gui = std::make_shared<CliCheckersGui>();

    // Create a CheckersGame with default setup (White=human)
    CheckersGame game;

    // Attach GUI and set time limit
    game.SetGui(gui);
    game.SetTimeLimit(600.0f);
    game.SetTimeLimitAi(5.0f);

    // Run the game
    game.Play(output_file);

    return 0;
}
