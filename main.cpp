#include <iostream>
#include "checkers_game.hpp"
#include "cli_gui.hpp"

using namespace CudaMctsCheckers;

int main(int argc, char** argv)
{
    // TODO: Bug: Not working multi-capture
    // TODO: Monte Carlo Descend into tree instead building from scratch

    std::string output_file;
    if (argc > 1) {
        output_file = argv[1];
    }

    // Create a CLI-based GUI
    auto gui = std::make_shared<CliCheckersGui>();

    CheckersGame game{};

    // Attach GUI and set time limit
    game.SetGui(gui);
    game.SetTimeLimit(600.0f);
    game.SetTimeLimitAi(10.0f);

    game.Play(output_file);

    return 0;
}
