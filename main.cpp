#include <iostream>
#include "cpu/board.hpp"
#include "game/checkers_game.hpp"
#include "game/cli_gui.hpp"

static checkers::cpu::Board CreateStandardBoard()
{
    // Standard 8x8 checkers layout:
    //   black in rows 0..2 (top), white in rows 5..7 (bottom).
    //   row=0 => squares 0..3, row=1 => squares 4..7, row=2 => squares 8..11
    //   row=5 => squares 20..23, row=6 => squares 24..27, row=7 => squares 28..31
    checkers::cpu::Board board;
    board.white = 0;
    board.black = 0;
    board.kings = 0;

    // place black pieces on squares 0..11
    for (int i = 0; i < 12; i++) {
        board.black |= (1U << i);
    }
    // place white pieces on squares 20..31
    for (int i = 20; i < 32; i++) {
        board.white |= (1U << i);
    }

    return board;
}

int main()
{
    // Create standard board
    checkers::cpu::Board board = CreateStandardBoard();

    // White to move first, White is human, Black is AI
    checkers::CheckersGame game(board, checkers::Turn::kWhite, checkers::Turn::kWhite);

    // Set time limits
    game.SetHumanTimeLimit(120.0f);  // 60 seconds for human
    game.SetAiTimeLimit(4.0f);       // 3 seconds for AI

    // Attach CLI GUI
    auto gui = std::make_shared<checkers::CliGui>();
    game.SetGui(gui);

    // Optional: Load partial game from a record file
    // game.LoadGameRecord("my_previous_game.txt");

    // Play until completion, record into "game_record.txt"
    game.Play("game_record.txt");

    return 0;
}
