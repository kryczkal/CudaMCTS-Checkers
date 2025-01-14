#include <iostream>
#include <monte_carlo_tree.hpp>

int main()
{
    CudaMctsCheckers::Board board;

    // Example setup: Initialize some pieces
    // Place white pieces
    board.SetPieceAt<CudaMctsCheckers::BoardCheckType::kWhite>(0);  // (0,1)
    board.SetPieceAt<CudaMctsCheckers::BoardCheckType::kWhite>(1);  // (0,3)
    board.SetPieceAt<CudaMctsCheckers::BoardCheckType::kWhite>(2);  // (0,5)
    board.SetPieceAt<CudaMctsCheckers::BoardCheckType::kWhite>(3);  // (0,7)

    // Place black pieces
    board.SetPieceAt<CudaMctsCheckers::BoardCheckType::kBlack>(28);  // (7,1)
    board.SetPieceAt<CudaMctsCheckers::BoardCheckType::kBlack>(29);  // (7,3)
    board.SetPieceAt<CudaMctsCheckers::BoardCheckType::kBlack>(30);  // (7,5)
    board.SetPieceAt<CudaMctsCheckers::BoardCheckType::kBlack>(31);  // (7,7)

    // Promote a white piece to king
    board.SetPieceAt<CudaMctsCheckers::BoardCheckType::kKings>(2);

    board.SetPieceAt<CudaMctsCheckers::BoardCheckType::kKings>(28);

    // Print the board
    std::cout << board;

    // Initialize Monte Carlo Tree (optional)
    CudaMctsCheckers::MonteCarloTree tree(board);

    return 0;
}
