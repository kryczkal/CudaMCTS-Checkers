#include <iostream>
#include <monte_carlo_tree.hpp>
int main()
{
    CudaMctsCheckers::Board board;

    // Example setup: Initialize some pieces
    // Place white pieces
    board.SetPieceAt<CudaMctsCheckers::BoardCheckType::kWhite>(28);  // (0,1)
    board.SetPieceAt<CudaMctsCheckers::BoardCheckType::kWhite>(29);  // (0,3)
    board.SetPieceAt<CudaMctsCheckers::BoardCheckType::kWhite>(30);  // (0,5)
    board.SetPieceAt<CudaMctsCheckers::BoardCheckType::kWhite>(31);  // (0,7)

    // Place black pieces
    board.SetPieceAt<CudaMctsCheckers::BoardCheckType::kBlack>(0);  // (7,1)
    board.SetPieceAt<CudaMctsCheckers::BoardCheckType::kBlack>(1);  // (7,3)
    board.SetPieceAt<CudaMctsCheckers::BoardCheckType::kBlack>(2);  // (7,5)
    board.SetPieceAt<CudaMctsCheckers::BoardCheckType::kBlack>(3);  // (7,7)

    // Promote a white piece to king
    board.SetPieceAt<CudaMctsCheckers::BoardCheckType::kKings>(2);

    board.SetPieceAt<CudaMctsCheckers::BoardCheckType::kKings>(28);

    // Print the board
    std::cout << board;

    CudaMctsCheckers::MonteCarloTree tree(board, CudaMctsCheckers::Turn::kWhite);
    CudaMctsCheckers::TrieDecodedMoveAsPair best_move = tree.Run(1.f);
    std::cout << "Best move: " << static_cast<u32>(best_move.first) << " -> "
              << static_cast<u32>(best_move.second) << std::endl;

    return 0;
}
