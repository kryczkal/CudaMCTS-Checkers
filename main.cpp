#include <iostream>
#include <monte_carlo_tree.hpp>

int main()
{
    CudaMctsCheckers::Board board;
    CudaMctsCheckers::MonteCarloTree tree(board);
    return 0;
}