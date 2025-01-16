#ifndef CUDA_MCTS_CHECKRS_INCLUDE_CAPTURE_LOOK_UP_TABLE_HPP_
#define CUDA_MCTS_CHECKRS_INCLUDE_CAPTURE_LOOK_UP_TABLE_HPP_

#include <array>
#include <board.hpp>
#include <numeric>

namespace CudaMctsCheckers
{

extern std::array<std::array<Board::HalfBoard, Board::kHalfBoardSize>, Board::kHalfBoardSize>
    kCaptureLookUpTable;

}  // namespace CudaMctsCheckers

#endif  // CUDA_MCTS_CHECKRS_INCLUDE_CAPTURE_LOOK_UP_TABLE_HPP_
