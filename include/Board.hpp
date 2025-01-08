#ifndef CUDA_MCTS_CHECKRS_INCLUDE_BOARD_HPP_
#define CUDA_MCTS_CHECKRS_INCLUDE_BOARD_HPP_

#include <bitset>
#include <defines.hpp>
#include <types.hpp>

namespace CudaMctsCheckers
{

struct PACK Board {
    //------------------------------------------------------------------------------//
    //                                Static Fields                                 //
    //------------------------------------------------------------------------------//
    static constexpr u8 BOARD_SIZE_X     = 8;  // Board size in the x direction
    static constexpr u8 BOARD_SIZE_Y     = 8;  // Board size in the y direction
    static constexpr u8 BOARD_SIZE_TOTAL = BOARD_SIZE_X * BOARD_SIZE_Y;  // Total board size
    static constexpr u8 BOARD_SIZE_REAL  = BOARD_SIZE_TOTAL / 2;  // Board size used by pieces

    //------------------------------------------------------------------------------//
    //                                    Fields                                    //
    //------------------------------------------------------------------------------//
    u32 white_pieces;  // Bitset of white pieces
    u32 black_pieces;  // Bitset of black pieces
    u32 kings;         // Bitset of white kings
};

}  // namespace CudaMctsCheckers

#endif  // CUDA_MCTS_CHECKRS_INCLUDE_BOARD_HPP_