#ifndef CUDA_MCTS_CHECKRS_INCLUDE_MOVE_GENERATION_HPP_
#define CUDA_MCTS_CHECKRS_INCLUDE_MOVE_GENERATION_HPP_

#include <move.hpp>

namespace CudaMctsCheckers
{

struct PACK MoveGenerationOutput {
    Move::MoveArrayForPlayer possible_moves;
    bool detected_capture;
};

class MoveGenerator
{
    public:
    //------------------------------------------------------------------------------//
    //                        Class Creation and Destruction                        //
    //------------------------------------------------------------------------------//
    // Static class
    MoveGenerator()                                 = delete;
    ~MoveGenerator()                                = delete;
    MoveGenerator(const MoveGenerator &)            = delete;
    MoveGenerator &operator=(const MoveGenerator &) = delete;
    MoveGenerator(MoveGenerator &&)                 = delete;

    //------------------------------------------------------------------------------//
    //                                Public Methods                                //
    //------------------------------------------------------------------------------//
    template <BoardCheckType type>
    static MoveGenerationOutput GenerateMovesForPlayerCpu(const Board &board);

    template <BoardCheckType type, MoveDirection direction>
    static void GenerateMovesDiagonalCpu(
        const Board &board, MoveGenerationOutput &output, Board::IndexType index,
        Board::IndexType &move_index
    );

    //------------------------------------------------------------------------------//
    //                               Public Variables                               //
    //------------------------------------------------------------------------------//

    private:
    //------------------------------------------------------------------------------//
    //                                Private Methods                               //
    //------------------------------------------------------------------------------//

    //------------------------------------------------------------------------------//
    //                               Private Variables                              //
    //------------------------------------------------------------------------------//
};

}  // namespace CudaMctsCheckers

#include <move_generation.tpp>

#endif  // CUDA_MCTS_CHECKRS_INCLUDE_MOVE_GENERATION_HPP_
