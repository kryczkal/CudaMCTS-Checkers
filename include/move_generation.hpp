#ifndef CUDA_MCTS_CHECKRS_INCLUDE_MOVE_GENERATION_HPP_
#define CUDA_MCTS_CHECKRS_INCLUDE_MOVE_GENERATION_HPP_

#include <move.hpp>

namespace CudaMctsCheckers
{

struct MoveGenerationOutput {
    static constexpr u32 CaptureFlagIndex = Move::kNumMoveArrayForPlayerSize;
    Move::MoveArrayForPlayer possible_moves;
    std::bitset<Move::kNumMoveArrayForPlayerSize + 1>
        capture_moves_bitmask;  // Last bit is for detected capture
};

class MoveGenerator
{
    public:
    //------------------------------------------------------------------------------//
    //                        Class Creation and Destruction                        //
    //------------------------------------------------------------------------------//

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

    template <BoardCheckType type>
    WRAP_CALL void GenerateMovesPieceCpu(
        const Board &board, MoveGenerationOutput &output, Board::IndexType i, u32 current_move_index
    );

    template <BoardCheckType type, MoveDirection direction>
    WRAP_CALL void GenerateMovesDiagonalCpu(
        const Board &board, MoveGenerationOutput &output, Board::IndexType index,
        u32 &current_move_index
    );

    template <BoardCheckType type>
    WRAP_CALL void GenerateMovesKingCpu(
        const Board &board, MoveGenerationOutput &output, Board::IndexType i, u32 current_move_index
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
