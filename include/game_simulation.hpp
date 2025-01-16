#ifndef CUDA_MCTS_CHECKRS_INCLUDE_GAME_SIMULATION_HPP_
#define CUDA_MCTS_CHECKRS_INCLUDE_GAME_SIMULATION_HPP_

#include <board.hpp>
#include <monte_carlo_tree.hpp>

namespace CudaMctsCheckers
{

class GameSimulation
{
    public:
    static constexpr f32 DrawScore = 0.5f;
    static constexpr f32 WinScore  = 1.0f;
    static constexpr f32 LoseScore = 0.0f;

    //------------------------------------------------------------------------------//
    //                        Class Creation and Destruction                        //
    //------------------------------------------------------------------------------//

    GameSimulation()                                  = delete;
    ~GameSimulation()                                 = delete;
    GameSimulation(const GameSimulation &)            = delete;
    GameSimulation &operator=(const GameSimulation &) = delete;
    GameSimulation(GameSimulation &&)                 = delete;

    //------------------------------------------------------------------------------//
    //                                Public Methods                                //
    //------------------------------------------------------------------------------//

    static f32 RunGame(Board &board, Turn turn, GameResult wanted_result);

    static f32 CalcGameScore(const GameResult &wanted_result, const GameResult &result);
};

}  // namespace CudaMctsCheckers

#endif  // CUDA_MCTS_CHECKRS_INCLUDE_GAME_SIMULATION_HPP_
