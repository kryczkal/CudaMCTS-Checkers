#include <board.hpp>
#include <checkers_engine.hpp>
#include <game_simulation.hpp>
#include <monte_carlo_tree.hpp>
#include <move_generation.hpp>

namespace CudaMctsCheckers
{

f32 GameSimulation::RunGame(Board &board, Turn turn, GameResult wanted_result)
{
    // Create a CheckersEngine
    CheckersEngine engine(board, turn);

    // Loop until game is over
    GameResult result;
    while ((result = engine.CheckGameResult()) == GameResult::kInProgress) {
        // Apply a random move; if none available => other side effectively loses
        if (!engine.ApplyRandomMove()) {
            return CalcGameScore(
                wanted_result,
                (turn == Turn::kWhite) ? GameResult::kBlackWin : GameResult::kWhiteWin
            );
        }
    }

    return CalcGameScore(wanted_result, result);
}

f32 GameSimulation::CalcGameScore(const GameResult &wanted_result, const GameResult &result)
{
    if (result == GameResult::kDraw) {
        return DrawScore;
    }

    if (wanted_result == GameResult::kBlackWin) {
        if (result == GameResult::kBlackWin) {
            return WinScore;
        } else if (result == GameResult::kWhiteWin) {
            return LoseScore;
        }
    } else {
        if (result == GameResult::kWhiteWin) {
            return WinScore;
        } else if (result == GameResult::kBlackWin) {
            return LoseScore;
        }
    }
}

}  // namespace CudaMctsCheckers
