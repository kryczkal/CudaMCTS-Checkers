#include <board.hpp>
#include <game_simulation.hpp>
#include <monte_carlo_tree.hpp>
#include <move_generation.hpp>

namespace CudaMctsCheckers
{

f32 GameSimulation::RunGame(Board &board, Turn turn)
{
    MoveGenerationOutput output;
    GameResult result;

    //        std::cout << "Running game" << std::endl;
    //        std::cout << board;

    bool capture      = false;
    bool prev_capture = false;

    while ((result = board.CheckGameResult()) == GameResult::kInProgress) {
        //            std::cout << "Turn: " << (turn == Turn::kWhite ? "White" : "Black") <<
        //            std::endl;
        if (turn == Turn::kWhite) {
            output = MoveGenerator::GenerateMovesForPlayerCpu<BoardCheckType::kWhite>(board);
        } else {
            output = MoveGenerator::GenerateMovesForPlayerCpu<BoardCheckType::kBlack>(board);
        }
        capture = output.capture_moves_bitmask[MoveGenerationOutput::CaptureFlagIndex];
        if (prev_capture && !capture) {
            turn         = turn == Turn::kWhite ? Turn::kBlack : Turn::kWhite;
            prev_capture = false;
            continue;
        }

        u32 random_move_index = rand() % Move::kNumMoveArrayForPlayerSize;
        while (output.possible_moves[random_move_index] == Move::kInvalidMove ||
               (capture && !output.capture_moves_bitmask[random_move_index])) {
            random_move_index = (++random_move_index) % Move::kNumMoveArrayForPlayerSize;
        }
        //            std::cout << "Making move: " <<
        //            static_cast<u32>(Move::DecodeOriginIndex(random_move_index)) << " -> " <<
        //            static_cast<u32>(output.possible_moves[random_move_index]) << std::endl;

        if (turn == Turn::kWhite) {
            board.ApplyMove<BoardCheckType::kWhite>(
                Move::DecodeOriginIndex(random_move_index),
                output.possible_moves[random_move_index], capture
            );
        } else {
            board.ApplyMove<BoardCheckType::kBlack>(
                Move::DecodeOriginIndex(random_move_index),
                output.possible_moves[random_move_index], capture
            );
        }

        board.PromoteAll();

        if (board.IsPieceAt<BoardCheckType::kKings>(Move::DecodeOriginIndex(random_move_index)) &&
            !capture) {
            board.time_from_non_reversible_move++;
        } else {
            board.time_from_non_reversible_move = 0;
        }

        turn         = capture ? turn : turn == Turn::kWhite ? Turn::kBlack : Turn::kWhite;
        prev_capture = capture;

        //            std::cout << board;
    }

    //        std::cout << "Game result: " << (result == GameResult::kWhiteWin ? "White wins" :
    //        result == GameResult::kBlackWin ? "Black wins" : "Draw") << std::endl;

    if (result == GameResult::kWhiteWin) {
        return GameSimulation::WinScore;
    } else if (result == GameResult::kBlackWin) {
        return GameSimulation::LoseScore;
    } else {
        return GameSimulation::DrawScore;
    }
}
}  // namespace CudaMctsCheckers
