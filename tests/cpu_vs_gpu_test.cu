#include "gtest/gtest.h"

#include "game/checkers_engine.hpp"
#include "game/checkers_game.hpp"
#include "game/cli_gui.hpp"
#include "gtest/gtest.h"

#include "iostream"

namespace checkers
{

static constexpr bool kRunCpuVsGpuTest = false;

class [[maybe_unused]] DummyGui : public checkers::ICheckersGui
{
    public:
    void DisplayBoard(const checkers::cpu::Board& board, const checkers::move_t move) override {}
    void DisplayMessage(const std::string& msg) override {}
    std::string PromptForMove() override { return ""; };
};

static Game SetupGame()
{
    cpu::Board board;
    board.CreateStandard();
    checkers::GameTypeInfo game_type_info;
    game_type_info.black_player_type = checkers::PlayerType::kAi;
    game_type_info.black_backend     = checkers::mcts::Backend::kSingleThreadedCpu;
    game_type_info.white_player_type = checkers::PlayerType::kAi;
    game_type_info.white_backend     = checkers::mcts::Backend::kGpu;
    game_type_info.start_side        = checkers::Turn::kWhite;
    game_type_info.gui               = std::make_shared<checkers::CliGui>();
    game_type_info.black_time_limit  = 0.5f;
    game_type_info.white_time_limit  = 0.5f;

    checkers::Game game(board, game_type_info);
    return game;
}

TEST(CpuVsGpuTest, RunSimulations)
{
    if constexpr (!kRunCpuVsGpuTest) {
        return;
    }
    static u64 kSampleCount = 1e1;
    u64 gpu_win_count       = 0;
    u64 cpu_win_count       = 0;
    for (u64 i = 0; i < kSampleCount; ++i) {
        auto game      = SetupGame();  // Cpu is black, Gpu is white
        GameResult res = game.Play();
        if (res == GameResult::kBlackWin) {
            cpu_win_count++;
        } else if (res == GameResult::kWhiteWin) {
            gpu_win_count++;
        }
    }
    std::cout << "CPU wins: " << cpu_win_count << std::endl;
    std::cout << "GPU wins: " << gpu_win_count << std::endl;
    std::cout << "Draws: " << kSampleCount - cpu_win_count - gpu_win_count << std::endl;
    std::cout << "CPU win rate: " << (double)cpu_win_count / kSampleCount << std::endl;
    std::cout << "GPU win rate: " << (double)gpu_win_count / kSampleCount << std::endl;

    EXPECT_GT(gpu_win_count, cpu_win_count);
}

}  // namespace checkers
