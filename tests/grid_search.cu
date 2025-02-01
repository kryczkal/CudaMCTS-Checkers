#include "common/checkers_defines.hpp"
#include "common/parallel.hpp"
#include "gtest/gtest.h"
#include "mcts/monte_carlo_tree.hpp"

static constexpr bool kRunSimulationSpeedTest = false;
TEST(ManualTest, SimulationSpeed)
{
    using namespace checkers::mcts;
    if constexpr (kRunSimulationSpeedTest) {
        // This test serves as a manual test to check the simulation speed of the MCTS algorithm.
        // Relevant values are kMaxTotalSimulations, kNumThreadsCPU, kNumBoardsPerBlock, kThreadsPerBoardInSimulation.
        static constexpr u64 kSimulationTimeSeconds = 4;
        static constexpr u64 kTestCount             = 6;
        checkers::cpu::Board board;
        board.CreateStandard();
        u64 total_simulation_count = 0;
        for (u64 i = 0; i < kTestCount; ++i) {
            board.CreateStandard();
            checkers::mcts::MonteCarloTree mcts(board, checkers::Turn::kWhite);
            mcts.RunParallel(kSimulationTimeSeconds, checkers::kNumThreadsCPU);
            total_simulation_count += mcts.GetTotalSimulations();
        }
        std::cout << "Total simulations: " << total_simulation_count << std::endl;
        std::cout << "Simulations per second: " << total_simulation_count / (kTestCount * kSimulationTimeSeconds)
                  << std::endl;
    }
}
