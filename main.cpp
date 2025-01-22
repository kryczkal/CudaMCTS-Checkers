#include <cuda/launchers.cuh>
#include "cuda/board_helpers.cuh"
#include "cuda/game_simulation.cuh"

int main()
{
    using namespace checkers;
    static constexpr u64 kNumGames      = 10000;
    static constexpr u64 kMaxIterations = 100;
    std::vector<board_t> whites(kNumGames, 0);
    std::vector<board_t> blacks(kNumGames, 0);
    std::vector<board_t> kings(kNumGames, 0);
    std::vector<u8> seeds(kNumGames);

    srand(time(nullptr));
    for (auto& seed : seeds) {
        seed = rand();
    }

    for (board_t& board : whites) {
        for (u8 i = 32 - 12; i < 32; i++) {
            board |= 1 << i;
        }
    }
    for (board_t& board : blacks) {
        for (u8 i = 0; i < 11; i++) {
            board |= 1 << i;
        }
    }

    auto outcome = checkers::gpu::launchers::HostSimulateCheckersGames(whites, blacks, kings, seeds, kMaxIterations);

    u64 wins  = 0;
    u64 loses = 0;
    u64 draws = 0;
    for (u8 var : outcome) {
        std::cout << static_cast<u32>(var) << " ";
        if (var == checkers::gpu::kOutcomeWhite) {
            wins++;
        }
        if (var == checkers::gpu::kOutcomeBlack) {
            loses++;
        }
        if (var == checkers::gpu::kOutcomeDraw) {
            draws++;
        }
    }
    std::cout << std::endl;
    std::cout << wins;
    std::cout << " " << loses;
    std::cout << " " << draws << std::endl;
}
