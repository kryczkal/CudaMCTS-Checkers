#include <cuda/launchers.cuh>
#include "cuda/board_helpers.cuh"

int main()
{
    using namespace checkers;
    static constexpr u64 kNumGames      = 100;
    static constexpr u64 kMaxIterations = 100;
    std::vector<board_t> whites(kNumGames, 0);
    std::vector<board_t> blacks(kNumGames, 0);
    std::vector<board_t> kings(kNumGames, 0);
    std::vector<u8> seeds(kNumGames);

    for (auto& seed : seeds) {
        seed = rand();
    }

    for (board_t& board : whites) {
        for (u8 i = 0; i < 4; i++) {
            board |= 1 << i;
        }
    }
    for (board_t& board : blacks) {
        for (u8 i = 28; i < 32; i++) {
            board |= 1 << i;
        }
    }

    auto outcome = checkers::gpu::launchers::HostSimulateCheckersGames(whites, blacks, kings, seeds, kMaxIterations);

    u64 wins  = 0;
    u64 loses = 0;
    u64 draws = 0;
    for (u8 var : outcome) {
        if (var == 1) {
            wins++;
        }
        if (var == 2) {
            loses++;
        }
        if (var == 3) {
            draws++;
        }
    }
    std::cout << wins;
    std::cout << " " << loses;
    std::cout << " " << draws << std::endl;
}
