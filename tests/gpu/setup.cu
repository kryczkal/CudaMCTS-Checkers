#include "cuda/capture_lookup_table.cuh"
#include "cuda/check_outcome.cuh"
#include "cuda/cuda_utils.cuh"
#include "cuda/game_simulation.cuh"

int init = []() {
    checkers::gpu::apply_move::InitializeCaptureLookupTable();
    CudaUtils::PrintCudaDeviceInfo();
    return 0;
}();
