#include "cuda/capture_lookup_table.cuh"
#include "cuda/cuda_utils.cuh"

int init = []() {
    checkers::gpu::apply_move::InitializeCaptureLookupTable();
    CudaUtils::PrintCudaDeviceInfo();
    return 0;
}();
