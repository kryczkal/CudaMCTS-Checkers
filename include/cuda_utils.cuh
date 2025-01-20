#ifndef MCTS_CHECKERS_INCLUDE_CUDA_UTILS_CUH_
#define MCTS_CHECKERS_INCLUDE_CUDA_UTILS_CUH_

/**
 * @brief Utility class for CUDA related functions
 */
class CudaUtils
{
    public:
    static void printCudaDeviceInfo();
    static void check_cuda_error(cudaError_t err, char const *func, char const *file, int line);
    static void check_last_cuda_error(char const *file, int line);
};

#define CHECK_CUDA_ERROR(val)   CudaUtils::check_cuda_error((val), #val, __FILE__, __LINE__)
#define CHECK_LAST_CUDA_ERROR() CudaUtils::check_last_cuda_error(__FILE__, __LINE__)
#define CEIL_DIV(x, y)          (((x) + (y) - 1) / (y))

#endif  // MCTS_CHECKERS_INCLUDE_CUDA_UTILS_CUH_
