#ifndef CUDA_MCTS_CHECKRS_INCLUDE_DEFINES_HPP_
#define CUDA_MCTS_CHECKRS_INCLUDE_DEFINES_HPP_

// https://github.com/Jlisowskyy/AlkOS/blob/dev/alkos/kernel/include/defines.hpp
// ------------------------------
// Attribute macros
// ------------------------------

/* Prevent the compiler from adding padding to structures */
#define PACK __attribute__((__packed__))

/* Indicate that the function will never return */
#define NO_RET __attribute__((noreturn))

/* Force the compiler to always inline the function */
#define FORCE_INLINE inline __attribute__((always_inline))

/* Declare a function as a static inline wrapper */
#define WRAP_CALL static FORCE_INLINE

#define MAYBE_UNUSED(x) (void)(x)

#endif  // CUDA_MCTS_CHECKRS_INCLUDE_DEFINES_HPP_
