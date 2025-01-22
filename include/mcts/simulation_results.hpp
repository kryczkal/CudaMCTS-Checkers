#ifndef MCTS_CHECKERS_INCLUDE_MCTS_SIMULATION_RESULTS_HPP_
#define MCTS_CHECKERS_INCLUDE_MCTS_SIMULATION_RESULTS_HPP_

#include "types.hpp"

/**
 * \brief Per-batch result: final averaged score + the number of simulations.
 */
struct SimulationResult {
    f64 score;
    u64 n_simulations;
};

#endif  // MCTS_CHECKERS_INCLUDE_SIMULATION_RESULTS_HPP_
