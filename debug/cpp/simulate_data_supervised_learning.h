/*!
 * Copyright (c) 2023 stochtree authors
 */
#ifndef STOCHTREE_DEBUG_SIMULATE_SUPERVISED_LEARNING_DATA_H_
#define STOCHTREE_DEBUG_SIMULATE_SUPERVISED_LEARNING_DATA_H_

#include <stochtree/config.h>
#include <stochtree/data.h>

#include <random>
#include <vector>

namespace StochTree {

/*! \brief Function that generates a Dataset for supervised learning tasks
 *  Feature X_i are U(0,1) and outcome is f(X) + epsilon, where
 *  epsilon ~ N(0,1) and f(X) = 5 X_1 - 10 X_2
 */
std::vector<double> SimulateTabularDataset(data_size_t n, int p, int seed = 1234) {
  // Random number generator
  std::mt19937 gen(seed);

  // Distribution for covariates
  std::uniform_real_distribution<double> covariate_dist(0., 1.);
  
  // Distribution for outcome noise
  std::normal_distribution<double> outcome_noise_dist(0., 1.);

  // Generate the data in-memory in row-major format
  data_size_t data_length = n*(p+1);
  std::vector<double> data_vector(data_length);

  for (data_size_t i = 0; i < n; i++) {
    for (int j = 1; j < (p+1); j++) {
      // Fill in all p covariates
      data_vector[i*(p+1) + j] = covariate_dist(gen);
    }
    // Generate outcome noise
    data_vector[i*(p+1)] = 5.*data_vector[i*(p+1) + 1] - 10.*data_vector[i*(p+1) + 2] + outcome_noise_dist(gen);
  }

  return data_vector;
}

} // namespace StochTree

#endif  // STOCHTREE_DEBUG_SIMULATE_SUPERVISED_LEARNING_DATA_H_
