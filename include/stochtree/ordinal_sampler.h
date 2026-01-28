/*!
 * Copyright (c) 2025 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_ORDINAL_SAMPLER_H_
#define STOCHTREE_ORDINAL_SAMPLER_H_

#include <stochtree/data.h>
#include <stochtree/ensemble.h>
#include <stochtree/gamma_sampler.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/tree.h>

#include <Eigen/Dense>
#include <random>
#include <vector>

namespace StochTree {

static double sample_truncated_exponential_low_high(double u, double rate, double low, double high) {
    return -std::log((1-u)*std::exp(-rate*low) + u*std::exp(-rate*high))/rate;
}

static double sample_truncated_exponential_low(double u, double rate, double low) {
    return -std::log((1-u)*std::exp(-rate*low))/rate;
}

static double sample_truncated_exponential_high(double u, double rate, double high) {
    return -std::log1p(u*std::expm1(-high*rate))/rate;
}

static double sample_exponential(double u, double rate) {
    return -std::log1p(-u)/rate;
}

/*!
 * \brief Sampler for ordinal model hyperparameters
 * 
 * This class handles MCMC sampling for ordinal-specific parameters:
 * - Truncated exponential latent variables (Z)
 * - Cutpoint parameters (gamma)
 * - Cumulative sum of exp(gamma) (seg) [derived parameter]
 */
class OrdinalSampler {
 public:
  OrdinalSampler() {
    gamma_sampler_ = GammaSampler();
  }
  ~OrdinalSampler() {}

  /*!
   * \brief Sample from truncated exponential distribution
   * 
   * Samples from exponential distribution truncated to [low,high]
   * 
   * \param gen Random number generator
   * \param rate Rate parameter for exponential distribution
   * \param low Lower truncation bound
   * \param high Upper truncation bound
   * \return Sampled value from truncated exponential
   */
  static double SampleTruncatedExponential(std::mt19937& gen, double rate, double low = 0.0, double high = 1.0);

  /*!
   * \brief Update truncated exponential latent variables (Z)
   * 
   * \param dataset Forest dataset containing training data (covariates) and auxiliary data needed for sampling
   * \param outcome Vector of outcome values
   * \param gen Random number generator
   */
  void UpdateLatentVariables(ForestDataset& dataset, Eigen::VectorXd& outcome, std::mt19937& gen);

  /*!
   * \brief Update gamma cutpoint parameters
   * 
   * \param dataset Forest dataset containing training data (covariates) and auxiliary data needed for sampling
   * \param outcome Vector of outcome values
   * \param alpha_gamma Shape parameter for log-gamma prior on cutpoints gamma
   * \param beta_gamma Rate parameter for log-gamma prior on cutpoints gamma
   * \param gamma_0 Fixed value for first cutpoint parameter (for identifiability)
   * \param gen Random number generator
   */
  void UpdateGammaParams(ForestDataset& dataset, Eigen::VectorXd& outcome, 
                         double alpha_gamma, double beta_gamma, 
                         double gamma_0, std::mt19937& gen);

  /*!
   * \brief Update cumulative exponential sums (seg)
   * 
   * \param dataset Forest dataset containing training data (covariates) and auxiliary data needed for sampling
   */
  void UpdateCumulativeExpSums(ForestDataset& dataset);

 private:
  GammaSampler gamma_sampler_;
};

} // namespace StochTree

#endif // STOCHTREE_ORDINAL_SAMPLER_H_
