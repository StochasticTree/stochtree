/*!
 * Copyright (c) 2024 stochtree authors. All rights reserved.
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
   * Samples from exponential distribution truncated to [0,1]
   * 
   * \param lambda Rate parameter for exponential distribution
   * \param gen Random number generator
   * \return Sampled value from truncated exponential
   */
  static double SampleTruncatedExponential(double lambda, std::mt19937& gen);

  /*!
   * \brief Update truncated exponential latent variables (Z)
   * 
   * \param dataset Forest dataset containing training data (covariates)
   * \param outcome Vector of outcome values
   * \param tracker Forest tracker containing auxiliary data 
   * \param gen Random number generator
   */
  void UpdateLatentVariables(ForestDataset& dataset, Eigen::VectorXd& outcome, ForestTracker& tracker,
                            std::mt19937& gen);

  /*!
   * \brief Update gamma cutpoint parameters
   * 
   * \param dataset Forest dataset containing training data (covariates)
   * \param outcome Vector of outcome values
   * \param tracker Forest tracker containing auxiliary data
   * \param alpha_gamma Shape parameter for log-gamma prior on cutpoints gamma
   * \param beta_gamma Rate parameter for log-gamma prior on cutpoints gamma
   * \param gamma_0 Fixed value for first cutpoint parameter (for identifiability)
   * \param gen Random number generator
   */
  void UpdateGammaParams(ForestDataset& dataset, Eigen::VectorXd& outcome, ForestTracker& tracker,
                        double alpha_gamma, double beta_gamma, double gamma_0,
                        std::mt19937& gen);

  /*!
   * \brief Update cumulative exponential sums (seg)
   * 
   * \param tracker Forest tracker containing auxiliary data
   */
  void UpdateCumulativeExpSums(ForestTracker& tracker);

 private:
  GammaSampler gamma_sampler_;
};

} // namespace StochTree

#endif // STOCHTREE_ORDINAL_SAMPLER_H_
