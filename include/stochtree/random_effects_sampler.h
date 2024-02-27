/*! Copyright (c) 2024 stochtree authors. All rights reserved. */
#ifndef STOCHTREE_RFX_SAMPLER_H_
#define STOCHTREE_RFX_SAMPLER_H_

#include <stochtree/cutpoint_candidates.h>
#include <stochtree/data.h>
#include <stochtree/ensemble.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/prior.h>

#include <cmath>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

namespace StochTree {

/*! \brief Forward declaration of RandomEffectsPersisted class */
class RandomEffectsPersisted;

/*! \brief Sampling, prediction, and group / component tracking for random effects */
class RandomEffectsSampler {
 friend RandomEffectsPersisted;
 public:
  RandomEffectsSampler() {}
  RandomEffectsSampler(RegressionRandomEffectsDataset& rfx_dataset, RandomEffectsRegressionGaussianPrior& rfx_prior);
  void InitializeParameters(RegressionRandomEffectsDataset& rfx_dataset, ColumnVector& residual);
  void SampleRandomEffects(RandomEffectsRegressionGaussianPrior& rfx_prior, RegressionRandomEffectsDataset& rfx_dataset, ColumnVector& residual, std::mt19937& gen);
  ~RandomEffectsSampler() {}
  std::vector<std::int32_t> GroupObservationIndices(std::int32_t group_num) const;
  void InitializeParameters(Eigen::MatrixXd& X, Eigen::MatrixXd& y);
  void SampleRandomEffects(Eigen::MatrixXd& X, Eigen::VectorXd& y, std::mt19937& gen, double a, double b);
  void SampleAlpha(Eigen::MatrixXd& X, Eigen::VectorXd& y, std::mt19937& gen);
  void SampleXi(Eigen::MatrixXd& X, Eigen::VectorXd& y, std::mt19937& gen);
  void SampleSigma(std::mt19937& gen, double a, double b);
  void SiftGroupIndices(std::vector<int32_t>& group_labels);
  Eigen::VectorXd PredictRandomEffects(Eigen::MatrixXd& X, std::vector<int32_t>& group_labels);
  
 private:
  int num_components_;
  int num_groups_;
  Eigen::MatrixXd W_beta_;
  Eigen::VectorXd alpha_;
  Eigen::MatrixXd xi_;
  Eigen::VectorXd sigma_xi_;
  Eigen::VectorXd sigma_alpha_;
  Eigen::MatrixXd Sigma_xi_;
  Eigen::MatrixXd Sigma_xi_inv_;
  std::vector<std::int32_t> sifted_group_observations_;
  std::vector<std::uint64_t> group_index_begin_;
  std::vector<std::uint64_t> group_index_end_;
  std::vector<std::int32_t> group_index_labels_;
  std::map<int32_t, uint32_t> label_map_;
};

} // namespace StochTree

#endif // STOCHTREE_RFX_SAMPLER_H_