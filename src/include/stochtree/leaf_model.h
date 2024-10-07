/*!
 * Copyright (c) 2024 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_LEAF_MODEL_H_
#define STOCHTREE_LEAF_MODEL_H_

#include <Eigen/Dense>
#include <stochtree/cutpoint_candidates.h>
#include <stochtree/data.h>
#include <stochtree/gamma_sampler.h>
#include <stochtree/ig_sampler.h>
#include <stochtree/log.h>
#include <stochtree/meta.h>
#include <stochtree/normal_sampler.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/prior.h>
#include <stochtree/tree.h>

#include <random>
#include <tuple>
#include <variant>

namespace StochTree {

enum ModelType {
  kConstantLeafGaussian, 
  kUnivariateRegressionLeafGaussian, 
  kMultivariateRegressionLeafGaussian, 
  kLogLinearVariance
};

/*! \brief Sufficient statistic and associated operations for gaussian homoskedastic constant leaf outcome model */
class GaussianConstantSuffStat {
 public:
  data_size_t n;
  double sum_w;
  double sum_yw;
  GaussianConstantSuffStat() {
    n = 0;
    sum_w = 0.0;
    sum_yw = 0.0;
  }
  void IncrementSuffStat(ForestDataset& dataset, Eigen::VectorXd& outcome, ForestTracker& tracker, data_size_t row_idx, int tree_idx) {
    n += 1;
    if (dataset.HasVarWeights()) {
      sum_w += 1/dataset.VarWeightValue(row_idx);
      sum_yw += outcome(row_idx, 0)/dataset.VarWeightValue(row_idx);
    } else {
      sum_w += 1.0;
      sum_yw += outcome(row_idx, 0);
    }
  }
  void ResetSuffStat() {
    n = 0;
    sum_w = 0.0;
    sum_yw = 0.0;
  }
  void AddSuffStat(GaussianConstantSuffStat& lhs, GaussianConstantSuffStat& rhs) {
    n = lhs.n + rhs.n;
    sum_w = lhs.sum_w + rhs.sum_w;
    sum_yw = lhs.sum_yw + rhs.sum_yw;
  }
  void SubtractSuffStat(GaussianConstantSuffStat& lhs, GaussianConstantSuffStat& rhs) {
    n = lhs.n - rhs.n;
    sum_w = lhs.sum_w - rhs.sum_w;
    sum_yw = lhs.sum_yw - rhs.sum_yw;
  }
  bool SampleGreaterThan(data_size_t threshold) {
    return n > threshold;
  }
  bool SampleGreaterThanEqual(data_size_t threshold) {
    return n >= threshold;
  }
  data_size_t SampleSize() {
    return n;
  }
};

/*! \brief Marginal likelihood and posterior computation for gaussian homoskedastic constant leaf outcome model */
class GaussianConstantLeafModel {
 public:
  GaussianConstantLeafModel(double tau) {tau_ = tau; normal_sampler_ = UnivariateNormalSampler();}
  ~GaussianConstantLeafModel() {}
  double SplitLogMarginalLikelihood(GaussianConstantSuffStat& left_stat, GaussianConstantSuffStat& right_stat, double global_variance);
  double NoSplitLogMarginalLikelihood(GaussianConstantSuffStat& suff_stat, double global_variance);
  double PosteriorParameterMean(GaussianConstantSuffStat& suff_stat, double global_variance);
  double PosteriorParameterVariance(GaussianConstantSuffStat& suff_stat, double global_variance);
  void SampleLeafParameters(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, Tree* tree, int tree_num, double global_variance, std::mt19937& gen);
  void SetEnsembleRootPredictedValue(ForestDataset& dataset, TreeEnsemble* ensemble, double root_pred_value);
  void SetScale(double tau) {tau_ = tau;}
  inline bool RequiresBasis() {return false;}
 private:
  double tau_;
  UnivariateNormalSampler normal_sampler_;
};

/*! \brief Sufficient statistic and associated operations for gaussian homoskedastic constant leaf outcome model */
class GaussianUnivariateRegressionSuffStat {
 public:
  data_size_t n;
  double sum_xxw;
  double sum_yxw;
  GaussianUnivariateRegressionSuffStat() {
    n = 0;
    sum_xxw = 0.0;
    sum_yxw = 0.0;
  }
  void IncrementSuffStat(ForestDataset& dataset, Eigen::VectorXd& outcome, ForestTracker& tracker, data_size_t row_idx, int tree_idx) {
    n += 1;
    if (dataset.HasVarWeights()) {
      sum_xxw += dataset.BasisValue(row_idx, 0)*dataset.BasisValue(row_idx, 0)/dataset.VarWeightValue(row_idx);
      sum_yxw += outcome(row_idx, 0)*dataset.BasisValue(row_idx, 0)/dataset.VarWeightValue(row_idx);
    } else {
      sum_xxw += dataset.BasisValue(row_idx, 0)*dataset.BasisValue(row_idx, 0);
      sum_yxw += outcome(row_idx, 0)*dataset.BasisValue(row_idx, 0);
    }
  }
  void ResetSuffStat() {
    n = 0;
    sum_xxw = 0.0;
    sum_yxw = 0.0;
  }
  void AddSuffStat(GaussianUnivariateRegressionSuffStat& lhs, GaussianUnivariateRegressionSuffStat& rhs) {
    n = lhs.n + rhs.n;
    sum_xxw = lhs.sum_xxw + rhs.sum_xxw;
    sum_yxw = lhs.sum_yxw + rhs.sum_yxw;
  }
  void SubtractSuffStat(GaussianUnivariateRegressionSuffStat& lhs, GaussianUnivariateRegressionSuffStat& rhs) {
    n = lhs.n - rhs.n;
    sum_xxw = lhs.sum_xxw - rhs.sum_xxw;
    sum_yxw = lhs.sum_yxw - rhs.sum_yxw;
  }
  bool SampleGreaterThan(data_size_t threshold) {
    return n > threshold;
  }
  bool SampleGreaterThanEqual(data_size_t threshold) {
    return n >= threshold;
  }
  data_size_t SampleSize() {
    return n;
  }
};

/*! \brief Marginal likelihood and posterior computation for gaussian homoskedastic constant leaf outcome model */
class GaussianUnivariateRegressionLeafModel {
 public:
  GaussianUnivariateRegressionLeafModel(double tau) {tau_ = tau; normal_sampler_ = UnivariateNormalSampler();}
  ~GaussianUnivariateRegressionLeafModel() {}
  double SplitLogMarginalLikelihood(GaussianUnivariateRegressionSuffStat& left_stat, GaussianUnivariateRegressionSuffStat& right_stat, double global_variance);
  double NoSplitLogMarginalLikelihood(GaussianUnivariateRegressionSuffStat& suff_stat, double global_variance);
  double PosteriorParameterMean(GaussianUnivariateRegressionSuffStat& suff_stat, double global_variance);
  double PosteriorParameterVariance(GaussianUnivariateRegressionSuffStat& suff_stat, double global_variance);
  void SampleLeafParameters(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, Tree* tree, int tree_num, double global_variance, std::mt19937& gen);
  void SetEnsembleRootPredictedValue(ForestDataset& dataset, TreeEnsemble* ensemble, double root_pred_value);
  void SetScale(double tau) {tau_ = tau;}
  inline bool RequiresBasis() {return true;}
 private:
  double tau_;
  UnivariateNormalSampler normal_sampler_;
};

/*! \brief Sufficient statistic and associated operations for gaussian homoskedastic constant leaf outcome model */
class GaussianMultivariateRegressionSuffStat {
 public:
  data_size_t n;
  int p;
  Eigen::MatrixXd XtWX;
  Eigen::MatrixXd ytWX;
  GaussianMultivariateRegressionSuffStat(int basis_dim) {
    n = 0;
    XtWX = Eigen::MatrixXd::Zero(basis_dim, basis_dim);
    ytWX = Eigen::MatrixXd::Zero(1, basis_dim);
    p = basis_dim;
  }
  void IncrementSuffStat(ForestDataset& dataset, Eigen::VectorXd& outcome, ForestTracker& tracker, data_size_t row_idx, int tree_idx) {
    n += 1;
    if (dataset.HasVarWeights()) {
      XtWX += dataset.GetBasis()(row_idx, Eigen::all).transpose()*dataset.GetBasis()(row_idx, Eigen::all)/dataset.VarWeightValue(row_idx);
      ytWX += (outcome(row_idx, 0)*(dataset.GetBasis()(row_idx, Eigen::all)))/dataset.VarWeightValue(row_idx);
    } else {
      XtWX += dataset.GetBasis()(row_idx, Eigen::all).transpose()*dataset.GetBasis()(row_idx, Eigen::all);
      ytWX += (outcome(row_idx, 0)*(dataset.GetBasis()(row_idx, Eigen::all)));
    }
  }
  void ResetSuffStat() {
    n = 0;
    XtWX = Eigen::MatrixXd::Zero(p, p);
    ytWX = Eigen::MatrixXd::Zero(1, p);
  }
  void AddSuffStat(GaussianMultivariateRegressionSuffStat& lhs, GaussianMultivariateRegressionSuffStat& rhs) {
    n = lhs.n + rhs.n;
    XtWX = lhs.XtWX + rhs.XtWX;
    ytWX = lhs.ytWX + rhs.ytWX;
  }
  void SubtractSuffStat(GaussianMultivariateRegressionSuffStat& lhs, GaussianMultivariateRegressionSuffStat& rhs) {
    n = lhs.n - rhs.n;
    XtWX = lhs.XtWX - rhs.XtWX;
    ytWX = lhs.ytWX - rhs.ytWX;
  }
  bool SampleGreaterThan(data_size_t threshold) {
    return n > threshold;
  }
  bool SampleGreaterThanEqual(data_size_t threshold) {
    return n >= threshold;
  }
  data_size_t SampleSize() {
    return n;
  }
};

/*! \brief Marginal likelihood and posterior computation for gaussian homoskedastic constant leaf outcome model */
class GaussianMultivariateRegressionLeafModel {
 public:
  GaussianMultivariateRegressionLeafModel(Eigen::MatrixXd& Sigma_0) {Sigma_0_ = Sigma_0; multivariate_normal_sampler_ = MultivariateNormalSampler();}
  ~GaussianMultivariateRegressionLeafModel() {}
  double SplitLogMarginalLikelihood(GaussianMultivariateRegressionSuffStat& left_stat, GaussianMultivariateRegressionSuffStat& right_stat, double global_variance);
  double NoSplitLogMarginalLikelihood(GaussianMultivariateRegressionSuffStat& suff_stat, double global_variance);
  Eigen::VectorXd PosteriorParameterMean(GaussianMultivariateRegressionSuffStat& suff_stat, double global_variance);
  Eigen::MatrixXd PosteriorParameterVariance(GaussianMultivariateRegressionSuffStat& suff_stat, double global_variance);
  void SampleLeafParameters(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, Tree* tree, int tree_num, double global_variance, std::mt19937& gen);
  void SetEnsembleRootPredictedValue(ForestDataset& dataset, TreeEnsemble* ensemble, double root_pred_value);
  void SetScale(Eigen::MatrixXd& Sigma_0) {Sigma_0_ = Sigma_0;}
  inline bool RequiresBasis() {return true;}
 private:
  Eigen::MatrixXd Sigma_0_;
  MultivariateNormalSampler multivariate_normal_sampler_;
};

/*! \brief Sufficient statistic and associated operations for heteroskedastic log-linear variance model */
class LogLinearVarianceSuffStat {
 public:
  data_size_t n;
  double weighted_sum_ei;
  LogLinearVarianceSuffStat() {
    n = 0;
    weighted_sum_ei = 0.0;
  }
  void IncrementSuffStat(ForestDataset& dataset, Eigen::VectorXd& outcome, ForestTracker& tracker, data_size_t row_idx, int tree_idx) {
    n += 1;
    weighted_sum_ei += std::exp(std::log(outcome(row_idx)*outcome(row_idx)) - tracker.GetSamplePrediction(row_idx) + tracker.GetTreeSamplePrediction(row_idx, tree_idx));
  }
  void ResetSuffStat() {
    n = 0;
    weighted_sum_ei = 0.0;
  }
  void AddSuffStat(LogLinearVarianceSuffStat& lhs, LogLinearVarianceSuffStat& rhs) {
    n = lhs.n + rhs.n;
    weighted_sum_ei = lhs.weighted_sum_ei + rhs.weighted_sum_ei;
  }
  void SubtractSuffStat(LogLinearVarianceSuffStat& lhs, LogLinearVarianceSuffStat& rhs) {
    n = lhs.n - rhs.n;
    weighted_sum_ei = lhs.weighted_sum_ei - rhs.weighted_sum_ei;
  }
  bool SampleGreaterThan(data_size_t threshold) {
    return n > threshold;
  }
  bool SampleGreaterThanEqual(data_size_t threshold) {
    return n >= threshold;
  }
  data_size_t SampleSize() {
    return n;
  }
};

/*! \brief Marginal likelihood and posterior computation for heteroskedastic log-linear variance model */
class LogLinearVarianceLeafModel {
 public:
  LogLinearVarianceLeafModel(double a, double b) {a_ = a; b_ = b; gamma_sampler_ = GammaSampler();}
  ~LogLinearVarianceLeafModel() {}
  double SplitLogMarginalLikelihood(LogLinearVarianceSuffStat& left_stat, LogLinearVarianceSuffStat& right_stat, double global_variance);
  double NoSplitLogMarginalLikelihood(LogLinearVarianceSuffStat& suff_stat, double global_variance);
  double SuffStatLogMarginalLikelihood(LogLinearVarianceSuffStat& suff_stat, double global_variance);
  double PosteriorParameterShape(LogLinearVarianceSuffStat& suff_stat, double global_variance);
  double PosteriorParameterScale(LogLinearVarianceSuffStat& suff_stat, double global_variance);
  void SampleLeafParameters(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, Tree* tree, int tree_num, double global_variance, std::mt19937& gen);
  void SetEnsembleRootPredictedValue(ForestDataset& dataset, TreeEnsemble* ensemble, double root_pred_value);
  void SetPriorShape(double a) {a_ = a;}
  void SetPriorRate(double b) {b_ = b;}
  inline bool RequiresBasis() {return false;}
 private:
  double a_;
  double b_;
  GammaSampler gamma_sampler_;
};

using SuffStatVariant = std::variant<GaussianConstantSuffStat, 
                                     GaussianUnivariateRegressionSuffStat, 
                                     GaussianMultivariateRegressionSuffStat, 
                                     LogLinearVarianceSuffStat>;

using LeafModelVariant = std::variant<GaussianConstantLeafModel, 
                                      GaussianUnivariateRegressionLeafModel, 
                                      GaussianMultivariateRegressionLeafModel, 
                                      LogLinearVarianceLeafModel>;

template<typename SuffStatType, typename... SuffStatConstructorArgs>
static inline SuffStatVariant createSuffStat(SuffStatConstructorArgs... leaf_suff_stat_args) {
  return SuffStatType(leaf_suff_stat_args...);
}

template<typename LeafModelType, typename... LeafModelConstructorArgs>
static inline LeafModelVariant createLeafModel(LeafModelConstructorArgs... leaf_model_args) {
  return LeafModelType(leaf_model_args...);
}

static inline SuffStatVariant suffStatFactory(ModelType model_type, int basis_dim = 0) {
  if (model_type == kConstantLeafGaussian) {
    return createSuffStat<GaussianConstantSuffStat>();
  } else if (model_type == kUnivariateRegressionLeafGaussian) {
    return createSuffStat<GaussianUnivariateRegressionSuffStat>();
  } else if (model_type == kMultivariateRegressionLeafGaussian) {
    return createSuffStat<GaussianMultivariateRegressionSuffStat, int>(basis_dim);
  } else {
    return createSuffStat<LogLinearVarianceSuffStat>();
  }
}

static inline LeafModelVariant leafModelFactory(ModelType model_type, double tau, Eigen::MatrixXd& Sigma0, double a, double b) {
  if (model_type == kConstantLeafGaussian) {
    return createLeafModel<GaussianConstantLeafModel, double>(tau);
  } else if (model_type == kUnivariateRegressionLeafGaussian) {
    return createLeafModel<GaussianUnivariateRegressionLeafModel, double>(tau);
  } else if (model_type == kMultivariateRegressionLeafGaussian) {
    return createLeafModel<GaussianMultivariateRegressionLeafModel, Eigen::MatrixXd>(Sigma0);
  } else if (model_type == kLogLinearVariance) {
    return createLeafModel<LogLinearVarianceLeafModel, double, double>(a, b);
  } else {
    Log::Fatal("Incompatible model type provided to leaf model factory");
  }
}

template<typename SuffStatType>
static inline void AccumulateSuffStatProposed(SuffStatType& node_suff_stat, SuffStatType& left_suff_stat, SuffStatType& right_suff_stat, ForestDataset& dataset, ForestTracker& tracker, 
                                ColumnVector& residual, double global_variance, TreeSplit& split, int tree_num, int leaf_num, int split_feature) {
  // Acquire iterators
  auto node_begin_iter = tracker.UnsortedNodeBeginIterator(tree_num, leaf_num);
  auto node_end_iter = tracker.UnsortedNodeEndIterator(tree_num, leaf_num);

  // Accumulate sufficient statistics
  for (auto i = node_begin_iter; i != node_end_iter; i++) {
    auto idx = *i;
    double feature_value = dataset.CovariateValue(idx, split_feature);
    node_suff_stat.IncrementSuffStat(dataset, residual.GetData(), tracker, idx, tree_num);
    if (split.SplitTrue(feature_value)) {
      left_suff_stat.IncrementSuffStat(dataset, residual.GetData(), tracker, idx, tree_num);
    } else {
      right_suff_stat.IncrementSuffStat(dataset, residual.GetData(), tracker, idx, tree_num);
    }
  }
}

template<typename SuffStatType>
static inline void AccumulateSuffStatExisting(SuffStatType& node_suff_stat, SuffStatType& left_suff_stat, SuffStatType& right_suff_stat, ForestDataset& dataset, ForestTracker& tracker, 
                                ColumnVector& residual, double global_variance, int tree_num, int split_node_id, int left_node_id, int right_node_id) {
  // Acquire iterators
  auto left_node_begin_iter = tracker.UnsortedNodeBeginIterator(tree_num, left_node_id);
  auto left_node_end_iter = tracker.UnsortedNodeEndIterator(tree_num, left_node_id);
  auto right_node_begin_iter = tracker.UnsortedNodeBeginIterator(tree_num, right_node_id);
  auto right_node_end_iter = tracker.UnsortedNodeEndIterator(tree_num, right_node_id);

  // Accumulate sufficient statistics for the left and split nodes
  for (auto i = left_node_begin_iter; i != left_node_end_iter; i++) {
    auto idx = *i;
    left_suff_stat.IncrementSuffStat(dataset, residual.GetData(), tracker, idx, tree_num);
    node_suff_stat.IncrementSuffStat(dataset, residual.GetData(), tracker, idx, tree_num);
  }

  // Accumulate sufficient statistics for the right and split nodes
  for (auto i = right_node_begin_iter; i != right_node_end_iter; i++) {
    auto idx = *i;
    right_suff_stat.IncrementSuffStat(dataset, residual.GetData(), tracker, idx, tree_num);
    node_suff_stat.IncrementSuffStat(dataset, residual.GetData(), tracker, idx, tree_num);
  }
}

template<typename SuffStatType, bool sorted>
static inline void AccumulateSingleNodeSuffStat(SuffStatType& node_suff_stat, ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, int tree_num, int node_id) {
  // Acquire iterators
  std::vector<data_size_t>::iterator node_begin_iter;
  std::vector<data_size_t>::iterator node_end_iter;
  if (sorted) {
    // Default to the first feature if we're using the presort tracker
    node_begin_iter = tracker.SortedNodeBeginIterator(node_id, 0);
    node_end_iter = tracker.SortedNodeEndIterator(node_id, 0);
  } else {
    node_begin_iter = tracker.UnsortedNodeBeginIterator(tree_num, node_id);
    node_end_iter = tracker.UnsortedNodeEndIterator(tree_num, node_id);
  }
  
  // Accumulate sufficient statistics
  for (auto i = node_begin_iter; i != node_end_iter; i++) {
    auto idx = *i;
    node_suff_stat.IncrementSuffStat(dataset, residual.GetData(), tracker, idx, tree_num);
  }
}

template<typename SuffStatType>
static inline void AccumulateCutpointBinSuffStat(SuffStatType& left_suff_stat, ForestTracker& tracker, CutpointGridContainer& cutpoint_grid_container, 
                                   ForestDataset& dataset, ColumnVector& residual, double global_variance, int tree_num, int node_id, 
                                   int feature_num, int cutpoint_num) {
  // Acquire iterators
  auto node_begin_iter = tracker.SortedNodeBeginIterator(node_id, feature_num);
  auto node_end_iter = tracker.SortedNodeEndIterator(node_id, feature_num);
  
  // Determine node start point
  data_size_t node_begin = tracker.SortedNodeBegin(node_id, feature_num);

  // Determine cutpoint bin start and end points
  data_size_t current_bin_begin = cutpoint_grid_container.BinStartIndex(cutpoint_num, feature_num);
  data_size_t current_bin_size = cutpoint_grid_container.BinLength(cutpoint_num, feature_num);
  data_size_t next_bin_begin = cutpoint_grid_container.BinStartIndex(cutpoint_num + 1, feature_num);

  // Cutpoint specific iterators
  // TODO: fix the hack of having to subtract off node_begin, probably by cleaning up the CutpointGridContainer interface
  auto cutpoint_begin_iter = node_begin_iter + (current_bin_begin - node_begin);
  auto cutpoint_end_iter = node_begin_iter + (next_bin_begin - node_begin);

  // Accumulate sufficient statistics
  for (auto i = cutpoint_begin_iter; i != cutpoint_end_iter; i++) {
    auto idx = *i;
    left_suff_stat.IncrementSuffStat(dataset, residual.GetData(), tracker, idx, tree_num);
  }
}

} // namespace StochTree

#endif // STOCHTREE_LEAF_MODEL_H_
