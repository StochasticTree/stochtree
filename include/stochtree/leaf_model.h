/*!
 * Copyright (c) 2024 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_LEAF_MODEL_H_
#define STOCHTREE_LEAF_MODEL_H_

#include <Eigen/Dense>
#include <stochtree/cutpoint_candidates.h>
#include <stochtree/data.h>
#include <stochtree/log.h>
#include <stochtree/meta.h>
#include <stochtree/normal_sampler.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/prior.h>
#include <stochtree/tree.h>

#include <random>
#include <tuple>

namespace StochTree {

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
  void IncrementSuffStat(ForestDataset& dataset, Eigen::VectorXd& outcome, data_size_t row_idx) {
    n += 1;
    if (dataset.HasVarWeights()) {
      sum_w += 1./dataset.VarWeightValue(row_idx);
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
  std::tuple<double, double, data_size_t, data_size_t> EvaluateProposedSplit(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, TreeSplit& split, int tree_num, int leaf_num, int split_feature, double global_variance);
  std::tuple<double, double, data_size_t, data_size_t> EvaluateExistingSplit(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, double global_variance, int tree_num, int split_node_id, int left_node_id, int right_node_id);
  void EvaluateAllPossibleSplits(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, TreePrior& tree_prior, double global_variance, int tree_num, int split_node_id, 
                                 std::vector<double>& log_cutpoint_evaluations, std::vector<int>& cutpoint_features, std::vector<double>& cutpoint_values, std::vector<FeatureType>& cutpoint_feature_types, 
                                 data_size_t& valid_cutpoint_count, CutpointGridContainer& cutpoint_grid_container, data_size_t node_begin, data_size_t node_end, std::vector<FeatureType>& feature_types);
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
  void IncrementSuffStat(ForestDataset& dataset, Eigen::VectorXd& outcome, data_size_t row_idx) {
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
  std::tuple<double, double, data_size_t, data_size_t> EvaluateProposedSplit(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, TreeSplit& split, int tree_num, int leaf_num, int split_feature, double global_variance);
  std::tuple<double, double, data_size_t, data_size_t> EvaluateExistingSplit(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, double global_variance, int tree_num, int split_node_id, int left_node_id, int right_node_id);
  void EvaluateAllPossibleSplits(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, TreePrior& tree_prior, double global_variance, int tree_num, int split_node_id, 
                                 std::vector<double>& log_cutpoint_evaluations, std::vector<int>& cutpoint_features, std::vector<double>& cutpoint_values, std::vector<FeatureType>& cutpoint_feature_types, 
                                 data_size_t& valid_cutpoint_count, CutpointGridContainer& cutpoint_grid_container, data_size_t node_begin, data_size_t node_end, std::vector<FeatureType>& feature_types);
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
    ytWX = Eigen::MatrixXd::Zero(basis_dim, 1);
    p = basis_dim;
  }
  void IncrementSuffStat(ForestDataset& dataset, Eigen::VectorXd& outcome, data_size_t row_idx) {
    n += 1;
    if (dataset.HasVarWeights()) {
      XtWX += dataset.GetBasis()(row_idx, Eigen::all).transpose()*dataset.GetBasis()(row_idx, Eigen::all)/dataset.VarWeightValue(row_idx);
      ytWX += dataset.GetBasis()(row_idx, Eigen::all).transpose()*outcome(row_idx, 0)/dataset.VarWeightValue(row_idx);
    } else {
      XtWX += dataset.GetBasis()(row_idx, Eigen::all).transpose()*dataset.GetBasis()(row_idx, Eigen::all);
      ytWX += dataset.GetBasis()(row_idx, Eigen::all).transpose()*outcome(row_idx, 0);
    }
  }
  void ResetSuffStat() {
    n = 0;
    XtWX = Eigen::MatrixXd::Zero(p, p);
    ytWX = Eigen::MatrixXd::Zero(p, 1);
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
  std::tuple<double, double, data_size_t, data_size_t> EvaluateProposedSplit(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, TreeSplit& split, int tree_num, int leaf_num, int split_feature, double global_variance);
  std::tuple<double, double, data_size_t, data_size_t> EvaluateExistingSplit(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, double global_variance, int tree_num, int split_node_id, int left_node_id, int right_node_id);
  void EvaluateAllPossibleSplits(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, TreePrior& tree_prior, double global_variance, int tree_num, int split_node_id, 
                                 std::vector<double>& log_cutpoint_evaluations, std::vector<int>& cutpoint_features, std::vector<double>& cutpoint_values, std::vector<FeatureType>& cutpoint_feature_types, 
                                 data_size_t& valid_cutpoint_count, CutpointGridContainer& cutpoint_grid_container, data_size_t node_begin, data_size_t node_end, std::vector<FeatureType>& feature_types);
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

} // namespace StochTree

#endif // STOCHTREE_LEAF_MODEL_H_
