/*!
 * Copyright (c) 2023 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_OUTCOME_MODEL_H_
#define STOCHTREE_OUTCOME_MODEL_H_

#include <stochtree/cutpoint_candidates.h>
#include <stochtree/ensemble.h>
#include <stochtree/log.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/tree.h>
#include "../../dependencies/boost_math/include/boost/math/special_functions/gamma.hpp"
#include <Eigen/Dense>

#include <cmath>
#include <random>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace StochTree {

enum NodeIndicator {
  SplitNode, 
  LeftNode, 
  RightNode
};

enum GlobalParamName {
  GlobalVariance, 
  LeafPriorVariance
};

/*! \brief Base class for models parameterizing tree sampling loop */
class ModelWrapper {
 public:
  ModelWrapper() {}
  virtual ~ModelWrapper() = default;
  virtual void IncrementNodeSuffStat(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, Eigen::MatrixXd& outcome, data_size_t row_idx, NodeIndicator node_indicator) {}
  virtual void ResetNodeSuffStat(NodeIndicator node_indicator) {}
  virtual void SubtractNodeSuffStat(NodeIndicator node_indicator, NodeIndicator lhs_node_indicator, NodeIndicator rhs_node_indicator) {}
  virtual bool NodeSampleGreaterThan(NodeIndicator node_indicator, data_size_t threshold) {}
  virtual data_size_t NodeSampleSize(NodeIndicator node_indicator) {}
  virtual double SplitLogMarginalLikelihood() {}
  virtual double NoSplitLogMarginalLikelihood() {}
  virtual void SampleLeafParameters(std::mt19937& gen, int leaf_num, Tree* tree) {}
  virtual void SetGlobalParameter(double param_value, GlobalParamName param_name) {}
};

/*! \brief Sufficient statistic and associated operations for gaussian homoskedastic constant leaf outcome model */
struct GaussianHomoskedasticConstantSuffStat {
  data_size_t n;
  double sum_y;
  double sum_y_squared;
  GaussianHomoskedasticConstantSuffStat() {
    n = 0;
    sum_y = 0.0;
    sum_y_squared = 0.0;
  }
  void IncrementSuffStat(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, Eigen::MatrixXd& outcome, data_size_t row_idx) {
    n += 1;
    sum_y += outcome(row_idx, 0);
    sum_y_squared += std::pow(outcome(row_idx, 0), 2.0);
  }
  void ResetSuffStat() {
    n = 0;
    sum_y = 0.0;
    sum_y_squared = 0.0;
  }
  void SubtractSuffStat(GaussianHomoskedasticConstantSuffStat& lhs, GaussianHomoskedasticConstantSuffStat& rhs) {
    n = lhs.n - rhs.n;
    sum_y = lhs.sum_y - rhs.sum_y;
    sum_y_squared = lhs.sum_y_squared - rhs.sum_y_squared;
  }
  bool SampleGreaterThan(data_size_t threshold) {
    return n > threshold;
  }
  data_size_t SampleSize() {
    return n;
  }
};

/*! \brief Global parameters for gaussian homoskedastic constant leaf outcome model */
struct GaussianHomoskedasticConstantGlobalParameters {
  double tau;
  double sigma_sq;
};

/*! \brief Marginal likelihood and posterior computation for gaussian homoskedastic constant leaf outcome model */
class GaussianHomoskedasticConstantOutcomeModel {
 public:
  GaussianHomoskedasticConstantOutcomeModel() {}
  ~GaussianHomoskedasticConstantOutcomeModel() {}
  double SplitLogMarginalLikelihood(GaussianHomoskedasticConstantSuffStat& left_stat, GaussianHomoskedasticConstantSuffStat& right_stat, GaussianHomoskedasticConstantGlobalParameters& global_params);
  double NoSplitLogMarginalLikelihood(GaussianHomoskedasticConstantSuffStat& suff_stat, GaussianHomoskedasticConstantGlobalParameters& global_params);
  double PosteriorParameterMean(GaussianHomoskedasticConstantSuffStat& suff_stat, GaussianHomoskedasticConstantGlobalParameters& global_params);
  double PosteriorParameterVariance(GaussianHomoskedasticConstantSuffStat& suff_stat, GaussianHomoskedasticConstantGlobalParameters& global_params);
  void SampleLeafParameters(GaussianHomoskedasticConstantSuffStat& suff_stat, GaussianHomoskedasticConstantGlobalParameters& global_params, std::mt19937& gen, int leaf_num, Tree* tree);
};

/*! \brief Tree model with conjugate Gaussian constant leaf node priors and homoskedastic Gaussian outcome likelihood */
class GaussianHomoskedasticConstantModelWrapper : public ModelWrapper {
 public:
  GaussianHomoskedasticConstantModelWrapper();
  ~GaussianHomoskedasticConstantModelWrapper();
  void IncrementNodeSuffStat(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, Eigen::MatrixXd& outcome, data_size_t row_idx, NodeIndicator node_indicator);
  void ResetNodeSuffStat(NodeIndicator node_indicator);
  void SubtractNodeSuffStat(NodeIndicator node_indicator, NodeIndicator lhs_node_indicator, NodeIndicator rhs_node_indicator);
  bool NodeSampleGreaterThan(NodeIndicator node_indicator, data_size_t threshold);
  data_size_t NodeSampleSize(NodeIndicator node_indicator);
  double SplitLogMarginalLikelihood();
  double NoSplitLogMarginalLikelihood();
  void SampleLeafParameters(std::mt19937& gen, int leaf_num, Tree* tree);
  void SetGlobalParameter(double param_value, GlobalParamName param_name);
 private:
  std::map<NodeIndicator, GaussianHomoskedasticConstantSuffStat> suff_stat_map_;
  GaussianHomoskedasticConstantGlobalParameters global_model_params_;
  GaussianHomoskedasticConstantOutcomeModel outcome_model_;
};

/*! \brief Sufficient statistic and associated operations for homoskedastic, univariate regression leaf outcome model */
struct GaussianHomoskedasticUnivariateRegressionSuffStat {
  data_size_t n;
  double sum_y;
  double sum_yx;
  double sum_x_squared;
  double sum_y_squared;
  GaussianHomoskedasticUnivariateRegressionSuffStat() {
    n = 0;
    sum_y = 0.0;
    sum_yx = 0.0;
    sum_x_squared = 0.0;
    sum_y_squared = 0.0;
  }
  void IncrementSuffStat(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, Eigen::MatrixXd& outcome, data_size_t row_idx) {
    n += 1;
    sum_y += outcome(row_idx, 0);
    sum_yx += outcome(row_idx, 0)*basis(row_idx, 0);
    sum_x_squared += std::pow(basis(row_idx, 0), 2.0);
    sum_y_squared += std::pow(outcome(row_idx, 0), 2.0);
  }
  void ResetSuffStat() {
    n = 0;
    sum_y = 0.0;
    sum_yx = 0.0;
    sum_x_squared = 0.0;
    sum_y_squared = 0.0;
  }
  void SubtractSuffStat(GaussianHomoskedasticUnivariateRegressionSuffStat& lhs, GaussianHomoskedasticUnivariateRegressionSuffStat& rhs) {
    n = lhs.n - rhs.n;
    sum_y = lhs.sum_y - rhs.sum_y;
    sum_yx = lhs.sum_yx - rhs.sum_yx;
    sum_x_squared = lhs.sum_x_squared - rhs.sum_x_squared;
    sum_y_squared = lhs.sum_y_squared - rhs.sum_y_squared;
  }
  bool SampleGreaterThan(data_size_t threshold) {
    return n > threshold;
  }
  data_size_t SampleSize() {
    return n;
  }
};

/*! \brief Global parameters for gaussian homoskedastic univariate regression leaf outcome model */
struct GaussianHomoskedasticUnivariateRegressionGlobalParameters {
  double tau;
  double sigma_sq;
};

/*! \brief Marginal likelihood and posterior computation for gaussian homoskedastic univariate regression leaf outcome model */
class GaussianHomoskedasticUnivariateRegressionOutcomeModel {
 public:
  GaussianHomoskedasticUnivariateRegressionOutcomeModel() {}
  ~GaussianHomoskedasticUnivariateRegressionOutcomeModel() {}
  double SplitLogMarginalLikelihood(GaussianHomoskedasticUnivariateRegressionSuffStat& left_stat, GaussianHomoskedasticUnivariateRegressionSuffStat& right_stat, GaussianHomoskedasticUnivariateRegressionGlobalParameters& global_params);
  double NoSplitLogMarginalLikelihood(GaussianHomoskedasticUnivariateRegressionSuffStat& suff_stat, GaussianHomoskedasticUnivariateRegressionGlobalParameters& global_params);
  double PosteriorParameterMean(GaussianHomoskedasticUnivariateRegressionSuffStat& suff_stat, GaussianHomoskedasticUnivariateRegressionGlobalParameters& global_params);
  double PosteriorParameterVariance(GaussianHomoskedasticUnivariateRegressionSuffStat& suff_stat, GaussianHomoskedasticUnivariateRegressionGlobalParameters& global_params);
  void SampleLeafParameters(GaussianHomoskedasticUnivariateRegressionSuffStat& suff_stat, GaussianHomoskedasticUnivariateRegressionGlobalParameters& global_params, std::mt19937& gen, int leaf_num, Tree* tree);
};

/*! \brief Tree model with conjugate Gaussian linear regression leaf node priors and homoskedastic Gaussian outcome likelihood */
class GaussianHomoskedasticUnivariateRegressionModelWrapper : public ModelWrapper {
 public:
  GaussianHomoskedasticUnivariateRegressionModelWrapper();
  ~GaussianHomoskedasticUnivariateRegressionModelWrapper();
  void IncrementNodeSuffStat(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, Eigen::MatrixXd& outcome, data_size_t row_idx, NodeIndicator node_indicator);
  void ResetNodeSuffStat(NodeIndicator node_indicator);
  void SubtractNodeSuffStat(NodeIndicator node_indicator, NodeIndicator lhs_node_indicator, NodeIndicator rhs_node_indicator);
  bool NodeSampleGreaterThan(NodeIndicator node_indicator, data_size_t threshold);
  data_size_t NodeSampleSize(NodeIndicator node_indicator);
  double SplitLogMarginalLikelihood();
  double NoSplitLogMarginalLikelihood();
  void SampleLeafParameters(std::mt19937& gen, int leaf_num, Tree* tree);
  void SetGlobalParameter(double param_value, GlobalParamName param_name);
 private:
  std::map<NodeIndicator, GaussianHomoskedasticUnivariateRegressionSuffStat> suff_stat_map_;
  GaussianHomoskedasticUnivariateRegressionGlobalParameters global_model_params_;
  GaussianHomoskedasticUnivariateRegressionOutcomeModel outcome_model_;
};

/*! \brief Sufficient statistic and associated operations for gaussian homoskedastic multivariate regression leaf outcome model */
struct GaussianHomoskedasticMultivariateRegressionSuffStat {
  data_size_t n;
  int basis_dim;
  double yty;
  Eigen::MatrixXd Xty;
  Eigen::MatrixXd XtX;
  GaussianHomoskedasticMultivariateRegressionSuffStat() {
    // Default to a basis dimension of 1, this is resettable
    basis_dim = 1;
    n = 0;
    yty = 0.0;
    Xty = Eigen::MatrixXd::Zero(basis_dim, 1);
    XtX = Eigen::MatrixXd::Zero(basis_dim, basis_dim);
  }
  GaussianHomoskedasticMultivariateRegressionSuffStat(int basis_dim) {
    basis_dim = basis_dim;
    n = 0;
    yty = 0.0;
    Xty = Eigen::MatrixXd::Zero(basis_dim, 1);
    XtX = Eigen::MatrixXd::Zero(basis_dim, basis_dim);
  }
  void AdjustBasisDim(int basis_dim) {
    basis_dim = basis_dim;
    n = 0;
    yty = 0.0;
    Xty = Eigen::MatrixXd::Zero(basis_dim, 1);
    XtX = Eigen::MatrixXd::Zero(basis_dim, basis_dim);
  }
  void IncrementSuffStat(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, Eigen::MatrixXd& outcome, data_size_t row_idx) {
    CHECK_EQ(basis_dim, basis.cols());
    n += 1;
    yty += std::pow(outcome(row_idx, 0), 2.0);
    Xty += basis(row_idx, Eigen::all).transpose()*outcome(row_idx, 0);
    XtX += basis(row_idx, Eigen::all).transpose()*basis(row_idx, Eigen::all);
  }
  void ResetSuffStat() {
    n = 0;
    yty = 0.0;
    Xty = Eigen::MatrixXd::Zero(basis_dim, 1);
    XtX = Eigen::MatrixXd::Zero(basis_dim, basis_dim);
  }
  void SubtractSuffStat(GaussianHomoskedasticMultivariateRegressionSuffStat& lhs, GaussianHomoskedasticMultivariateRegressionSuffStat& rhs) {
    n = lhs.n - rhs.n;
    yty = lhs.yty - rhs.yty;
    Xty = lhs.Xty - rhs.Xty;
    XtX = lhs.XtX - rhs.XtX;
  }
  bool SampleGreaterThan(data_size_t threshold) {
    return n > threshold;
  }
  data_size_t SampleSize() {
    return n;
  }
};

/*! \brief Global parameters for gaussian homoskedastic multivariate regression leaf outcome model */
struct GaussianHomoskedasticMultivariateRegressionGlobalParameters {
  Eigen::MatrixXd Sigma;
  double sigma_sq;
};

/*! \brief Marginal likelihood and posterior computation for gaussian homoskedastic multivariate regression leaf outcome model */
class GaussianHomoskedasticMultivariateRegressionOutcomeModel {
 public:
  GaussianHomoskedasticMultivariateRegressionOutcomeModel() {}
  ~GaussianHomoskedasticMultivariateRegressionOutcomeModel() {}
  double SplitLogMarginalLikelihood(GaussianHomoskedasticMultivariateRegressionSuffStat& left_stat, GaussianHomoskedasticMultivariateRegressionSuffStat& right_stat, GaussianHomoskedasticMultivariateRegressionGlobalParameters& global_params);
  double NoSplitLogMarginalLikelihood(GaussianHomoskedasticMultivariateRegressionSuffStat& suff_stat, GaussianHomoskedasticMultivariateRegressionGlobalParameters& global_params);
  Eigen::MatrixXd PosteriorParameterMean(GaussianHomoskedasticMultivariateRegressionSuffStat& suff_stat, GaussianHomoskedasticMultivariateRegressionGlobalParameters& global_params);
  Eigen::MatrixXd PosteriorParameterVariance(GaussianHomoskedasticMultivariateRegressionSuffStat& suff_stat, GaussianHomoskedasticMultivariateRegressionGlobalParameters& global_params);
  void SampleLeafParameters(GaussianHomoskedasticMultivariateRegressionSuffStat& suff_stat, GaussianHomoskedasticMultivariateRegressionGlobalParameters& global_params, std::mt19937& gen, int leaf_num, Tree* tree);
};

/*! \brief Tree model with conjugate Gaussian (multivariate) linear regression leaf node priors and homoskedastic Gaussian outcome likelihood */
class GaussianHomoskedasticMultivariateRegressionModelWrapper : public ModelWrapper {
 public:
  GaussianHomoskedasticMultivariateRegressionModelWrapper();
  GaussianHomoskedasticMultivariateRegressionModelWrapper(int basis_dim);
  ~GaussianHomoskedasticMultivariateRegressionModelWrapper();
  void IncrementNodeSuffStat(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, Eigen::MatrixXd& outcome, data_size_t row_idx, NodeIndicator node_indicator);
  void ResetNodeSuffStat(NodeIndicator node_indicator);
  void SubtractNodeSuffStat(NodeIndicator node_indicator, NodeIndicator lhs_node_indicator, NodeIndicator rhs_node_indicator);
  bool NodeSampleGreaterThan(NodeIndicator node_indicator, data_size_t threshold);
  data_size_t NodeSampleSize(NodeIndicator node_indicator);
  double SplitLogMarginalLikelihood();
  double NoSplitLogMarginalLikelihood();
  void SampleLeafParameters(std::mt19937& gen, int leaf_num, Tree* tree);
  void SetGlobalParameter(double param_value, GlobalParamName param_name);
 private:
  std::map<NodeIndicator, GaussianHomoskedasticMultivariateRegressionSuffStat> suff_stat_map_;
  GaussianHomoskedasticMultivariateRegressionGlobalParameters global_model_params_;
  GaussianHomoskedasticMultivariateRegressionOutcomeModel outcome_model_;
  int basis_dim_;
};

} // namespace StochTree

#endif // STOCHTREE_OUTCOME_MODEL_H_