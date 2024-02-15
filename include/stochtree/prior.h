/*!
 * Copyright (c) 2024 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_PRIOR_H_
#define STOCHTREE_PRIOR_H_

#include <Eigen/Dense>
#include <stochtree/log.h>

namespace StochTree {

class LeafGaussianPrior {
 public:
  LeafGaussianPrior() {}
  virtual ~LeafGaussianPrior() = default;
};

class LeafConstantGaussianPrior : public LeafGaussianPrior {
 public:
  LeafConstantGaussianPrior(double mu_bar, double tau) {
    mu_bar_ = mu_bar;
    tau_ = tau;
  }
  ~LeafConstantGaussianPrior() {}
  double GetPriorMean() {return mu_bar_;}
  double GetPriorScale() {return tau_;}
  void SetPriorMean(double mu_bar) {mu_bar_ = mu_bar;}
  void SetPriorScale(double tau) {tau_ = tau;}
 private:
  double mu_bar_;
  double tau_;
};

class LeafUnivariateRegressionGaussianPrior : public LeafGaussianPrior {
 public:
  LeafUnivariateRegressionGaussianPrior(double beta_bar, double tau) {
    beta_bar_ = beta_bar;
    tau_ = tau;
  }
  ~LeafUnivariateRegressionGaussianPrior() {}
  double GetPriorMean() {return beta_bar_;}
  double GetPriorScale() {return tau_;}
  void SetPriorMean(double beta_bar) {beta_bar_ = beta_bar;}
  void SetPriorScale(double tau) {tau_ = tau;}
 private:
  double beta_bar_;
  double tau_;
};

class LeafMultivariateRegressionGaussianPrior : public LeafGaussianPrior {
 public:
  LeafMultivariateRegressionGaussianPrior(Eigen::VectorXd& Beta, Eigen::MatrixXd& Sigma, int basis_dim) {
    Beta_ = Beta;
    Sigma_ = Sigma;
    basis_dim_ = basis_dim;
  }
  ~LeafMultivariateRegressionGaussianPrior() {}
  Eigen::VectorXd GetPriorMean() {return Beta_;}
  Eigen::MatrixXd GetPriorScale() {return Sigma_;}
  void SetPriorMean(Eigen::VectorXd& Beta) {Beta_ = Beta;}
  void SetPriorScale(Eigen::MatrixXd& Sigma) {Sigma_ = Sigma;}
 private:
  Eigen::VectorXd Beta_;
  Eigen::MatrixXd Sigma_;
  int basis_dim_;
};

class RandomEffectsGaussianPrior {
 public:
  RandomEffectsGaussianPrior() {}
  virtual ~RandomEffectsGaussianPrior() = default;
};

class RandomEffectsRegressionGaussianPrior : public RandomEffectsGaussianPrior {
 public:
  RandomEffectsRegressionGaussianPrior(double a, double b, int32_t num_components, int32_t num_groups) {
    a_ = a;
    b_ = b;
    num_components_ = num_components;
    num_groups_ = num_groups;
  }
  ~RandomEffectsRegressionGaussianPrior() {}
  double GetPriorVarianceShape() {return a_;}
  double GetPriorVarianceScale() {return b_;}
  int32_t GetNumComponents() {return num_components_;}
  int32_t GetNumGroups() {return num_groups_;}
  void SetPriorVarianceShape(double a) {a_ = a;}
  void SetPriorVarianceScale(double b) {b_ = b;}
  void SetNumComponents(int32_t num_components) {num_components_ = num_components;}
  void SetNumGroups(int32_t num_groups) {num_groups_ = num_groups;}
 private:
  double a_;
  double b_;
  int32_t num_components_; 
  int32_t num_groups_;
};

class TreePrior {
 public:
  TreePrior(double alpha, double beta, int32_t min_samples_in_leaf) {
    alpha_ = alpha;
    beta_ = beta;
    min_samples_in_leaf_ = min_samples_in_leaf;
  }
  ~TreePrior() {}
  double GetAlpha() {return alpha_;}
  double GetBeta() {return beta_;}
  double GetMinSamplesLeaf() {return beta_;}
  void SetAlpha(double alpha) {alpha_ = alpha;}
  void SetBeta(double beta) {beta_ = beta;}
  void SetMinSamplesLeaf(int32_t min_samples_in_leaf) {min_samples_in_leaf_ = min_samples_in_leaf;}
 private:
  double alpha_;
  double beta_;
  int32_t min_samples_in_leaf_;
};

class IGVariancePrior {
 public:
  IGVariancePrior(double shape, double scale) {
    shape_ = shape;
    scale_ = scale;
  }
  ~IGVariancePrior() {}
  double GetShape() {return shape_;}
  double GetScale() {return scale_;}
  void SetShape(double shape) {shape_ = shape;}
  void SetScale(double scale) {scale_ = scale;}
 private:
  double shape_;
  double scale_;
};

/*! \brief Sufficient statistic and associated operations for gaussian homoskedastic constant leaf outcome model */
struct LeafConstantGaussianSuffStat {
  data_size_t n;
  double sum_y;
  double sum_y_squared;
  LeafConstantGaussianSuffStat() {
    n = 0;
    sum_y = 0.0;
    sum_y_squared = 0.0;
  }
  template <typename LeafForestDatasetType>
  void IncrementSuffStat(LeafForestDatasetType* data, UnivariateResidual* residual, data_size_t row_idx) {
    n += 1;
    sum_y += residual->residual(row_idx, 0);
    sum_y_squared += std::pow(residual->residual(row_idx, 0), 2.0);
  }
  void ResetSuffStat() {
    n = 0;
    sum_y = 0.0;
    sum_y_squared = 0.0;
  }
  void SubtractSuffStat(LeafConstantGaussianSuffStat& lhs, LeafConstantGaussianSuffStat& rhs) {
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

/*! \brief Sufficient statistic and associated operations for homoskedastic, univariate regression leaf outcome model */
struct LeafUnivariateRegressionGaussianSuffStat {
  data_size_t n;
  double sum_y;
  double sum_yx;
  double sum_x_squared;
  double sum_y_squared;
  LeafUnivariateRegressionGaussianSuffStat() {
    n = 0;
    sum_y = 0.0;
    sum_yx = 0.0;
    sum_x_squared = 0.0;
    sum_y_squared = 0.0;
  }
  template <typename LeafForestDatasetType>
  void IncrementSuffStat(RegressionLeafForestDataset* data, UnivariateResidual* residual, data_size_t row_idx) {
    n += 1;
    sum_y += residual->residual(row_idx, 0);
    sum_yx += residual->residual(row_idx, 0)*data->basis(row_idx, 0);
    sum_x_squared += std::pow(data->basis(row_idx, 0), 2.0);
    sum_y_squared += std::pow(residual->residual(row_idx, 0), 2.0);
  }
  void ResetSuffStat() {
    n = 0;
    sum_y = 0.0;
    sum_yx = 0.0;
    sum_x_squared = 0.0;
    sum_y_squared = 0.0;
  }
  void SubtractSuffStat(LeafUnivariateRegressionGaussianSuffStat& lhs, LeafUnivariateRegressionGaussianSuffStat& rhs) {
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

/*! \brief Sufficient statistic and associated operations for gaussian homoskedastic multivariate regression leaf outcome model */
struct LeafMultivariateRegressionGaussianSuffStat {
  data_size_t n;
  int basis_dim;
  double yty;
  Eigen::MatrixXd Xty;
  Eigen::MatrixXd XtX;
  LeafMultivariateRegressionGaussianSuffStat(int basis_dim) {
    basis_dim = basis_dim;
    n = 0;
    yty = 0.0;
    Xty = Eigen::MatrixXd::Zero(basis_dim, 1);
    XtX = Eigen::MatrixXd::Zero(basis_dim, basis_dim);
  }
  template <typename LeafForestDatasetType>
  void IncrementSuffStat(LeafForestDatasetType* data, UnivariateResidual* residual, data_size_t row_idx) {
    CHECK_EQ(basis_dim, data->basis.cols());
    n += 1;
    yty += std::pow(residual->residual(row_idx, 0), 2.0);
    Xty += data->basis(row_idx, Eigen::all).transpose()*residual->residual(row_idx, 0);
    XtX += data->basis(row_idx, Eigen::all).transpose()*data->basis(row_idx, Eigen::all);
  }
  void ResetSuffStat() {
    n = 0;
    yty = 0.0;
    Xty = Eigen::MatrixXd::Zero(basis_dim, 1);
    XtX = Eigen::MatrixXd::Zero(basis_dim, basis_dim);
  }
  void SubtractSuffStat(LeafMultivariateRegressionGaussianSuffStat& lhs, LeafMultivariateRegressionGaussianSuffStat& rhs) {
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

} // namespace StochTree

#endif // STOCHTREE_PRIOR_H_