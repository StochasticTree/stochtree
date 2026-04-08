/*! Copyright (c) 2024 stochtree authors. All rights reserved. */
#ifndef STOCHTREE_NORMAL_SAMPLER_H_
#define STOCHTREE_NORMAL_SAMPLER_H_

#include <Eigen/Dense>
#include <stochtree/distributions.h>
#include <stochtree/log.h>
#include <boost/math/special_functions/erf.hpp>
#include <cmath>
#include <random>
#include <vector>

namespace StochTree {

// --- Normal CDF / quantile via boost error functions ---
// Phi(x)      = 0.5 * erfc(-x / sqrt(2))
// Phi^{-1}(p) = -sqrt(2) * erfc_inv(2p)

inline double norm_cdf(double x) {
  return 0.5 * boost::math::erfc(-x / std::sqrt(2.0));
}

inline double norm_inv_cdf(double p) {
  return -std::sqrt(2.0) * boost::math::erfc_inv(2.0 * p);
}

class UnivariateNormalSampler {
 public:
  UnivariateNormalSampler() {std_normal_dist_ = standard_normal();}
  ~UnivariateNormalSampler() {}
  double Sample(double mean, double variance, std::mt19937& gen) {
    return mean + std::sqrt(variance) * std_normal_dist_(gen);
  }
 private:
  /*! \brief Standard normal distribution */
  standard_normal std_normal_dist_;
};

class MultivariateNormalSampler {
 public:
  MultivariateNormalSampler() {std_normal_dist_ = standard_normal();}
  ~MultivariateNormalSampler() {}
  std::vector<double> Sample(Eigen::VectorXd& mean, Eigen::MatrixXd& covariance, std::mt19937& gen) {
    // Dimension extraction and checks
    int mean_cols = mean.size();
    int cov_rows = covariance.rows();
    int cov_cols = covariance.cols();
    CHECK_EQ(mean_cols, cov_cols);
    
    // Variance cholesky decomposition
    Eigen::LLT<Eigen::MatrixXd> decomposition(covariance);
    Eigen::MatrixXd covariance_chol = decomposition.matrixL();

    // Sample a vector of standard normal random variables
    Eigen::VectorXd std_norm_vec(cov_rows);
    for (int i = 0; i < cov_rows; i++) {
      std_norm_vec(i) = std_normal_dist_(gen);
    }

    // Compute and return the sampled value
    Eigen::VectorXd sampled_values_raw = mean + covariance_chol * std_norm_vec;
    std::vector<double> result(cov_rows);
    for (int i = 0; i < cov_rows; i++) {
      result[i] = sampled_values_raw(i, 0);
    }
    return result;
  }
  Eigen::VectorXd SampleEigen(Eigen::VectorXd& mean, Eigen::MatrixXd& covariance, std::mt19937& gen) {
    // Dimension extraction and checks
    int mean_cols = mean.size();
    int cov_rows = covariance.rows();
    int cov_cols = covariance.cols();
    CHECK_EQ(mean_cols, cov_cols);
    
    // Variance cholesky decomposition
    Eigen::LLT<Eigen::MatrixXd> decomposition(covariance);
    Eigen::MatrixXd covariance_chol = decomposition.matrixL();

    // Sample a vector of standard normal random variables
    Eigen::VectorXd std_norm_vec(cov_rows);
    for (int i = 0; i < cov_rows; i++) {
      std_norm_vec(i) = std_normal_dist_(gen);
    }

    // Compute and return the sampled value
    return mean + covariance_chol * std_norm_vec;
  }
 private:
  /*! \brief Standard normal distribution */
  standard_normal std_normal_dist_;
};

} // namespace StochTree

#endif // STOCHTREE_NORMAL_SAMPLER_H_