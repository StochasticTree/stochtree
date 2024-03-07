/*! Copyright (c) 2024 stochtree authors. All rights reserved. */
#ifndef STOCHTREE_NORMAL_SAMPLER_H_
#define STOCHTREE_NORMAL_SAMPLER_H_

#include <Eigen/Dense>
#include <stochtree/log.h>
#include <random>
#include <vector>

namespace StochTree {

class UnivariateNormalSampler {
 public:
  UnivariateNormalSampler() {std_normal_dist_ = std::normal_distribution<double>(0.,1.);}
  ~UnivariateNormalSampler() {}
  double Sample(double mean, double variance, std::mt19937& gen) {
    return mean + std::sqrt(variance) * std_normal_dist_(gen);
  }
 private:
  /*! \brief Standard normal distribution */
  std::normal_distribution<double> std_normal_dist_;
};

class MultivariateNormalSampler {
 public:
  MultivariateNormalSampler() {std_normal_dist_ = std::normal_distribution<double>(0.,1.);}
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
  std::normal_distribution<double> std_normal_dist_;
};

} // namespace StochTree

#endif // STOCHTREE_NORMAL_SAMPLER_H_