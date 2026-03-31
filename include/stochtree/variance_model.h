/*!
 * Copyright (c) 2023 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_VARIANCE_MODEL_H_
#define STOCHTREE_VARIANCE_MODEL_H_

#include <Eigen/Dense>
#include <stochtree/data.h>
#include <stochtree/ensemble.h>
#include <stochtree/gamma_sampler.h>
#include <stochtree/ig_sampler.h>
#include <stochtree/meta.h>

#include <random>

namespace StochTree {

/*! \brief Marginal likelihood and posterior computation for gaussian homoskedastic constant leaf outcome model */
class GlobalHomoskedasticVarianceModel {
 public:
  GlobalHomoskedasticVarianceModel() {ig_sampler_ = InverseGammaSampler();}
  ~GlobalHomoskedasticVarianceModel() {}
  double PosteriorShape(Eigen::VectorXd& residuals, double a, double b) {
    data_size_t n = residuals.rows();
    return a + (0.5 * n);
  }
  double PosteriorScale(Eigen::VectorXd& residuals, double a, double b) {
    data_size_t n = residuals.rows();
    double sum_sq_resid = 0.;
    for (data_size_t i = 0; i < n; i++) {
      sum_sq_resid += (residuals(i) * residuals(i));
    }
    return b + (0.5 * sum_sq_resid);
  }
  double PosteriorShape(Eigen::VectorXd& residuals, Eigen::VectorXd& weights, double a, double b) {
    data_size_t n = residuals.rows();
    return a + (0.5 * n);
  }
  double PosteriorScale(Eigen::VectorXd& residuals, Eigen::VectorXd& weights, double a, double b) {
    data_size_t n = residuals.rows();
    double sum_sq_resid = 0.;
    for (data_size_t i = 0; i < n; i++) {
      sum_sq_resid += (residuals(i) * residuals(i)) / weights(i);
    }
    return b + (0.5 * sum_sq_resid);
  }
  double SampleVarianceParameter(Eigen::VectorXd& residuals, double a, double b, std::mt19937& gen) {
    double ig_shape = PosteriorShape(residuals, a, b);
    double ig_scale = PosteriorScale(residuals, a, b);
    return ig_sampler_.Sample(ig_shape, ig_scale, gen);
  }
  double SampleVarianceParameter(Eigen::VectorXd& residuals, Eigen::VectorXd& weights, double a, double b, std::mt19937& gen) {
    double ig_shape = PosteriorShape(residuals, weights, a, b);
    double ig_scale = PosteriorScale(residuals, weights, a, b);
    return ig_sampler_.Sample(ig_shape, ig_scale, gen);
  }
 private:
  InverseGammaSampler ig_sampler_;
};

/*! \brief Marginal likelihood and posterior computation for gaussian homoskedastic constant leaf outcome model */
class LeafNodeHomoskedasticVarianceModel {
 public:
  LeafNodeHomoskedasticVarianceModel() {ig_sampler_ = InverseGammaSampler();}
  ~LeafNodeHomoskedasticVarianceModel() {}
  double PosteriorShape(TreeEnsemble* ensemble, double a, double b) {
    data_size_t num_leaves = ensemble->NumLeaves();
    return (a/2.0) + (num_leaves/2.0);
  }
  double PosteriorScale(TreeEnsemble* ensemble, double a, double b) {
    double mu_sq = ensemble->SumLeafSquared();
    return (b/2.0) + (mu_sq/2.0);
  }
  double SampleVarianceParameter(TreeEnsemble* ensemble, double a, double b, std::mt19937& gen) {
    double ig_shape = PosteriorShape(ensemble, a, b);
    double ig_scale = PosteriorScale(ensemble, a, b);
    return ig_sampler_.Sample(ig_shape, ig_scale, gen);
  }
 private:
  InverseGammaSampler ig_sampler_;
};

} // namespace StochTree

#endif // STOCHTREE_VARIANCE_MODEL_H_