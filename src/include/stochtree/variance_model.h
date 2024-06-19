/*!
 * Copyright (c) 2023 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_VARIANCE_MODEL_H_
#define STOCHTREE_VARIANCE_MODEL_H_

#include <Eigen/Dense>
#include <stochtree/data.h>
#include <stochtree/ensemble.h>
#include <stochtree/ig_sampler.h>
#include <stochtree/meta.h>

#include <cmath>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace StochTree {

/*! \brief Marginal likelihood and posterior computation for gaussian homoskedastic constant leaf outcome model */
class GlobalHomoskedasticVarianceModel {
 public:
  GlobalHomoskedasticVarianceModel() {ig_sampler_ = InverseGammaSampler();}
  ~GlobalHomoskedasticVarianceModel() {}
  double PosteriorShape(Eigen::VectorXd& residuals, double nu, double lambda) {
    data_size_t n = residuals.rows();
    return (nu/2.0) + n;
  }
  double PosteriorScale(Eigen::VectorXd& residuals, double nu, double lambda) {
    data_size_t n = residuals.rows();
    double nu_lambda = nu*lambda;
    double sum_sq_resid = 0.;
    for (data_size_t i = 0; i < n; i++) {
      sum_sq_resid += std::pow(residuals(i, 0), 2);
    }
    return (nu_lambda/2.0) + sum_sq_resid;
  }
  double SampleVarianceParameter(Eigen::VectorXd& residuals, double nu, double lambda, std::mt19937& gen) {
    double ig_shape = PosteriorShape(residuals, nu, lambda);
    double ig_scale = PosteriorScale(residuals, nu, lambda);
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
    return (a/2.0) + num_leaves;
  }
  double PosteriorScale(TreeEnsemble* ensemble, double a, double b) {
    double mu_sq = ensemble->SumLeafSquared();
    return (b/2.0) + mu_sq;
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