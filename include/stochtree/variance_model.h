/*!
 * Copyright (c) 2023 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_VARIANCE_MODEL_H_
#define STOCHTREE_VARIANCE_MODEL_H_

#include <Eigen/Dense>

#include <cmath>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace StochTree {

/*! \brief Marginal likelihood and posterior computation for gaussian homoskedastic constant leaf outcome model */
class GlobalHomoskedasticVarianceModel {
 public:
  GlobalHomoskedasticVarianceModel() {}
  ~GlobalHomoskedasticVarianceModel() {}
  double PosteriorShape(Eigen::MatrixXd& residuals, double nu, double lambda) {
    data_size_t n = residuals.rows();
    return (nu/2.0) + n;
  }
  double PosteriorScale(Eigen::MatrixXd& residuals, double nu, double lambda) {
    data_size_t n = residuals.rows();
    double nu_lambda = nu*lambda;
    double sum_sq_resid = 0.;
    for (data_size_t i = 0; i < n; i++) {
      sum_sq_resid += std::pow(residuals(i, 0), 2);
    }
    return (nu_lambda/2.0) + sum_sq_resid;
  }
  double SampleVarianceParameter(Eigen::MatrixXd& residuals, double nu, double lambda, std::mt19937& gen) {
    double ig_shape = PosteriorShape(residuals, nu, lambda);
    double ig_scape = PosteriorScale(residuals, nu, lambda);

    // C++ standard library provides a gamma distribution with scale
    // parameter, but the correspondence between gamma and IG is that 
    // 1 / gamma(a,b) ~ IG(a,b) when b is a __rate__ parameter.
    // Before sampling, we convert ig_scale to a gamma scale parameter by 
    // taking its multiplicative inverse.
    double gamma_scale = 1./ig_scape;
    std::gamma_distribution<double> residual_variance_dist(ig_shape, gamma_scale);
    return (1/residual_variance_dist(gen));
  }
};


/*! \brief Marginal likelihood and posterior computation for gaussian homoskedastic constant leaf outcome model */
class LeafNodeHomoskedasticVarianceModel {
 public:
  LeafNodeHomoskedasticVarianceModel() {}
  ~LeafNodeHomoskedasticVarianceModel() {}
  double PosteriorShape(ModelDraw* model_draw, double a, double b) {
    data_size_t num_leaves = model_draw->NumLeaves();
    return (a/2.0) + num_leaves;
  }
  double PosteriorScale(ModelDraw* model_draw, double a, double b) {
    double mu_sq = model_draw->SumLeafSquared();
    return (b/2.0) + mu_sq;
  }
  double SampleVarianceParameter(ModelDraw* model_draw, double a, double b, std::mt19937& gen) {
    double ig_shape = PosteriorShape(model_draw, a, b);
    double ig_scape = PosteriorScale(model_draw, a, b);

    // C++ standard library provides a gamma distribution with scale
    // parameter, but the correspondence between gamma and IG is that 
    // 1 / gamma(a,b) ~ IG(a,b) when b is a __rate__ parameter.
    // Before sampling, we convert ig_scale to a gamma scale parameter by 
    // taking its multiplicative inverse.
    double gamma_scale = 1./ig_scape;
    std::gamma_distribution<double> residual_variance_dist(ig_shape, gamma_scale);
    return (1/residual_variance_dist(gen));
  }
};

} // namespace StochTree

#endif // STOCHTREE_VARIANCE_MODEL_H_