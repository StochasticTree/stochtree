/*!
 * Copyright (c) 2026 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_REGRESSION_H_
#define STOCHTREE_REGRESSION_H_

#include <Eigen/Dense>
#include <stochtree/distributions.h>
#include <stochtree/normal_sampler.h>

#include <random>
#include "Eigen/src/Core/Matrix.h"

namespace StochTree {

/*!
 * \brief Sample a regression coefficient from the posterior distribution of a univariate Gaussian regression model with known error variance and known prior variance.
 *
 * \param y Pointer to outcome array of length n
 * \param x Pointer to covariate array of length n
 * \param error_variance Known error variance (sigma^2)
 * \param prior_variance Known prior variance (tau^2)
 * \param n Number of observations
 * \param gen Random number generator
 * \return double
 */
static double sample_univariate_gaussian_regression_coefficient(double* y, double* x, double error_variance, double prior_variance, int n, std::mt19937& gen) {
  double sum_xx = 0.0;
  double sum_yx = 0.0;
  for (int i = 0; i < n; i++) {
    sum_xx += x[i] * x[i];
    sum_yx += y[i] * x[i];
  }
  double post_var = (prior_variance * error_variance) / (sum_xx * prior_variance + error_variance);
  double post_mean = post_var * (sum_yx / error_variance);
  return sample_standard_normal(post_mean, std::sqrt(post_var), gen);
}

/*!
 * \brief Sample regression coefficients from the posterior distribution of a bivariate Gaussian regression model with known error variance and known prior variance.
 *
 * \param output Pointer to sampled coefficient array of length 2, where the sampled coefficients will be stored
 * \param y Pointer to outcome array of length n
 * \param x1 Pointer to first covariate array of length n
 * \param x2 Pointer to second covariate array of length n
 * \param error_variance Known error variance (sigma^2)
 * \param prior_variance_11 First diagonal element of the known prior covariance matrix (tau^2 for the first coefficient)
 * \param prior_variance_12 Off-diagonal element of the known prior covariance matrix (covariance between the two coefficients) -- note that this is the same as prior_variance_21 so only one off-diagonal element is needed
 * \param prior_variance_22 Second diagonal element of the known prior covariance matrix (tau^2 for the second coefficient)
 * \param n Number of observations
 * \param gen Random number generator
 */
static void sample_general_bivariate_gaussian_regression_coefficients(double* output, double* y, double* x1, double* x2, double error_variance, double prior_variance_11, double prior_variance_12, double prior_variance_22, int n, std::mt19937& gen) {
  double det_prior_var = prior_variance_11 * prior_variance_22 - prior_variance_12 * prior_variance_12;
  double inv_prior_var_11 = prior_variance_22 / det_prior_var;
  double inv_prior_var_12 = -prior_variance_12 / det_prior_var;
  double inv_prior_var_22 = prior_variance_11 / det_prior_var;
  double sum_x1x1 = 0.0;
  double sum_x1x2 = 0.0;
  double sum_x2x2 = 0.0;
  double sum_yx1 = 0.0;
  double sum_yx2 = 0.0;
  for (int i = 0; i < n; i++) {
    sum_x1x1 += x1[i] * x1[i];
    sum_x1x2 += x1[i] * x2[i];
    sum_x2x2 += x2[i] * x2[i];
    sum_yx1 += y[i] * x1[i];
    sum_yx2 += y[i] * x2[i];
  }
  double post_var_pre_inv_11 = inv_prior_var_11 + sum_x1x1 / error_variance;
  double post_var_pre_inv_12 = inv_prior_var_12 + sum_x1x2 / error_variance;
  double post_var_pre_inv_22 = inv_prior_var_22 + sum_x2x2 / error_variance;
  double det_post_var_pre_inv = post_var_pre_inv_11 * post_var_pre_inv_22 - post_var_pre_inv_12 * post_var_pre_inv_12;
  double post_var_11 = post_var_pre_inv_22 / det_post_var_pre_inv;
  double post_var_12 = -post_var_pre_inv_12 / det_post_var_pre_inv;
  double post_var_22 = post_var_pre_inv_11 / det_post_var_pre_inv;
  double post_mean_1 = post_var_11 * (sum_yx1 / error_variance) + post_var_12 * (sum_yx2 / error_variance);
  double post_mean_2 = post_var_12 * (sum_yx1 / error_variance) + post_var_22 * (sum_yx2 / error_variance);
  double chol_var_11 = std::sqrt(post_var_11);
  double chol_var_12 = post_var_12 / chol_var_11;
  double chol_var_22 = std::sqrt(post_var_22 - chol_var_12 * chol_var_12);
  double z1 = sample_standard_normal(0.0, 1.0, gen);
  double z2 = sample_standard_normal(0.0, 1.0, gen);
  output[0] = post_mean_1 + chol_var_11 * z1;
  output[1] = post_mean_2 + chol_var_12 * z1 + chol_var_22 * z2;
}

/*!
 * \brief Sample regression coefficients from the posterior distribution of a bivariate Gaussian regression model with known error variance and known diagonal prior variance.
 *
 * \param output Pointer to sampled coefficient array of length 2, where the sampled coefficients will be stored
 * \param y Pointer to outcome array of length n
 * \param x1 Pointer to first covariate array of length n
 * \param x2 Pointer to second covariate array of length n
 * \param error_variance Known error variance (sigma^2)
 * \param prior_variance_11 First diagonal element of the known prior covariance matrix (tau^2 for the first coefficient)
 * \param prior_variance_22 Second diagonal element of the known prior covariance matrix (tau^2 for the second coefficient)
 * \param n Number of observations
 * \param gen Random number generator
 */
static void sample_diagonal_bivariate_gaussian_regression_coefficients(double* output, double* y, double* x1, double* x2, double error_variance, double prior_variance_11, double prior_variance_22, int n, std::mt19937& gen) {
  double inv_prior_var_11 = 1.0 / prior_variance_11;
  double inv_prior_var_22 = 1.0 / prior_variance_22;
  double sum_x1x1 = 0.0;
  double sum_x1x2 = 0.0;
  double sum_x2x2 = 0.0;
  double sum_yx1 = 0.0;
  double sum_yx2 = 0.0;
  for (int i = 0; i < n; i++) {
    sum_x1x1 += x1[i] * x1[i];
    sum_x1x2 += x1[i] * x2[i];
    sum_x2x2 += x2[i] * x2[i];
    sum_yx1 += y[i] * x1[i];
    sum_yx2 += y[i] * x2[i];
  }
  double post_var_pre_inv_11 = inv_prior_var_11 + sum_x1x1 / error_variance;
  double post_var_pre_inv_12 = sum_x1x2 / error_variance;
  double post_var_pre_inv_22 = inv_prior_var_22 + sum_x2x2 / error_variance;
  double det_post_var_pre_inv = post_var_pre_inv_11 * post_var_pre_inv_22 - post_var_pre_inv_12 * post_var_pre_inv_12;
  double post_var_11 = post_var_pre_inv_22 / det_post_var_pre_inv;
  double post_var_12 = -post_var_pre_inv_12 / det_post_var_pre_inv;
  double post_var_22 = post_var_pre_inv_11 / det_post_var_pre_inv;
  double post_mean_1 = post_var_11 * (sum_yx1 / error_variance) + post_var_12 * (sum_yx2 / error_variance);
  double post_mean_2 = post_var_12 * (sum_yx1 / error_variance) + post_var_22 * (sum_yx2 / error_variance);
  double chol_var_11 = std::sqrt(post_var_11);
  double chol_var_12 = post_var_12 / chol_var_11;
  double chol_var_22 = std::sqrt(post_var_22 - chol_var_12 * chol_var_12);
  double z1 = sample_standard_normal(0.0, 1.0, gen);
  double z2 = sample_standard_normal(0.0, 1.0, gen);
  output[0] = post_mean_1 + chol_var_11 * z1;
  output[1] = post_mean_2 + chol_var_12 * z1 + chol_var_22 * z2;
}

/*!
 * \brief Sample regression coefficients from the posterior distribution of a bivariate Gaussian regression model with known error variance and known diagonal prior variance.
 *
 * \param y Eigen::VectorXd of outcomes of length n
 * \param X Eigen::MatrixXd of covariates with n rows and p columns
 * \param error_variance Known error variance (sigma^2)
 * \param prior_variance Eigen::MatrixXd of known prior covariance matrix (tau^2 for the coefficients) of dimension p x p
 * \param n Number of observations
 * \param gen Random number generator
 */
static Eigen::VectorXd sample_general_gaussian_regression_coefficients(Eigen::VectorXd& y, Eigen::MatrixXd& X, double error_variance, Eigen::MatrixXd& prior_variance, int n, std::mt19937& gen) {
  int p = X.cols();
  Eigen::MatrixXd inv_prior_var = prior_variance.inverse();
  Eigen::MatrixXd XtX = X.transpose() * X;
  Eigen::VectorXd Xty = X.transpose() * y;
  Eigen::MatrixXd post_var_pre_inv = inv_prior_var + XtX / error_variance;
  Eigen::MatrixXd post_var = post_var_pre_inv.inverse();
  Eigen::VectorXd post_mean = post_var * (Xty / error_variance);
  Eigen::LLT<Eigen::MatrixXd> chol(post_var);
  Eigen::MatrixXd L = chol.matrixL();
  Eigen::VectorXd z(p);
  for (int i = 0; i < p; i++) {
    z(i) = sample_standard_normal(0.0, 1.0, gen);
  }
  return post_mean + L * z;
}

}  // namespace StochTree

#endif  // STOCHTREE_REGRESSION_H_
