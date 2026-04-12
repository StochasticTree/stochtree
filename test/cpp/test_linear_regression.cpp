#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <testutils.h>
#include <stochtree/distributions.h>
#include <stochtree/linear_regression.h>
#include <numeric>
#include <random>
#include <vector>

TEST(LinearRegression, UnivariateDegeneratePosteriorMeanCorrectness) {
  // Test that the posterior mean of the regression coefficient is correct in a degenerate case where the outcome has no variance
  // and the prior variance is nearly infinite (i.e. the posterior mean should equal the OLS estimate).

  // Generate data
  std::mt19937 gen(1234);
  int n = 100;
  std::vector<double> x(n, 0.0);
  std::vector<double> y(n, 0.0);
  for (int i = 0; i < n; i++) {
    x[i] = StochTree::standard_uniform_draw_53bit(gen);
    y[i] = 2.0 * x[i];
  }
  double sigma2 = 1e-6;  // near-zero outcome variance
  double tau2 = 1e6;     // near-infinite prior variance

  // Sample from the regression model
  int num_samples = 1000;
  std::vector<double> beta_samples(num_samples);
  for (int i = 0; i < num_samples; i++) {
    beta_samples[i] = StochTree::sample_univariate_gaussian_regression_coefficient(y.data(), x.data(), sigma2, tau2, n, gen);
  }

  // Check posterior mean is close to true value (which should also be the OLS estimate without noise)
  double beta_mean = std::accumulate(beta_samples.begin(), beta_samples.end(), 0.0) / num_samples;
  double ols_estimate = 2.0;
  EXPECT_NEAR(beta_mean, ols_estimate, 1e-2);
}

TEST(LinearRegression, UnivariatePosteriorMeanCorrectness) {
  // Test that the sampled regression coefficients average out close to the expected posterior mean with enough samples

  // Generate data
  std::mt19937 gen(1234);
  int n = 100;
  std::vector<double> x(n, 0.0);
  std::vector<double> y(n, 0.0);
  for (int i = 0; i < n; i++) {
    x[i] = StochTree::standard_uniform_draw_53bit(gen);
    y[i] = 2.0 * x[i] + StochTree::sample_standard_normal(0.0, 0.1, gen);
  }
  double sigma2 = 1;
  double tau2 = 1;

  // Compute the "true" posterior mean analytically for comparison
  double sum_xx = 0.0;
  double sum_yx = 0.0;
  for (int i = 0; i < n; i++) {
    sum_xx += x[i] * x[i];
    sum_yx += y[i] * x[i];
  }
  double post_mean = (sum_yx * sigma2) / (sum_xx * tau2 + sigma2);

  // Draw many samples from the posterior and compute their average
  int num_samples = 10000;
  std::vector<double> beta_samples(num_samples);
  for (int i = 0; i < num_samples; i++) {
    beta_samples[i] = StochTree::sample_univariate_gaussian_regression_coefficient(y.data(), x.data(), sigma2, tau2, n, gen);
  }
  double beta_mean = std::accumulate(beta_samples.begin(), beta_samples.end(), 0.0) / num_samples;
  EXPECT_NEAR(beta_mean, post_mean, 1e-2);
}

TEST(LinearRegression, BivariatePosteriorMeanCorrectness) {
  // Test that the sampled regression coefficients average out close to the expected posterior mean with enough samples

  // Generate data
  std::mt19937 gen(1234);
  int n = 100;
  std::vector<double> x1(n, 0.0);
  std::vector<double> x2(n, 0.0);
  std::vector<double> y(n, 0.0);
  for (int i = 0; i < n; i++) {
    x1[i] = StochTree::standard_uniform_draw_53bit(gen);
    x2[i] = StochTree::standard_uniform_draw_53bit(gen);
    y[i] = 2.0 * x1[i] + 3.0 * x2[i] + StochTree::sample_standard_normal(0.0, 0.1, gen);
  }
  double sigma2 = 1;
  double prior_variance_11 = 1;
  double prior_variance_12 = 0.5;
  double prior_variance_22 = 1;

  // Compute the "true" posterior mean analytically for comparison
  double det_prior_var = prior_variance_11 * prior_variance_22 - prior_variance_12 * prior_variance_12;
  double inv_prior_var_11 = prior_variance_22 / det_prior_var;
  double inv_prior_var_12 = -prior_variance_12 / det_prior_var;
  double inv_prior_var_22 = prior_variance_11 / det_prior_var;
  double sum_x1x1 = 0.0;
  double sum_x2x2 = 0.0;
  double sum_x1x2 = 0.0;
  double sum_yx1 = 0.0;
  double sum_yx2 = 0.0;
  for (int i = 0; i < n; i++) {
    sum_x1x1 += x1[i] * x1[i];
    sum_x2x2 += x2[i] * x2[i];
    sum_x1x2 += x1[i] * x2[i];
    sum_yx1 += y[i] * x1[i];
    sum_yx2 += y[i] * x2[i];
  }
  double post_var_pre_inv_11 = inv_prior_var_11 + sum_x1x1 / sigma2;
  double post_var_pre_inv_12 = inv_prior_var_12 + sum_x1x2 / sigma2;
  double post_var_pre_inv_22 = inv_prior_var_22 + sum_x2x2 / sigma2;
  double det_post_var_pre_inv = post_var_pre_inv_11 * post_var_pre_inv_22 - post_var_pre_inv_12 * post_var_pre_inv_12;
  double post_var_11 = post_var_pre_inv_22 / det_post_var_pre_inv;
  double post_var_12 = -post_var_pre_inv_12 / det_post_var_pre_inv;
  double post_var_22 = post_var_pre_inv_11 / det_post_var_pre_inv;
  double post_mean_1 = post_var_11 * (sum_yx1 / sigma2) + post_var_12 * (sum_yx2 / sigma2);
  double post_mean_2 = post_var_12 * (sum_yx1 / sigma2) + post_var_22 * (sum_yx2 / sigma2);

  // Draw many samples from the posterior and compute their average
  int num_samples = 10000;
  double beta_mean_1_sum = 0.0;
  double beta_mean_2_sum = 0.0;
  std::vector<double> beta_samples(num_samples * 2);
  for (int i = 0; i < num_samples; i++) {
    StochTree::sample_general_bivariate_gaussian_regression_coefficients(beta_samples.data() + 2 * i, y.data(), x1.data(), x2.data(), sigma2, prior_variance_11, prior_variance_12, prior_variance_22, n, gen);
    beta_mean_1_sum += beta_samples[2 * i];
    beta_mean_2_sum += beta_samples[2 * i + 1];
  }
  double beta_mean_1 = beta_mean_1_sum / num_samples;
  double beta_mean_2 = beta_mean_2_sum / num_samples;
  EXPECT_NEAR(beta_mean_1, post_mean_1, 1e-2);
  EXPECT_NEAR(beta_mean_2, post_mean_2, 1e-2);
}

TEST(LinearRegression, BivariateMatchWhenDiagonalPrior) {
  // Test that the sampled regression coefficients for the general bivariate and specialized diagonal bivariate samplers are close to each other with enough samples when the covariance is diagonal

  // Generate data
  std::mt19937 gen(1234);
  int n = 100;
  std::vector<double> x1(n, 0.0);
  std::vector<double> x2(n, 0.0);
  std::vector<double> y(n, 0.0);
  for (int i = 0; i < n; i++) {
    x1[i] = StochTree::standard_uniform_draw_53bit(gen);
    x2[i] = StochTree::standard_uniform_draw_53bit(gen);
    y[i] = 2.0 * x1[i] + 3.0 * x2[i] + StochTree::sample_standard_normal(0.0, 0.1, gen);
  }
  double sigma2 = 1;
  double prior_variance_11 = 1;
  double prior_variance_12 = 0;
  double prior_variance_22 = 1;

  // Draw many samples from the posterior and compute their average
  int num_samples = 10000;
  double beta_mean_1_sum_general = 0.0;
  double beta_mean_2_sum_general = 0.0;
  double beta_mean_1_sum_diagonal = 0.0;
  double beta_mean_2_sum_diagonal = 0.0;
  std::vector<double> beta_samples_general(num_samples * 2);
  std::vector<double> beta_samples_diagonal(num_samples * 2);
  for (int i = 0; i < num_samples; i++) {
    StochTree::sample_general_bivariate_gaussian_regression_coefficients(beta_samples_general.data() + 2 * i, y.data(), x1.data(), x2.data(), sigma2, prior_variance_11, prior_variance_12, prior_variance_22, n, gen);
    StochTree::sample_diagonal_bivariate_gaussian_regression_coefficients(beta_samples_diagonal.data() + 2 * i, y.data(), x1.data(), x2.data(), sigma2, prior_variance_11, prior_variance_22, n, gen);
    beta_mean_1_sum_general += beta_samples_general[2 * i];
    beta_mean_2_sum_general += beta_samples_general[2 * i + 1];
    beta_mean_1_sum_diagonal += beta_samples_diagonal[2 * i];
    beta_mean_2_sum_diagonal += beta_samples_diagonal[2 * i + 1];
  }
  double beta_mean_1_general = beta_mean_1_sum_general / num_samples;
  double beta_mean_2_general = beta_mean_2_sum_general / num_samples;
  double beta_mean_1_diagonal = beta_mean_1_sum_diagonal / num_samples;
  double beta_mean_2_diagonal = beta_mean_2_sum_diagonal / num_samples;
  EXPECT_NEAR(beta_mean_1_general, beta_mean_1_diagonal, 1e-2);
  EXPECT_NEAR(beta_mean_2_general, beta_mean_2_diagonal, 1e-2);
}

TEST(LinearRegression, MultivariateBivariateMatch) {
  // Test that the sampled regression coefficients for the general bivariate and multivariate samplers are close to each other with enough samples when covariates are bivariate

  // Generate data
  std::mt19937 gen(1234);
  int n = 100;
  std::vector<double> x1(n, 0.0);
  std::vector<double> x2(n, 0.0);
  std::vector<double> y(n, 0.0);
  Eigen::VectorXd y_eigen(n);
  Eigen::MatrixXd X_eigen(n, 2);
  double x1_elem, x2_elem, y_elem;
  for (int i = 0; i < n; i++) {
    x1_elem = StochTree::standard_uniform_draw_53bit(gen);
    x2_elem = StochTree::standard_uniform_draw_53bit(gen);
    y_elem = 2.0 * x1_elem + 3.0 * x2_elem + StochTree::sample_standard_normal(0.0, 0.1, gen);
    x1[i] = x1_elem;
    x2[i] = x2_elem;
    y[i] = y_elem;
    y_eigen(i) = y_elem;
    X_eigen(i, 0) = x1_elem;
    X_eigen(i, 1) = x2_elem;
  }
  double sigma2 = 1;
  double prior_variance_11 = 1;
  double prior_variance_12 = 0;
  double prior_variance_22 = 1;
  Eigen::MatrixXd prior_variance(2, 2);
  prior_variance(0, 0) = prior_variance_11;
  prior_variance(0, 1) = prior_variance_12;
  prior_variance(1, 0) = prior_variance_12;
  prior_variance(1, 1) = prior_variance_22;

  // Draw many samples from the posterior and compute their average
  int num_samples = 10000;
  double beta_mean_1_sum_bivariate = 0.0;
  double beta_mean_2_sum_bivariate = 0.0;
  double beta_mean_1_sum_multivariate = 0.0;
  double beta_mean_2_sum_multivariate = 0.0;
  std::vector<double> beta_samples_bivariate(num_samples * 2);
  Eigen::VectorXd beta(2);
  for (int i = 0; i < num_samples; i++) {
    StochTree::sample_general_bivariate_gaussian_regression_coefficients(beta_samples_bivariate.data() + 2 * i, y.data(), x1.data(), x2.data(), sigma2, prior_variance_11, prior_variance_12, prior_variance_22, n, gen);
    beta = StochTree::sample_general_gaussian_regression_coefficients(y_eigen, X_eigen, sigma2, prior_variance, n, gen);
    beta_mean_1_sum_bivariate += beta_samples_bivariate[2 * i];
    beta_mean_2_sum_bivariate += beta_samples_bivariate[2 * i + 1];
    beta_mean_1_sum_multivariate += beta(0);
    beta_mean_2_sum_multivariate += beta(1);
  }
  double beta_mean_1_bivariate = beta_mean_1_sum_bivariate / num_samples;
  double beta_mean_2_bivariate = beta_mean_2_sum_bivariate / num_samples;
  double beta_mean_1_multivariate = beta_mean_1_sum_multivariate / num_samples;
  double beta_mean_2_multivariate = beta_mean_2_sum_multivariate / num_samples;
  EXPECT_NEAR(beta_mean_1_bivariate, beta_mean_1_multivariate, 1e-2);
  EXPECT_NEAR(beta_mean_2_bivariate, beta_mean_2_multivariate, 1e-2);
}
