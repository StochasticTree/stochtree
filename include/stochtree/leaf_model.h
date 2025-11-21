/*!
 * Copyright (c) 2024 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_LEAF_MODEL_H_
#define STOCHTREE_LEAF_MODEL_H_

#include <Eigen/Dense>
#include <stochtree/cutpoint_candidates.h>
#include <stochtree/data.h>
#include <stochtree/gamma_sampler.h>
#include <stochtree/ig_sampler.h>
#include <stochtree/log.h>
#include <stochtree/meta.h>
#include <stochtree/normal_sampler.h>
#include <stochtree/openmp_utils.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/prior.h>
#include <stochtree/tree.h>

#include <random>
#include <variant>

namespace StochTree {

/*! 
 * \defgroup leaf_model_group Leaf Model API
 * 
 * \brief Classes / functions for implementing leaf models.
 * 
 * Stochastic tree algorithms are all essentially hierarchical 
 * models with an adaptive group structure defined by an ensemble 
 * of decision trees. Each novel model is governed by 
 * 
 * - A `LeafModel` class, defining the integrated likelihood and posterior, conditional on a particular tree structure
 * - A `SuffStat` class that tracks and accumulates sufficient statistics necessary for a `LeafModel`
 * 
 * To provide a thorough overview of this interface (and, importantly, how to extend it), we must introduce some mathematical notation. 
 * Any forest-based regression model involves an outcome, which we'll call \f$y\f$, and features (or "covariates"), which we'll call \f$X\f$.
 * Our goal is to predict \f$y\f$ as a function of \f$X\f$, which we'll call \f$f(X)\f$. 
 * 
 * <i>NOTE:</i> if we have a more complicated, but still additive, model, such as \f$y = X\beta + f(X)\f$, then we can just model 
 * \f$y - X\beta = f(X)\f$, treating the residual \f$y - X\beta\f$ as the outcome data, and we are back to the general setting above.
 * 
 * Now, since \f$f(X)\f$ is an additive tree ensemble, we can think of it as the sum of \f$b\f$ separate decision tree functions, 
 * where \f$b\f$ is the number of trees in an ensemble, so that
 * 
 *  \f[
 *    f(X) = f_1(X) + \dots + f_b(X)
 *  \f]
 * 
 * and each decision tree function \f$f_j\f$ has the property that features \f$X\f$ are used to determine which leaf node an observation 
 * falls into, and then the parameters attached to that leaf node are used to compute \f$f_j(X)\f$. The exact mechanics of this process 
 * are model-dependent, so now we introduce the "leaf node" models that `stochtree` supports.
 *
 * \section gaussian_constant_leaf_model Gaussian Constant Leaf Model
 * 
 * The most standard and common tree ensemble is a sum of "constant leaf" trees, in which a leaf node's parameter uniquely determines the prediction
 * for all observations that fall into that leaf. For example, if leaf 2 for a tree is reached by the conditions that \f$X_1 < 0.4 \; \& \; X_2 > 0.6\f$, then 
 * every observation whose first feature is less than 0.4 and whose second feature is greater than 0.6 will receive the same prediction. Mathematically, 
 * for an observation \f$i\f$ this looks like
 * 
 *  \f[
 *    f_j(X_i) = \sum_{\ell \in L} \mathbb{1}(X_i \in \ell) \mu_{\ell}
 *  \f]
 * 
 * where \f$L\f$ denotes the indices of every leaf node, \f$\mu_{\ell}\f$ is the parameter attached to leaf node \f$\ell\f$, and \f$\mathbb{1}(X \in \ell)\f$
 * checks whether \f$X_i\f$ falls into leaf node \f$\ell\f$.
 * 
 * The way that we make such a model "stochastic" is by attaching to the leaf node parameters \f$\mu_{\ell}\f$ a "prior" distribution.
 * This leaf model corresponds to the "classic" BART model of <a href="https://projecteuclid.org/journals/annals-of-applied-statistics/volume-4/issue-1/BART-Bayesian-additive-regression-trees/10.1214/09-AOAS285.full">Chipman et al (2010)</a> 
 * as well as its "XBART" extension (<a href="https://www.tandfonline.com/doi/full/10.1080/01621459.2021.1942012">He and Hahn (2023)</a>). 
 * We assign each leaf node parameter a prior
 * 
 *  \f[
 *    \mu \sim N\left(0, \tau\right)
 *  \f]
 * 
 * Assuming a homoskedastic Gaussian outcome likelihood (i.e. \f$y_i \sim N\left(f(X_i),\sigma^2\right)\f$), 
 * the log marginal likelihood in this model, for the outcome data in node \f$\ell\f$ of tree \f$j\f$ is given by 
 * 
 *  \f[
 *    L(y) = -\frac{n_{\ell}}{2}\log(2\pi) - n_{\ell}\log(\sigma) + \frac{1}{2} \log\left(\frac{\sigma^2}{n_{\ell} \tau + \sigma^2}\right) - \frac{s_{yy,\ell}}{2\sigma^2} + \frac{\tau s_{y,\ell}^2}{2\sigma^2(n_{\ell} \tau + \sigma^2)}
 *  \f]
 * 
 * where
 * 
 *  \f[
 *    n_{\ell} = \sum_{i : X_i \in \ell} 1
 *  \f]
 * 
 *  \f[
 *    s_{y,\ell} = \sum_{i : X_i \in \ell} r_i
 *  \f]
 * 
 *  \f[
 *    s_{yy,\ell} = \sum_{i : X_i \in \ell} r_i^2
 *  \f]
 * 
 *  \f[
 *    r_i = y_i - \sum_{k \neq j} f_k(X_i)
 *  \f]
 *
 * In words, this model depends on the data for a given leaf node only through three sufficient statistics, \f$n_{\ell}\f$, \f$s_{y,\ell}\f$, and \f$s_{yy,\ell}\f$, 
 * and it only depends on the other trees in the ensemble through the "partial residual" \f$r_i\f$. The posterior distribution for 
 * node \f$\ell\f$'s leaf parameter is similarly defined as:
 * 
 *  \f[
 *    \mu_{\ell} \mid - \sim N\left(\frac{\tau s_{y,\ell}}{n_{\ell} \tau + \sigma^2}, \frac{\tau \sigma^2}{n_{\ell} \tau + \sigma^2}\right)
 *  \f]
 * 
 * Now, consider the possibility that each observation carries a unique weight \f$w_i\f$. These could be "case weights" in a survey context or 
 * individual-level variances ("heteroskedasticity"). These case weights transform the outcome distribution (and associated likelihood) to
 * 
 *  \f[
 *    y_i \mid - \sim N\left(\mu(X_i), \frac{\sigma^2}{w_i}\right) 
 *  \f]
 * 
 * This gives a modified log marginal likelihood of 
 * 
 *  \f[
 *    L(y) = -\frac{n_{\ell}}{2}\log(2\pi) - \frac{1}{2} \sum_{i : X_i \in \ell} \log\left(\frac{\sigma^2}{w_i}\right) + \frac{1}{2} \log\left(\frac{\sigma^2}{s_{w,\ell} \tau + \sigma^2}\right) - \frac{s_{wyy,\ell}}{2\sigma^2} + \frac{\tau s_{wy,\ell}^2}{2\sigma^2(s_{w,\ell} \tau + \sigma^2)}
 *  \f]
 * 
 * where
 * 
 *  \f[
 *    s_{w,\ell} = \sum_{i : X_i \in \ell} w_i
 *  \f]
 * 
 *  \f[
 *    s_{wy,\ell} = \sum_{i : X_i \in \ell} w_i r_i
 *  \f]
 * 
 *  \f[
 *    s_{wyy,\ell} = \sum_{i : X_i \in \ell} w_i r_i^2
 *  \f]
 * 
 * Finally, note that when we consider splitting leaf \f$\ell\f$ into new left and right leaves, or pruning two nodes into a single leaf node, 
 * we compute the log marginal likelihood of the combined data and the log marginal likelihoods of the left and right leaves and compare these three values. 
 * 
 * The terms \f$\frac{n_{\ell}}{2}\log(2\pi)\f$, \f$\sum_{i : X_i \in \ell} \log\left(\frac{\sigma^2}{w_i}\right)\f$, and \f$\frac{s_{wyy,\ell}}{2\sigma^2}\f$ 
 * are such that their left and right node values will always sum to the respective value in the combined log marginal likelihood, so they can be ignored 
 * when evaluating splits or prunes and thus the reduced log marginal likelihood is
 * 
 *  \f[
 *    L(y) \propto \frac{1}{2} \log\left(\frac{\sigma^2}{s_{w,\ell} \tau + \sigma^2}\right) + \frac{\tau s_{wy,\ell}^2}{2\sigma^2(n_{\ell} \tau + \sigma^2)}
 *  \f]
 * 
 * So the \ref StochTree::GaussianConstantSuffStat "GaussianConstantSuffStat" class tracks a generalized version of these three statistics
 * (which allows for each observation to have a weight \f$w_i \neq 1\f$):
 * 
 * - \f$n_{\ell}\f$: `data_size_t n`
 * - \f$s_{w,\ell}\f$: `double sum_w`
 * - \f$s_{wy,\ell}\f$: `double sum_yw`
 * 
 * And these values are used by the \ref StochTree::GaussianConstantLeafModel "GaussianConstantLeafModel" class in the 
 * \ref StochTree::GaussianConstantLeafModel::SplitLogMarginalLikelihood "SplitLogMarginalLikelihood", 
 * \ref StochTree::GaussianConstantLeafModel::NoSplitLogMarginalLikelihood "NoSplitLogMarginalLikelihood", 
 * \ref StochTree::GaussianConstantLeafModel::PosteriorParameterMean "PosteriorParameterMean", and 
 * \ref StochTree::GaussianConstantLeafModel::PosteriorParameterVariance "PosteriorParameterVariance" methods. 
 * To give one example, below is the implementation of \ref StochTree::GaussianConstantLeafModel::SplitLogMarginalLikelihood "SplitLogMarginalLikelihood":
 * 
 * \code{.cpp}
 * double left_log_ml = (
 *    -0.5*std::log(1 + tau_*(left_stat.sum_w/global_variance)) + ((tau_*left_stat.sum_yw*left_stat.sum_yw)/(2.0*global_variance*(tau_*left_stat.sum_w + global_variance)))
 * );
 * 
 * double right_log_ml = (
 *    -0.5*std::log(1 + tau_*(right_stat.sum_w/global_variance)) + ((tau_*right_stat.sum_yw*right_stat.sum_yw)/(2.0*global_variance*(tau_*right_stat.sum_w + global_variance)))
 * );
 * 
 * return left_log_ml + right_log_ml;
 * \endcode 
 * 
 * \section gaussian_multivariate_regression_leaf_model Gaussian Multivariate Regression Leaf Model
 * 
 * In this model, the tree defines a "partitioned linear model" in which leaf node parameters define regression weights 
 * that are multiplied by a "basis" \f$\Omega\f$ to determine the prediction for an observation.
 * 
 *  \f[
 *    f_j(X_i) = \sum_{\ell \in L} \mathbb{1}(X_i \in \ell) \Omega_i \vec{\beta_{\ell}}
 *  \f]
 * 
 * and we assign \f$\beta_{\ell}\f$ a prior of
 * 
 *  \f[
 *    \vec{\beta_{\ell}} \sim N\left(\vec{\beta_0}, \Sigma_0\right)
 *  \f]
 * 
 * where \f$\vec{\beta_0}\f$ is typically a vector of zeros. The outcome likelihood is still
 * 
 *  \f[
 *    y_i \sim N\left(f(X_i), \sigma^2\right)
 *  \f]
 * 
 * This gives a reduced log integrated likelihood of
 * 
 *  \f[
 *    L(y) \propto - \frac{1}{2} \log\left(\textrm{det}\left(I_p + \frac{\Sigma_0\Omega'\Omega}{\sigma^2}\right)\right) + \frac{1}{2}\frac{y'\Omega}{\sigma^2}\left(\Sigma_0^{-1} + \frac{\Omega'\Omega}{\sigma^2}\right)^{-1}\frac{\Omega'y}{\sigma^2}
 *  \f]
 * 
 * where \f$\Omega\f$ is a matrix of bases for every observation in leaf \f$\ell\f$ and \f$p\f$ is the dimension of \f$\Omega\f$. The posterior for \f$\vec{\beta_{\ell}}\f$ is 
 * 
 *  \f[
 *    \vec{\beta_{\ell}} \sim N\left(\left(\Sigma_0^{-1} + \frac{\Omega'\Omega}{\sigma^2}\right)^{-1}\left(\frac{\Omega'y}{\sigma^2}\right),\left(\Sigma_0^{-1} + \frac{\Omega'\Omega}{\sigma^2}\right)^{-1}\right)
 *  \f]
 * 
 * This is an extension of the single-tree model of <a href="https://link.springer.com/article/10.1023/A:1013916107446">Chipman et al (2002)</a>, with:
 * 
 * - Support for using a separate basis for leaf model than the partitioning (i.e. tree) model (i.e. \f$X \neq \Omega\f$)
 * - Support for multiple trees and sampling via grow-from-root (GFR) or MCMC
 * 
 * We can also enable heteroskedasticity by defining a (diagonal) covariance matrix for the outcome likelihood
 * 
 *  \f[
 *    \Sigma_y = \text{diag}\left(\sigma^2 / w_1,\sigma^2 / w_2,\dots,\sigma^2 / w_n\right)
 *  \f]
 * 
 * This updates the reduced log integrated likelihood to
 * 
 *  \f[
 *    L(y) \propto - \frac{1}{2} \log\left(\textrm{det}\left(I_p + \Sigma_{0}\Omega'\Sigma_y^{-1}\Omega\right)\right) + \frac{1}{2}y'\Sigma_{y}^{-1}\Omega\left(\Sigma_{0}^{-1} + \Omega'\Sigma_{y}^{-1}\Omega\right)^{-1}\Omega'\Sigma_{y}^{-1}y
 *  \f]
 * 
 * and a posterior for \f$\vec{\beta_{\ell}}\f$ of 
 * 
 *  \f[
 *    \vec{\beta_{\ell}} \sim N\left(\left(\Sigma_{0}^{-1} + \Omega'\Sigma_{y}^{-1}\Omega\right)^{-1}\left(\Omega'\Sigma_{y}^{-1}y\right),\left(\Sigma_{0}^{-1} + \Omega'\Sigma_{y}^{-1}\Omega\right)^{-1}\right)
 *  \f]
 *  
 * \section gaussian_univariate_regression_leaf_model Gaussian Univariate Regression Leaf Model
 * 
 * This specializes the Gaussian Multivariate Regression Leaf Model for a univariate leaf basis, which allows for several computational speedups (replacing generalized matrix operations with simple summation or sum-product operations).
 * We simplify \f$\Omega\f$ to \f$\omega\f$, a univariate basis for every observation, so that \f$\Omega'\Omega = \sum_{i:i \in \ell}\omega_i^2\f$ and \f$\Omega'y = \sum_{i:i \in \ell}\omega_ir_i\f$. Similarly, the prior for the leaf 
 * parameter becomes univariate normal as in \ref gaussian_constant_leaf_model: 
 * 
 *  \f[
 *    \beta \sim N\left(0, \tau\right)
 *  \f]
 * 
 * Allowing for case / variance weights \f$w_i\f$ as above, we derive a reduced log marginal likelihood of 
 * 
 *  \f[
 *    L(y) \propto \frac{1}{2} \log\left(\frac{\sigma^2}{s_{wxx,\ell} \tau + \sigma^2}\right) + \frac{\tau s_{wyx,\ell}^2}{2\sigma^2(s_{wxx,\ell} \tau + \sigma^2)}
 *  \f]
 * 
 * where
 * 
 *  \f[
 *    s_{wxx,\ell} = \sum_{i : X_i \in \ell} w_i \omega_i \omega_i
 *  \f]
 * 
 *  \f[
 *    s_{wyx,\ell} = \sum_{i : X_i \in \ell} w_i r_i \omega_i
 *  \f]
 * 
 * and a posterior of 
 * 
 *  \f[
 *    \beta_{\ell} \mid - \sim N\left(\frac{\tau s_{wyx,\ell}}{s_{wxx,\ell} \tau + \sigma^2}, \frac{\tau \sigma^2}{s_{wxx,\ell} \tau + \sigma^2}\right)
 *  \f]
 * 
 * \section inverse_gamma_leaf_model Inverse Gamma Leaf Model
 * 
 * Each of the above models is a variation on a theme: a conjugate, partitioned Gaussian leaf model. 
 * The inverse gamma leaf model allows for forest-based heteroskedasticity modeling using an inverse gamma prior on the exponentiated leaf parameter, as discussed in <a href="https://www.tandfonline.com/doi/full/10.1080/01621459.2020.1813587">Murray (2021)</a>
 * Define a variance function based on an ensemble of \f$b\f$ trees as
 * 
 *  \f[
 *    \sigma^2(X) = \exp\left(s_1(X) + \dots + s_b(X)\right)
 *  \f]
 * 
 * where each tree function \f$s_j(X)\f$ is defined as 
 * 
 *  \f[
 *    s_j(X_i) = \sum_{\ell \in L} \mathbb{1}(X_i \in \ell) \lambda_{\ell}
 *  \f]
 * 
 * We reparameterize \f$\lambda_{\ell} = \log(\mu_{\ell})\f$ and we place an inverse gamma prior on \f$\mu_{\ell}\f$
 * 
 *  \f[
 *    \mu_{\ell} \sim \text{IG}\left(a, b\right)
 *  \f]
 * 
 * As noted in <a href="https://www.tandfonline.com/doi/full/10.1080/01621459.2020.1813587">Murray (2021)</a>, this model no longer enables the "Bayesian backfitting" simplification 
 * of conjugated Gaussian leaf models, in which sampling updates for a given tree only depend on other trees in the ensemble via their imprint on the partial residual 
 * \f$r_i = y_i - \sum_{k \neq j} \mu_k(X_i)\f$. 
 * However, this model is part of a broader class of models with convenient "blocked MCMC" sampling updates (another important example being multinomial classification).
 * 
 * Under an outcome model
 * 
 *  \f[
 *    y \sim N\left(f(X), \sigma_0^2 \sigma^2(X)\right)
 *  \f]
 * 
 * updates to \f$\mu_{\ell}\f$ for a given tree \f$j\f$ are based on a reduced log marginal likelihood of
 * 
 *  \f[
 *    L(y) \propto a \log (b) - \log \Gamma (a) + \log \Gamma \left(a + \frac{n_{\ell}}{2}\right) - \left(a + \frac{n_{\ell}}{2}\right) \left(b + \frac{s_{\sigma,\ell}}{2\sigma_0^2}\right)
 *  \f]
 * 
 * where
 * 
 *  \f[
 *    n_{\ell} = \sum_{i : X_i \in \ell} 1
 *  \f]
 * 
 *  \f[
 *    s_{\sigma,\ell} = \sum_{i: i \in \ell} \frac{(y_i - f(X_i))^2}{\prod_{k \neq j} s_k(X_i)}
 *  \f]
 * 
 * and a posterior of 
 * 
 *  \f[
 *    \mu_{\ell} \mid - \sim \text{IG}\left( a + \frac{n_{\ell}}{2} , b + \frac{s_{\sigma,\ell}}{2\sigma_0^2} \right)
 *  \f]
 * 
 * Thus, as above, we implement a sufficient statistic class (\ref StochTree::LogLinearVarianceSuffStat "LogLinearVarianceSuffStat"), which tracks
 * 
 * - \f$n_{\ell}\f$: `data_size_t n`
 * - \f$s_{\sigma,\ell}\f$: `double weighted_sum_ei`
 * 
 * And these values are used by the \ref StochTree::LogLinearVarianceLeafModel "LogLinearVarianceLeafModel" class in the 
 * \ref StochTree::LogLinearVarianceLeafModel::SplitLogMarginalLikelihood "SplitLogMarginalLikelihood", 
 * \ref StochTree::LogLinearVarianceLeafModel::NoSplitLogMarginalLikelihood "NoSplitLogMarginalLikelihood", 
 * \ref StochTree::LogLinearVarianceLeafModel::PosteriorParameterShape "PosteriorParameterShape", and 
 * \ref StochTree::LogLinearVarianceLeafModel::PosteriorParameterScale "PosteriorParameterScale" methods. 
 * To give one example, below is the implementation of \ref StochTree::LogLinearVarianceLeafModel::NoSplitLogMarginalLikelihood "NoSplitLogMarginalLikelihood":
 * 
 * \code{.cpp}
 * double prior_terms = a_ * std::log(b_) - boost::math::lgamma(a_);
 * double a_term = a_ + 0.5 * suff_stat.n;
 * double b_term = b_ + ((0.5 * suff_stat.weighted_sum_ei) / global_variance);
 * double log_b_term = std::log(b_term);
 * double lgamma_a_term = boost::math::lgamma(a_term);
 * double resid_term = a_term * log_b_term;
 * double log_ml = prior_terms + lgamma_a_term - resid_term;
 * return log_ml;
 * \endcode 
 * 
 * \{
 */

/*! \brief Leaf models for the forest sampler:
 * 1. `kConstantLeafGaussian`: Every leaf node has a zero-centered univariate normal prior and every leaf is constant.
 * 2. `kUnivariateRegressionLeafGaussian`: Every leaf node has a zero-centered univariate normal prior and every leaf is a linear model, multiplying the leaf parameter by a (fixed) basis.
 * 3. `kMultivariateRegressionLeafGaussian`: Every leaf node has a multivariate normal prior, centered around the zero vector, and every leaf is a linear model, matrix-multiplying the leaf parameters by a (fixed) basis vector.
 * 4. `kLogLinearVariance`: Every leaf node has a inverse gamma prior and every leaf is constant.
 */
enum ModelType {
  kConstantLeafGaussian, 
  kUnivariateRegressionLeafGaussian, 
  kMultivariateRegressionLeafGaussian, 
  kLogLinearVariance
};

/*! \brief Sufficient statistic and associated operations for gaussian homoskedastic constant leaf outcome model */
class GaussianConstantSuffStat {
 public:
  data_size_t n;
  double sum_w;
  double sum_yw;
  /*!
   * \brief Construct a new GaussianConstantSuffStat object, setting all sufficient statistics to zero
   */
  GaussianConstantSuffStat() {
    n = 0;
    sum_w = 0.0;
    sum_yw = 0.0;
  }
  /*!
   * \brief Accumulate data from observation `row_idx` into the sufficient statistics
   * 
   * \param dataset Data object containining training data, including covariates, leaf regression bases, and case weights
   * \param outcome Data object containing the "partial" residual net of all the model's other mean terms, aside from `tree`
   * \param tracker Tracking data structures that speed up sampler operations, synchronized with `active_forest` tracking a forest's state
   * \param row_idx Index of the training data observation from which the sufficient statistics should be updated
   * \param tree_idx Index of the tree being updated in the course of this sufficient statistic update
   */
  void IncrementSuffStat(ForestDataset& dataset, Eigen::VectorXd& outcome, ForestTracker& tracker, data_size_t row_idx, int tree_idx) {
    n += 1;
    if (dataset.HasVarWeights()) {
      sum_w += 1/dataset.VarWeightValue(row_idx);
      sum_yw += outcome(row_idx, 0)/dataset.VarWeightValue(row_idx);
    } else {
      sum_w += 1.0;
      sum_yw += outcome(row_idx, 0);
    }
  }
  /*!
   * \brief Reset all of the sufficient statistics to zero
   */
  void ResetSuffStat() {
    n = 0;
    sum_w = 0.0;
    sum_yw = 0.0;
  }
  /*!
   * \brief Increment the value of each sufficient statistic by the values provided by `suff_stat`
   * 
   * \param suff_stat Sufficient statistic to be added to the current sufficient statistics
   */
  void AddSuffStatInplace(GaussianConstantSuffStat& suff_stat) {
    n += suff_stat.n;
    sum_w += suff_stat.sum_w;
    sum_yw += suff_stat.sum_yw;
  }
  /*!
   * \brief Set the value of each sufficient statistic to the sum of the values provided by `lhs` and `rhs`
   * 
   * \param lhs First sufficient statistic ("left hand side")
   * \param rhs Second sufficient statistic ("right hand side")
   */
  void AddSuffStat(GaussianConstantSuffStat& lhs, GaussianConstantSuffStat& rhs) {
    n = lhs.n + rhs.n;
    sum_w = lhs.sum_w + rhs.sum_w;
    sum_yw = lhs.sum_yw + rhs.sum_yw;
  }
  /*!
   * \brief Set the value of each sufficient statistic to the difference between the values provided by `lhs` and those provided by `rhs`
   * 
   * \param lhs First sufficient statistic ("left hand side")
   * \param rhs Second sufficient statistic ("right hand side")
   */
  void SubtractSuffStat(GaussianConstantSuffStat& lhs, GaussianConstantSuffStat& rhs) {
    n = lhs.n - rhs.n;
    sum_w = lhs.sum_w - rhs.sum_w;
    sum_yw = lhs.sum_yw - rhs.sum_yw;
  }
  /*!
   * \brief Check whether accumulated sample size, `n`, is greater than some threshold
   * 
   * \param threshold Value used to compute `n > threshold`
   */
  bool SampleGreaterThan(data_size_t threshold) {
    return n > threshold;
  }
  /*!
   * \brief Check whether accumulated sample size, `n`, is greater than or equal to some threshold
   * 
   * \param threshold Value used to compute `n >= threshold`
   */
  bool SampleGreaterThanEqual(data_size_t threshold) {
    return n >= threshold;
  }
  /*!
   * \brief Return the sample size accumulated by a sufficient stat object
   */
  data_size_t SampleSize() {
    return n;
  }
};

/*! \brief Marginal likelihood and posterior computation for gaussian homoskedastic constant leaf outcome model */
class GaussianConstantLeafModel {
 public:
  /*!
   * \brief Construct a new GaussianConstantLeafModel object
   * 
   * \param tau Leaf node prior scale parameter
   */
  GaussianConstantLeafModel(double tau) {tau_ = tau; normal_sampler_ = UnivariateNormalSampler();}
  ~GaussianConstantLeafModel() {}
  /*!
   * \brief Log marginal likelihood for a proposed split, evaluated only for observations that fall into the node being split.
   * 
   * \param left_stat Sufficient statistics of the left node formed by the proposed split
   * \param right_stat Sufficient statistics of the right node formed by the proposed split
   * \param global_variance Global error variance parameter
   */
  double SplitLogMarginalLikelihood(GaussianConstantSuffStat& left_stat, GaussianConstantSuffStat& right_stat, double global_variance);
  /*!
   * \brief Log marginal likelihood of a node, evaluated only for observations that fall into the node being split.
   * 
   * \param suff_stat Sufficient statistics of the node being evaluated
   * \param global_variance Global error variance parameter
   */
  double NoSplitLogMarginalLikelihood(GaussianConstantSuffStat& suff_stat, double global_variance);
  /*!
   * \brief Leaf node posterior mean.
   * 
   * \param suff_stat Sufficient statistics of the node being evaluated
   * \param global_variance Global error variance parameter
   */
  double PosteriorParameterMean(GaussianConstantSuffStat& suff_stat, double global_variance);
  /*!
   * \brief Leaf node posterior variance.
   * 
   * \param suff_stat Sufficient statistics of the node being evaluated
   * \param global_variance Global error variance parameter
   */
  double PosteriorParameterVariance(GaussianConstantSuffStat& suff_stat, double global_variance);
  /*!
   * \brief Draw new parameters for every leaf node in `tree`, using a Gibbs update that conditions on the data, every other tree in the forest, and all model parameters
   * 
   * \param dataset Data object containining training data, including covariates, leaf regression bases, and case weights
   * \param tracker Tracking data structures that speed up sampler operations, synchronized with `active_forest` tracking a forest's state
   * \param residual Data object containing the "partial" residual net of all the model's other mean terms, aside from `tree`
   * \param tree Tree to be updated
   * \param tree_num Integer index of tree to be updated
   * \param global_variance Value of the global error variance parameter
   * \param gen C++ random number generator
   */
  void SampleLeafParameters(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, Tree* tree, int tree_num, double global_variance, std::mt19937& gen);
  void SetEnsembleRootPredictedValue(ForestDataset& dataset, TreeEnsemble* ensemble, double root_pred_value);
  /*!
   * \brief Set a new value for the leaf node scale parameter
   * 
   * \param tau Leaf node prior scale parameter
   */
  void SetScale(double tau) {tau_ = tau;}
  /*!
   * \brief Whether this model requires a basis vector for posterior inference and prediction
   */
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
  /*!
   * \brief Construct a new GaussianUnivariateRegressionSuffStat object, setting all sufficient statistics to zero
   */
  GaussianUnivariateRegressionSuffStat() {
    n = 0;
    sum_xxw = 0.0;
    sum_yxw = 0.0;
  }
  /*!
   * \brief Accumulate data from observation `row_idx` into the sufficient statistics
   * 
   * \param dataset Data object containining training data, including covariates, leaf regression bases, and case weights
   * \param outcome Data object containing the "partial" residual net of all the model's other mean terms, aside from `tree`
   * \param tracker Tracking data structures that speed up sampler operations, synchronized with `active_forest` tracking a forest's state
   * \param row_idx Index of the training data observation from which the sufficient statistics should be updated
   * \param tree_idx Index of the tree being updated in the course of this sufficient statistic update
   */
  void IncrementSuffStat(ForestDataset& dataset, Eigen::VectorXd& outcome, ForestTracker& tracker, data_size_t row_idx, int tree_idx) {
    n += 1;
    if (dataset.HasVarWeights()) {
      sum_xxw += dataset.BasisValue(row_idx, 0)*dataset.BasisValue(row_idx, 0)/dataset.VarWeightValue(row_idx);
      sum_yxw += outcome(row_idx, 0)*dataset.BasisValue(row_idx, 0)/dataset.VarWeightValue(row_idx);
    } else {
      sum_xxw += dataset.BasisValue(row_idx, 0)*dataset.BasisValue(row_idx, 0);
      sum_yxw += outcome(row_idx, 0)*dataset.BasisValue(row_idx, 0);
    }
  }
  /*!
   * \brief Reset all of the sufficient statistics to zero
   */
  void ResetSuffStat() {
    n = 0;
    sum_xxw = 0.0;
    sum_yxw = 0.0;
  }
  /*!
   * \brief Increment the value of each sufficient statistic by the values provided by `suff_stat`
   * 
   * \param suff_stat Sufficient statistic to be added to the current sufficient statistics
   */
  void AddSuffStatInplace(GaussianUnivariateRegressionSuffStat& suff_stat) {
    n += suff_stat.n;
    sum_xxw += suff_stat.sum_xxw;
    sum_yxw += suff_stat.sum_yxw;
  }
  /*!
   * \brief Set the value of each sufficient statistic to the sum of the values provided by `lhs` and `rhs`
   * 
   * \param lhs First sufficient statistic ("left hand side")
   * \param rhs Second sufficient statistic ("right hand side")
   */
  void AddSuffStat(GaussianUnivariateRegressionSuffStat& lhs, GaussianUnivariateRegressionSuffStat& rhs) {
    n = lhs.n + rhs.n;
    sum_xxw = lhs.sum_xxw + rhs.sum_xxw;
    sum_yxw = lhs.sum_yxw + rhs.sum_yxw;
  }
  /*!
   * \brief Set the value of each sufficient statistic to the difference between the values provided by `lhs` and those provided by `rhs`
   * 
   * \param lhs First sufficient statistic ("left hand side")
   * \param rhs Second sufficient statistic ("right hand side")
   */
  void SubtractSuffStat(GaussianUnivariateRegressionSuffStat& lhs, GaussianUnivariateRegressionSuffStat& rhs) {
    n = lhs.n - rhs.n;
    sum_xxw = lhs.sum_xxw - rhs.sum_xxw;
    sum_yxw = lhs.sum_yxw - rhs.sum_yxw;
  }
  /*!
   * \brief Check whether accumulated sample size, `n`, is greater than some threshold
   * 
   * \param threshold Value used to compute `n > threshold`
   */
  bool SampleGreaterThan(data_size_t threshold) {
    return n > threshold;
  }
  /*!
   * \brief Check whether accumulated sample size, `n`, is greater than or equal to some threshold
   * 
   * \param threshold Value used to compute `n >= threshold`
   */
  bool SampleGreaterThanEqual(data_size_t threshold) {
    return n >= threshold;
  }
  /*!
   * \brief Return the sample size accumulated by a sufficient stat object
   */
  data_size_t SampleSize() {
    return n;
  }
};

/*! \brief Marginal likelihood and posterior computation for gaussian homoskedastic constant leaf outcome model */
class GaussianUnivariateRegressionLeafModel {
 public:
  GaussianUnivariateRegressionLeafModel(double tau) {tau_ = tau; normal_sampler_ = UnivariateNormalSampler();}
  ~GaussianUnivariateRegressionLeafModel() {}
  /*!
   * \brief Log marginal likelihood for a proposed split, evaluated only for observations that fall into the node being split.
   * 
   * \param left_stat Sufficient statistics of the left node formed by the proposed split
   * \param right_stat Sufficient statistics of the right node formed by the proposed split
   * \param global_variance Global error variance parameter
   */
  double SplitLogMarginalLikelihood(GaussianUnivariateRegressionSuffStat& left_stat, GaussianUnivariateRegressionSuffStat& right_stat, double global_variance);
  /*!
   * \brief Log marginal likelihood of a node, evaluated only for observations that fall into the node being split.
   * 
   * \param suff_stat Sufficient statistics of the node being evaluated
   * \param global_variance Global error variance parameter
   */
  double NoSplitLogMarginalLikelihood(GaussianUnivariateRegressionSuffStat& suff_stat, double global_variance);
  /*!
   * \brief Leaf node posterior mean.
   * 
   * \param suff_stat Sufficient statistics of the node being evaluated
   * \param global_variance Global error variance parameter
   */
  double PosteriorParameterMean(GaussianUnivariateRegressionSuffStat& suff_stat, double global_variance);
  /*!
   * \brief Leaf node posterior variance.
   * 
   * \param suff_stat Sufficient statistics of the node being evaluated
   * \param global_variance Global error variance parameter
   */
  double PosteriorParameterVariance(GaussianUnivariateRegressionSuffStat& suff_stat, double global_variance);
  /*!
   * \brief Draw new parameters for every leaf node in `tree`, using a Gibbs update that conditions on the data, every other tree in the forest, and all model parameters
   * 
   * \param dataset Data object containining training data, including covariates, leaf regression bases, and case weights
   * \param tracker Tracking data structures that speed up sampler operations, synchronized with `active_forest` tracking a forest's state
   * \param residual Data object containing the "partial" residual net of all the model's other mean terms, aside from `tree`
   * \param tree Tree to be updated
   * \param tree_num Integer index of tree to be updated
   * \param global_variance Value of the global error variance parameter
   * \param gen C++ random number generator
   */
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
  /*!
   * \brief Construct a new GaussianMultivariateRegressionSuffStat object
   * 
   * \param basis_dim Size of the basis vector that defines the leaf regression
   */
  GaussianMultivariateRegressionSuffStat(int basis_dim) {
    n = 0;
    XtWX = Eigen::MatrixXd::Zero(basis_dim, basis_dim);
    ytWX = Eigen::MatrixXd::Zero(1, basis_dim);
    p = basis_dim;
  }
  /*!
   * \brief Accumulate data from observation `row_idx` into the sufficient statistics
   * 
   * \param dataset Data object containining training data, including covariates, leaf regression bases, and case weights
   * \param outcome Data object containing the "partial" residual net of all the model's other mean terms, aside from `tree`
   * \param tracker Tracking data structures that speed up sampler operations, synchronized with `active_forest` tracking a forest's state
   * \param row_idx Index of the training data observation from which the sufficient statistics should be updated
   * \param tree_idx Index of the tree being updated in the course of this sufficient statistic update
   */
  void IncrementSuffStat(ForestDataset& dataset, Eigen::VectorXd& outcome, ForestTracker& tracker, data_size_t row_idx, int tree_idx) {
    n += 1;
    if (dataset.HasVarWeights()) {
      XtWX += dataset.GetBasis()(row_idx, Eigen::all).transpose()*dataset.GetBasis()(row_idx, Eigen::all)/dataset.VarWeightValue(row_idx);
      ytWX += (outcome(row_idx, 0)*(dataset.GetBasis()(row_idx, Eigen::all)))/dataset.VarWeightValue(row_idx);
    } else {
      XtWX += dataset.GetBasis()(row_idx, Eigen::all).transpose()*dataset.GetBasis()(row_idx, Eigen::all);
      ytWX += (outcome(row_idx, 0)*(dataset.GetBasis()(row_idx, Eigen::all)));
    }
  }
  /*!
   * \brief Reset all of the sufficient statistics to zero
   */
  void ResetSuffStat() {
    n = 0;
    XtWX = Eigen::MatrixXd::Zero(p, p);
    ytWX = Eigen::MatrixXd::Zero(1, p);
  }
  /*!
   * \brief Increment the value of each sufficient statistic by the values provided by `suff_stat`
   * 
   * \param suff_stat Sufficient statistic to be added to the current sufficient statistics
   */
  void AddSuffStatInplace(GaussianMultivariateRegressionSuffStat& suff_stat) {
    n += suff_stat.n;
    XtWX += suff_stat.XtWX;
    ytWX += suff_stat.ytWX;
  }
  /*!
   * \brief Set the value of each sufficient statistic to the sum of the values provided by `lhs` and `rhs`
   * 
   * \param lhs First sufficient statistic ("left hand side")
   * \param rhs Second sufficient statistic ("right hand side")
   */
  void AddSuffStat(GaussianMultivariateRegressionSuffStat& lhs, GaussianMultivariateRegressionSuffStat& rhs) {
    n = lhs.n + rhs.n;
    XtWX = lhs.XtWX + rhs.XtWX;
    ytWX = lhs.ytWX + rhs.ytWX;
  }
  /*!
   * \brief Set the value of each sufficient statistic to the difference between the values provided by `lhs` and those provided by `rhs`
   * 
   * \param lhs First sufficient statistic ("left hand side")
   * \param rhs Second sufficient statistic ("right hand side")
   */
  void SubtractSuffStat(GaussianMultivariateRegressionSuffStat& lhs, GaussianMultivariateRegressionSuffStat& rhs) {
    n = lhs.n - rhs.n;
    XtWX = lhs.XtWX - rhs.XtWX;
    ytWX = lhs.ytWX - rhs.ytWX;
  }
  /*!
   * \brief Check whether accumulated sample size, `n`, is greater than some threshold
   * 
   * \param threshold Value used to compute `n > threshold`
   */
  bool SampleGreaterThan(data_size_t threshold) {
    return n > threshold;
  }
  /*!
   * \brief Check whether accumulated sample size, `n`, is greater than or equal to some threshold
   * 
   * \param threshold Value used to compute `n >= threshold`
   */
  bool SampleGreaterThanEqual(data_size_t threshold) {
    return n >= threshold;
  }
  /*!
   * \brief Return the sample size accumulated by a sufficient stat object
   */
  data_size_t SampleSize() {
    return n;
  }
};

/*! \brief Marginal likelihood and posterior computation for gaussian homoskedastic constant leaf outcome model */
class GaussianMultivariateRegressionLeafModel {
 public:
  /*!
   * \brief Construct a new GaussianMultivariateRegressionLeafModel object
   * 
   * \param Sigma_0 Prior covariance, must have the same number of rows and columns as dimensions of the basis vector for the multivariate regression problem
   */
  GaussianMultivariateRegressionLeafModel(Eigen::MatrixXd& Sigma_0) {Sigma_0_ = Sigma_0; multivariate_normal_sampler_ = MultivariateNormalSampler();}
  ~GaussianMultivariateRegressionLeafModel() {}
  /*!
   * \brief Log marginal likelihood for a proposed split, evaluated only for observations that fall into the node being split.
   * 
   * \param left_stat Sufficient statistics of the left node formed by the proposed split
   * \param right_stat Sufficient statistics of the right node formed by the proposed split
   * \param global_variance Global error variance parameter
   */
  double SplitLogMarginalLikelihood(GaussianMultivariateRegressionSuffStat& left_stat, GaussianMultivariateRegressionSuffStat& right_stat, double global_variance);
  /*!
   * \brief Log marginal likelihood of a node, evaluated only for observations that fall into the node being split.
   * 
   * \param suff_stat Sufficient statistics of the node being evaluated
   * \param global_variance Global error variance parameter
   */
  double NoSplitLogMarginalLikelihood(GaussianMultivariateRegressionSuffStat& suff_stat, double global_variance);
  /*!
   * \brief Leaf node posterior mean.
   * 
   * \param suff_stat Sufficient statistics of the node being evaluated
   * \param global_variance Global error variance parameter
   */
  Eigen::VectorXd PosteriorParameterMean(GaussianMultivariateRegressionSuffStat& suff_stat, double global_variance);
  /*!
   * \brief Leaf node posterior variance.
   * 
   * \param suff_stat Sufficient statistics of the node being evaluated
   * \param global_variance Global error variance parameter
   */
  Eigen::MatrixXd PosteriorParameterVariance(GaussianMultivariateRegressionSuffStat& suff_stat, double global_variance);
  /*!
   * \brief Draw new parameters for every leaf node in `tree`, using a Gibbs update that conditions on the data, every other tree in the forest, and all model parameters
   * 
   * \param dataset Data object containining training data, including covariates, leaf regression bases, and case weights
   * \param tracker Tracking data structures that speed up sampler operations, synchronized with `active_forest` tracking a forest's state
   * \param residual Data object containing the "partial" residual net of all the model's other mean terms, aside from `tree`
   * \param tree Tree to be updated
   * \param tree_num Integer index of tree to be updated
   * \param global_variance Value of the global error variance parameter
   * \param gen C++ random number generator
   */
  void SampleLeafParameters(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, Tree* tree, int tree_num, double global_variance, std::mt19937& gen);
  void SetEnsembleRootPredictedValue(ForestDataset& dataset, TreeEnsemble* ensemble, double root_pred_value);
  void SetScale(Eigen::MatrixXd& Sigma_0) {Sigma_0_ = Sigma_0;}
  inline bool RequiresBasis() {return true;}
 private:
  Eigen::MatrixXd Sigma_0_;
  MultivariateNormalSampler multivariate_normal_sampler_;
};

/*! \brief Sufficient statistic and associated operations for heteroskedastic log-linear variance model */
class LogLinearVarianceSuffStat {
 public:
  data_size_t n;
  double weighted_sum_ei;
  LogLinearVarianceSuffStat() {
    n = 0;
    weighted_sum_ei = 0.0;
  }
  /*!
   * \brief Accumulate data from observation `row_idx` into the sufficient statistics
   * 
   * \param dataset Data object containining training data, including covariates, leaf regression bases, and case weights
   * \param outcome Data object containing the "partial" residual net of all the model's other mean terms, aside from `tree`
   * \param tracker Tracking data structures that speed up sampler operations, synchronized with `active_forest` tracking a forest's state
   * \param row_idx Index of the training data observation from which the sufficient statistics should be updated
   * \param tree_idx Index of the tree being updated in the course of this sufficient statistic update
   */
  void IncrementSuffStat(ForestDataset& dataset, Eigen::VectorXd& outcome, ForestTracker& tracker, data_size_t row_idx, int tree_idx) {
    n += 1;
    weighted_sum_ei += std::exp(std::log(outcome(row_idx)*outcome(row_idx)) - tracker.GetSamplePrediction(row_idx) + tracker.GetTreeSamplePrediction(row_idx, tree_idx));
  }
  /*!
   * \brief Reset all of the sufficient statistics to zero
   */
  void ResetSuffStat() {
    n = 0;
    weighted_sum_ei = 0.0;
  }
  /*!
   * \brief Increment the value of each sufficient statistic by the values provided by `suff_stat`
   * 
   * \param suff_stat Sufficient statistic to be added to the current sufficient statistics
   */
  void AddSuffStatInplace(LogLinearVarianceSuffStat& suff_stat) {
    n += suff_stat.n;
    weighted_sum_ei += suff_stat.weighted_sum_ei;
  }
  /*!
   * \brief Set the value of each sufficient statistic to the sum of the values provided by `lhs` and `rhs`
   * 
   * \param lhs First sufficient statistic ("left hand side")
   * \param rhs Second sufficient statistic ("right hand side")
   */
  void AddSuffStat(LogLinearVarianceSuffStat& lhs, LogLinearVarianceSuffStat& rhs) {
    n = lhs.n + rhs.n;
    weighted_sum_ei = lhs.weighted_sum_ei + rhs.weighted_sum_ei;
  }
  /*!
   * \brief Set the value of each sufficient statistic to the difference between the values provided by `lhs` and those provided by `rhs`
   * 
   * \param lhs First sufficient statistic ("left hand side")
   * \param rhs Second sufficient statistic ("right hand side")
   */
  void SubtractSuffStat(LogLinearVarianceSuffStat& lhs, LogLinearVarianceSuffStat& rhs) {
    n = lhs.n - rhs.n;
    weighted_sum_ei = lhs.weighted_sum_ei - rhs.weighted_sum_ei;
  }
  /*!
   * \brief Check whether accumulated sample size, `n`, is greater than some threshold
   * 
   * \param threshold Value used to compute `n > threshold`
   */
  bool SampleGreaterThan(data_size_t threshold) {
    return n > threshold;
  }
  /*!
   * \brief Check whether accumulated sample size, `n`, is greater than or equal to some threshold
   * 
   * \param threshold Value used to compute `n >= threshold`
   */
  bool SampleGreaterThanEqual(data_size_t threshold) {
    return n >= threshold;
  }
  /*!
   * \brief Return the sample size accumulated by a sufficient stat object
   */
  data_size_t SampleSize() {
    return n;
  }
};

/*! \brief Marginal likelihood and posterior computation for heteroskedastic log-linear variance model */
class LogLinearVarianceLeafModel {
 public:
  LogLinearVarianceLeafModel(double a, double b) {a_ = a; b_ = b; gamma_sampler_ = GammaSampler();}
  ~LogLinearVarianceLeafModel() {}
  /*!
   * \brief Log marginal likelihood for a proposed split, evaluated only for observations that fall into the node being split.
   * 
   * \param left_stat Sufficient statistics of the left node formed by the proposed split
   * \param right_stat Sufficient statistics of the right node formed by the proposed split
   * \param global_variance Global error variance parameter
   */
  double SplitLogMarginalLikelihood(LogLinearVarianceSuffStat& left_stat, LogLinearVarianceSuffStat& right_stat, double global_variance);
  /*!
   * \brief Log marginal likelihood of a node, evaluated only for observations that fall into the node being split.
   * 
   * \param suff_stat Sufficient statistics of the node being evaluated
   * \param global_variance Global error variance parameter
   */
  double NoSplitLogMarginalLikelihood(LogLinearVarianceSuffStat& suff_stat, double global_variance);
  double SuffStatLogMarginalLikelihood(LogLinearVarianceSuffStat& suff_stat, double global_variance);
  /*!
   * \brief Leaf node posterior shape parameter.
   * 
   * \param suff_stat Sufficient statistics of the node being evaluated
   * \param global_variance Global error variance parameter
   */
  double PosteriorParameterShape(LogLinearVarianceSuffStat& suff_stat, double global_variance);
  /*!
   * \brief Leaf node posterior scale parameter.
   * 
   * \param suff_stat Sufficient statistics of the node being evaluated
   * \param global_variance Global error variance parameter
   */
  double PosteriorParameterScale(LogLinearVarianceSuffStat& suff_stat, double global_variance);
  /*!
   * \brief Draw new parameters for every leaf node in `tree`, using a Gibbs update that conditions on the data, every other tree in the forest, and all model parameters
   * 
   * \param dataset Data object containining training data, including covariates, leaf regression bases, and case weights
   * \param tracker Tracking data structures that speed up sampler operations, synchronized with `active_forest` tracking a forest's state
   * \param residual Data object containing the "full" residual net of all the model's mean terms
   * \param tree Tree to be updated
   * \param tree_num Integer index of tree to be updated
   * \param global_variance Value of the global error variance parameter
   * \param gen C++ random number generator
   */
  void SampleLeafParameters(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, Tree* tree, int tree_num, double global_variance, std::mt19937& gen);
  void SetEnsembleRootPredictedValue(ForestDataset& dataset, TreeEnsemble* ensemble, double root_pred_value);
  void SetPriorShape(double a) {a_ = a;}
  void SetPriorRate(double b) {b_ = b;}
  inline bool RequiresBasis() {return false;}
 private:
  double a_;
  double b_;
  GammaSampler gamma_sampler_;
};

/*!
 * \brief Unifying layer for disparate sufficient statistic class types
 * 
 * Joins together GaussianConstantSuffStat, GaussianUnivariateRegressionSuffStat, 
 * GaussianMultivariateRegressionSuffStat, and LogLinearVarianceSuffStat 
 * as a combined "variant" type. See <a href="https://en.cppreference.com/w/cpp/utility/variant">the std::variant documentation</a> 
 * for more detail.
 */
using SuffStatVariant = std::variant<GaussianConstantSuffStat, 
                                     GaussianUnivariateRegressionSuffStat, 
                                     GaussianMultivariateRegressionSuffStat, 
                                     LogLinearVarianceSuffStat>;

/*!
 * \brief Unifying layer for disparate leaf model class types
 * 
 * Joins together GaussianConstantLeafModel, GaussianUnivariateRegressionLeafModel, 
 * GaussianMultivariateRegressionLeafModel, and LogLinearVarianceLeafModel 
 * as a combined "variant" type. See <a href="https://en.cppreference.com/w/cpp/utility/variant">the std::variant documentation</a> 
 * for more detail.
 */
using LeafModelVariant = std::variant<GaussianConstantLeafModel, 
                                      GaussianUnivariateRegressionLeafModel, 
                                      GaussianMultivariateRegressionLeafModel, 
                                      LogLinearVarianceLeafModel>;

template<typename SuffStatType, typename... SuffStatConstructorArgs>
static inline SuffStatVariant createSuffStat(SuffStatConstructorArgs... leaf_suff_stat_args) {
  return SuffStatType(leaf_suff_stat_args...);
}

template<typename LeafModelType, typename... LeafModelConstructorArgs>
static inline LeafModelVariant createLeafModel(LeafModelConstructorArgs... leaf_model_args) {
  return LeafModelType(leaf_model_args...);
}

/*!
 * \brief Factory function that creates a new `SuffStat` object for the specified model type
 * 
 * \param model_type Enumeration storing the model type
 * \param basis_dim [Optional] dimension of the basis vector, only used if `model_type = kMultivariateRegressionLeafGaussian`
 */
static inline SuffStatVariant suffStatFactory(ModelType model_type, int basis_dim = 0) {
  if (model_type == kConstantLeafGaussian) {
    return createSuffStat<GaussianConstantSuffStat>();
  } else if (model_type == kUnivariateRegressionLeafGaussian) {
    return createSuffStat<GaussianUnivariateRegressionSuffStat>();
  } else if (model_type == kMultivariateRegressionLeafGaussian) {
    return createSuffStat<GaussianMultivariateRegressionSuffStat, int>(basis_dim);
  } else {
    return createSuffStat<LogLinearVarianceSuffStat>();
  }
}

/*!
 * \brief Factory function that creates a new `LeafModel` object for the specified model type
 * 
 * \param model_type Enumeration storing the model type
 * \param tau Value of the leaf node prior scale parameter, only used if `model_type = kConstantLeafGaussian` or `model_type = kUnivariateRegressionLeafGaussian`
 * \param Sigma0 Value of the leaf node prior covariance matrix, only used if `model_type = kMultivariateRegressionLeafGaussian`
 * \param a Value of the leaf node inverse gamma prior shape parameter, only used if `model_type = kLogLinearVariance`
 * \param b Value of the leaf node inverse gamma prior scale parameter, only used if `model_type = kLogLinearVariance`
 */
static inline LeafModelVariant leafModelFactory(ModelType model_type, double tau, Eigen::MatrixXd& Sigma0, double a, double b) {
  if (model_type == kConstantLeafGaussian) {
    return createLeafModel<GaussianConstantLeafModel, double>(tau);
  } else if (model_type == kUnivariateRegressionLeafGaussian) {
    return createLeafModel<GaussianUnivariateRegressionLeafModel, double>(tau);
  } else if (model_type == kMultivariateRegressionLeafGaussian) {
    return createLeafModel<GaussianMultivariateRegressionLeafModel, Eigen::MatrixXd>(Sigma0);
  } else {
    return createLeafModel<LogLinearVarianceLeafModel, double, double>(a, b);
  }
}

template<typename SuffStatType, typename... SuffStatConstructorArgs>
static inline void AccumulateSuffStatProposed(
  SuffStatType& node_suff_stat, SuffStatType& left_suff_stat, SuffStatType& right_suff_stat, ForestDataset& dataset, ForestTracker& tracker, 
  ColumnVector& residual, double global_variance, TreeSplit& split, int tree_num, int leaf_num, int split_feature, int num_threads, 
  SuffStatConstructorArgs&... suff_stat_args
) {
  // Determine the position of the node's indices in the forest tracking data structure
  int node_begin_index = tracker.UnsortedNodeBegin(tree_num, leaf_num);
  int node_end_index = tracker.UnsortedNodeEnd(tree_num, leaf_num);

  // Extract pointer to the feature partition for tree_num
  UnsortedNodeSampleTracker* unsorted_node_sample_tracker = tracker.GetUnsortedNodeSampleTracker();
  FeatureUnsortedPartition* feature_partition = unsorted_node_sample_tracker->GetFeaturePartition(tree_num);

  // Determine the number of threads to use
  int chunk_size = (node_end_index - node_begin_index) / num_threads;
  if (chunk_size < 100) {
    num_threads = 1;
    chunk_size = node_end_index - node_begin_index;
  }

  if (num_threads > 1) {
    // Split the work into num_threads chunks
    std::vector<std::pair<int, int>> thread_ranges(num_threads);
    std::vector<SuffStatType> thread_suff_stats_node;
    std::vector<SuffStatType> thread_suff_stats_left;
    std::vector<SuffStatType> thread_suff_stats_right;
    for (int i = 0; i < num_threads; i++) {
      thread_ranges[i] = std::make_pair(node_begin_index + i * chunk_size, 
                                        node_begin_index + (i + 1) * chunk_size);
      thread_suff_stats_node.emplace_back(suff_stat_args...);
      thread_suff_stats_left.emplace_back(suff_stat_args...);
      thread_suff_stats_right.emplace_back(suff_stat_args...);
    }
    
    // Accumulate sufficient statistics
    StochTree::ParallelFor(0, num_threads, num_threads, [&](int i) {
      int start_idx = thread_ranges[i].first;
      int end_idx = thread_ranges[i].second;
      for (int idx = start_idx; idx < end_idx; idx++) {
        int obs_num = feature_partition->indices_[idx];
        double feature_value = dataset.CovariateValue(obs_num, split_feature);
        thread_suff_stats_node[i].IncrementSuffStat(dataset, residual.GetData(), tracker, obs_num, tree_num);
        if (split.SplitTrue(feature_value)) {
          thread_suff_stats_left[i].IncrementSuffStat(dataset, residual.GetData(), tracker, obs_num, tree_num);
        } else {
          thread_suff_stats_right[i].IncrementSuffStat(dataset, residual.GetData(), tracker, obs_num, tree_num);
        }
      }
    });

    // Combine the thread-local sufficient statistics
    for (int i = 0; i < num_threads; i++) {
      node_suff_stat.AddSuffStatInplace(thread_suff_stats_node[i]);
      left_suff_stat.AddSuffStatInplace(thread_suff_stats_left[i]);
      right_suff_stat.AddSuffStatInplace(thread_suff_stats_right[i]);
    }
  } else {
    for (int idx = node_begin_index; idx < node_end_index; idx++) {
      int obs_num = feature_partition->indices_[idx];
      double feature_value = dataset.CovariateValue(obs_num, split_feature);
      node_suff_stat.IncrementSuffStat(dataset, residual.GetData(), tracker, obs_num, tree_num);
      if (split.SplitTrue(feature_value)) {
        left_suff_stat.IncrementSuffStat(dataset, residual.GetData(), tracker, obs_num, tree_num);
      } else {
        right_suff_stat.IncrementSuffStat(dataset, residual.GetData(), tracker, obs_num, tree_num);
      }
    }
  }
}

template<typename SuffStatType>
static inline void AccumulateSuffStatExisting(SuffStatType& node_suff_stat, SuffStatType& left_suff_stat, SuffStatType& right_suff_stat, ForestDataset& dataset, ForestTracker& tracker, 
                                ColumnVector& residual, double global_variance, int tree_num, int split_node_id, int left_node_id, int right_node_id) {
  // Acquire iterators
  auto left_node_begin_iter = tracker.UnsortedNodeBeginIterator(tree_num, left_node_id);
  auto left_node_end_iter = tracker.UnsortedNodeEndIterator(tree_num, left_node_id);
  auto right_node_begin_iter = tracker.UnsortedNodeBeginIterator(tree_num, right_node_id);
  auto right_node_end_iter = tracker.UnsortedNodeEndIterator(tree_num, right_node_id);

  // Accumulate sufficient statistics for the left and split nodes
  for (auto i = left_node_begin_iter; i != left_node_end_iter; i++) {
    auto idx = *i;
    left_suff_stat.IncrementSuffStat(dataset, residual.GetData(), tracker, idx, tree_num);
    node_suff_stat.IncrementSuffStat(dataset, residual.GetData(), tracker, idx, tree_num);
  }

  // Accumulate sufficient statistics for the right and split nodes
  for (auto i = right_node_begin_iter; i != right_node_end_iter; i++) {
    auto idx = *i;
    right_suff_stat.IncrementSuffStat(dataset, residual.GetData(), tracker, idx, tree_num);
    node_suff_stat.IncrementSuffStat(dataset, residual.GetData(), tracker, idx, tree_num);
  }
}

template<typename SuffStatType, bool sorted>
static inline void AccumulateSingleNodeSuffStat(SuffStatType& node_suff_stat, ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, int tree_num, int node_id) {
  // Acquire iterators
  std::vector<data_size_t>::iterator node_begin_iter;
  std::vector<data_size_t>::iterator node_end_iter;
  if (sorted) {
    // Default to the first feature if we're using the presort tracker
    node_begin_iter = tracker.SortedNodeBeginIterator(node_id, 0);
    node_end_iter = tracker.SortedNodeEndIterator(node_id, 0);
  } else {
    node_begin_iter = tracker.UnsortedNodeBeginIterator(tree_num, node_id);
    node_end_iter = tracker.UnsortedNodeEndIterator(tree_num, node_id);
  }
  
  // Accumulate sufficient statistics
  for (auto i = node_begin_iter; i != node_end_iter; i++) {
    auto idx = *i;
    node_suff_stat.IncrementSuffStat(dataset, residual.GetData(), tracker, idx, tree_num);
  }
}

template<typename SuffStatType>
static inline void AccumulateCutpointBinSuffStat(SuffStatType& left_suff_stat, ForestTracker& tracker, CutpointGridContainer& cutpoint_grid_container, 
                                   ForestDataset& dataset, ColumnVector& residual, double global_variance, int tree_num, int node_id, 
                                   int feature_num, int cutpoint_num) {
  // Acquire iterators
  auto node_begin_iter = tracker.SortedNodeBeginIterator(node_id, feature_num);
  auto node_end_iter = tracker.SortedNodeEndIterator(node_id, feature_num);
  
  // Determine node start point
  data_size_t node_begin = tracker.SortedNodeBegin(node_id, feature_num);

  // Determine cutpoint bin start and end points
  data_size_t current_bin_begin = cutpoint_grid_container.BinStartIndex(cutpoint_num, feature_num);
  data_size_t current_bin_size = cutpoint_grid_container.BinLength(cutpoint_num, feature_num);
  data_size_t next_bin_begin = cutpoint_grid_container.BinStartIndex(cutpoint_num + 1, feature_num);

  // Cutpoint specific iterators
  // TODO: fix the hack of having to subtract off node_begin, probably by cleaning up the CutpointGridContainer interface
  auto cutpoint_begin_iter = node_begin_iter + (current_bin_begin - node_begin);
  auto cutpoint_end_iter = node_begin_iter + (next_bin_begin - node_begin);

  // Accumulate sufficient statistics
  for (auto i = cutpoint_begin_iter; i != cutpoint_end_iter; i++) {
    auto idx = *i;
    left_suff_stat.IncrementSuffStat(dataset, residual.GetData(), tracker, idx, tree_num);
  }
}

/*! \} */ // end of leaf_model_group

} // namespace StochTree

#endif // STOCHTREE_LEAF_MODEL_H_
