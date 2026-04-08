/*!
 * Copyright (c) 2026 stochtree authors. All rights reserved.
 *
 * Conjugate Bayesian linear regression samplers used by BCF for:
 *   - adaptive coding: sample (b0, b1) from diagonal bivariate posterior
 *   - sample_intercept: sample tau_0 from univariate or multivariate posterior
 *
 * All samplers assume a normal likelihood and a normal prior, giving a
 * closed-form normal posterior. The formulas follow the standard conjugate
 * update:
 *
 *   Prior:      β ~ N(μ₀, Σ₀)
 *   Likelihood: r ~ N(X β, σ² I)
 *   Posterior:  β | r ~ N(μ_n, Σ_n)
 *     Λ_n = Σ₀⁻¹ + X'X / σ²      (posterior precision)
 *     μ_n = Λ_n⁻¹ (Σ₀⁻¹ μ₀ + X'r / σ²)
 */
#ifndef STOCHTREE_LINEAR_REGRESSION_H_
#define STOCHTREE_LINEAR_REGRESSION_H_

#include <stochtree/distributions.h>

#include <Eigen/Dense>
#include <cmath>
#include <random>

namespace StochTree {

// ── Univariate ─────────────────────────────────────────────────────────────
//
// Sample scalar β from the conjugate posterior N(μ_n, σ²_n) where:
//   σ²_n = 1 / (1/prior_var + x'x / σ²)
//   μ_n  = σ²_n * (prior_mean / prior_var + x'r / σ²)
//
// Parameters
//   r          residual vector, length n
//   x          covariate (basis) vector, length n
//   n          number of observations
//   sigma2     current error variance σ²
//   prior_var  prior variance (scalar prior N(prior_mean, prior_var))
//   prior_mean prior mean
//   rng        Mersenne Twister RNG

inline double SampleBayesLinReg1D(
    const double* r,
    const double* x,
    int           n,
    double        sigma2,
    double        prior_var,
    double        prior_mean,
    std::mt19937& rng)
{
    double xr = 0.0, xx = 0.0;
    for (int i = 0; i < n; i++) { xx += x[i] * x[i]; xr += x[i] * r[i]; }
    double post_prec = 1.0 / prior_var + xx / sigma2;
    double post_var  = 1.0 / post_prec;
    double post_mean = post_var * (prior_mean / prior_var + xr / sigma2);
    return sample_standard_normal(post_mean, std::sqrt(post_var), rng);
}

// ── Bivariate diagonal ──────────────────────────────────────────────────────
//
// Sample (b0, b1) independently when x0 and x1 are orthogonal (non-overlapping
// support). This is the case for adaptive coding in BCF, where
//   x0_i = τ_total(X_i) * (1 - Z_i)   (control observations only)
//   x1_i = τ_total(X_i) * Z_i          (treated observations only)
// so x0 ⊥ x1 and the bivariate posterior factorises into two independent 1D
// updates, each with the shared scalar prior N(0, prior_var).
//
// In the standard BCF adaptive coding prior, prior_var = 0.5 (equivalently,
// the prior precision is 2), which corresponds to the `2 * sigma2` term in the
// existing R BCF implementation:
//   posterior variance = sigma2 / (x'x + sigma2 / prior_var)
//                      = sigma2 / (x'x + 2 * sigma2)  when prior_var = 0.5.

inline void SampleBayesLinReg2DDiag(
    const double* r,
    const double* x0,
    const double* x1,
    int           n,
    double        sigma2,
    double        prior_var,
    double&       b0_out,
    double&       b1_out,
    std::mt19937& rng)
{
    b0_out = SampleBayesLinReg1D(r, x0, n, sigma2, prior_var, 0.0, rng);
    b1_out = SampleBayesLinReg1D(r, x1, n, sigma2, prior_var, 0.0, rng);
}

// ── Multivariate ────────────────────────────────────────────────────────────
//
// Sample β ∈ ℝᵈ from N(μ_n, Λ_n⁻¹) where:
//   Λ_n = prior_prec + X'X / σ²
//   μ_n = Λ_n⁻¹ (prior_prec * prior_mean + X'r / σ²)
//
// Accepts pre-computed sufficient statistics X'r and X'X to avoid re-scanning
// the data (callers typically already have these from the current-basis update).
// Sampling uses the precision-Cholesky trick:
//   β = μ_n + (L^T)⁻¹ z,  z ~ N(0, I),  Λ_n = L L^T
//
// Parameters
//   xtr        X'r  (d-vector of sufficient statistics)
//   xtx        X'X  (d×d matrix of sufficient statistics)
//   sigma2     current error variance σ²
//   prior_prec Σ₀⁻¹ (d×d prior precision matrix)
//   prior_mean μ₀   (d-vector prior mean)
//   rng        Mersenne Twister RNG

inline Eigen::VectorXd SampleBayesLinRegMulti(
    const Eigen::VectorXd& xtr,
    const Eigen::MatrixXd& xtx,
    double                 sigma2,
    const Eigen::MatrixXd& prior_prec,
    const Eigen::VectorXd& prior_mean,
    std::mt19937&          rng)
{
    int d = static_cast<int>(prior_mean.size());

    // Posterior precision and mean
    Eigen::MatrixXd post_prec = prior_prec + xtx / sigma2;
    Eigen::VectorXd rhs       = prior_prec * prior_mean + xtr / sigma2;
    Eigen::LLT<Eigen::MatrixXd> llt(post_prec);
    Eigen::VectorXd post_mean = llt.solve(rhs);

    // Sample β = post_mean + (L^T)⁻¹ z  [precision-Cholesky trick]
    standard_normal sn;
    Eigen::VectorXd z(d);
    for (int i = 0; i < d; i++) z(i) = sn(rng);
    Eigen::VectorXd beta = post_mean + llt.matrixU().solve(z);
    return beta;
}

} // namespace StochTree

#endif // STOCHTREE_LINEAR_REGRESSION_H_
