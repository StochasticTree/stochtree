/*!
 * Copyright (c) 2024 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_SLICE_SAMPLER_H_
#define STOCHTREE_SLICE_SAMPLER_H_

#include <random>
#include <limits>
#include <cmath>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/polygamma.hpp>

#ifndef M_LN2
#define M_LN2 0.6931471805599453  // ln(2)
#endif

namespace StochTree {

/*!
 * \brief Abstract base class for log-likelihood functions used in slice sampling
 */
class LoglikFunction {
 public:
  virtual ~LoglikFunction() {}
  
  /*!
   * \brief Evaluate the log-likelihood function at point x
   * \param x Input value
   * \return Log-likelihood value
   */
  virtual double Evaluate(double x) = 0;
};

/*!
 * \brief Log-likelihood function for scale_lambda parameter in ordinal models
 */
class ScaleLambdaLoglik : public LoglikFunction {
 public:
  /*!
   * \brief Constructor for scale lambda log-likelihood
   * \param n Number of observations (lambda values)
   * \param sum_lambda Sum of all lambda values
   * \param sum_exp_lambda Sum of exp(lambda) values
   * \param scale Prior scale parameter for scale_lambda
   */
  ScaleLambdaLoglik(double n, double sum_lambda, double sum_exp_lambda, double scale) 
    : n_(n), sum_lambda_(sum_lambda), sum_exp_lambda_(sum_exp_lambda), scale_(scale) {}
  
  /*!
   * \brief Evaluate log-likelihood of scale_lambda parameter
   * \param sigma Input scale parameter value (scale_lambda)
   * \return Log-likelihood value
   */
  double Evaluate(double sigma) override {
    if (sigma <= 0) return -std::numeric_limits<double>::infinity();
    
    // Convert scale_lambda to alpha and beta parameters
    double alpha, beta;
    ScaleLambdaToAlphaBeta(alpha, beta, sigma);
    
    // Log-likelihood contribution from lambda values (gamma prior)
    double loglik = n_ * alpha * std::log(beta) 
                   - n_ * boost::math::lgamma(alpha)
                   + alpha * sum_lambda_ 
                   - beta * sum_exp_lambda_;
    
    // Add constants and prior terms
    loglik += M_LN2 - 0.5 * std::log(2.0 * M_PI);  // M_LN2 - LN_2_BY_PI approximation
    
    // Prior on scale_lambda (half-normal or similar)
    double scale_ratio = sigma / scale_;
    loglik -= 0.5 * scale_ratio * scale_ratio;
    
    return loglik;
  }

 private:
  double n_;
  double sum_lambda_;
  double sum_exp_lambda_;
  double scale_;
  
  /*!
   * \brief Convert scale_lambda to alpha and beta parameters for the gamma prior
   */
  void ScaleLambdaToAlphaBeta(double& alpha, double& beta, const double sigma) {
    double sigma_sq = sigma * sigma;
    alpha = TrigammaInverse(sigma_sq);
    beta = std::exp(boost::math::digamma(alpha));
  }
  
  /*!
   * \brief Compute inverse trigamma function using Newton's method
   */
  double TrigammaInverse(double x) {
    if (x > 1E7) return 1.0 / std::sqrt(x);
    if (x < 1E-6) return 1.0 / x;

    double y = 0.5 + 1.0 / x;
    for (int i = 0; i < 50; i++) {
      double tri = boost::math::trigamma(y);
      double dif = tri * (1.0 - tri / x) / boost::math::polygamma(3, y);
      y += dif;
      if (-dif / y < 1E-8) break;
    }
    return y;
  }
};

/*!
 * \brief Slice sampler implementation
 */
class SliceSampler {
 public:
  SliceSampler() {}
  ~SliceSampler() {}
  
  /*!
   * \brief Sample from a distribution using slice sampling
   * \param x0 Initial value
   * \param loglik_func Log-likelihood function
   * \param w Step size for expanding interval
   * \param lower Lower bound
   * \param upper Upper bound  
   * \param gen Random number generator
   * \return Sampled value
   */
  double Sample(double x0, LoglikFunction* loglik_func, double w, 
                double lower, double upper, std::mt19937& gen) {
    
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::exponential_distribution<double> exp_dist(1.0);
    
    // Find the log density at the initial point
    double gx0 = loglik_func->Evaluate(x0);
    
    // Determine the slice level, in log terms
    double logy = gx0 - exp_dist(gen);
    
    // Find the initial interval to sample from
    double u = w * unif(gen);
    double L = x0 - u;
    double R = x0 + (w - u);
    
    // Expand the interval until its ends are outside the slice
    while (L > lower && loglik_func->Evaluate(L) > logy) {
      L -= w;
    }
    
    while (R < upper && loglik_func->Evaluate(R) > logy) {
      R += w;
    }
    
    // Shrink interval to bounds
    if (L < lower) L = lower;
    if (R > upper) R = upper;
    
    // Sample from the interval, shrinking it on each rejection
    double x1;
    do {
      x1 = L + (R - L) * unif(gen);
      double gx1 = loglik_func->Evaluate(x1);
      
      if (gx1 >= logy) break;
      
      if (x1 > x0) {
        R = x1;
      } else {
        L = x1;
      }
    } while (true);
    
    return x1;
  }
};

} // namespace StochTree

#endif // STOCHTREE_SLICE_SAMPLER_H_
