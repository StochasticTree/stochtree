/*! Copyright (c) 2024 stochtree authors. All rights reserved. */
#ifndef STOCHTREE_IG_SAMPLER_H_
#define STOCHTREE_IG_SAMPLER_H_

#include <random>

namespace StochTree {

class InverseGammaSampler {
 public:
  InverseGammaSampler() {}
  ~InverseGammaSampler() {}
  double Sample(double a, double b, std::mt19937& gen, bool scale_param = true) {
    // C++ standard library provides a gamma distribution with scale
    // parameter, but the correspondence between gamma and IG is that 
    // 1 / gamma(a,b) ~ IG(a,b) when b is a __rate__ parameter.
    // Before sampling, we convert ig_scale to a gamma scale parameter by 
    // taking its multiplicative inverse.
    double gamma_scale = scale_param ? 1./b : b;
    gamma_dist_ = std::gamma_distribution<double>(a, gamma_scale);
    return (1/gamma_dist_(gen));
  }
 private:
  /*! \brief Standard normal distribution */
  std::gamma_distribution<double> gamma_dist_;
};

} // namespace StochTree

#endif // STOCHTREE_IG_SAMPLER_H_