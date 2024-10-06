/*! Copyright (c) 2024 stochtree authors. All rights reserved. */
#ifndef STOCHTREE_GAMMA_SAMPLER_H_
#define STOCHTREE_GAMMA_SAMPLER_H_

#include <random>

namespace StochTree {

class GammaSampler {
 public:
  GammaSampler() {}
  ~GammaSampler() {}
  double Sample(double a, double b, std::mt19937& gen, bool rate_param = true) {
    double scale = rate_param ? 1./b : b;
    gamma_dist_ = std::gamma_distribution<double>(a, scale);
    return gamma_dist_(gen);
  }
 private:
  /*! \brief Standard normal distribution */
  std::gamma_distribution<double> gamma_dist_;
};

} // namespace StochTree

#endif // STOCHTREE_IG_SAMPLER_H_