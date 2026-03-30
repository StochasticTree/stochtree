/*! Copyright (c) 2024 stochtree authors. All rights reserved. */
#ifndef STOCHTREE_GAMMA_SAMPLER_H_
#define STOCHTREE_GAMMA_SAMPLER_H_

#include <stochtree/distributions.h>
#include <random>

namespace StochTree {

class GammaSampler {
 public:
  GammaSampler() {}
  ~GammaSampler() {}
  double Sample(double a, double b, std::mt19937& gen, bool rate_param = true) {
    double scale = rate_param ? 1./b : b;
    return sample_gamma(gen, a, scale);
  }
};

} // namespace StochTree

#endif // STOCHTREE_GAMMA_SAMPLER_H_