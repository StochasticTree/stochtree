/*!
 * Copyright (c) 2024 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_PROBIT_H_
#define STOCHTREE_PROBIT_H_

#include <stochtree/distributions.h>

namespace StochTree {

void sample_probit_latent_outcome(std::mt19937& gen, double* outcome, double* conditional_mean, double* latent_outcome, int n) {
  double uniform_draw_std;
  double uniform_draw_trunc;
  double quantile;
  for (int i = 0; i < n; i++) {
    uniform_draw_std = standard_uniform_draw_53bit(gen);
    quantile = norm_cdf(0 - conditional_mean[i]);
    if (outcome[i] == 1.0) {
      uniform_draw_trunc = quantile + uniform_draw_std * (1.0 - quantile);
      latent_outcome[i] = norm_inv_cdf(uniform_draw_trunc) + conditional_mean[i];
    } else {
      uniform_draw_trunc = uniform_draw_std * quantile;
      latent_outcome[i] = norm_inv_cdf(uniform_draw_trunc) + conditional_mean[i];
    }
  }
}

}  // namespace StochTree

#endif  // STOCHTREE_PROBIT_H_