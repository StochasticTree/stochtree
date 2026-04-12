/*!
 * Copyright (c) 2024 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_PROBIT_H_
#define STOCHTREE_PROBIT_H_

#include <stochtree/distributions.h>

namespace StochTree {

void sample_probit_latent_outcome(std::mt19937& gen, double* outcome, double* conditional_mean, double* partial_residual, double y_bar, int n) {
  double uniform_draw_std;
  double uniform_draw_trunc;
  double quantile;
  double cond_mean;
  double latent_outcome;
  for (int i = 0; i < n; i++) {
    cond_mean = conditional_mean[i] + y_bar;
    uniform_draw_std = standard_uniform_draw_53bit(gen);
    quantile = norm_cdf(0 - cond_mean);
    if (outcome[i] == 1.0) {
      uniform_draw_trunc = quantile + uniform_draw_std * (1.0 - quantile);
      latent_outcome = norm_inv_cdf(uniform_draw_trunc) + cond_mean;
    } else {
      uniform_draw_trunc = uniform_draw_std * quantile;
      latent_outcome = norm_inv_cdf(uniform_draw_trunc) + cond_mean;
    }
    partial_residual[i] = latent_outcome - cond_mean;
  }
}

}  // namespace StochTree

#endif  // STOCHTREE_PROBIT_H_