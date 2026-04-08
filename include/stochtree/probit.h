/*! Copyright (c) 2026 stochtree authors. All rights reserved. */
#ifndef STOCHTREE_PROBIT_H_
#define STOCHTREE_PROBIT_H_

#include <stochtree/data.h>
#include <stochtree/distributions.h>
#include <stochtree/normal_sampler.h>
#include <stochtree/partition_tracker.h>
#include <random>

namespace StochTree {

/*!
 * \brief Albert and Chib (1993) latent Z update for one probit BART iteration.
 *
 * Samples z_i ~ TruncatedNormal(eta_i, 1, (-inf,0]) for y_i = 0 or TruncatedNormal(eta_i, 1, [0,inf))
 * for y_i=1, where eta_i = forest_pred_i + y_bar. 
 * Uses inverse-transform sampling with the identity that truncating N(mu,1) at 0 gives:
 *   y=0: z = mu + Phi^{-1}(u * Phi(-mu))
 *   y=1: z = mu + Phi^{-1}(Phi(-mu) + u * (1 - Phi(-mu)))
 *
 * After sampling, the residual for observation i is set to z_i - eta_i,
 * which equals z_i - y_bar - forest_pred_i, the zero-centered signal
 * the forest needs to learn.
 *
 * \param residual   Column vector of forest residuals; updated in place.
 * \param tracker    ForestTracker holding current cumulative forest predictions.
 * \param y_int      Binary labels (0 or 1), length n_train.
 * \param n_train    Number of training observations.
 * \param y_bar      Probit-scale intercept: Phi^{-1}(mean(y)).
 * \param rng        Mersenne Twister RNG.
 */
inline void sample_probit_latent_outcome(
    ColumnVector&   residual,
    ForestTracker&  tracker,
    const int*      y_int,
    int             n_train,
    double          y_bar,
    std::mt19937&   rng)
{
  for (int i = 0; i < n_train; i++) {
    double eta_i = tracker.GetSamplePrediction(i) + y_bar;
    double phi_neg_eta = norm_cdf(-eta_i);
    double u  = standard_uniform_draw_53bit(rng);
    double z_i;
    if (y_int[i] == 0) {
      z_i = eta_i + norm_inv_cdf(u * phi_neg_eta);
    } else {
      z_i = eta_i + norm_inv_cdf(phi_neg_eta + u * (1.0 - phi_neg_eta));
    }
    // residual_i = (z_i - y_bar) - forest_pred_i = z_i - eta_i
    residual.SetElement(i, z_i - eta_i);
  }
}

} // namespace StochTree

#endif // STOCHTREE_PROBIT_H_
