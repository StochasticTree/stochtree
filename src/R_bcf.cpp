/*!
 * cpp11 binding for BCFSamplerFit — R fast path.
 *
 * Returns a named list that R/bcf.R assembles into a bcfmodel object.
 * Optional inputs are passed as SEXP so that NULL can be used from R.
 */
#include <cpp11.hpp>
#include "stochtree_types.h"
#include <stochtree/bcf.h>
#include <stochtree/bcf_sampler.h>
#include <stochtree/random_effects.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

static bool unpack_matrix(SEXP sexp, const double*& ptr, int& nrow, int& ncol) {
    if (Rf_isNull(sexp)) return false;
    ptr  = REAL(sexp);
    nrow = Rf_nrows(sexp);
    ncol = Rf_ncols(sexp);
    return true;
}
static bool unpack_dvec(SEXP sexp, const double*& ptr, int& len) {
    if (Rf_isNull(sexp)) return false;
    ptr = REAL(sexp);
    len = Rf_length(sexp);
    return true;
}
static bool unpack_ivec(SEXP sexp, const int*& ptr, int& len) {
    if (Rf_isNull(sexp)) return false;
    ptr = INTEGER(sexp);
    len = Rf_length(sexp);
    return true;
}

[[cpp11::register]]
cpp11::list bcf_sampler_fit_r_cpp(
    // ── Required training data ─────────────────────────────────────────────
    cpp11::doubles_matrix<> X_train_r,    // n_train × p
    cpp11::doubles          y_train_r,    // n_train
    cpp11::doubles_matrix<> Z_train_r,    // n_train × treatment_dim
    // ── Optional data (SEXP so NULL is valid) ──────────────────────────────
    SEXP pi_hat_train_r,    // n_train propensity scores or R_NilValue
    SEXP X_test_r,          // n_test × p or R_NilValue
    SEXP Z_test_r,          // n_test × treatment_dim or R_NilValue
    SEXP pi_hat_test_r,     // n_test propensity scores or R_NilValue
    SEXP weights_r,         // n_train or R_NilValue
    SEXP feature_types_r,   // p integers or R_NilValue
    // ── Sampler counts ─────────────────────────────────────────────────────
    int  num_gfr,
    int  num_burnin,
    int  num_mcmc,
    int  num_chains,
    int  keep_every,
    bool keep_gfr,
    bool keep_burnin,
    // ── Global variance ────────────────────────────────────────────────────
    bool   sample_sigma2_global,
    double a_global,
    double b_global,
    double sigma2_init,
    // ── Mu-forest params ───────────────────────────────────────────────────
    int    num_trees_mu,
    double alpha_mu,
    double beta_mu,
    int    min_samples_leaf_mu,
    int    max_depth_mu,
    bool   sample_sigma2_leaf_mu,
    double a_leaf_mu,
    double b_leaf_mu,
    double leaf_scale_mu,
    cpp11::doubles variable_weights_mu_r,
    // ── Tau-forest params ──────────────────────────────────────────────────
    int    num_trees_tau,
    double alpha_tau,
    double beta_tau,
    int    min_samples_leaf_tau,
    int    max_depth_tau,
    bool   sample_sigma2_leaf_tau,
    double a_leaf_tau,
    double b_leaf_tau,
    double leaf_scale_tau,
    cpp11::doubles variable_weights_tau_r,
    // ── Treatment effect intercept ─────────────────────────────────────────
    bool   sample_intercept,
    double tau_0_prior_var,
    // ── Adaptive coding ────────────────────────────────────────────────────
    bool   adaptive_coding,
    double b0_init,
    double b1_init,
    double coding_prior_var,
    // ── Propensity routing ─────────────────────────────────────────────────
    std::string propensity_covariate,    // "mu", "tau", "both", "none"
    // ── Variance forest ────────────────────────────────────────────────────
    bool   include_variance_forest,
    int    num_trees_variance,
    double alpha_variance,
    double beta_variance,
    int    min_samples_leaf_variance,
    int    max_depth_variance,
    double a_forest,
    double b_forest,
    double variance_forest_leaf_init,
    // ── Misc ───────────────────────────────────────────────────────────────
    bool   standardize,
    int    random_seed,
    int    num_threads
) {
    int n_train = X_train_r.nrow();
    int p       = X_train_r.ncol();
    int treatment_dim = Z_train_r.ncol();

    // Unpack optional SEXP args
    const double* pi_hat_train_ptr = nullptr; int n_pi = 0;
    const double* X_test_ptr       = nullptr; int n_test = 0, p_test = 0;
    const double* Z_test_ptr       = nullptr; int n_zt = 0, td_t = 0;
    const double* pi_hat_test_ptr  = nullptr; int n_pit = 0;
    const double* weights_ptr      = nullptr; int n_wt = 0;
    const int*    ft_ptr           = nullptr; int n_ft = 0;

    bool has_pi      = unpack_dvec  (pi_hat_train_r, pi_hat_train_ptr, n_pi);
    bool has_test    = unpack_matrix(X_test_r,        X_test_ptr,  n_test, p_test);
    bool has_z_test  = unpack_matrix(Z_test_r,        Z_test_ptr,  n_zt,   td_t);
    bool has_pi_test = unpack_dvec  (pi_hat_test_r,   pi_hat_test_ptr, n_pit);
    bool has_weights = unpack_dvec  (weights_r,        weights_ptr, n_wt);
    bool has_ft      = unpack_ivec  (feature_types_r,  ft_ptr,      n_ft);

    // ── BCFConfig ────────────────────────────────────────────────────────
    StochTree::BCFConfig cfg;
    cfg.num_gfr     = num_gfr;
    cfg.num_burnin  = num_burnin;
    cfg.num_mcmc    = num_mcmc;
    cfg.num_chains  = num_chains;
    cfg.keep_every  = keep_every;
    cfg.keep_gfr    = keep_gfr;
    cfg.keep_burnin = keep_burnin;
    cfg.profile_phases = false;

    cfg.sample_sigma2_global = sample_sigma2_global;
    cfg.a_global             = a_global;
    cfg.b_global             = b_global;
    cfg.sigma2_init          = sigma2_init;

    cfg.mu_forest.num_trees         = num_trees_mu;
    cfg.mu_forest.alpha             = alpha_mu;
    cfg.mu_forest.beta              = beta_mu;
    cfg.mu_forest.min_samples_leaf  = min_samples_leaf_mu;
    cfg.mu_forest.max_depth         = max_depth_mu;
    cfg.mu_forest.sample_sigma2_leaf = sample_sigma2_leaf_mu;
    cfg.mu_forest.a_leaf            = a_leaf_mu;
    cfg.mu_forest.b_leaf            = b_leaf_mu;
    cfg.mu_forest.leaf_scale        = leaf_scale_mu;
    if (variable_weights_mu_r.size() > 0)
        cfg.mu_forest.variable_weights.assign(variable_weights_mu_r.begin(),
                                              variable_weights_mu_r.end());

    cfg.tau_forest.num_trees         = num_trees_tau;
    cfg.tau_forest.alpha             = alpha_tau;
    cfg.tau_forest.beta              = beta_tau;
    cfg.tau_forest.min_samples_leaf  = min_samples_leaf_tau;
    cfg.tau_forest.max_depth         = max_depth_tau;
    cfg.tau_forest.sample_sigma2_leaf = sample_sigma2_leaf_tau;
    cfg.tau_forest.a_leaf            = a_leaf_tau;
    cfg.tau_forest.b_leaf            = b_leaf_tau;
    cfg.tau_forest.leaf_scale        = leaf_scale_tau;
    if (variable_weights_tau_r.size() > 0)
        cfg.tau_forest.variable_weights.assign(variable_weights_tau_r.begin(),
                                               variable_weights_tau_r.end());

    cfg.sample_intercept    = sample_intercept;
    cfg.tau_0_prior_var     = tau_0_prior_var;
    cfg.adaptive_coding     = adaptive_coding;
    cfg.b0_init             = b0_init;
    cfg.b1_init             = b1_init;
    cfg.coding_prior_var    = coding_prior_var;
    cfg.propensity_covariate = propensity_covariate;

    cfg.include_variance_forest      = include_variance_forest;
    cfg.num_trees_variance           = num_trees_variance;
    cfg.alpha_variance               = alpha_variance;
    cfg.beta_variance                = beta_variance;
    cfg.min_samples_leaf_variance    = min_samples_leaf_variance;
    cfg.max_depth_variance           = max_depth_variance;
    cfg.a_forest                     = a_forest;
    cfg.b_forest                     = b_forest;
    cfg.variance_forest_leaf_init    = variance_forest_leaf_init;

    cfg.standardize  = standardize;
    cfg.random_seed  = random_seed;
    cfg.num_threads  = num_threads;

    // ── BCFData ─────────────────────────────────────────────────────────
    StochTree::BCFData data;
    data.X_train       = REAL(X_train_r);
    data.n_train       = n_train;
    data.p             = p;
    data.y_train       = REAL(y_train_r);
    data.Z_train       = REAL(Z_train_r);
    data.treatment_dim = treatment_dim;

    if (has_pi)      data.pi_hat_train = pi_hat_train_ptr;
    if (has_test)  { data.X_test = X_test_ptr;  data.n_test = n_test; }
    if (has_z_test)  data.Z_test = Z_test_ptr;
    if (has_pi_test) data.pi_hat_test = pi_hat_test_ptr;
    if (has_weights) data.weights = weights_ptr;
    if (has_ft)      data.feature_types = ft_ptr;

    // ── Run sampler ──────────────────────────────────────────────────────
    StochTree::BCFResult result;
    StochTree::BCFSamplerFit(&result, cfg, data);

    // ── Build return list ────────────────────────────────────────────────
    cpp11::writable::list out;

    // Forest containers
    auto push_fc = [&](const char* name,
                        std::unique_ptr<StochTree::ForestContainer>& fc) {
        if (fc) {
            out.push_back(cpp11::named_arg(name) =
                cpp11::external_pointer<StochTree::ForestContainer>(fc.release()));
        } else {
            out.push_back(cpp11::named_arg(name) = R_NilValue);
        }
    };
    push_fc("mu_forest_container_ptr",       result.mu_forest_container);
    push_fc("tau_forest_container_ptr",      result.tau_forest_container);
    push_fc("variance_forest_container_ptr", result.variance_forest_container);

    auto to_r_doubles = [](const std::vector<double>& v) {
        cpp11::writable::doubles out(static_cast<int>(v.size()));
        std::copy(v.begin(), v.end(), out.begin());
        return out;
    };

    out.push_back(cpp11::named_arg("y_hat_train")           = to_r_doubles(result.y_hat_train));
    out.push_back(cpp11::named_arg("y_hat_test")            = to_r_doubles(result.y_hat_test));
    out.push_back(cpp11::named_arg("mu_hat_train")          = to_r_doubles(result.mu_hat_train));
    out.push_back(cpp11::named_arg("mu_hat_test")           = to_r_doubles(result.mu_hat_test));
    out.push_back(cpp11::named_arg("tau_hat_train")         = to_r_doubles(result.tau_hat_train));
    out.push_back(cpp11::named_arg("tau_hat_test")          = to_r_doubles(result.tau_hat_test));
    out.push_back(cpp11::named_arg("sigma2_x_hat_train")    = to_r_doubles(result.sigma2_x_hat_train));
    out.push_back(cpp11::named_arg("sigma2_x_hat_test")     = to_r_doubles(result.sigma2_x_hat_test));
    out.push_back(cpp11::named_arg("sigma2_global_samples") = to_r_doubles(result.sigma2_global_samples));
    out.push_back(cpp11::named_arg("leaf_scale_mu_samples") = to_r_doubles(result.leaf_scale_mu_samples));
    out.push_back(cpp11::named_arg("leaf_scale_tau_samples")= to_r_doubles(result.leaf_scale_tau_samples));
    out.push_back(cpp11::named_arg("tau_0_samples")         = to_r_doubles(result.tau_0_samples));
    out.push_back(cpp11::named_arg("b0_samples")            = to_r_doubles(result.b0_samples));
    out.push_back(cpp11::named_arg("b1_samples")            = to_r_doubles(result.b1_samples));

    out.push_back(cpp11::named_arg("y_bar")             = result.y_bar);
    out.push_back(cpp11::named_arg("y_std")             = result.y_std);
    out.push_back(cpp11::named_arg("num_total_samples") = result.num_total_samples);

    return out;
}
