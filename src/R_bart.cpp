/*!
 * cpp11 binding for BARTSamplerFit — R fast path.
 *
 * Returns a named list that R/bart.R assembles into a bartmodel object.
 * All optional inputs (X_test, weights, basis, rfx) are passed as SEXP
 * so that NULL can be used from R to signal "not present".
 */
#include <cpp11.hpp>
#include "stochtree_types.h"
#include <stochtree/bart.h>
#include <stochtree/bart_sampler.h>
#include <stochtree/random_effects.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

// Helper: convert a cpp11::doubles_matrix SEXP to (pointer, nrow, ncol) without copy.
// Returns false if sexp is NULL/nil.
static bool unpack_matrix(SEXP sexp, const double*& ptr, int& nrow, int& ncol) {
    if (Rf_isNull(sexp)) return false;
    ptr  = REAL(sexp);
    nrow = Rf_nrows(sexp);
    ncol = Rf_ncols(sexp);
    return true;
}

static bool unpack_ivec(SEXP sexp, const int*& ptr, int& len) {
    if (Rf_isNull(sexp)) return false;
    ptr = INTEGER(sexp);
    len = Rf_length(sexp);
    return true;
}

static bool unpack_dvec(SEXP sexp, const double*& ptr, int& len) {
    if (Rf_isNull(sexp)) return false;
    ptr = REAL(sexp);
    len = Rf_length(sexp);
    return true;
}

[[cpp11::register]]
cpp11::list bart_sampler_fit_r_cpp(
    // ── Required training data ────────────────────────────────────────────────
    cpp11::doubles_matrix<> X_train_r,    // n_train × p, column-major
    cpp11::doubles          y_train_r,    // n_train
    // ── Optional data (SEXP so NULL is valid) ─────────────────────────────────
    SEXP X_test_r,            // n_test × p matrix or R_NilValue
    SEXP weights_r,           // n_train doubles or R_NilValue
    SEXP feature_types_r,     // p integers or R_NilValue  (NULL = all numeric)
    SEXP basis_train_r,       // n_train × basis_dim matrix or R_NilValue
    SEXP basis_test_r,        // n_test × basis_dim matrix or R_NilValue
    SEXP rfx_group_ids_train_r,// n_train integers (1-indexed) or R_NilValue
    SEXP rfx_group_ids_test_r, // n_test  integers or R_NilValue
    SEXP rfx_basis_train_r,    // n_train × rfx_basis_dim matrix or R_NilValue
    SEXP rfx_basis_test_r,     // n_test  × rfx_basis_dim matrix or R_NilValue
    // ── Tree config ───────────────────────────────────────────────────────────
    int  num_trees,
    int  num_gfr,
    int  num_burnin,
    int  num_mcmc,
    int  num_chains,
    int  keep_every,
    bool keep_gfr,
    bool keep_burnin,
    // ── Tree prior ────────────────────────────────────────────────────────────
    double alpha_mean,
    double beta_mean,
    int    min_samples_leaf_mean,
    int    max_depth_mean,
    // ── Variance / global prior ───────────────────────────────────────────────
    bool   sample_sigma2_global,
    bool   sample_sigma2_leaf,
    bool   standardize,
    int    cutpoint_grid_size,
    double a_global,
    double b_global,
    double a_leaf,
    double b_leaf,
    double sigma2_init,
    // ── Link / leaf model ─────────────────────────────────────────────────────
    std::string link_function_str,     // "identity", "probit", "cloglog"
    std::string leaf_model_str,        // "constant", "univariate_regression", "multivariate_regression"
    int  cloglog_num_categories,
    // ── Variance forest ───────────────────────────────────────────────────────
    bool   include_variance_forest,
    int    num_trees_variance,
    double alpha_variance,
    double beta_variance,
    int    min_samples_leaf_variance,
    int    max_depth_variance,
    double a_forest,
    double b_forest,
    double variance_forest_leaf_init,
    // ── RFX priors ────────────────────────────────────────────────────────────
    std::string rfx_model_spec_str,   // "none", "intercept_only", "custom"
    int    rfx_num_components,
    double rfx_alpha_init,
    double rfx_xi_init,
    double rfx_sigma_alpha_init,
    double rfx_sigma_xi_init,
    double rfx_variance_prior_shape,
    double rfx_variance_prior_scale,
    // ── Variable weights ──────────────────────────────────────────────────────
    cpp11::doubles variable_weights_mean_r,
    cpp11::doubles variable_weights_variance_r,
    // ── Execution ─────────────────────────────────────────────────────────────
    int random_seed,
    int num_threads
) {
    // ── Sizes ────────────────────────────────────────────────────────────────
    int n_train = X_train_r.nrow();
    int p       = X_train_r.ncol();

    // ── Unpack optional SEXP args ─────────────────────────────────────────────
    const double* X_test_ptr     = nullptr; int n_test = 0, p_test = 0;
    const double* weights_ptr    = nullptr; int n_wt = 0;
    const int*    ft_ptr         = nullptr; int n_ft = 0;
    const double* basis_train_ptr = nullptr; int basis_n = 0, basis_dim = 0;
    const double* basis_test_ptr  = nullptr; int basis_nt = 0, basis_dim_t = 0;
    const int*    rfx_gids_train_ptr = nullptr; int rfx_n = 0;
    const int*    rfx_gids_test_ptr  = nullptr; int rfx_nt = 0;
    const double* rfx_basis_train_ptr = nullptr; int rfx_bn = 0, rfx_bd = 0;
    const double* rfx_basis_test_ptr  = nullptr; int rfx_bnt = 0, rfx_bdt = 0;

    bool has_test    = unpack_matrix(X_test_r,         X_test_ptr,         n_test, p_test);
    bool has_weights = unpack_dvec  (weights_r,         weights_ptr,        n_wt);
    bool has_ft      = unpack_ivec  (feature_types_r,   ft_ptr,             n_ft);
    bool has_basis   = unpack_matrix(basis_train_r,     basis_train_ptr,    basis_n, basis_dim);
    bool has_basis_test = has_test && unpack_matrix(basis_test_r, basis_test_ptr, basis_nt, basis_dim_t);
    bool has_rfx     = unpack_ivec  (rfx_group_ids_train_r, rfx_gids_train_ptr, rfx_n);
    bool has_rfx_test = has_rfx && has_test &&
                        unpack_ivec(rfx_group_ids_test_r, rfx_gids_test_ptr, rfx_nt);
    bool has_rfx_basis = has_rfx &&
                         unpack_matrix(rfx_basis_train_r, rfx_basis_train_ptr, rfx_bn, rfx_bd);
    bool has_rfx_basis_test = has_rfx_test &&
                              unpack_matrix(rfx_basis_test_r, rfx_basis_test_ptr, rfx_bnt, rfx_bdt);

    // ── BARTConfig ────────────────────────────────────────────────────────────
    StochTree::BARTConfig cfg;
    cfg.num_trees   = num_trees;
    cfg.num_gfr     = num_gfr;
    cfg.num_burnin  = num_burnin;
    cfg.num_mcmc    = num_mcmc;
    cfg.num_chains  = num_chains;
    cfg.keep_every  = keep_every;
    cfg.keep_gfr    = keep_gfr;
    cfg.keep_burnin = keep_burnin;

    cfg.alpha           = alpha_mean;
    cfg.beta            = beta_mean;
    cfg.min_samples_leaf = min_samples_leaf_mean;
    cfg.max_depth       = max_depth_mean;

    cfg.sample_sigma2_global = sample_sigma2_global;
    cfg.sample_sigma2_leaf   = sample_sigma2_leaf;
    cfg.standardize          = standardize;
    cfg.cutpoint_grid_size   = cutpoint_grid_size;
    cfg.a_global             = a_global;
    cfg.b_global             = b_global;
    cfg.a_leaf               = a_leaf;
    cfg.b_leaf               = b_leaf;
    cfg.sigma2_init          = sigma2_init;
    cfg.random_seed          = random_seed;
    cfg.num_threads          = num_threads;

    // Link function
    if      (link_function_str == "probit")  cfg.link_function = StochTree::LinkFunction::Probit;
    else if (link_function_str == "cloglog") cfg.link_function = StochTree::LinkFunction::Cloglog;
    else                                     cfg.link_function = StochTree::LinkFunction::Identity;
    cfg.cloglog_num_categories = cloglog_num_categories;

    // Leaf model
    if      (leaf_model_str == "univariate_regression")   cfg.leaf_model = StochTree::LeafModel::UnivariateRegression;
    else if (leaf_model_str == "multivariate_regression") cfg.leaf_model = StochTree::LeafModel::MultivariateRegression;
    else                                                  cfg.leaf_model = StochTree::LeafModel::Constant;

    // Variance forest
    cfg.include_variance_forest   = include_variance_forest;
    cfg.num_trees_variance        = num_trees_variance;
    cfg.alpha_variance            = alpha_variance;
    cfg.beta_variance             = beta_variance;
    cfg.min_samples_leaf_variance = min_samples_leaf_variance;
    cfg.max_depth_variance        = max_depth_variance;
    cfg.a_forest                  = a_forest;
    cfg.b_forest                  = b_forest;
    cfg.variance_forest_leaf_init = variance_forest_leaf_init;

    // RFX
    if      (rfx_model_spec_str == "intercept_only") cfg.rfx_model_spec = StochTree::RFXModelSpec::InterceptOnly;
    else if (rfx_model_spec_str == "custom")         cfg.rfx_model_spec = StochTree::RFXModelSpec::Custom;
    else                                             cfg.rfx_model_spec = StochTree::RFXModelSpec::None;
    cfg.rfx_num_components         = rfx_num_components;
    cfg.rfx_alpha_init             = rfx_alpha_init;
    cfg.rfx_xi_init                = rfx_xi_init;
    cfg.rfx_sigma_alpha_init       = rfx_sigma_alpha_init;
    cfg.rfx_sigma_xi_init          = rfx_sigma_xi_init;
    cfg.rfx_variance_prior_shape   = rfx_variance_prior_shape;
    cfg.rfx_variance_prior_scale   = rfx_variance_prior_scale;
    cfg.profile_phases             = false;  // never emit profiling output via R path

    // Variable weights
    if (variable_weights_mean_r.size() > 0) {
        cfg.variable_weights_mean.assign(variable_weights_mean_r.begin(),
                                         variable_weights_mean_r.end());
    }
    if (variable_weights_variance_r.size() > 0) {
        cfg.variable_weights_variance.assign(variable_weights_variance_r.begin(),
                                             variable_weights_variance_r.end());
    }

    // Feature types go on BARTData, not BARTConfig — handled below.

    // ── BARTData ──────────────────────────────────────────────────────────────
    StochTree::BARTData data;
    data.X_train = REAL(X_train_r);
    data.n_train = n_train;
    data.p       = p;
    data.y_train = REAL(y_train_r);

    if (has_test) {
        data.X_test = X_test_ptr;
        data.n_test = n_test;
    }
    if (has_weights) {
        data.weights = weights_ptr;
    }
    if (has_ft) {
        data.feature_types = ft_ptr;
    }
    if (has_basis) {
        data.basis_train = basis_train_ptr;
        data.basis_dim   = basis_dim;
        if (has_basis_test)
            data.basis_test = basis_test_ptr;
    }

    // R group IDs are 1-indexed; BARTData expects 0-indexed.
    std::vector<int> rfx_gids_0idx_train, rfx_gids_0idx_test;
    if (has_rfx) {
        rfx_gids_0idx_train.resize(rfx_n);
        for (int i = 0; i < rfx_n; ++i)
            rfx_gids_0idx_train[i] = rfx_gids_train_ptr[i] - 1;
        data.rfx_groups = rfx_gids_0idx_train.data();
        if (has_rfx_basis) {
            data.rfx_basis_train = rfx_basis_train_ptr;
        }
        if (has_rfx_test) {
            rfx_gids_0idx_test.resize(rfx_nt);
            for (int i = 0; i < rfx_nt; ++i)
                rfx_gids_0idx_test[i] = rfx_gids_test_ptr[i] - 1;
            data.rfx_groups_test = rfx_gids_0idx_test.data();
            if (has_rfx_basis_test)
                data.rfx_basis_test = rfx_basis_test_ptr;
        }
    }

    // ── Run sampler ───────────────────────────────────────────────────────────
    StochTree::BARTResult result;
    StochTree::BARTSamplerFit(&result, cfg, data);

    // ── Build return list ─────────────────────────────────────────────────────
    cpp11::writable::list out;

    // Forest container (always present when num_mcmc > 0 with a mean forest)
    if (result.forest_container) {
        out.push_back(cpp11::named_arg("forest_container_ptr") =
            cpp11::external_pointer<StochTree::ForestContainer>(
                result.forest_container.release()));
    } else {
        out.push_back(cpp11::named_arg("forest_container_ptr") = R_NilValue);
    }

    // Variance forest container
    if (result.variance_forest_container) {
        out.push_back(cpp11::named_arg("variance_forest_container_ptr") =
            cpp11::external_pointer<StochTree::ForestContainer>(
                result.variance_forest_container.release()));
    } else {
        out.push_back(cpp11::named_arg("variance_forest_container_ptr") = R_NilValue);
    }

    // RFX container + label mapper
    if (result.rfx_container) {
        out.push_back(cpp11::named_arg("rfx_container_ptr") =
            cpp11::external_pointer<StochTree::RandomEffectsContainer>(
                result.rfx_container.release()));

        // Build label mapper from the sorted unique group IDs stored in result.
        // R uses 1-indexed group IDs; result.rfx_group_ids are 0-indexed.
        const auto& gids = result.rfx_group_ids; // 0-indexed unique IDs
        std::map<int32_t, int32_t> label_map;
        for (int i = 0; i < static_cast<int>(gids.size()); ++i)
            label_map[static_cast<int32_t>(gids[i])] = static_cast<int32_t>(i);
        auto lm_ptr = std::make_unique<StochTree::LabelMapper>(label_map);
        out.push_back(cpp11::named_arg("rfx_label_mapper_ptr") =
            cpp11::external_pointer<StochTree::LabelMapper>(lm_ptr.release()));

        // Return 1-indexed unique group IDs to R
        cpp11::writable::integers r_gids(static_cast<int>(gids.size()));
        for (int i = 0; i < static_cast<int>(gids.size()); ++i)
            r_gids[i] = gids[i] + 1; // convert back to 1-indexed
        out.push_back(cpp11::named_arg("rfx_group_ids") = r_gids);
    } else {
        out.push_back(cpp11::named_arg("rfx_container_ptr")   = R_NilValue);
        out.push_back(cpp11::named_arg("rfx_label_mapper_ptr") = R_NilValue);
        out.push_back(cpp11::named_arg("rfx_group_ids")        = R_NilValue);
    }

    // Helper: copy std::vector<double> to cpp11::doubles
    auto to_r_doubles = [](const std::vector<double>& v) {
        cpp11::writable::doubles out(static_cast<int>(v.size()));
        std::copy(v.begin(), v.end(), out.begin());
        return out;
    };

    out.push_back(cpp11::named_arg("y_hat_train")            = to_r_doubles(result.y_hat_train));
    out.push_back(cpp11::named_arg("y_hat_test")             = to_r_doubles(result.y_hat_test));
    out.push_back(cpp11::named_arg("sigma2_x_hat_train")     = to_r_doubles(result.sigma2_x_hat_train));
    out.push_back(cpp11::named_arg("sigma2_x_hat_test")      = to_r_doubles(result.sigma2_x_hat_test));
    out.push_back(cpp11::named_arg("sigma2_global_samples")  = to_r_doubles(result.sigma2_global_samples));
    out.push_back(cpp11::named_arg("leaf_scale_samples")     = to_r_doubles(result.leaf_scale_samples));
    out.push_back(cpp11::named_arg("cloglog_cutpoint_samples") = to_r_doubles(result.cloglog_cutpoint_samples));

    // Scalar metadata
    out.push_back(cpp11::named_arg("y_bar")             = result.y_bar);
    out.push_back(cpp11::named_arg("y_std")             = result.y_std);
    out.push_back(cpp11::named_arg("num_total_samples") = result.num_total_samples);

    return out;
}
