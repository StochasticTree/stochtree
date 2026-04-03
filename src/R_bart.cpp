/*!
 * cpp11 R binding for the C++ BART dispatch API (RFC 0004).
 *
 * Stage 1 scope: continuous outcome, identity link, constant leaf, no variance
 * forest, no RFX.  The fast path in R/bart.R calls bart_fit_cpp() and unpacks
 * the result with the accessor functions below.
 */

#include "R_bart.h"
#include <cpp11.hpp>
#include <stochtree/container.h>

#include <algorithm>
#include <memory>
#include <vector>

// ── Config helpers ────────────────────────────────────────────────────────────

// Reads a scalar double from a list element (SEXP), returning `sentinel` if NULL.
static double nullable_dbl(SEXP x, double sentinel = -1.0) {
    if (x == R_NilValue) return sentinel;
    return cpp11::as_cpp<double>(x);
}

static StochTree::BARTConfig bart_config_from_r(
    cpp11::list sampler_cfg,
    cpp11::list mean_forest_cfg
) {
    StochTree::BARTConfig cfg;

    // ── Sampler / general params ──────────────────────────────────────────
    cfg.num_gfr              = cpp11::as_cpp<int>(sampler_cfg["num_gfr"]);
    cfg.num_burnin           = cpp11::as_cpp<int>(sampler_cfg["num_burnin"]);
    cfg.num_mcmc             = cpp11::as_cpp<int>(sampler_cfg["num_mcmc"]);
    cfg.num_chains           = cpp11::as_cpp<int>(sampler_cfg["num_chains"]);
    cfg.keep_every           = cpp11::as_cpp<int>(sampler_cfg["keep_every"]);
    cfg.keep_gfr             = cpp11::as_cpp<bool>(sampler_cfg["keep_gfr"]);
    cfg.keep_burnin          = cpp11::as_cpp<bool>(sampler_cfg["keep_burnin"]);
    cfg.standardize          = cpp11::as_cpp<bool>(sampler_cfg["standardize"]);
    cfg.random_seed          = cpp11::as_cpp<int>(sampler_cfg["random_seed"]);
    cfg.num_threads          = cpp11::as_cpp<int>(sampler_cfg["num_threads"]);
    cfg.a_global             = cpp11::as_cpp<double>(sampler_cfg["sigma2_global_shape"]);
    cfg.b_global             = cpp11::as_cpp<double>(sampler_cfg["sigma2_global_scale"]);
    cfg.sigma2_init          = nullable_dbl(sampler_cfg["sigma2_global_init"]);
    cfg.sample_sigma2_global = cpp11::as_cpp<bool>(sampler_cfg["sample_sigma2_global"]);

    // ── Link function ─────────────────────────────────────────────────────
    {
        std::string lf = cpp11::as_cpp<std::string>(sampler_cfg["link_function"]);
        if      (lf == "probit")   cfg.link_function = StochTree::LinkFunction::Probit;
        else if (lf == "cloglog")  cfg.link_function = StochTree::LinkFunction::Cloglog;
        else                       cfg.link_function = StochTree::LinkFunction::Identity;
    }

    // ── Mean forest params ────────────────────────────────────────────────
    cfg.num_trees            = cpp11::as_cpp<int>(mean_forest_cfg["num_trees"]);
    cfg.alpha                = cpp11::as_cpp<double>(mean_forest_cfg["alpha"]);
    cfg.beta                 = cpp11::as_cpp<double>(mean_forest_cfg["beta"]);
    cfg.min_samples_leaf     = cpp11::as_cpp<int>(mean_forest_cfg["min_samples_leaf"]);
    cfg.max_depth            = cpp11::as_cpp<int>(mean_forest_cfg["max_depth"]);
    cfg.sample_sigma2_leaf   = cpp11::as_cpp<bool>(mean_forest_cfg["sample_sigma2_leaf"]);
    cfg.a_leaf               = cpp11::as_cpp<double>(mean_forest_cfg["sigma2_leaf_shape"]);
    cfg.b_leaf               = nullable_dbl(mean_forest_cfg["sigma2_leaf_scale"]);
    cfg.leaf_scale           = nullable_dbl(mean_forest_cfg["sigma2_leaf_init"]);

    // Optional variable weights (empty = uniform 1/p)
    SEXP vw = mean_forest_cfg["variable_weights"];
    if (vw != R_NilValue) {
        cpp11::doubles vw_r(vw);
        cfg.variable_weights_mean.assign(vw_r.begin(), vw_r.end());
    }

    return cfg;
}

// Helper: copy a std::vector<double> into a new R numeric vector.
static cpp11::doubles vec_to_r(const std::vector<double>& v) {
    cpp11::writable::doubles out(static_cast<R_xlen_t>(v.size()));
    std::copy(v.begin(), v.end(), out.begin());
    return out;
}

// ── bart_fit_cpp ──────────────────────────────────────────────────────────────

[[cpp11::register]]
cpp11::external_pointer<BARTResultR> bart_fit_cpp(
    cpp11::list             sampler_cfg,
    cpp11::list             mean_forest_cfg,
    cpp11::doubles_matrix<> X_train_r,
    cpp11::doubles          y_train_r,
    SEXP                    X_test_r,
    SEXP                    feature_types_r,
    SEXP                    weights_r
) {
    StochTree::BARTConfig config = bart_config_from_r(sampler_cfg, mean_forest_cfg);

    StochTree::BARTData data;
    data.n_train = X_train_r.nrow();
    data.p       = X_train_r.ncol();

    // Input objects are on the R protection stack for the call duration;
    // PROTECT/UNPROTECT here follows the defensive style of R_data.cpp.
    int nprotect = 0;
    data.X_train = REAL(PROTECT(static_cast<SEXP>(X_train_r))); ++nprotect;
    data.y_train = REAL(PROTECT(static_cast<SEXP>(y_train_r))); ++nprotect;

    // Optional test covariates
    if (X_test_r != R_NilValue) {
        data.X_test = REAL(PROTECT(X_test_r)); ++nprotect;
        data.n_test = Rf_nrows(X_test_r);
    }

    // Optional feature types (int vector, length p)
    std::vector<int> ft_storage;
    if (feature_types_r != R_NilValue) {
        cpp11::integers ft(feature_types_r);
        ft_storage.assign(ft.begin(), ft.end());
        data.feature_types = ft_storage.data();
    }

    // Optional observation weights
    if (weights_r != R_NilValue) {
        data.weights = REAL(PROTECT(weights_r)); ++nprotect;
    }

    // Run BART
    auto res = std::make_unique<StochTree::BARTResult>();
    StochTree::BARTFit(res.get(), config, data);

    UNPROTECT(nprotect);

    auto wrapper = std::make_unique<BARTResultR>(std::move(res));
    return cpp11::external_pointer<BARTResultR>(wrapper.release());
}

// ── Accessors ─────────────────────────────────────────────────────────────────

[[cpp11::register]]
int bart_result_num_samples_cpp(cpp11::external_pointer<BARTResultR> ptr) {
    return ptr->result->num_total_samples;
}

[[cpp11::register]]
double bart_result_y_bar_cpp(cpp11::external_pointer<BARTResultR> ptr) {
    return ptr->result->y_bar;
}

[[cpp11::register]]
double bart_result_y_std_cpp(cpp11::external_pointer<BARTResultR> ptr) {
    return ptr->result->y_std;
}

[[cpp11::register]]
cpp11::doubles bart_result_y_hat_train_cpp(cpp11::external_pointer<BARTResultR> ptr) {
    return vec_to_r(ptr->result->y_hat_train);
}

[[cpp11::register]]
cpp11::doubles bart_result_y_hat_test_cpp(cpp11::external_pointer<BARTResultR> ptr) {
    return vec_to_r(ptr->result->y_hat_test);
}

[[cpp11::register]]
cpp11::doubles bart_result_sigma2_global_samples_cpp(cpp11::external_pointer<BARTResultR> ptr) {
    return vec_to_r(ptr->result->sigma2_global_samples);
}

[[cpp11::register]]
cpp11::doubles bart_result_leaf_scale_samples_cpp(cpp11::external_pointer<BARTResultR> ptr) {
    return vec_to_r(ptr->result->leaf_scale_samples);
}

// Transfers ownership of the ForestContainer out of BARTResult.
// After this call, ptr->result->forest_container is null.
// The returned external_pointer owns the ForestContainer; wrap it in a
// ForestSamples R6 object on the R side to tie its lifetime to that object.
[[cpp11::register]]
cpp11::external_pointer<StochTree::ForestContainer> bart_result_steal_forest_samples_cpp(
    cpp11::external_pointer<BARTResultR> ptr
) {
    StochTree::ForestContainer* fc = ptr->result->forest_container.release();
    return cpp11::external_pointer<StochTree::ForestContainer>(fc);
}
