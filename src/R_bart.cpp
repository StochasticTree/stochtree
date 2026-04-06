/*!
 * cpp11 R binding for the C++ BART dispatch API (RFC 0004).
 *
 * Supports all BARTFit models: identity, probit, cloglog/ordinal,
 * variance forest, leaf regression, and random effects (intercept-only
 * and custom).
 */

#include "R_bart.h"
#include <cpp11.hpp>
#include <stochtree/container.h>
#include <stochtree/random_effects.h>

#include <algorithm>
#include <cstdint>
#include <map>
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
    cpp11::list mean_forest_cfg,
    cpp11::list variance_forest_cfg,
    cpp11::list rfx_cfg
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

    // ── Cloglog / ordinal params ──────────────────────────────────────────
    cfg.cloglog_num_categories = cpp11::as_cpp<int>(sampler_cfg["cloglog_num_categories"]);
    cfg.cloglog_forest_shape   = cpp11::as_cpp<double>(sampler_cfg["cloglog_forest_shape"]);
    cfg.cloglog_forest_rate    = cpp11::as_cpp<double>(sampler_cfg["cloglog_forest_rate"]);
    cfg.cloglog_cutpoint_0     = cpp11::as_cpp<double>(sampler_cfg["cloglog_cutpoint_0"]);

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

    // Leaf model type (0 = constant, 1 = univariate regression, 2 = multivariate regression)
    cfg.leaf_model = static_cast<StochTree::LeafModel>(
        cpp11::as_cpp<int>(mean_forest_cfg["leaf_model"]));

    // Optional variable weights (empty = uniform 1/p)
    SEXP vw = mean_forest_cfg["variable_weights"];
    if (vw != R_NilValue) {
        cpp11::doubles vw_r(vw);
        cfg.variable_weights_mean.assign(vw_r.begin(), vw_r.end());
    }

    // ── Variance forest params ─────────────────────────────────────────────
    cfg.include_variance_forest   = cpp11::as_cpp<bool>(variance_forest_cfg["include_variance_forest"]);
    cfg.num_trees_variance        = cpp11::as_cpp<int>(variance_forest_cfg["num_trees"]);
    cfg.alpha_variance            = cpp11::as_cpp<double>(variance_forest_cfg["alpha"]);
    cfg.beta_variance             = cpp11::as_cpp<double>(variance_forest_cfg["beta"]);
    cfg.min_samples_leaf_variance = cpp11::as_cpp<int>(variance_forest_cfg["min_samples_leaf"]);
    cfg.max_depth_variance        = cpp11::as_cpp<int>(variance_forest_cfg["max_depth"]);
    cfg.a_forest                  = nullable_dbl(variance_forest_cfg["a_forest"]);
    cfg.b_forest                  = nullable_dbl(variance_forest_cfg["b_forest"]);
    cfg.variance_forest_leaf_init = nullable_dbl(variance_forest_cfg["var_forest_leaf_init"]);
    SEXP vwv = variance_forest_cfg["variable_weights"];
    if (vwv != R_NilValue) {
        cpp11::doubles vwv_r(vwv);
        cfg.variable_weights_variance.assign(vwv_r.begin(), vwv_r.end());
    }

    // ── Random effects config ─────────────────────────────────────────────
    {
        std::string spec = cpp11::as_cpp<std::string>(rfx_cfg["rfx_model_spec"]);
        if      (spec == "intercept_only") cfg.rfx_model_spec = StochTree::RFXModelSpec::InterceptOnly;
        else if (spec == "custom")         cfg.rfx_model_spec = StochTree::RFXModelSpec::Custom;
        else                               cfg.rfx_model_spec = StochTree::RFXModelSpec::None;
    }
    cfg.rfx_num_components       = cpp11::as_cpp<int>(rfx_cfg["rfx_num_components"]);
    cfg.rfx_alpha_init           = cpp11::as_cpp<double>(rfx_cfg["rfx_alpha_init"]);
    cfg.rfx_xi_init              = cpp11::as_cpp<double>(rfx_cfg["rfx_xi_init"]);
    cfg.rfx_sigma_alpha_init     = cpp11::as_cpp<double>(rfx_cfg["rfx_sigma_alpha_init"]);
    cfg.rfx_sigma_xi_init        = cpp11::as_cpp<double>(rfx_cfg["rfx_sigma_xi_init"]);
    cfg.rfx_variance_prior_shape = cpp11::as_cpp<double>(rfx_cfg["rfx_variance_prior_shape"]);
    cfg.rfx_variance_prior_scale = cpp11::as_cpp<double>(rfx_cfg["rfx_variance_prior_scale"]);

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
    cpp11::list             variance_forest_cfg,
    cpp11::list             rfx_cfg,
    cpp11::doubles_matrix<> X_train_r,
    cpp11::doubles          y_train_r,
    SEXP                    X_test_r,
    SEXP                    feature_types_r,
    SEXP                    weights_r,
    SEXP                    basis_train_r,
    SEXP                    basis_test_r,
    SEXP                    rfx_groups_train_r,
    SEXP                    rfx_basis_train_r,
    SEXP                    rfx_groups_test_r,
    SEXP                    rfx_basis_test_r
) {
    StochTree::BARTConfig config = bart_config_from_r(
        sampler_cfg, mean_forest_cfg, variance_forest_cfg, rfx_cfg);

    StochTree::BARTData data;
    data.n_train = X_train_r.nrow();
    data.p       = X_train_r.ncol();

    // Input objects are on the R protection stack for the call duration.
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

    // Optional leaf regression basis (column-major matrix)
    if (basis_train_r != R_NilValue) {
        data.basis_train = REAL(PROTECT(basis_train_r)); ++nprotect;
        data.basis_dim   = Rf_ncols(basis_train_r);
    }
    if (basis_test_r != R_NilValue) {
        data.basis_test = REAL(PROTECT(basis_test_r)); ++nprotect;
    }

    // Optional RFX data (group labels as int, basis as column-major double)
    std::vector<int> rfx_groups_train_storage, rfx_groups_test_storage;
    if (rfx_groups_train_r != R_NilValue) {
        cpp11::integers g(rfx_groups_train_r);
        rfx_groups_train_storage.assign(g.begin(), g.end());
        data.rfx_groups = rfx_groups_train_storage.data();
    }
    if (rfx_basis_train_r != R_NilValue) {
        data.rfx_basis_train = REAL(PROTECT(rfx_basis_train_r)); ++nprotect;
    }
    if (rfx_groups_test_r != R_NilValue) {
        cpp11::integers g(rfx_groups_test_r);
        rfx_groups_test_storage.assign(g.begin(), g.end());
        data.rfx_groups_test = rfx_groups_test_storage.data();
    }
    if (rfx_basis_test_r != R_NilValue) {
        data.rfx_basis_test = REAL(PROTECT(rfx_basis_test_r)); ++nprotect;
    }

    // Run BART
    auto res = std::make_unique<StochTree::BARTResult>();
    StochTree::BARTFit(res.get(), config, data);

    UNPROTECT(nprotect);

    auto wrapper = std::make_unique<BARTResultR>(std::move(res));
    return cpp11::external_pointer<BARTResultR>(wrapper.release());
}

// ── Scalar accessors ──────────────────────────────────────────────────────────

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

// ── Cloglog accessors ─────────────────────────────────────────────────────────

[[cpp11::register]]
cpp11::doubles bart_result_cloglog_cutpoint_samples_cpp(cpp11::external_pointer<BARTResultR> ptr) {
    return vec_to_r(ptr->result->cloglog_cutpoint_samples);
}

// ── Forest accessors ──────────────────────────────────────────────────────────

// Transfers ownership of the ForestContainer out of BARTResult (one-shot).
[[cpp11::register]]
cpp11::external_pointer<StochTree::ForestContainer> bart_result_steal_forest_samples_cpp(
    cpp11::external_pointer<BARTResultR> ptr
) {
    StochTree::ForestContainer* fc = ptr->result->forest_container.release();
    return cpp11::external_pointer<StochTree::ForestContainer>(fc);
}

// ── Variance forest accessors ─────────────────────────────────────────────────

[[cpp11::register]]
bool bart_result_has_variance_forest_cpp(cpp11::external_pointer<BARTResultR> ptr) {
    return ptr->result->variance_forest_container != nullptr;
}

[[cpp11::register]]
cpp11::doubles bart_result_sigma2_x_hat_train_cpp(cpp11::external_pointer<BARTResultR> ptr) {
    return vec_to_r(ptr->result->sigma2_x_hat_train);
}

[[cpp11::register]]
cpp11::doubles bart_result_sigma2_x_hat_test_cpp(cpp11::external_pointer<BARTResultR> ptr) {
    return vec_to_r(ptr->result->sigma2_x_hat_test);
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::ForestContainer> bart_result_steal_variance_forest_samples_cpp(
    cpp11::external_pointer<BARTResultR> ptr
) {
    StochTree::ForestContainer* fc = ptr->result->variance_forest_container.release();
    return cpp11::external_pointer<StochTree::ForestContainer>(fc);
}

// ── Random effects accessors ──────────────────────────────────────────────────

[[cpp11::register]]
bool bart_result_has_rfx_cpp(cpp11::external_pointer<BARTResultR> ptr) {
    return ptr->result->rfx_container != nullptr;
}

[[cpp11::register]]
int bart_result_rfx_num_groups_cpp(cpp11::external_pointer<BARTResultR> ptr) {
    return ptr->result->rfx_num_groups;
}

[[cpp11::register]]
int bart_result_rfx_num_components_cpp(cpp11::external_pointer<BARTResultR> ptr) {
    return ptr->result->rfx_num_components;
}

[[cpp11::register]]
cpp11::integers bart_result_rfx_group_ids_cpp(cpp11::external_pointer<BARTResultR> ptr) {
    const auto& ids = ptr->result->rfx_group_ids;
    cpp11::writable::integers out(static_cast<R_xlen_t>(ids.size()));
    for (R_xlen_t i = 0; i < static_cast<R_xlen_t>(ids.size()); ++i)
        out[i] = static_cast<int>(ids[i]);
    return out;
}

// Transfers ownership of the RandomEffectsContainer out of BARTResult (one-shot).
[[cpp11::register]]
cpp11::external_pointer<StochTree::RandomEffectsContainer> bart_result_steal_rfx_container_cpp(
    cpp11::external_pointer<BARTResultR> ptr
) {
    if (!ptr->result->rfx_container)
        StochTree::Log::Fatal("BARTResult: rfx_container is null or already taken");
    StochTree::RandomEffectsContainer* raw = ptr->result->rfx_container.release();
    return cpp11::external_pointer<StochTree::RandomEffectsContainer>(raw);
}
