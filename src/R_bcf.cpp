#include <cpp11.hpp>
#include <stochtree/bcf.h>
#include <stochtree/bcf_sampler.h>
#include <stochtree/container.h>
#include <stochtree/data.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/tree_sampler.h>
#include "stochtree/prediction.h"
#include "stochtree_types.h"
#include <memory>
#include <string>

StochTree::BCFConfig convert_list_to_bcf_config(cpp11::list config) {
  StochTree::BCFConfig output;

  // Global model parameters
  output.standardize_outcome = get_config_scalar_default<bool>(config, "standardize_outcome", true);
  output.num_threads = get_config_scalar_default<int>(config, "num_threads", 1);
  output.verbose = get_config_scalar_default<bool>(config, "verbose", false);
  output.cutpoint_grid_size = get_config_scalar_default<int>(config, "cutpoint_grid_size", 100);
  output.link_function = static_cast<StochTree::LinkFunction>(get_config_scalar_default<int>(config, "link_function", 0));
  output.outcome_type = static_cast<StochTree::OutcomeType>(get_config_scalar_default<int>(config, "outcome_type", 0));
  output.random_seed = get_config_scalar_default<int>(config, "random_seed", 1);
  output.keep_gfr = get_config_scalar_default<bool>(config, "keep_gfr", true);
  output.keep_burnin = get_config_scalar_default<bool>(config, "keep_burnin", false);
  output.adaptive_coding = get_config_scalar_default<bool>(config, "adaptive_coding", false);
  output.b_0_init = get_config_scalar_default<double>(config, "b_0_init", 0.0);
  output.b_1_init = get_config_scalar_default<double>(config, "b_1_init", 1.0);

  // Global error variance parameters
  output.a_sigma2_global = get_config_scalar_default<double>(config, "a_sigma2_global", 0.0);
  output.b_sigma2_global = get_config_scalar_default<double>(config, "b_sigma2_global", 0.0);
  output.sigma2_global_init = get_config_scalar_default<double>(config, "sigma2_global_init", 1.0);
  output.sample_sigma2_global = get_config_scalar_default<bool>(config, "sample_sigma2_global", true);

  // Prognostic forest parameters
  output.num_trees_mu = get_config_scalar_default<int>(config, "num_trees_mu", 200);
  output.alpha_mu = get_config_scalar_default<double>(config, "alpha_mu", 0.95);
  output.beta_mu = get_config_scalar_default<double>(config, "beta_mu", 2.0);
  output.min_samples_leaf_mu = get_config_scalar_default<int>(config, "min_samples_leaf_mu", 5);
  output.max_depth_mu = get_config_scalar_default<int>(config, "max_depth_mu", -1);
  output.leaf_constant_mu = get_config_scalar_default<bool>(config, "leaf_constant_mu", true);
  output.leaf_dim_mu = get_config_scalar_default<int>(config, "leaf_dim_mu", 1);
  output.exponentiated_leaf_mu = get_config_scalar_default<bool>(config, "exponentiated_leaf_mu", false);
  output.num_features_subsample_mu = get_config_scalar_default<int>(config, "num_features_subsample_mu", 0);
  output.a_sigma2_mu = get_config_scalar_default<double>(config, "a_sigma2_mu", 3.0);
  output.b_sigma2_mu = get_config_scalar_default<double>(config, "b_sigma2_mu", -1.0);
  output.sigma2_mu_init = get_config_scalar_default<double>(config, "sigma2_mu_init", -1.0);
  output.sample_sigma2_leaf_mu = get_config_scalar_default<bool>(config, "sample_sigma2_leaf_mu", false);

  // Treatment effect forest parameters
  output.num_trees_tau = get_config_scalar_default<int>(config, "num_trees_tau", 50);
  output.alpha_tau = get_config_scalar_default<double>(config, "alpha_tau", 0.95);
  output.beta_tau = get_config_scalar_default<double>(config, "beta_tau", 2.0);
  output.min_samples_leaf_tau = get_config_scalar_default<int>(config, "min_samples_leaf_tau", 5);
  output.max_depth_tau = get_config_scalar_default<int>(config, "max_depth_tau", -1);
  output.leaf_constant_tau = get_config_scalar_default<bool>(config, "leaf_constant_tau", true);
  output.leaf_dim_tau = get_config_scalar_default<int>(config, "leaf_dim_tau", 1);
  output.exponentiated_leaf_tau = get_config_scalar_default<bool>(config, "exponentiated_leaf_tau", false);
  output.num_features_subsample_tau = get_config_scalar_default<int>(config, "num_features_subsample_tau", 0);
  output.a_sigma2_tau = get_config_scalar_default<double>(config, "a_sigma2_tau", 3.0);
  output.b_sigma2_tau = get_config_scalar_default<double>(config, "b_sigma2_tau", -1.0);
  output.sigma2_tau_init = get_config_scalar_default<double>(config, "sigma2_tau_init", -1.0);
  output.sample_sigma2_leaf_tau = get_config_scalar_default<bool>(config, "sample_sigma2_leaf_tau", false);
  output.tau_leaf_model_type = static_cast<StochTree::MeanLeafModelType>(get_config_scalar_default<int>(config, "tau_leaf_model_type", 1));
  output.sample_tau_0 = get_config_scalar_default<bool>(config, "sample_tau_0", true);
  output.tau_0_prior_var_scalar = get_config_scalar_default<double>(config, "tau_0_prior_var_scalar", -1.0);

  // Variance forest parameters
  output.num_trees_variance = get_config_scalar_default<int>(config, "num_trees_variance", 0);
  output.leaf_prior_calibration_param = get_config_scalar_default<double>(config, "leaf_prior_calibration_param", 1.5);
  output.shape_variance_forest = get_config_scalar_default<double>(config, "shape_variance_forest", -1.0);
  output.scale_variance_forest = get_config_scalar_default<double>(config, "scale_variance_forest", -1.0);
  output.variance_forest_leaf_init = get_config_scalar_default<double>(config, "variance_forest_leaf_init", -1.0);
  output.alpha_variance = get_config_scalar_default<double>(config, "alpha_variance", 0.5);
  output.beta_variance = get_config_scalar_default<double>(config, "beta_variance", 2.0);
  output.min_samples_leaf_variance = get_config_scalar_default<int>(config, "min_samples_leaf_variance", 5);
  output.max_depth_variance = get_config_scalar_default<int>(config, "max_depth_variance", -1);
  output.leaf_constant_variance = get_config_scalar_default<bool>(config, "leaf_constant_variance", true);
  output.leaf_dim_variance = get_config_scalar_default<int>(config, "leaf_dim_variance", 1);
  output.exponentiated_leaf_variance = get_config_scalar_default<bool>(config, "exponentiated_leaf_variance", true);
  output.num_features_subsample_variance = get_config_scalar_default<int>(config, "num_features_subsample_variance", 0);

  // Random effect parameters
  output.has_random_effects = get_config_scalar_default<bool>(config, "has_random_effects", false);
  output.rfx_model_spec = static_cast<StochTree::BCFRFXModelSpec>(get_config_scalar_default<int>(config, "rfx_model_spec", 0));
  output.rfx_variance_prior_shape = get_config_scalar_default<double>(config, "rfx_variance_prior_shape", 1.0);
  output.rfx_variance_prior_scale = get_config_scalar_default<double>(config, "rfx_variance_prior_scale", 1.0);

  // Handle vector conversions separately
  SEXP feature_type_raw = static_cast<SEXP>(config["feature_types"]);
  if (!Rf_isNull(feature_type_raw)) {
    cpp11::integers feature_types_r_vec(feature_type_raw);
    for (auto i : feature_types_r_vec) {
      output.feature_types.push_back(static_cast<StochTree::FeatureType>(i));
    }
  }
  SEXP sweep_update_indices_mu_raw = static_cast<SEXP>(config["sweep_update_indices_mu"]);
  if (!Rf_isNull(sweep_update_indices_mu_raw)) {
    cpp11::integers sweep_update_indices_mu_r_vec(sweep_update_indices_mu_raw);
    output.sweep_update_indices_mu.assign(sweep_update_indices_mu_r_vec.begin(), sweep_update_indices_mu_r_vec.end());
  }
  SEXP sweep_update_indices_tau_raw = static_cast<SEXP>(config["sweep_update_indices_tau"]);
  if (!Rf_isNull(sweep_update_indices_tau_raw)) {
    cpp11::integers sweep_update_indices_tau_r_vec(sweep_update_indices_tau_raw);
    output.sweep_update_indices_tau.assign(sweep_update_indices_tau_r_vec.begin(), sweep_update_indices_tau_r_vec.end());
  }
  SEXP sweep_update_indices_variance_raw = static_cast<SEXP>(config["sweep_update_indices_variance"]);
  if (!Rf_isNull(sweep_update_indices_variance_raw)) {
    cpp11::integers sweep_update_indices_variance_r_vec(sweep_update_indices_variance_raw);
    output.sweep_update_indices_variance.assign(sweep_update_indices_variance_r_vec.begin(), sweep_update_indices_variance_r_vec.end());
  }
  SEXP var_weights_mu_raw = static_cast<SEXP>(config["var_weights_mu"]);
  if (!Rf_isNull(var_weights_mu_raw)) {
    cpp11::doubles var_weights_mu_r_vec(var_weights_mu_raw);
    output.var_weights_mu.assign(var_weights_mu_r_vec.begin(), var_weights_mu_r_vec.end());
  }
  SEXP var_weights_tau_raw = static_cast<SEXP>(config["var_weights_tau"]);
  if (!Rf_isNull(var_weights_tau_raw)) {
    cpp11::doubles var_weights_tau_r_vec(var_weights_tau_raw);
    output.var_weights_tau.assign(var_weights_tau_r_vec.begin(), var_weights_tau_r_vec.end());
  }
  SEXP sigma2_leaf_tau_matrix_raw = static_cast<SEXP>(config["sigma2_leaf_tau_matrix"]);
  if (!Rf_isNull(sigma2_leaf_tau_matrix_raw)) {
    cpp11::doubles sigma2_leaf_tau_matrix_r_vec(sigma2_leaf_tau_matrix_raw);
    output.sigma2_leaf_tau_matrix.assign(sigma2_leaf_tau_matrix_r_vec.begin(), sigma2_leaf_tau_matrix_r_vec.end());
  }
  SEXP tau_0_prior_var_raw = static_cast<SEXP>(config["tau_0_prior_var_multivariate"]);
  if (!Rf_isNull(tau_0_prior_var_raw)) {
    cpp11::doubles tau_0_prior_var_r_vec(tau_0_prior_var_raw);
    output.tau_0_prior_var_multivariate.assign(tau_0_prior_var_r_vec.begin(), tau_0_prior_var_r_vec.end());
  }
  SEXP var_weights_variance_raw = static_cast<SEXP>(config["var_weights_variance"]);
  if (!Rf_isNull(var_weights_variance_raw)) {
    cpp11::doubles var_weights_variance_r_vec(var_weights_variance_raw);
    output.var_weights_variance.assign(var_weights_variance_r_vec.begin(), var_weights_variance_r_vec.end());
  }
  SEXP rfx_working_parameter_mean_prior_raw = static_cast<SEXP>(config["rfx_working_parameter_mean_prior"]);
  if (!Rf_isNull(rfx_working_parameter_mean_prior_raw)) {
    cpp11::doubles rfx_working_parameter_mean_prior_r_vec(rfx_working_parameter_mean_prior_raw);
    output.rfx_working_parameter_mean_prior.assign(rfx_working_parameter_mean_prior_r_vec.begin(), rfx_working_parameter_mean_prior_r_vec.end());
  }
  SEXP rfx_group_parameter_mean_prior_raw = static_cast<SEXP>(config["rfx_group_parameter_mean_prior"]);
  if (!Rf_isNull(rfx_group_parameter_mean_prior_raw)) {
    cpp11::doubles rfx_group_parameter_mean_prior_r_vec(rfx_group_parameter_mean_prior_raw);
    output.rfx_group_parameter_mean_prior.assign(rfx_group_parameter_mean_prior_r_vec.begin(), rfx_group_parameter_mean_prior_r_vec.end());
  }
  SEXP rfx_working_parameter_cov_prior_raw = static_cast<SEXP>(config["rfx_working_parameter_cov_prior"]);
  if (!Rf_isNull(rfx_working_parameter_cov_prior_raw)) {
    cpp11::doubles rfx_working_parameter_cov_prior_r_vec(rfx_working_parameter_cov_prior_raw);
    output.rfx_working_parameter_cov_prior.assign(rfx_working_parameter_cov_prior_r_vec.begin(), rfx_working_parameter_cov_prior_r_vec.end());
  }
  SEXP rfx_group_parameter_cov_prior_raw = static_cast<SEXP>(config["rfx_group_parameter_cov_prior"]);
  if (!Rf_isNull(rfx_group_parameter_cov_prior_raw)) {
    cpp11::doubles rfx_group_parameter_cov_prior_r_vec(rfx_group_parameter_cov_prior_raw);
    output.rfx_group_parameter_cov_prior.assign(rfx_group_parameter_cov_prior_r_vec.begin(), rfx_group_parameter_cov_prior_r_vec.end());
  }
  return output;
}

cpp11::writable::list create_bcf_metadata(StochTree::BCFConfig& config) {
  cpp11::writable::list result;
  result.push_back(cpp11::named_arg("sigma2_global_init") = config.sigma2_global_init);
  result.push_back(cpp11::named_arg("sigma2_mu_init") = config.sigma2_mu_init);
  result.push_back(cpp11::named_arg("sigma2_tau_init") = config.sigma2_tau_init);
  result.push_back(cpp11::named_arg("b_sigma2_mu") = config.b_sigma2_mu);
  result.push_back(cpp11::named_arg("b_sigma2_tau") = config.b_sigma2_tau);
  result.push_back(cpp11::named_arg("shape_variance_forest") = config.shape_variance_forest);
  result.push_back(cpp11::named_arg("scale_variance_forest") = config.scale_variance_forest);
  return result;
}

[[cpp11::register]]
cpp11::writable::list bcf_sample_cpp(
    cpp11::external_pointer<StochTree::BCFSamples> samples,
    cpp11::sexp X_train,
    cpp11::sexp Z_train,
    cpp11::sexp y_train,
    cpp11::sexp X_test,
    cpp11::sexp Z_test,
    int n_train,
    int n_test,
    int p,
    int treatment_dim,
    cpp11::sexp obs_weights_train,
    cpp11::sexp obs_weights_test,
    cpp11::sexp rfx_group_ids_train,
    cpp11::sexp rfx_group_ids_test,
    cpp11::sexp rfx_basis_train,
    cpp11::sexp rfx_basis_test,
    int rfx_num_groups,
    int rfx_basis_dim,
    int num_gfr,
    int num_burnin,
    int keep_every,
    int num_mcmc,
    int num_chains,
    bool adaptive_coding,
    cpp11::sexp warmstart_samples,
    int warmstart_sample_num,
    cpp11::list config_input) {
  // Extract pointers to raw data
  int protect_count = 0;
  double* X_train_ptr = extract_numeric_pointer(X_train, "X_train", protect_count);
  double* Z_train_ptr = extract_numeric_pointer(Z_train, "Z_train", protect_count);
  double* y_train_ptr = extract_numeric_pointer(y_train, "y_train", protect_count);
  double* X_test_ptr = extract_numeric_pointer(X_test, "X_test", protect_count);
  double* Z_test_ptr = extract_numeric_pointer(Z_test, "Z_test", protect_count);
  double* obs_weights_train_ptr = extract_numeric_pointer(obs_weights_train, "obs_weights_train", protect_count);
  double* obs_weights_test_ptr = extract_numeric_pointer(obs_weights_test, "obs_weights_test", protect_count);
  int* rfx_group_ids_train_ptr = extract_integer_pointer(rfx_group_ids_train, "rfx_group_ids_train", protect_count);
  int* rfx_group_ids_test_ptr = extract_integer_pointer(rfx_group_ids_test, "rfx_group_ids_test", protect_count);
  double* rfx_basis_train_ptr = extract_numeric_pointer(rfx_basis_train, "rfx_basis_train", protect_count);
  double* rfx_basis_test_ptr = extract_numeric_pointer(rfx_basis_test, "rfx_basis_test", protect_count);

  // Load the BCFData struct
  // Consider reading directly from the R objects or at least checking for matches with the R object dimensions)
  StochTree::BCFData data;
  data.X_train = X_train_ptr;
  data.treatment_train = Z_train_ptr;
  data.y_train = y_train_ptr;
  data.X_test = X_test_ptr;
  data.treatment_test = Z_test_ptr;
  data.n_train = n_train;
  data.p = p;
  data.n_test = n_test;
  data.treatment_dim = treatment_dim;
  data.obs_weights_train = obs_weights_train_ptr;
  data.obs_weights_test = obs_weights_test_ptr;
  data.rfx_group_ids_train = rfx_group_ids_train_ptr;
  data.rfx_group_ids_test = rfx_group_ids_test_ptr;
  data.rfx_basis_train = rfx_basis_train_ptr;
  data.rfx_basis_test = rfx_basis_test_ptr;
  data.rfx_num_groups = rfx_num_groups;
  data.rfx_basis_dim = rfx_basis_dim;

  // Create the BCFConfig object
  StochTree::BCFConfig config = convert_list_to_bcf_config(config_input);

  // Optional previous-model warm-start source (bcf(previous_model_json=...)). The R wrapper passes the
  // deserialized model's BCFSamples external pointer (or NULL); the borrowed pointer need only outlive
  // this call.
  StochTree::BCFSamples* warmstart_ptr = nullptr;
  if (!Rf_isNull(warmstart_samples)) {
    cpp11::external_pointer<StochTree::BCFSamples> warmstart_ext(warmstart_samples);
    warmstart_ptr = warmstart_ext.get();
  }

  // Initialize a BCF sampler (warm-started from the previous model when warmstart_ptr != nullptr)
  StochTree::BCFSampler bcf_sampler(*samples, config, data, /*continuation=*/false, warmstart_ptr, warmstart_sample_num);

  // Probit warm-start: regenerate the (unpersisted) latent so the seeded state is stationary before the
  // first draw. No-op for Gaussian or a non-warm-start run.
  if (warmstart_ptr != nullptr) {
    bcf_sampler.RegenerateProbitLatent(*samples);
  }

  // Run the sampler
  bcf_sampler.run_gfr(*samples, num_gfr, config.keep_gfr, num_chains);
  if (num_chains > 1) {
    bcf_sampler.run_mcmc_chains(*samples, num_chains, num_burnin, keep_every, num_mcmc);
  } else {
    bcf_sampler.run_mcmc(*samples, num_burnin, keep_every, num_mcmc);
  }
  bcf_sampler.postprocess_samples(*samples);

  // Unprotect protected R objects
  UNPROTECT(protect_count);

  // Unpack outputs
  cpp11::writable::list output_list = create_bcf_metadata(config);
  // Final RNG state, so continueSampling() can resume the stream (single-chain runs).
  output_list.push_back(cpp11::named_arg("rng_state") = bcf_sampler.GetRngState());
  return output_list;
}

[[cpp11::register]]
cpp11::writable::list bcf_continue_sample_cpp(
    cpp11::external_pointer<StochTree::BCFSamples> samples,
    cpp11::sexp X_train,
    cpp11::sexp Z_train,
    cpp11::sexp y_train,
    cpp11::sexp X_test,
    cpp11::sexp Z_test,
    int n_train,
    int n_test,
    int p,
    int treatment_dim,
    cpp11::sexp obs_weights_train,
    cpp11::sexp rfx_group_ids_train,
    cpp11::sexp rfx_basis_train,
    cpp11::sexp rfx_group_ids_test,
    cpp11::sexp rfx_basis_test,
    int rfx_num_groups,
    int rfx_basis_dim,
    int num_burnin,
    int keep_every,
    int num_mcmc,
    std::string rng_state_in,
    bool override_seed,
    cpp11::list config_input) {
  // Extract pointers to raw (re-supplied) data
  int protect_count = 0;
  double* X_train_ptr = extract_numeric_pointer(X_train, "X_train", protect_count);
  double* Z_train_ptr = extract_numeric_pointer(Z_train, "Z_train", protect_count);
  double* y_train_ptr = extract_numeric_pointer(y_train, "y_train", protect_count);
  double* X_test_ptr = extract_numeric_pointer(X_test, "X_test", protect_count);
  double* Z_test_ptr = extract_numeric_pointer(Z_test, "Z_test", protect_count);
  double* obs_weights_train_ptr = extract_numeric_pointer(obs_weights_train, "obs_weights_train", protect_count);
  int* rfx_group_ids_train_ptr = extract_integer_pointer(rfx_group_ids_train, "rfx_group_ids_train", protect_count);
  double* rfx_basis_train_ptr = extract_numeric_pointer(rfx_basis_train, "rfx_basis_train", protect_count);
  int* rfx_group_ids_test_ptr = extract_integer_pointer(rfx_group_ids_test, "rfx_group_ids_test", protect_count);
  double* rfx_basis_test_ptr = extract_numeric_pointer(rfx_basis_test, "rfx_basis_test", protect_count);

  // Load the BCFData struct from re-supplied data. A test set is optional on continuation; when
  // supplied, postprocess_samples recomputes the full test-prediction trace from all retained forests.
  StochTree::BCFData data;
  data.X_train = X_train_ptr;
  data.treatment_train = Z_train_ptr;
  data.y_train = y_train_ptr;
  data.X_test = X_test_ptr;
  data.treatment_test = Z_test_ptr;
  data.n_train = n_train;
  data.p = p;
  data.n_test = n_test;
  data.treatment_dim = treatment_dim;
  data.obs_weights_train = obs_weights_train_ptr;
  data.rfx_group_ids_train = rfx_group_ids_train_ptr;
  data.rfx_basis_train = rfx_basis_train_ptr;
  data.rfx_group_ids_test = rfx_group_ids_test_ptr;
  data.rfx_basis_test = rfx_basis_test_ptr;
  data.rfx_num_groups = rfx_num_groups;
  data.rfx_basis_dim = rfx_basis_dim;

  // Create the BCFConfig object
  StochTree::BCFConfig config = convert_list_to_bcf_config(config_input);

  // Continuation appends new MCMC draws in place onto the model's single-owner samples object. The
  // warm-start reads the last retained forests (mu/tau/variance), rfx, and scalar state (tau_0, b0/b1)
  // directly off `samples`; postprocess_samples(samples, num_history) rescales only the new draws.
  StochTree::BCFSamples& bcf_samples = *samples;
  const int num_history = bcf_samples.num_samples;

  // Initialize a BCF sampler in continuation mode (warm-start from last sample)
  StochTree::BCFSampler bcf_sampler(bcf_samples, config, data, /*continuation=*/true);

  // Resume the RNG stream unless the user supplied a new seed (override_seed).
  if (!override_seed && !rng_state_in.empty()) {
    bcf_sampler.SetRngState(rng_state_in);
  }

  // Probit warm-start: regenerate the (unpersisted) latent outcome now that the RNG is positioned at
  // the resumed/re-seeded stream, so the first continued draw starts from a valid stationary state.
  bcf_sampler.RegenerateProbitLatent(bcf_samples);

  // Append new MCMC draws (continuation does not run GFR), then post-process only the new range.
  bcf_sampler.run_mcmc(bcf_samples, num_burnin, keep_every, num_mcmc);
  bcf_sampler.postprocess_samples(bcf_samples, num_history);

  // Unprotect protected R objects
  UNPROTECT(protect_count);

  // Metadata only; the extended samples live on the passed-in samples object.
  cpp11::writable::list metadata_list = create_bcf_metadata(config);
  metadata_list.push_back(cpp11::named_arg("rng_state") = bcf_sampler.GetRngState());
  return metadata_list;
}

cpp11::writable::list convert_bcf_preds_to_list(StochTree::BCFPredictionResult& bcf_preds) {
  cpp11::writable::list output;

  // Predictions
  SEXP y_hat_sexp = !bcf_preds.y_hat.empty()
                        ? static_cast<SEXP>(cpp11::writable::doubles(bcf_preds.y_hat.begin(), bcf_preds.y_hat.end()))
                        : R_NilValue;
  output.push_back(cpp11::named_arg("y_hat") = y_hat_sexp);

  SEXP mu_x_sexp = !bcf_preds.mu_x.empty()
                       ? static_cast<SEXP>(cpp11::writable::doubles(bcf_preds.mu_x.begin(), bcf_preds.mu_x.end()))
                       : R_NilValue;
  output.push_back(cpp11::named_arg("mu_x") = mu_x_sexp);

  SEXP tau_x_sexp = !bcf_preds.tau_x.empty()
                        ? static_cast<SEXP>(cpp11::writable::doubles(bcf_preds.tau_x.begin(), bcf_preds.tau_x.end()))
                        : R_NilValue;
  output.push_back(cpp11::named_arg("tau_x") = tau_x_sexp);

  SEXP prognostic_function_sexp = !bcf_preds.prognostic_function.empty()
                                      ? static_cast<SEXP>(cpp11::writable::doubles(bcf_preds.prognostic_function.begin(), bcf_preds.prognostic_function.end()))
                                      : R_NilValue;
  output.push_back(cpp11::named_arg("prognostic_function") = prognostic_function_sexp);

  SEXP cate_sexp = !bcf_preds.cate.empty()
                       ? static_cast<SEXP>(cpp11::writable::doubles(bcf_preds.cate.begin(), bcf_preds.cate.end()))
                       : R_NilValue;
  output.push_back(cpp11::named_arg("cate") = cate_sexp);

  SEXP conditional_variance_sexp = !bcf_preds.conditional_variance.empty()
                                       ? static_cast<SEXP>(cpp11::writable::doubles(bcf_preds.conditional_variance.begin(), bcf_preds.conditional_variance.end()))
                                       : R_NilValue;
  output.push_back(cpp11::named_arg("conditional_variance") = conditional_variance_sexp);

  SEXP random_effects_sexp = !bcf_preds.random_effects.empty()
                                 ? static_cast<SEXP>(cpp11::writable::doubles(bcf_preds.random_effects.begin(), bcf_preds.random_effects.end()))
                                 : R_NilValue;
  output.push_back(cpp11::named_arg("random_effects") = random_effects_sexp);

  return output;
}

[[cpp11::register]]
cpp11::writable::list bcf_predict_cpp(
    cpp11::external_pointer<StochTree::BCFSamples> bcf_samples_ptr,
    cpp11::list bcf_model_metadata,
    cpp11::sexp X,
    cpp11::sexp Z,
    int n,
    int p,
    int treatment_dim,
    cpp11::sexp obs_weights,
    cpp11::sexp rfx_group_ids,
    cpp11::sexp rfx_basis,
    int rfx_num_groups,
    int rfx_basis_dim,
    bool posterior,
    int scale,
    bool predict_y_hat,
    bool predict_mu_x,
    bool predict_tau_x,
    bool predict_prognostic_function,
    bool predict_cate,
    bool predict_conditional_variance,
    bool predict_random_effects) {
  // Extract pointers to raw data
  int protect_count = 0;
  double* X_ptr = extract_numeric_pointer(X, "X", protect_count);
  double* Z_ptr = extract_numeric_pointer(Z, "Z", protect_count);
  double* obs_weights_ptr = extract_numeric_pointer(obs_weights, "obs_weights", protect_count);
  int* rfx_group_ids_ptr = extract_integer_pointer(rfx_group_ids, "rfx_group_ids", protect_count);
  double* rfx_basis_ptr = extract_numeric_pointer(rfx_basis, "rfx_basis", protect_count);

  // Load the BCFData struct
  // Consider reading directly from the R objects or at least checking for matches with the R object dimensions)
  StochTree::BCFData data;
  data.X_test = X_ptr;
  data.treatment_test = Z_ptr;
  data.p = p;
  data.n_test = n;
  data.treatment_dim = treatment_dim;
  data.obs_weights_test = obs_weights_ptr;
  data.rfx_group_ids_test = rfx_group_ids_ptr;
  data.rfx_basis_test = rfx_basis_ptr;
  data.rfx_num_groups = rfx_num_groups;
  data.rfx_basis_dim = rfx_basis_dim;

  // Load the BCF model and config from the model list
  StochTree::BCFPredictionMetadata pred_metadata;
  pred_metadata.num_samples = Rf_asInteger(bcf_model_metadata["num_samples"]);
  pred_metadata.num_obs = n;
  pred_metadata.treatment_dim = treatment_dim;
  pred_metadata.y_bar = Rf_asReal(bcf_model_metadata["y_bar"]);
  pred_metadata.y_std = Rf_asReal(bcf_model_metadata["y_std"]);
  pred_metadata.has_variance_forest = (bool)Rf_asLogical(bcf_model_metadata["include_variance_forest"]);
  pred_metadata.has_rfx = (bool)Rf_asLogical(bcf_model_metadata["has_rfx"]);
  {
    SEXP rfx_spec_sexp = bcf_model_metadata["rfx_model_spec"];
    std::string rfx_model_spec_str = Rf_isNull(rfx_spec_sexp) ? "" : std::string(CHAR(STRING_ELT(rfx_spec_sexp, 0)));
    if (rfx_model_spec_str == "intercept_only") {
      pred_metadata.rfx_model_spec = StochTree::BCFRFXModelSpec::InterceptOnly;
    } else if (rfx_model_spec_str == "intercept_plus_treatment") {
      pred_metadata.rfx_model_spec = StochTree::BCFRFXModelSpec::InterceptPlusTreatment;
    } else {
      pred_metadata.rfx_model_spec = StochTree::BCFRFXModelSpec::Custom;
    }
  }
  pred_metadata.adaptive_coding = (bool)Rf_asLogical(bcf_model_metadata["adaptive_coding"]);
  pred_metadata.sample_tau_0 = (bool)Rf_asLogical(bcf_model_metadata["sample_tau_0"]);
  pred_metadata.pred_type = posterior ? StochTree::PredType::kPosterior : StochTree::PredType::kMean;
  if (scale == 0) {
    pred_metadata.pred_scale = StochTree::PredScale::kLinear;
  } else if (scale == 1) {
    pred_metadata.pred_scale = StochTree::PredScale::kProbability;
  } else {
    pred_metadata.pred_scale = StochTree::PredScale::kClass;
  }
  pred_metadata.pred_terms.y_hat = predict_y_hat;
  pred_metadata.pred_terms.mu_x = predict_mu_x;
  pred_metadata.pred_terms.tau_x = predict_tau_x;
  pred_metadata.pred_terms.prognostic_function = predict_prognostic_function;
  pred_metadata.pred_terms.cate = predict_cate;
  pred_metadata.pred_terms.conditional_variance = predict_conditional_variance;
  pred_metadata.pred_terms.random_effects = predict_random_effects;

  // Run the prediction function
  StochTree::BCFPredictionResult pred_results = predict_bcf_model(data, *bcf_samples_ptr, pred_metadata);

  // Unprotect protected R objects
  UNPROTECT(protect_count);

  // Unpack outputs
  cpp11::writable::list output_list = convert_bcf_preds_to_list(pred_results);
  return output_list;
}
