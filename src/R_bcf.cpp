#include <cpp11.hpp>
#include <stochtree/bcf.h>
#include <stochtree/bcf_sampler.h>
#include <stochtree/container.h>
#include <stochtree/data.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/tree_sampler.h>
#include "stochtree_types.h"
#include <memory>
#include <string>

StochTree::BCFConfig convert_list_to_bcf_config(cpp11::list config) {
  StochTree::BCFConfig output;

  // Global model parameters
  output.standardize_outcome = get_config_scalar_default<bool>(config, "standardize_outcome", true);
  output.num_threads = get_config_scalar_default<int>(config, "num_threads", 1);
  output.cutpoint_grid_size = get_config_scalar_default<int>(config, "cutpoint_grid_size", 100);
  output.link_function = static_cast<StochTree::LinkFunction>(get_config_scalar_default<int>(config, "link_function", 0));
  output.outcome_type = static_cast<StochTree::OutcomeType>(get_config_scalar_default<int>(config, "outcome_type", 0));
  output.random_seed = get_config_scalar_default<int>(config, "random_seed", 1);
  output.keep_gfr = get_config_scalar_default<bool>(config, "keep_gfr", true);
  output.keep_burnin = get_config_scalar_default<bool>(config, "keep_burnin", false);
  output.adaptive_coding = get_config_scalar_default<bool>(config, "adaptive_coding", false);

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

  // Variance forest parameters
  output.num_trees_variance = get_config_scalar_default<int>(config, "num_trees_variance", 0);
  output.leaf_prior_calibration_param = get_config_scalar_default<double>(config, "leaf_prior_calibration_param", 1.5);
  output.shape_variance_forest = get_config_scalar_default<double>(config, "shape_variance_forest", -1.0);
  output.scale_variance_forest = get_config_scalar_default<double>(config, "scale_variance_forest", -1.0);
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

cpp11::writable::list convert_bcf_results_to_list(StochTree::BCFSamples& bcf_samples) {
  cpp11::writable::list output;

  // Pointers to forests
  SEXP mu_forests_sexp = (bcf_samples.mu_forests.get() != nullptr)
                             ? static_cast<SEXP>(cpp11::external_pointer<StochTree::ForestContainer>(bcf_samples.mu_forests.release()))
                             : R_NilValue;
  output.push_back(cpp11::named_arg("mu_forests") = mu_forests_sexp);

  SEXP tau_forests_sexp = (bcf_samples.tau_forests.get() != nullptr)
                              ? static_cast<SEXP>(cpp11::external_pointer<StochTree::ForestContainer>(bcf_samples.tau_forests.release()))
                              : R_NilValue;
  output.push_back(cpp11::named_arg("tau_forests") = tau_forests_sexp);

  SEXP variance_forests_sexp = (bcf_samples.variance_forests.get() != nullptr)
                                   ? static_cast<SEXP>(cpp11::external_pointer<StochTree::ForestContainer>(bcf_samples.variance_forests.release()))
                                   : R_NilValue;
  output.push_back(cpp11::named_arg("variance_forests") = variance_forests_sexp);

  // Pointers to RFX model terms
  SEXP rfx_container_sexp = (bcf_samples.rfx_container.get() != nullptr)
                                ? static_cast<SEXP>(cpp11::external_pointer<StochTree::RandomEffectsContainer>(bcf_samples.rfx_container.release()))
                                : R_NilValue;
  output.push_back(cpp11::named_arg("rfx_container") = rfx_container_sexp);
  SEXP rfx_label_mapper_sexp = (bcf_samples.rfx_label_mapper.get() != nullptr)
                                   ? static_cast<SEXP>(cpp11::external_pointer<StochTree::LabelMapper>(bcf_samples.rfx_label_mapper.release()))
                                   : R_NilValue;
  output.push_back(cpp11::named_arg("rfx_label_mapper") = rfx_label_mapper_sexp);

  // Predictions
  SEXP mu_forest_predictions_train_sexp = !bcf_samples.mu_forest_predictions_train.empty()
                                              ? static_cast<SEXP>(cpp11::writable::doubles(bcf_samples.mu_forest_predictions_train.begin(), bcf_samples.mu_forest_predictions_train.end()))
                                              : R_NilValue;
  output.push_back(cpp11::named_arg("mu_forest_predictions_train") = mu_forest_predictions_train_sexp);

  SEXP tau_forest_predictions_train_sexp = !bcf_samples.tau_forest_predictions_train.empty()
                                               ? static_cast<SEXP>(cpp11::writable::doubles(bcf_samples.tau_forest_predictions_train.begin(), bcf_samples.tau_forest_predictions_train.end()))
                                               : R_NilValue;
  output.push_back(cpp11::named_arg("tau_forest_predictions_train") = tau_forest_predictions_train_sexp);

  SEXP var_preds_train_sexp = !bcf_samples.variance_forest_predictions_train.empty()
                                  ? static_cast<SEXP>(cpp11::writable::doubles(bcf_samples.variance_forest_predictions_train.begin(), bcf_samples.variance_forest_predictions_train.end()))
                                  : R_NilValue;
  output.push_back(cpp11::named_arg("variance_forest_predictions_train") = var_preds_train_sexp);

  SEXP y_hat_train_sexp = !bcf_samples.y_hat_train.empty()
                                  ? static_cast<SEXP>(cpp11::writable::doubles(bcf_samples.y_hat_train.begin(), bcf_samples.y_hat_train.end()))
                                  : R_NilValue;
  output.push_back(cpp11::named_arg("y_hat_train") = y_hat_train_sexp);

  SEXP mu_forest_predictions_test_sexp = !bcf_samples.mu_forest_predictions_test.empty()
                                             ? static_cast<SEXP>(cpp11::writable::doubles(bcf_samples.mu_forest_predictions_test.begin(), bcf_samples.mu_forest_predictions_test.end()))
                                             : R_NilValue;
  output.push_back(cpp11::named_arg("mu_forest_predictions_test") = mu_forest_predictions_test_sexp);

  SEXP tau_forest_predictions_test_sexp = !bcf_samples.tau_forest_predictions_test.empty()
                                              ? static_cast<SEXP>(cpp11::writable::doubles(bcf_samples.tau_forest_predictions_test.begin(), bcf_samples.tau_forest_predictions_test.end()))
                                              : R_NilValue;
  output.push_back(cpp11::named_arg("tau_forest_predictions_test") = tau_forest_predictions_test_sexp);

  SEXP var_preds_test_sexp = !bcf_samples.variance_forest_predictions_test.empty()
                                 ? static_cast<SEXP>(cpp11::writable::doubles(bcf_samples.variance_forest_predictions_test.begin(), bcf_samples.variance_forest_predictions_test.end()))
                                 : R_NilValue;
  output.push_back(cpp11::named_arg("variance_forest_predictions_test") = var_preds_test_sexp);

  SEXP y_hat_test_sexp = !bcf_samples.y_hat_test.empty()
                                  ? static_cast<SEXP>(cpp11::writable::doubles(bcf_samples.y_hat_test.begin(), bcf_samples.y_hat_test.end()))
                                  : R_NilValue;
  output.push_back(cpp11::named_arg("y_hat_test") = y_hat_test_sexp);

  // RFX predictions
  SEXP rfx_preds_train_sexp = !bcf_samples.rfx_predictions_train.empty()
                                  ? static_cast<SEXP>(cpp11::writable::doubles(bcf_samples.rfx_predictions_train.begin(), bcf_samples.rfx_predictions_train.end()))
                                  : R_NilValue;
  output.push_back(cpp11::named_arg("rfx_predictions_train") = rfx_preds_train_sexp);

  SEXP rfx_preds_test_sexp = !bcf_samples.rfx_predictions_test.empty()
                                 ? static_cast<SEXP>(cpp11::writable::doubles(bcf_samples.rfx_predictions_test.begin(), bcf_samples.rfx_predictions_test.end()))
                                 : R_NilValue;
  output.push_back(cpp11::named_arg("rfx_predictions_test") = rfx_preds_test_sexp);

  // Parameter samples
  SEXP global_var_sexp = !bcf_samples.global_error_variance_samples.empty()
                             ? static_cast<SEXP>(cpp11::writable::doubles(bcf_samples.global_error_variance_samples.begin(), bcf_samples.global_error_variance_samples.end()))
                             : R_NilValue;
  output.push_back(cpp11::named_arg("global_error_variance_samples") = global_var_sexp);

  SEXP leaf_scale_mu_sexp = !bcf_samples.leaf_scale_mu_samples.empty()
                                ? static_cast<SEXP>(cpp11::writable::doubles(bcf_samples.leaf_scale_mu_samples.begin(), bcf_samples.leaf_scale_mu_samples.end()))
                                : R_NilValue;
  output.push_back(cpp11::named_arg("leaf_scale_mu_samples") = leaf_scale_mu_sexp);

  SEXP leaf_scale_tau_sexp = !bcf_samples.leaf_scale_tau_samples.empty()
                                 ? static_cast<SEXP>(cpp11::writable::doubles(bcf_samples.leaf_scale_tau_samples.begin(), bcf_samples.leaf_scale_tau_samples.end()))
                                 : R_NilValue;
  output.push_back(cpp11::named_arg("leaf_scale_tau_samples") = leaf_scale_tau_sexp);

  SEXP adaptive_coding_samples_sexp = !bcf_samples.adaptive_coding_samples.empty()
                                          ? static_cast<SEXP>(cpp11::writable::doubles(bcf_samples.adaptive_coding_samples.begin(), bcf_samples.adaptive_coding_samples.end()))
                                          : R_NilValue;
  output.push_back(cpp11::named_arg("adaptive_coding_samples") = adaptive_coding_samples_sexp);

  // Metadata about the model that was sampled
  double y_bar_sexp = bcf_samples.y_bar;
  output.push_back(cpp11::named_arg("y_bar") = y_bar_sexp);
  double y_std_sexp = bcf_samples.y_std;
  output.push_back(cpp11::named_arg("y_std") = y_std_sexp);
  int num_samples_sexp = bcf_samples.num_samples;
  output.push_back(cpp11::named_arg("num_samples") = num_samples_sexp);
  int num_train_sexp = bcf_samples.num_train;
  output.push_back(cpp11::named_arg("num_train") = num_train_sexp);
  int num_test_sexp = bcf_samples.num_test;
  output.push_back(cpp11::named_arg("num_test") = num_test_sexp);
  return output;
}

void add_config_to_bcf_result_list(cpp11::writable::list& result, StochTree::BCFConfig& config) {
  // Unpack more metadata about the model that was sampled
  result.push_back(cpp11::named_arg("sigma2_global_init") = config.sigma2_global_init);
  result.push_back(cpp11::named_arg("sigma2_mu_init") = config.sigma2_mu_init);
  result.push_back(cpp11::named_arg("sigma2_tau_init") = config.sigma2_tau_init);
  result.push_back(cpp11::named_arg("b_sigma2_mu") = config.b_sigma2_mu);
  result.push_back(cpp11::named_arg("b_sigma2_tau") = config.b_sigma2_tau);
  result.push_back(cpp11::named_arg("shape_variance_forest") = config.shape_variance_forest);
  result.push_back(cpp11::named_arg("scale_variance_forest") = config.scale_variance_forest);
  return;
}

[[cpp11::register]]
cpp11::writable::list bcf_sample_cpp(
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
    cpp11::list config_input) {
  // Create outcome object
  StochTree::BCFSamples results_raw = StochTree::BCFSamples();

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

  // Initialize a BCF sampler
  StochTree::BCFSampler bcf_sampler(results_raw, config, data);

  // Run the sampler
  bcf_sampler.run_gfr(results_raw, num_gfr, config.keep_gfr, num_chains);
  if (num_chains > 1) {
    bcf_sampler.run_mcmc_chains(results_raw, num_chains, num_burnin, keep_every, num_mcmc);
  } else {
    bcf_sampler.run_mcmc(results_raw, num_burnin, keep_every, num_mcmc);
  }
  bcf_sampler.postprocess_samples(results_raw);

  // Unprotect protected R objects
  UNPROTECT(protect_count);

  // Unpack outputs
  cpp11::writable::list output_list = convert_bcf_results_to_list(results_raw);
  add_config_to_bcf_result_list(output_list, config);
  return output_list;
}
