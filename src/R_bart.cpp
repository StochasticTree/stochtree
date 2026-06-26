#include <cpp11.hpp>
#include <stochtree/bart.h>
#include <stochtree/bart_sampler.h>
#include <stochtree/container.h>
#include <stochtree/data.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/prediction.h>
#include <stochtree/tree_sampler.h>
#include "stochtree_types.h"

StochTree::BARTConfig convert_list_to_bart_config(cpp11::list config) {
  StochTree::BARTConfig output;

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

  // Global error variance parameters
  output.a_sigma2_global = get_config_scalar_default<double>(config, "a_sigma2_global", 0.0);
  output.b_sigma2_global = get_config_scalar_default<double>(config, "b_sigma2_global", 0.0);
  output.sigma2_global_init = get_config_scalar_default<double>(config, "sigma2_global_init", 1.0);
  output.sample_sigma2_global = get_config_scalar_default<bool>(config, "sample_sigma2_global", true);

  // Mean forest parameters
  output.num_trees_mean = get_config_scalar_default<int>(config, "num_trees_mean", 200);
  output.alpha_mean = get_config_scalar_default<double>(config, "alpha_mean", 0.95);
  output.beta_mean = get_config_scalar_default<double>(config, "beta_mean", 2.0);
  output.min_samples_leaf_mean = get_config_scalar_default<int>(config, "min_samples_leaf_mean", 5);
  output.max_depth_mean = get_config_scalar_default<int>(config, "max_depth_mean", -1);
  output.leaf_constant_mean = get_config_scalar_default<bool>(config, "leaf_constant_mean", true);
  output.leaf_dim_mean = get_config_scalar_default<int>(config, "leaf_dim_mean", 1);
  output.exponentiated_leaf_mean = get_config_scalar_default<bool>(config, "exponentiated_leaf_mean", false);
  output.num_features_subsample_mean = get_config_scalar_default<int>(config, "num_features_subsample_mean", 0);
  output.a_sigma2_mean = get_config_scalar_default<double>(config, "a_sigma2_mean", 3.0);
  output.b_sigma2_mean = get_config_scalar_default<double>(config, "b_sigma2_mean", -1.0);
  output.sigma2_mean_init = get_config_scalar_default<double>(config, "sigma2_mean_init", -1.0);
  output.sample_sigma2_leaf_mean = get_config_scalar_default<bool>(config, "sample_sigma2_leaf_mean", false);
  output.mean_leaf_model_type = static_cast<StochTree::MeanLeafModelType>(get_config_scalar_default<int>(config, "mean_leaf_model_type", 0));
  output.num_classes_cloglog = get_config_scalar_default<int>(config, "num_classes_cloglog", 0);
  output.cloglog_leaf_prior_shape = get_config_scalar_default<double>(config, "cloglog_leaf_prior_shape", 2.0);
  output.cloglog_leaf_prior_scale = get_config_scalar_default<double>(config, "cloglog_leaf_prior_scale", 2.0);
  output.cloglog_cutpoint_0 = get_config_scalar_default<double>(config, "cloglog_cutpoint_0", 0.0);

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
  output.rfx_model_spec = static_cast<StochTree::BARTRFXModelSpec>(get_config_scalar_default<int>(config, "rfx_model_spec", 0));
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
  SEXP sweep_update_indices_mean_raw = static_cast<SEXP>(config["sweep_update_indices_mean"]);
  if (!Rf_isNull(sweep_update_indices_mean_raw)) {
    cpp11::integers sweep_update_indices_mean_r_vec(sweep_update_indices_mean_raw);
    output.sweep_update_indices_mean.assign(sweep_update_indices_mean_r_vec.begin(), sweep_update_indices_mean_r_vec.end());
  }
  SEXP sweep_update_indices_variance_raw = static_cast<SEXP>(config["sweep_update_indices_variance"]);
  if (!Rf_isNull(sweep_update_indices_variance_raw)) {
    cpp11::integers sweep_update_indices_variance_r_vec(sweep_update_indices_variance_raw);
    output.sweep_update_indices_variance.assign(sweep_update_indices_variance_r_vec.begin(), sweep_update_indices_variance_r_vec.end());
  }
  SEXP var_weights_mean_raw = static_cast<SEXP>(config["var_weights_mean"]);
  if (!Rf_isNull(var_weights_mean_raw)) {
    cpp11::doubles var_weights_mean_r_vec(var_weights_mean_raw);
    output.var_weights_mean.assign(var_weights_mean_r_vec.begin(), var_weights_mean_r_vec.end());
  }
  SEXP sigma2_leaf_mean_matrix_raw = static_cast<SEXP>(config["sigma2_leaf_mean_matrix"]);
  if (!Rf_isNull(sigma2_leaf_mean_matrix_raw)) {
    cpp11::doubles sigma2_leaf_mean_matrix_r_vec(sigma2_leaf_mean_matrix_raw);
    output.sigma2_leaf_mean_matrix.assign(sigma2_leaf_mean_matrix_r_vec.begin(), sigma2_leaf_mean_matrix_r_vec.end());
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

cpp11::writable::list convert_bart_results_to_list(StochTree::BARTSamples& bart_samples) {
  cpp11::writable::list output;

  // Pointers to forests
  SEXP mean_forests_sexp = (bart_samples.mean_forests.get() != nullptr)
                               ? static_cast<SEXP>(cpp11::external_pointer<StochTree::ForestContainer>(bart_samples.mean_forests.release()))
                               : R_NilValue;
  output.push_back(cpp11::named_arg("mean_forests") = mean_forests_sexp);

  SEXP variance_forests_sexp = (bart_samples.variance_forests.get() != nullptr)
                                   ? static_cast<SEXP>(cpp11::external_pointer<StochTree::ForestContainer>(bart_samples.variance_forests.release()))
                                   : R_NilValue;
  output.push_back(cpp11::named_arg("variance_forests") = variance_forests_sexp);

  // Pointers to RFX model terms
  SEXP rfx_container_sexp = (bart_samples.rfx_container.get() != nullptr)
                                ? static_cast<SEXP>(cpp11::external_pointer<StochTree::RandomEffectsContainer>(bart_samples.rfx_container.release()))
                                : R_NilValue;
  output.push_back(cpp11::named_arg("rfx_container") = rfx_container_sexp);
  SEXP rfx_label_mapper_sexp = (bart_samples.rfx_label_mapper.get() != nullptr)
                                   ? static_cast<SEXP>(cpp11::external_pointer<StochTree::LabelMapper>(bart_samples.rfx_label_mapper.release()))
                                   : R_NilValue;
  output.push_back(cpp11::named_arg("rfx_label_mapper") = rfx_label_mapper_sexp);

  // Predictions
  SEXP mean_preds_train_sexp = !bart_samples.mean_forest_predictions_train.empty()
                                   ? static_cast<SEXP>(cpp11::writable::doubles(bart_samples.mean_forest_predictions_train.begin(), bart_samples.mean_forest_predictions_train.end()))
                                   : R_NilValue;
  output.push_back(cpp11::named_arg("mean_forest_predictions_train") = mean_preds_train_sexp);

  SEXP var_preds_train_sexp = !bart_samples.variance_forest_predictions_train.empty()
                                  ? static_cast<SEXP>(cpp11::writable::doubles(bart_samples.variance_forest_predictions_train.begin(), bart_samples.variance_forest_predictions_train.end()))
                                  : R_NilValue;
  output.push_back(cpp11::named_arg("variance_forest_predictions_train") = var_preds_train_sexp);

  SEXP mean_preds_test_sexp = !bart_samples.mean_forest_predictions_test.empty()
                                  ? static_cast<SEXP>(cpp11::writable::doubles(bart_samples.mean_forest_predictions_test.begin(), bart_samples.mean_forest_predictions_test.end()))
                                  : R_NilValue;
  output.push_back(cpp11::named_arg("mean_forest_predictions_test") = mean_preds_test_sexp);

  SEXP var_preds_test_sexp = !bart_samples.variance_forest_predictions_test.empty()
                                 ? static_cast<SEXP>(cpp11::writable::doubles(bart_samples.variance_forest_predictions_test.begin(), bart_samples.variance_forest_predictions_test.end()))
                                 : R_NilValue;
  output.push_back(cpp11::named_arg("variance_forest_predictions_test") = var_preds_test_sexp);

  // RFX predictions
  SEXP rfx_preds_train_sexp = !bart_samples.rfx_predictions_train.empty()
                                  ? static_cast<SEXP>(cpp11::writable::doubles(bart_samples.rfx_predictions_train.begin(), bart_samples.rfx_predictions_train.end()))
                                  : R_NilValue;
  output.push_back(cpp11::named_arg("rfx_predictions_train") = rfx_preds_train_sexp);

  SEXP rfx_preds_test_sexp = !bart_samples.rfx_predictions_test.empty()
                                 ? static_cast<SEXP>(cpp11::writable::doubles(bart_samples.rfx_predictions_test.begin(), bart_samples.rfx_predictions_test.end()))
                                 : R_NilValue;
  output.push_back(cpp11::named_arg("rfx_predictions_test") = rfx_preds_test_sexp);

  // Parameter samples
  SEXP global_var_sexp = !bart_samples.global_error_variance_samples.empty()
                             ? static_cast<SEXP>(cpp11::writable::doubles(bart_samples.global_error_variance_samples.begin(), bart_samples.global_error_variance_samples.end()))
                             : R_NilValue;
  output.push_back(cpp11::named_arg("global_error_variance_samples") = global_var_sexp);

  SEXP leaf_scale_sexp = !bart_samples.leaf_scale_samples.empty()
                             ? static_cast<SEXP>(cpp11::writable::doubles(bart_samples.leaf_scale_samples.begin(), bart_samples.leaf_scale_samples.end()))
                             : R_NilValue;
  output.push_back(cpp11::named_arg("leaf_scale_samples") = leaf_scale_sexp);

  SEXP cloglog_cutpoints_sexp = !bart_samples.cloglog_cutpoint_samples.empty()
                                    ? static_cast<SEXP>(cpp11::writable::doubles(bart_samples.cloglog_cutpoint_samples.begin(), bart_samples.cloglog_cutpoint_samples.end()))
                                    : R_NilValue;
  output.push_back(cpp11::named_arg("cloglog_cutpoint_samples") = cloglog_cutpoints_sexp);

  // Metadata about the model that was sampled
  double y_bar_sexp = bart_samples.y_bar;
  output.push_back(cpp11::named_arg("y_bar") = y_bar_sexp);
  double y_std_sexp = bart_samples.y_std;
  output.push_back(cpp11::named_arg("y_std") = y_std_sexp);
  int num_samples_sexp = bart_samples.num_samples;
  output.push_back(cpp11::named_arg("num_samples") = num_samples_sexp);
  int num_train_sexp = bart_samples.num_train;
  output.push_back(cpp11::named_arg("num_train") = num_train_sexp);
  int num_test_sexp = bart_samples.num_test;
  output.push_back(cpp11::named_arg("num_test") = num_test_sexp);
  return output;
}

void add_config_to_bart_result_list(cpp11::writable::list& result, StochTree::BARTConfig& config) {
  // Unpack more metadata about the model that was sampled
  result.push_back(cpp11::named_arg("sigma2_global_init") = config.sigma2_global_init);
  result.push_back(cpp11::named_arg("sigma2_mean_init") = config.sigma2_mean_init);
  result.push_back(cpp11::named_arg("b_sigma2_mean") = config.b_sigma2_mean);
  result.push_back(cpp11::named_arg("shape_variance_forest") = config.shape_variance_forest);
  result.push_back(cpp11::named_arg("scale_variance_forest") = config.scale_variance_forest);
  return;
}

[[cpp11::register]]
cpp11::writable::list bart_sample_cpp(
    cpp11::sexp X_train,
    cpp11::sexp y_train,
    cpp11::sexp X_test,
    int n_train,
    int n_test,
    int p,
    cpp11::sexp basis_train,
    cpp11::sexp basis_test,
    int basis_dim,
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
    cpp11::list config_input) {
  // Create outcome object
  StochTree::BARTSamples results_raw = StochTree::BARTSamples();

  // Extract pointers to raw data
  int protect_count = 0;
  double* X_train_ptr = extract_numeric_pointer(X_train, "X_train", protect_count);
  double* y_train_ptr = extract_numeric_pointer(y_train, "y_train", protect_count);
  double* X_test_ptr = extract_numeric_pointer(X_test, "X_test", protect_count);
  double* basis_train_ptr = extract_numeric_pointer(basis_train, "basis_train", protect_count);
  double* basis_test_ptr = extract_numeric_pointer(basis_test, "basis_test", protect_count);
  double* obs_weights_train_ptr = extract_numeric_pointer(obs_weights_train, "obs_weights_train", protect_count);
  double* obs_weights_test_ptr = extract_numeric_pointer(obs_weights_test, "obs_weights_test", protect_count);
  int* rfx_group_ids_train_ptr = extract_integer_pointer(rfx_group_ids_train, "rfx_group_ids_train", protect_count);
  int* rfx_group_ids_test_ptr = extract_integer_pointer(rfx_group_ids_test, "rfx_group_ids_test", protect_count);
  double* rfx_basis_train_ptr = extract_numeric_pointer(rfx_basis_train, "rfx_basis_train", protect_count);
  double* rfx_basis_test_ptr = extract_numeric_pointer(rfx_basis_test, "rfx_basis_test", protect_count);

  // Load the BARTData struct
  // Consider reading directly from the R objects or at least checking for matches with the R object dimensions)
  StochTree::BARTData data;
  data.X_train = X_train_ptr;
  data.y_train = y_train_ptr;
  data.X_test = X_test_ptr;
  data.n_train = n_train;
  data.p = p;
  data.n_test = n_test;
  data.basis_train = basis_train_ptr;
  data.basis_test = basis_test_ptr;
  data.basis_dim = basis_dim;
  data.obs_weights_train = obs_weights_train_ptr;
  data.obs_weights_test = obs_weights_test_ptr;
  data.rfx_group_ids_train = rfx_group_ids_train_ptr;
  data.rfx_group_ids_test = rfx_group_ids_test_ptr;
  data.rfx_basis_train = rfx_basis_train_ptr;
  data.rfx_basis_test = rfx_basis_test_ptr;
  data.rfx_num_groups = rfx_num_groups;
  data.rfx_basis_dim = rfx_basis_dim;

  // Create the BARTConfig object
  StochTree::BARTConfig config = convert_list_to_bart_config(config_input);

  // Initialize a BART sampler
  StochTree::BARTSampler bart_sampler(results_raw, config, data);

  // Run the sampler
  bart_sampler.run_gfr(results_raw, num_gfr, config.keep_gfr, num_chains);
  if (num_chains > 1) {
    bart_sampler.run_mcmc_chains(results_raw, num_chains, num_burnin, keep_every, num_mcmc);
  } else {
    bart_sampler.run_mcmc(results_raw, num_burnin, keep_every, num_mcmc);
  }
  bart_sampler.postprocess_samples(results_raw);

  // Unprotect protected R objects
  UNPROTECT(protect_count);

  // Unpack outputs
  cpp11::writable::list output_list = convert_bart_results_to_list(results_raw);
  add_config_to_bart_result_list(output_list, config);
  return output_list;
}

cpp11::writable::list convert_bart_preds_to_list(StochTree::BARTPredictionResult& bart_preds) {
  cpp11::writable::list output;

  // Predictions
  SEXP y_hat_sexp = !bart_preds.y_hat.empty()
                        ? static_cast<SEXP>(cpp11::writable::doubles(bart_preds.y_hat.begin(), bart_preds.y_hat.end()))
                        : R_NilValue;
  output.push_back(cpp11::named_arg("y_hat") = y_hat_sexp);

  SEXP mean_forest_pred_sexp = !bart_preds.mean_forest_predictions.empty()
                                   ? static_cast<SEXP>(cpp11::writable::doubles(bart_preds.mean_forest_predictions.begin(), bart_preds.mean_forest_predictions.end()))
                                   : R_NilValue;
  output.push_back(cpp11::named_arg("mean_forest_predictions") = mean_forest_pred_sexp);

  SEXP variance_forest_pred_sexp = !bart_preds.variance_forest_predictions.empty()
                                       ? static_cast<SEXP>(cpp11::writable::doubles(bart_preds.variance_forest_predictions.begin(), bart_preds.variance_forest_predictions.end()))
                                       : R_NilValue;
  output.push_back(cpp11::named_arg("variance_forest_predictions") = variance_forest_pred_sexp);

  SEXP rfx_predictions_sexp = !bart_preds.rfx_predictions.empty()
                                  ? static_cast<SEXP>(cpp11::writable::doubles(bart_preds.rfx_predictions.begin(), bart_preds.rfx_predictions.end()))
                                  : R_NilValue;
  output.push_back(cpp11::named_arg("rfx_predictions") = rfx_predictions_sexp);

  return output;
}

[[cpp11::register]]
cpp11::writable::list bart_predict_cpp(
    cpp11::list bart_model_list,
    cpp11::sexp X,
    cpp11::sexp leaf_basis,
    int n,
    int p,
    int num_basis,
    cpp11::sexp obs_weights,
    cpp11::sexp rfx_group_ids,
    cpp11::sexp rfx_basis,
    int rfx_num_groups,
    int rfx_basis_dim,
    bool posterior,
    int scale,
    bool predict_y_hat,
    bool predict_mean_forest,
    bool predict_variance_forest,
    bool predict_random_effects) {
  // Extract pointers to raw data
  int protect_count = 0;
  double* X_ptr = extract_numeric_pointer(X, "X", protect_count);
  double* leaf_basis_ptr = extract_numeric_pointer(leaf_basis, "leaf_basis", protect_count);
  double* obs_weights_ptr = extract_numeric_pointer(obs_weights, "obs_weights", protect_count);
  int* rfx_group_ids_ptr = extract_integer_pointer(rfx_group_ids, "rfx_group_ids", protect_count);
  double* rfx_basis_ptr = extract_numeric_pointer(rfx_basis, "rfx_basis", protect_count);

  // Load the BARTData struct
  // Consider reading directly from the R objects or at least checking for matches with the R object dimensions)
  StochTree::BARTData data;
  data.X_test = X_ptr;
  data.basis_test = leaf_basis_ptr;
  data.p = p;
  data.n_test = n;
  data.basis_dim = num_basis;
  data.obs_weights_test = obs_weights_ptr;
  data.rfx_group_ids_test = rfx_group_ids_ptr;
  data.rfx_basis_test = rfx_basis_ptr;
  data.rfx_num_groups = rfx_num_groups;
  data.rfx_basis_dim = rfx_basis_dim;

  // Load the BCF model and config from the model list
  StochTree::BARTPredictionInput pred_input;
  pred_input.global_error_variance_samples = extract_numeric_pointer(bart_model_list["sigma2_global_samples"], "sigma2_global_samples", protect_count);
  pred_input.leaf_scale_samples = extract_numeric_pointer(bart_model_list["sigma2_leaf_samples"], "sigma2_leaf_samples", protect_count);
  SEXP mean_forests_sexp = static_cast<SEXP>(bart_model_list["mean_forests"]);
  if (!Rf_isNull(mean_forests_sexp)) {
    pred_input.mean_forests = cpp11::external_pointer<StochTree::ForestContainer>(mean_forests_sexp).get();
  }
  SEXP variance_forests_sexp = static_cast<SEXP>(bart_model_list["variance_forests"]);
  if (!Rf_isNull(variance_forests_sexp)) {
    pred_input.variance_forests = cpp11::external_pointer<StochTree::ForestContainer>(variance_forests_sexp).get();
  }
  SEXP rfx_container_sexp = static_cast<SEXP>(bart_model_list["rfx_container"]);
  if (!Rf_isNull(rfx_container_sexp)) {
    pred_input.rfx_container = cpp11::external_pointer<StochTree::RandomEffectsContainer>(rfx_container_sexp).get();
  }
  SEXP rfx_label_mapper_sexp = static_cast<SEXP>(bart_model_list["rfx_label_mapper"]);
  if (!Rf_isNull(rfx_label_mapper_sexp)) {
    pred_input.rfx_label_mapper = cpp11::external_pointer<StochTree::LabelMapper>(rfx_label_mapper_sexp).get();
  }
  pred_input.num_samples = Rf_asInteger(bart_model_list["num_samples"]);
  pred_input.num_obs = n;
  pred_input.num_basis = num_basis;
  pred_input.y_bar = Rf_asReal(bart_model_list["y_bar"]);
  pred_input.y_std = Rf_asReal(bart_model_list["y_std"]);
  pred_input.has_variance_forest = (bool)Rf_asLogical(bart_model_list["include_variance_forest"]);
  pred_input.has_rfx = (bool)Rf_asLogical(bart_model_list["has_rfx"]);
  pred_input.cloglog_cutpoint_samples = extract_numeric_pointer(bart_model_list["cloglog_cutpoint_samples"], "cloglog_cutpoint_samples", protect_count);
  pred_input.cloglog_num_classes = Rf_asInteger(bart_model_list["cloglog_num_classes"]);
  {
    SEXP rfx_spec_sexp = bart_model_list["rfx_model_spec"];
    std::string rfx_model_spec_str = Rf_isNull(rfx_spec_sexp) ? "" : std::string(CHAR(STRING_ELT(rfx_spec_sexp, 0)));
    if (rfx_model_spec_str == "intercept_only") {
      pred_input.rfx_model_spec = StochTree::BARTRFXModelSpec::InterceptOnly;
    } else {
      pred_input.rfx_model_spec = StochTree::BARTRFXModelSpec::Custom;
    }
  }
  pred_input.pred_type = posterior ? StochTree::PredType::kPosterior : StochTree::PredType::kMean;
  if (scale == 0) {
    pred_input.pred_scale = StochTree::PredScale::kLinear;
  } else if (scale == 1) {
    pred_input.pred_scale = StochTree::PredScale::kProbability;
  } else {
    pred_input.pred_scale = StochTree::PredScale::kClass;
  }
  pred_input.pred_terms.y_hat = predict_y_hat;
  pred_input.pred_terms.mean_forest = predict_mean_forest;
  pred_input.pred_terms.variance_forest = predict_variance_forest;
  pred_input.pred_terms.random_effects = predict_random_effects;
  {
    SEXP link_function_sexp = bart_model_list["link_function"];
    std::string link_function_str = Rf_isNull(link_function_sexp) ? "" : std::string(CHAR(STRING_ELT(link_function_sexp, 0)));
    if (link_function_str == "identity") {
      pred_input.link_function = StochTree::LinkFunction::Identity;
    } else if (link_function_str == "probit") {
      pred_input.link_function = StochTree::LinkFunction::Probit;
    } else if (link_function_str == "cloglog") {
      pred_input.link_function = StochTree::LinkFunction::Cloglog;
    } else {
      StochTree::Log::Fatal("Unsupported link function specified in model list");
    }
  }
  {
    SEXP outcome_type_sexp = bart_model_list["outcome_type"];
    std::string outcome_type_str = Rf_isNull(outcome_type_sexp) ? "" : std::string(CHAR(STRING_ELT(outcome_type_sexp, 0)));
    if (outcome_type_str == "continuous") {
      pred_input.outcome_type = StochTree::OutcomeType::Continuous;
    } else if (outcome_type_str == "binary") {
      pred_input.outcome_type = StochTree::OutcomeType::Binary;
    } else if (outcome_type_str == "ordinal") {
      pred_input.outcome_type = StochTree::OutcomeType::Ordinal;
    } else {
      StochTree::Log::Fatal("Unsupported outcome type specified in model list");
    }
  }

  // Run the prediction function
  StochTree::BARTPredictionResult pred_results = predict_bart_model(data, pred_input);

  // Unprotect protected R objects
  UNPROTECT(protect_count);

  // Unpack outputs
  cpp11::writable::list output_list = convert_bart_preds_to_list(pred_results);
  return output_list;
}
