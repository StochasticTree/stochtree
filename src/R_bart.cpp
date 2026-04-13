#include <cpp11.hpp>
#include <stochtree/bart.h>
#include <stochtree/bart_sampler.h>
#include <stochtree/container.h>
#include <stochtree/data.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/tree_sampler.h>
#include <memory>
#include <string>

void check_numeric(cpp11::sexp input, const char* input_name) {
  if (TYPEOF(input) != REALSXP && !Rf_isInteger(input)) {
    cpp11::stop("Parameter %s must be a numeric array (integer or floating point)", input_name);
  }
}

double* extract_numeric_pointer(cpp11::sexp input, const char* input_name, int& protect_count) {
  if (input == R_NilValue) return nullptr;
  check_numeric(input, input_name);
  cpp11::sexp input_converted = PROTECT(Rf_coerceVector(input, REALSXP));
  protect_count++;
  return REAL(input_converted);
}

void check_integer(cpp11::sexp input, const char* input_name) {
  if (!Rf_isInteger(input)) {
    cpp11::stop("Parameter %s must be an integer array", input_name);
  }
}

int* extract_integer_pointer(cpp11::sexp input, const char* input_name, int& protect_count) {
  if (input == R_NilValue) return nullptr;
  check_integer(input, input_name);
  protect_count++;
  return INTEGER(input);
}

template <typename T>
T get_config_scalar_default(cpp11::list& config_list, const char* config_key, T default_value) {
  cpp11::sexp val = config_list[config_key];
  if (Rf_isNull(val)) return default_value;
  return cpp11::as_cpp<T>(val);
}

template <>
int get_config_scalar_default<int>(cpp11::list& config_list, const char* config_key, int default_value) {
  cpp11::sexp val = config_list[config_key];
  if (Rf_isNull(val)) return default_value;
  return Rf_asInteger(val);
}

StochTree::BARTConfig convert_list_to_config(cpp11::list config) {
  StochTree::BARTConfig output;

  // Global model parameters
  output.standardize_outcome = get_config_scalar_default<bool>(config, "standardize_outcome", true);
  output.num_threads = get_config_scalar_default<int>(config, "num_threads", 1);
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
  output.exponentiated_leaf_mean = get_config_scalar_default<bool>(config, "exponentiated_leaf_mean", true);
  output.num_features_subsample_mean = get_config_scalar_default<int>(config, "num_features_subsample_mean", 0);
  output.a_sigma2_mean = get_config_scalar_default<double>(config, "a_sigma2_mean", 3.0);
  output.b_sigma2_mean = get_config_scalar_default<double>(config, "b_sigma2_mean", -1.0);
  output.sigma2_mean_init = get_config_scalar_default<double>(config, "sigma2_mean_init", -1.0);
  output.sample_sigma2_leaf_mean = get_config_scalar_default<bool>(config, "sample_sigma2_leaf_mean", false);

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
  SEXP var_weights_variance_raw = static_cast<SEXP>(config["var_weights_variance"]);
  if (!Rf_isNull(var_weights_variance_raw)) {
    cpp11::doubles var_weights_variance_r_vec(var_weights_variance_raw);
    output.var_weights_variance.assign(var_weights_variance_r_vec.begin(), var_weights_variance_r_vec.end());
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

  // Parameter samples
  SEXP global_var_sexp = !bart_samples.global_error_variance_samples.empty()
                             ? static_cast<SEXP>(cpp11::writable::doubles(bart_samples.global_error_variance_samples.begin(), bart_samples.global_error_variance_samples.end()))
                             : R_NilValue;
  output.push_back(cpp11::named_arg("global_error_variance_samples") = global_var_sexp);

  SEXP leaf_scale_sexp = !bart_samples.leaf_scale_samples.empty()
                             ? static_cast<SEXP>(cpp11::writable::doubles(bart_samples.leaf_scale_samples.begin(), bart_samples.leaf_scale_samples.end()))
                             : R_NilValue;
  output.push_back(cpp11::named_arg("leaf_scale_samples") = leaf_scale_sexp);

  // Sample metadata
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
  StochTree::BARTConfig config = convert_list_to_config(config_input);

  // Initialize a BART sampler
  StochTree::BARTSampler bart_sampler(results_raw, config, data);

  // Run the sampler
  bart_sampler.run_gfr(results_raw, num_gfr, true);
  bart_sampler.run_mcmc(results_raw, num_burnin, keep_every, num_mcmc);
  bart_sampler.postprocess_samples(results_raw);

  // Unprotect protected R objects
  UNPROTECT(protect_count);

  return convert_bart_results_to_list(results_raw);
}
