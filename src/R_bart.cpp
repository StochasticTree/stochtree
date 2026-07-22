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

cpp11::writable::list create_bart_metadata(StochTree::BARTConfig& config) {
  // Unpack metadata about the model that was sampled
  cpp11::writable::list model_metadata;
  model_metadata.push_back(cpp11::named_arg("sigma2_global_init") = config.sigma2_global_init);
  model_metadata.push_back(cpp11::named_arg("sigma2_mean_init") = config.sigma2_mean_init);
  model_metadata.push_back(cpp11::named_arg("b_sigma2_mean") = config.b_sigma2_mean);
  model_metadata.push_back(cpp11::named_arg("shape_variance_forest") = config.shape_variance_forest);
  model_metadata.push_back(cpp11::named_arg("scale_variance_forest") = config.scale_variance_forest);
  return model_metadata;
}

[[cpp11::register]]
cpp11::writable::list bart_sample_cpp(
    cpp11::external_pointer<StochTree::BARTSamples> samples,
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
  StochTree::BARTSampler bart_sampler(*samples, config, data);

  // Run the sampler
  bart_sampler.run_gfr(*samples, num_gfr, config.keep_gfr, num_chains);
  if (num_chains > 1) {
    bart_sampler.run_mcmc_chains(*samples, num_chains, num_burnin, keep_every, num_mcmc);
  } else {
    bart_sampler.run_mcmc(*samples, num_burnin, keep_every, num_mcmc);
  }
  bart_sampler.postprocess_samples(*samples);

  // Unprotect protected R objects
  UNPROTECT(protect_count);

  // Unpack metadata (including the final RNG state, so continued sampling can resume the stream)
  cpp11::writable::list metadata_list = create_bart_metadata(config);
  metadata_list.push_back(cpp11::named_arg("rng_state") = bart_sampler.GetRngState());
  return metadata_list;
}

[[cpp11::register]]
cpp11::writable::list bart_continue_sample_cpp(
    cpp11::external_pointer<StochTree::BARTSamples> samples,
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
    bool keep_gfr,
    std::string rng_state_in,
    bool override_seed,
    cpp11::list config_input) {
  // Extract pointers to raw (re-supplied) data
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

  // Load the BARTData struct from re-supplied data
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

  // Continuation appends new draws in place onto the model's single-owner samples object. The
  // warm-start reads the last retained forests / rfx / scalar state directly off `samples`
  // (mean, variance, and random effects are all warm-started from their last sample), and
  // postprocess_samples(samples, num_history) rescales only the newly appended draws.
  StochTree::BARTSamples& bart_samples = *samples;
  const int num_history = bart_samples.num_samples;

  // Initialize a BART sampler in continuation mode (warm-start from last sample)
  StochTree::BARTSampler bart_sampler(bart_samples, config, data, /*continuation=*/true);

  // Resume the RNG stream unless the user supplied a new seed (override_seed). The warm-start init
  // consumes no RNG draws, so the restored state is positioned exactly at the next draw.
  if (!override_seed && !rng_state_in.empty()) {
    bart_sampler.SetRngState(rng_state_in);
  }

  // Probit warm-start: regenerate the (unpersisted) latent outcome now that the RNG is positioned at
  // the resumed/re-seeded stream, so the first continued draw starts from a valid stationary state.
  bart_sampler.RegenerateLatentOutcome(bart_samples);

  // Optionally append GFR warm-start draws (num_gfr, retained iff keep_gfr), then MCMC draws, then
  // post-process only the newly appended range. Single-chain continuation, so num_chains = 1.
  bart_sampler.run_gfr(bart_samples, num_gfr, keep_gfr, /*num_chains=*/1);
  bart_sampler.run_mcmc(bart_samples, num_burnin, keep_every, num_mcmc);
  bart_sampler.postprocess_samples(bart_samples, num_history);

  // Unprotect protected R objects
  UNPROTECT(protect_count);

  // Metadata only; the extended samples live on the passed-in samples object.
  cpp11::writable::list metadata_list = create_bart_metadata(config);
  metadata_list.push_back(cpp11::named_arg("rng_state") = bart_sampler.GetRngState());
  return metadata_list;
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
    cpp11::external_pointer<StochTree::BARTSamples> bart_samples_ptr,
    cpp11::list bart_model_metadata,
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
  StochTree::BARTPredictionMetadata pred_metadata;
  pred_metadata.num_samples = Rf_asInteger(bart_model_metadata["num_samples"]);
  pred_metadata.num_obs = n;
  pred_metadata.num_basis = num_basis;
  pred_metadata.y_bar = Rf_asReal(bart_model_metadata["y_bar"]);
  pred_metadata.y_std = Rf_asReal(bart_model_metadata["y_std"]);
  pred_metadata.has_variance_forest = (bool)Rf_asLogical(bart_model_metadata["include_variance_forest"]);
  pred_metadata.has_rfx = (bool)Rf_asLogical(bart_model_metadata["has_rfx"]);
  pred_metadata.cloglog_num_classes = Rf_asInteger(bart_model_metadata["cloglog_num_classes"]);
  {
    SEXP rfx_spec_sexp = bart_model_metadata["rfx_model_spec"];
    std::string rfx_model_spec_str = Rf_isNull(rfx_spec_sexp) ? "" : std::string(CHAR(STRING_ELT(rfx_spec_sexp, 0)));
    if (rfx_model_spec_str == "intercept_only") {
      pred_metadata.rfx_model_spec = StochTree::BARTRFXModelSpec::InterceptOnly;
    } else {
      pred_metadata.rfx_model_spec = StochTree::BARTRFXModelSpec::Custom;
    }
  }
  pred_metadata.pred_type = posterior ? StochTree::PredType::kPosterior : StochTree::PredType::kMean;
  if (scale == 0) {
    pred_metadata.pred_scale = StochTree::PredScale::kLinear;
  } else if (scale == 1) {
    pred_metadata.pred_scale = StochTree::PredScale::kProbability;
  } else {
    pred_metadata.pred_scale = StochTree::PredScale::kClass;
  }
  pred_metadata.pred_terms.y_hat = predict_y_hat;
  pred_metadata.pred_terms.mean_forest = predict_mean_forest;
  pred_metadata.pred_terms.variance_forest = predict_variance_forest;
  pred_metadata.pred_terms.random_effects = predict_random_effects;
  {
    SEXP link_function_sexp = bart_model_metadata["link_function"];
    std::string link_function_str = Rf_isNull(link_function_sexp) ? "" : std::string(CHAR(STRING_ELT(link_function_sexp, 0)));
    if (link_function_str == "identity") {
      pred_metadata.link_function = StochTree::LinkFunction::Identity;
    } else if (link_function_str == "probit") {
      pred_metadata.link_function = StochTree::LinkFunction::Probit;
    } else if (link_function_str == "cloglog") {
      pred_metadata.link_function = StochTree::LinkFunction::Cloglog;
    } else {
      StochTree::Log::Fatal("Unsupported link function specified in model list");
    }
  }
  {
    SEXP outcome_type_sexp = bart_model_metadata["outcome_type"];
    std::string outcome_type_str = Rf_isNull(outcome_type_sexp) ? "" : std::string(CHAR(STRING_ELT(outcome_type_sexp, 0)));
    if (outcome_type_str == "continuous") {
      pred_metadata.outcome_type = StochTree::OutcomeType::Continuous;
    } else if (outcome_type_str == "binary") {
      pred_metadata.outcome_type = StochTree::OutcomeType::Binary;
    } else if (outcome_type_str == "ordinal") {
      pred_metadata.outcome_type = StochTree::OutcomeType::Ordinal;
    } else {
      StochTree::Log::Fatal("Unsupported outcome type specified in model list");
    }
  }

  // Run the prediction function
  StochTree::BARTPredictionResult pred_results = predict_bart_model(data, *bart_samples_ptr, pred_metadata);

  // Unprotect protected R objects
  UNPROTECT(protect_count);

  // Unpack outputs
  cpp11::writable::list output_list = convert_bart_preds_to_list(pred_results);
  return output_list;
}
