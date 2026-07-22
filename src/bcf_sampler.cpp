/*! Copyright (c) 2026 by stochtree authors */
#include <stochtree/bcf.h>
#include <stochtree/bcf_sampler.h>
#include <stochtree/data.h>
#include <stochtree/distributions.h>
#include <stochtree/leaf_model.h>
#include <stochtree/linear_regression.h>
#include <stochtree/meta.h>
#include <stochtree/probit.h>
#include <stochtree/random_effects.h>
#include <stochtree/tree_sampler.h>
#include <stochtree/variance_model.h>
#include <memory>
#include <random>
#include <variant>

namespace StochTree {

void AddModelTermsForProbit(double* outcome_preds, ForestTracker* mu_forest_tracker, ForestTracker* tau_forest_tracker, RandomEffectsTracker* random_effects_tracker, int n) {
  double* mu_preds = mu_forest_tracker->GetSumPredictions();
  double* tau_preds = tau_forest_tracker->GetSumPredictions();
  if (random_effects_tracker != nullptr) {
    double* rfx_preds = random_effects_tracker->GetPredictions();
    for (int i = 0; i < n; i++) {
      outcome_preds[i] = mu_preds[i] + tau_preds[i] + rfx_preds[i];
    }
  } else {
    for (int i = 0; i < n; i++) {
      outcome_preds[i] = mu_preds[i] + tau_preds[i];
    }
  }
}

BCFSampler::BCFSampler(BCFSamples& samples, BCFConfig& config, BCFData& data, bool continuation) : config_{config}, data_{data}, mu_leaf_model_(GaussianConstantLeafModel(0.0)), tau_leaf_model_(GaussianUnivariateRegressionLeafModel(0.0)), variance_leaf_model_(0.0, 0.0) {
  InitializeState(samples, continuation);
}

void BCFSampler::InitializeState(BCFSamples& samples, bool continuation) {
  // Validate y_train values match the expected support for discrete link functions
  if (config_.link_function == LinkFunction::Probit) {
    for (int i = 0; i < data_.n_train; i++) {
      if (data_.y_train[i] != 0.0 && data_.y_train[i] != 1.0) {
        Log::Fatal("Outcomes must be 0 or 1 for probit link function");
      }
    }
    // Initialize model_preds_ vector for probit latent outcome sampling
    model_preds_.resize(data_.n_train, 0.0);
  } else if (config_.link_function == LinkFunction::Cloglog) {
    Log::Fatal("Cloglog link function is not currently supported in BCF");
  }

  // Validate that both num_trees_mu and num_trees_tau are positive
  if (config_.num_trees_mu <= 0) {
    Log::Fatal("num_trees_mu must be >0");
  }
  if (config_.num_trees_tau <= 0) {
    Log::Fatal("num_trees_tau must be >0");
  }

  // Validate outcome type
  if (config_.outcome_type == OutcomeType::Ordinal) {
    Log::Fatal("Ordinal outcome type is not currently supported in BCF");
  }

  // Switch off treatment forest leaf scale sampling if treatment is multivariate
  if (config_.sample_sigma2_leaf_tau && config_.tau_leaf_model_type != MeanLeafModelType::GaussianUnivariateRegression) {
    Log::Info("sample_sigma2_leaf_tau can only be true when tau_leaf_model_type is GaussianUnivariateRegression, setting sample_sigma2_leaf_tau to false");
    config_.sample_sigma2_leaf_tau = false;
  }

  // Adaptive coding model
  if (config_.adaptive_coding) {
    if (data_.treatment_dim != 1) {
      Log::Fatal("Adaptive coding is currently only supported for binary treatments (treatment_dim=1)");
    }
    adaptive_coding_ = true;
    // On continuation, warm-start b0/b1 from the last retained sample (stored raw; postprocess does
    // not touch them) so the adaptive basis built below + the tau-forest warm-start use them.
    if (continuation && !samples.b0_samples.empty()) {
      b_0_ = samples.b0_samples.back();
      b_1_ = samples.b1_samples.back();
    } else {
      b_0_ = config_.b_0_init;
      b_1_ = config_.b_1_init;
    }
  }

  // Load data from BARTData object into ForestDataset object
  forest_dataset_ = std::make_unique<ForestDataset>();
  forest_dataset_->AddCovariates(data_.X_train, data_.n_train, data_.p, /*row_major=*/false);
  if (data_.treatment_train != nullptr) {
    if (adaptive_coding_) {
      // Basis becomes b_0 * (1-Z) + b_1 * Z
      tau_basis_vector_train_.resize(data_.n_train);
      for (int i = 0; i < data_.n_train; i++) {
        double z = data_.treatment_train[i];
        tau_basis_vector_train_[i] = b_0_ * (1.0 - z) + b_1_ * z;
      }
      forest_dataset_->AddBasis(tau_basis_vector_train_.data(), data_.n_train, data_.treatment_dim, /*row_major=*/false);
    } else {
      forest_dataset_->AddBasis(data_.treatment_train, data_.n_train, data_.treatment_dim, /*row_major=*/false);
    }
  }
  if (data_.obs_weights_train != nullptr) {
    forest_dataset_->AddVarianceWeights(data_.obs_weights_train, data_.n_train);
  }
  samples.num_train = data_.n_train;
  samples.num_test = data_.n_test;
  samples.treatment_dim = data_.treatment_dim;
  residual_ = std::make_unique<ColumnVector>(data_.y_train, data_.n_train);
  outcome_raw_ = std::make_unique<ColumnVector>(data_.y_train, data_.n_train);
  if (data_.X_test != nullptr) {
    forest_dataset_test_ = std::make_unique<ForestDataset>();
    forest_dataset_test_->AddCovariates(data_.X_test, data_.n_test, data_.p, /*row_major=*/false);
    if (data_.treatment_test != nullptr) {
      if (adaptive_coding_) {
        // Basis becomes b_0 * (1-Z) + b_1 * Z
        tau_basis_vector_test_.resize(data_.n_test);
        for (int i = 0; i < data_.n_test; i++) {
          double z = data_.treatment_test[i];
          tau_basis_vector_test_[i] = b_0_ * (1.0 - z) + b_1_ * z;
        }
        forest_dataset_test_->AddBasis(tau_basis_vector_test_.data(), data_.n_test, data_.treatment_dim, /*row_major=*/false);
      } else {
        forest_dataset_test_->AddBasis(data_.treatment_test, data_.n_test, data_.treatment_dim, /*row_major=*/false);
      }
    }
    if (data_.obs_weights_test != nullptr) {
      forest_dataset_test_->AddVarianceWeights(data_.obs_weights_test, data_.n_test);
    }
    has_test_ = true;
  }

  // Precompute outcome mean and variance for standardization and calibration
  double y_mean = 0.0, M2 = 0.0, y_mean_prev = 0.0;
  for (int i = 0; i < data_.n_train; i++) {
    y_mean_prev = y_mean;
    y_mean = y_mean_prev + (data_.y_train[i] - y_mean_prev) / (i + 1);
    M2 = M2 + (data_.y_train[i] - y_mean_prev) * (data_.y_train[i] - y_mean);
  }
  double y_var = M2 / data_.n_train;

  // Outcome standardization and forest initial value setup
  if (config_.link_function == LinkFunction::Probit) {
    // Initialize forests to 0, no scaling, but offset by the probit transform of the mean outcome to improve mixing
    samples.y_std = 1.0;
    samples.y_bar = norm_inv_cdf(y_mean);
    init_val_mu_ = 0.0;
    init_val_tau_ = 0.0;
    if (config_.tau_leaf_model_type == MeanLeafModelType::GaussianMultivariateRegression) {
      init_val_tau_vec_.assign(config_.leaf_dim_tau, 0.0);
    }
  } else {
    if (config_.standardize_outcome) {
      samples.y_bar = y_mean;
      samples.y_std = std::sqrt(y_var);
      init_val_mu_ = 0.0;
      init_val_tau_ = 0.0;
      if (config_.tau_leaf_model_type == MeanLeafModelType::GaussianMultivariateRegression) {
        init_val_tau_vec_.assign(config_.leaf_dim_tau, 0.0);
      }
    } else {
      samples.y_bar = 0.0;
      samples.y_std = 1.0;
      init_val_mu_ = y_mean;
      init_val_tau_ = 0.0;
      if (config_.tau_leaf_model_type == MeanLeafModelType::GaussianMultivariateRegression) {
        init_val_tau_vec_.assign(config_.leaf_dim_tau, 0.0);
      }
    }
  }

  // Calibration for mu forest
  if (config_.sigma2_mu_init < 0.0) {
    if (config_.link_function == LinkFunction::Probit) {
      config_.sigma2_mu_init = 1.0 / config_.num_trees_mu;
    } else {
      if (config_.standardize_outcome)
        config_.sigma2_mu_init = 1.0 / config_.num_trees_mu;
      else
        config_.sigma2_mu_init = y_var / config_.num_trees_mu;
    }
  }
  if (config_.sample_sigma2_leaf_mu) {
    if (config_.b_sigma2_mu <= 0.0) {
      if (config_.link_function == LinkFunction::Probit) {
        config_.b_sigma2_mu = 1.0 / (2 * config_.num_trees_mu);
      } else {
        if (config_.standardize_outcome)
          config_.b_sigma2_mu = 1.0 / (2 * config_.num_trees_mu);
        else
          config_.b_sigma2_mu = y_var / (2 * config_.num_trees_mu);
      }
    }
  }

  // Calibration for tau forest
  if (config_.sigma2_tau_init < 0.0) {
    if (config_.link_function == LinkFunction::Probit) {
      config_.sigma2_tau_init = 1.0 / config_.num_trees_tau;
    } else {
      if (config_.standardize_outcome)
        config_.sigma2_tau_init = 1.0 / config_.num_trees_tau;
      else
        config_.sigma2_tau_init = y_var / config_.num_trees_tau;
    }
  }
  if (config_.sample_sigma2_leaf_tau) {
    if (config_.b_sigma2_tau <= 0.0) {
      if (config_.link_function == LinkFunction::Probit) {
        config_.b_sigma2_tau = 1.0 / (2 * config_.num_trees_tau);
      } else {
        if (config_.standardize_outcome)
          config_.b_sigma2_tau = 1.0 / (2 * config_.num_trees_tau);
        else
          config_.b_sigma2_tau = y_var / (2 * config_.num_trees_tau);
      }
    }
  }

  // Initialize mu leaf model
  mu_leaf_model_ = GaussianConstantLeafModel(config_.sigma2_mu_init);

  // Initialize tau leaf model
  if (config_.tau_leaf_model_type == MeanLeafModelType::GaussianUnivariateRegression) {
    tau_leaf_model_ = GaussianUnivariateRegressionLeafModel(config_.sigma2_tau_init);
  } else if (config_.tau_leaf_model_type == MeanLeafModelType::GaussianMultivariateRegression) {
    Eigen::MatrixXd Sigma_0;
    if (!config_.sigma2_leaf_tau_matrix.empty()) {
      if ((int)config_.sigma2_leaf_tau_matrix.size() != config_.leaf_dim_tau * config_.leaf_dim_tau) {
        Log::Fatal("sigma2_leaf_tau_matrix must have leaf_dim_tau * leaf_dim_tau = %d elements, but has %zu",
                   config_.leaf_dim_tau * config_.leaf_dim_tau, config_.sigma2_leaf_tau_matrix.size());
      }
      // Column-major interpretation matches both R and Eigen (python must be reordered before passing to C++)
      Sigma_0 = Eigen::Map<const Eigen::MatrixXd>(config_.sigma2_leaf_tau_matrix.data(), config_.leaf_dim_tau, config_.leaf_dim_tau);
    } else {
      Sigma_0 = config_.sigma2_tau_init * Eigen::MatrixXd::Identity(config_.leaf_dim_tau, config_.leaf_dim_tau);
    }
    tau_leaf_model_ = GaussianMultivariateRegressionLeafModel(Sigma_0);
  } else {
    Log::Fatal("Unsupported leaf model type for treatment forest");
  }

  // Calibration for variance forests
  if (config_.num_trees_variance > 0) {
    // NOTE: calibration only works for standardized outcomes
    if (config_.shape_variance_forest <= 0.0 || config_.scale_variance_forest <= 0.0) {
      if (config_.leaf_prior_calibration_param <= 0.0) {
        config_.leaf_prior_calibration_param = 1.5;
      }
      if (config_.shape_variance_forest <= 0.0) {
        config_.shape_variance_forest = config_.num_trees_variance / (config_.leaf_prior_calibration_param * config_.leaf_prior_calibration_param) + 0.5;
      }
      if (config_.scale_variance_forest <= 0.0) {
        config_.scale_variance_forest = config_.num_trees_variance / (config_.leaf_prior_calibration_param * config_.leaf_prior_calibration_param);
      }
    }
    if (config_.variance_forest_leaf_init > 0.0) {
      init_val_variance_ = config_.variance_forest_leaf_init;
    } else if (config_.standardize_outcome) {
      init_val_variance_ = 1.0;
    } else {
      init_val_variance_ = y_var;
    }
  }

  // Standardize partial residuals in place; these are updated in each iteration but initialized to standardized outcomes
  // Works for:
  //  1. Standardized outcomes (since y_bar = mean(y) and y_std = sd(y))
  //  2. Non-standardized outcomes (since y_bar = 0 and y_std = 1, so this just transfers y_train as-is)
  //  3. Probit link (since y_bar = norm_inv_cdf(mean(y)) and y_std = 1)
  for (int i = 0; i < data_.n_train; i++) residual_->GetData()[i] = (data_.y_train[i] - samples.y_bar) / samples.y_std;

  // Initialize mean forest state.
  // On continuation, do NOT re-create samples.mu_forests so that new draws are appended to the
  // existing container; otherwise allocate a fresh container.
  mu_forest_ = std::make_unique<TreeEnsemble>(config_.num_trees_mu, config_.leaf_dim_mu, config_.leaf_constant_mu, config_.exponentiated_leaf_mu);
  if (!continuation) {
    samples.mu_forests = std::make_unique<ForestContainer>(config_.num_trees_mu, config_.leaf_dim_mu, config_.leaf_constant_mu, config_.exponentiated_leaf_mu);
  }
  mu_forest_tracker_ = std::make_unique<ForestTracker>(forest_dataset_->GetCovariates(), config_.feature_types, config_.num_trees_mu, data_.n_train);
  tree_prior_mu_ = std::make_unique<TreePrior>(config_.alpha_mu, config_.beta_mu, config_.min_samples_leaf_mu, config_.max_depth_mu);
  mu_forest_->SetLeafValue(init_val_mu_ / config_.num_trees_mu);
  UpdateResidualEntireForest(*mu_forest_tracker_, *forest_dataset_, *residual_, mu_forest_.get(), !config_.leaf_constant_mu, std::minus<double>());
  mu_forest_tracker_->UpdatePredictions(mu_forest_.get(), *forest_dataset_.get());
  if (continuation) {
    // Warm-start from the last retained mu sample. The reset requires the already-populated
    // tracker established by the root init above (mirrors RestoreStateFromGFRSnapshot).
    int last_mu_idx = samples.mu_forests->NumSamples() - 1;
    TreeEnsemble& last_mu = *samples.mu_forests->GetEnsemble(last_mu_idx);
    mu_forest_->ReconstituteFromForest(last_mu);
    mu_forest_tracker_->ReconstituteFromForest(last_mu, *forest_dataset_, *residual_, true);
    mu_forest_tracker_->UpdatePredictions(mu_forest_.get(), *forest_dataset_.get());
  }

  // Initialize treatment effect forest state.
  // On continuation, do NOT re-create samples.tau_forests (append to the existing container).
  if (config_.tau_leaf_model_type == MeanLeafModelType::GaussianUnivariateRegression) {
    tau_forest_ = std::make_unique<TreeEnsemble>(config_.num_trees_tau, config_.leaf_dim_tau, config_.leaf_constant_tau, config_.exponentiated_leaf_tau);
    if (!continuation) {
      samples.tau_forests = std::make_unique<ForestContainer>(config_.num_trees_tau, config_.leaf_dim_tau, config_.leaf_constant_tau, config_.exponentiated_leaf_tau);
    }
    tau_forest_tracker_ = std::make_unique<ForestTracker>(forest_dataset_->GetCovariates(), config_.feature_types, config_.num_trees_tau, data_.n_train);
    tree_prior_tau_ = std::make_unique<TreePrior>(config_.alpha_tau, config_.beta_tau, config_.min_samples_leaf_tau, config_.max_depth_tau);
    tau_forest_->SetLeafValue(init_val_tau_ / config_.num_trees_tau);
    UpdateResidualEntireForest(*tau_forest_tracker_, *forest_dataset_, *residual_, tau_forest_.get(), !config_.leaf_constant_tau, std::minus<double>());
    tau_forest_tracker_->UpdatePredictions(tau_forest_.get(), *forest_dataset_.get());
    if (continuation) {
      // Warm-start from the last retained tau sample. At this point the residual already has the
      // warm-started mu contribution removed; reconstitution swaps the root tau contribution out
      // and the last-sample tau contribution in. The treatment intercept tau_0 is restored
      // separately (below), after both forests are warm-started.
      int last_tau_idx = samples.tau_forests->NumSamples() - 1;
      TreeEnsemble& last_tau = *samples.tau_forests->GetEnsemble(last_tau_idx);
      tau_forest_->ReconstituteFromForest(last_tau);
      tau_forest_tracker_->ReconstituteFromForest(last_tau, *forest_dataset_, *residual_, true);
      tau_forest_tracker_->UpdatePredictions(tau_forest_.get(), *forest_dataset_.get());
    }
  } else if (config_.tau_leaf_model_type == MeanLeafModelType::GaussianMultivariateRegression) {
    tau_forest_ = std::make_unique<TreeEnsemble>(config_.num_trees_tau, config_.leaf_dim_tau, config_.leaf_constant_tau, config_.exponentiated_leaf_tau);
    // On continuation, preserve the existing sample container (append new draws).
    if (!continuation) {
      samples.tau_forests = std::make_unique<ForestContainer>(config_.num_trees_tau, config_.leaf_dim_tau, config_.leaf_constant_tau, config_.exponentiated_leaf_tau);
    }
    tau_forest_tracker_ = std::make_unique<ForestTracker>(forest_dataset_->GetCovariates(), config_.feature_types, config_.num_trees_tau, data_.n_train);
    tree_prior_tau_ = std::make_unique<TreePrior>(config_.alpha_tau, config_.beta_tau, config_.min_samples_leaf_tau, config_.max_depth_tau);
    tau_forest_->SetLeafVector(init_val_tau_vec_);
    UpdateResidualEntireForest(*tau_forest_tracker_, *forest_dataset_, *residual_, tau_forest_.get(), true, std::minus<double>());
    tau_forest_tracker_->UpdatePredictions(tau_forest_.get(), *forest_dataset_.get());
    if (continuation) {
      // Warm-start the multivariate treatment forest from the last retained sample.
      int last_tau_idx = samples.tau_forests->NumSamples() - 1;
      TreeEnsemble& last_tau = *samples.tau_forests->GetEnsemble(last_tau_idx);
      tau_forest_->ReconstituteFromForest(last_tau);
      tau_forest_tracker_->ReconstituteFromForest(last_tau, *forest_dataset_, *residual_, true);
      tau_forest_tracker_->UpdatePredictions(tau_forest_.get(), *forest_dataset_.get());
    }
  } else {
    Log::Fatal("Unsupported leaf model type for treatment forest");
  }

  // Initialize variance forest state (if present)
  if (config_.num_trees_variance > 0) {
    variance_leaf_model_ = LogLinearVarianceLeafModel(config_.shape_variance_forest, config_.scale_variance_forest);
    variance_forest_ = std::make_unique<TreeEnsemble>(config_.num_trees_variance, config_.leaf_dim_variance, config_.leaf_constant_variance, config_.exponentiated_leaf_variance);
    // On continuation, preserve the existing sample container so new draws append to it.
    if (!continuation) {
      samples.variance_forests = std::make_unique<ForestContainer>(config_.num_trees_variance, config_.leaf_dim_variance, config_.leaf_constant_variance, config_.exponentiated_leaf_variance);
    }
    variance_forest_tracker_ = std::make_unique<ForestTracker>(forest_dataset_->GetCovariates(), config_.feature_types, config_.num_trees_variance, data_.n_train);
    tree_prior_variance_ = std::make_unique<TreePrior>(config_.alpha_variance, config_.beta_variance, config_.min_samples_leaf_variance, config_.max_depth_variance);
    // Leaf values for the log-linear variance model are on the log scale; the ensemble sums
    // log(sigma^2_i) contributions, so each tree starts at log(init_val) / num_trees.
    variance_forest_->SetLeafValue(std::log(init_val_variance_) / config_.num_trees_variance);
    variance_forest_tracker_->UpdatePredictions(variance_forest_.get(), *forest_dataset_.get());
    // UpdateVarModelTree (called inside GFRSampleOneIter / MCMCSampleOneIter) unconditionally
    // reads and writes the dataset variance weight slot via VarWeightValue / SetVarWeightValue.
    // This slot tracks the cumulative per-observation variance prediction
    // (sigma^2_i = exp(sum of tree leaf values)) and is incompatible with case weights, which
    // would need to be reapplied after every per-tree update. The R/Python APIs enforce this
    // as a hard error; guard here for callers that use BARTSampler directly.
    if (forest_dataset_->HasVarWeights()) {
      Log::Fatal("observation_weights and a variance forest cannot be used together.");
    }
    std::vector<double> initial_variance_preds(data_.n_train, init_val_variance_);
    forest_dataset_->AddVarianceWeights(initial_variance_preds.data(), data_.n_train);
    has_variance_forest_ = true;
    // Continuation: warm-start the active variance forest from the last retained sample. The
    // reconstitution (is_mean_model=false) swaps the flat-init leaves out of the variance-weight
    // slot and the last forest's leaves in, so the slot lands at exp(sum of last-forest leaves) =
    // the last sample's per-observation variance prediction. Mirrors the mu/tau warm-start + BART.
    if (continuation) {
      int last_var_idx = samples.variance_forests->NumSamples() - 1;
      TreeEnsemble& last_variance_forest = *samples.variance_forests->GetEnsemble(last_var_idx);
      variance_forest_->ReconstituteFromForest(last_variance_forest);
      variance_forest_tracker_->ReconstituteFromForest(last_variance_forest, *forest_dataset_, *residual_, /*is_mean_model=*/false);
    }
  }

  // Global error variance model
  if (config_.sample_sigma2_global) {
    var_model_ = std::make_unique<GlobalHomoskedasticVarianceModel>();
    sample_sigma2_global_ = true;
  }

  // Leaf scale models
  if (config_.sample_sigma2_leaf_mu) {
    leaf_scale_model_mu_ = std::make_unique<LeafNodeHomoskedasticVarianceModel>();
    sample_sigma2_leaf_mu_ = true;
  }
  if (config_.sample_sigma2_leaf_tau) {
    leaf_scale_model_tau_ = std::make_unique<LeafNodeHomoskedasticVarianceModel>();
    sample_sigma2_leaf_tau_ = true;
  }

  // Treatment intercept model
  if (config_.sample_tau_0) {
    sample_tau_0_ = true;
    if (data_.treatment_dim > 1) {
      tau_0_vector_.assign(data_.treatment_dim, 0.0);
      if (config_.tau_0_prior_var_multivariate.empty()) {
        config_.tau_0_prior_var_multivariate.assign(data_.treatment_dim, config_.sigma2_tau_init * config_.num_trees_tau);
      } else {
        if ((int)config_.tau_0_prior_var_multivariate.size() != data_.treatment_dim) {
          Log::Fatal("tau_0_prior_var_multivariate must have treatment_dim = %d elements, but has %zu",
                     data_.treatment_dim, config_.tau_0_prior_var_multivariate.size());
        }
      }
    } else {
      tau_0_scalar_ = 0.0;
      if (config_.tau_0_prior_var_scalar <= 0.0) {
        config_.tau_0_prior_var_scalar = config_.sigma2_tau_init * config_.num_trees_tau;
      }
    }
  }

  // Random effects model
  if (config_.has_random_effects) {
    random_effects_dataset_ = std::make_unique<RandomEffectsDataset>();
    random_effects_dataset_->AddGroupLabels(data_.rfx_group_ids_train, data_.n_train);
    if (data_.rfx_basis_train != nullptr) {
      random_effects_dataset_->AddBasis(data_.rfx_basis_train, data_.n_train, data_.rfx_basis_dim, /*row_major=*/false);
    } else {
      if (config_.rfx_model_spec == BCFRFXModelSpec::InterceptOnly) {
        // If no basis is provided, add an intercept basis (column of 1s)
        // TODO: do we need to do this before we determine rfx_basis_dim and initialize the RFX data structures?
        std::vector<double> rfx_basis(data_.n_train, 1.0);
        random_effects_dataset_->AddBasis(rfx_basis.data(), data_.n_train, 1, /*row_major=*/false);
        // Override rfx_basis_dim to 1 for intercept-only model the basis is a 1-dimensional vector of ones
        data_.rfx_basis_dim = 1;
      } else if (config_.rfx_model_spec == BCFRFXModelSpec::InterceptPlusTreatment) {
        // If no basis is provided, add an intercept basis (column of 1s) and the treatment variable(s) as the basis
        // TODO: do we need to do this before we determine rfx_basis_dim and initialize the RFX data structures?
        std::vector<double> rfx_basis(data_.n_train * (1 + data_.treatment_dim), 1.0);
        for (int i = 0; i < data_.n_train; i++) {
          for (int j = 0; j < data_.treatment_dim; j++) {
            rfx_basis[(j + 1) * data_.n_train + i] = data_.treatment_train[j * data_.n_train + i];
          }
        }
        random_effects_dataset_->AddBasis(rfx_basis.data(), data_.n_train, 1 + data_.treatment_dim, /*row_major=*/false);
        // Override rfx_basis_dim to 1 for intercept-only model the basis is a 1-dimensional vector of ones
        data_.rfx_basis_dim = 1 + data_.treatment_dim;
      } else {
        Log::Fatal("Random effects basis data must be provided for non-intercept-only random effects model");
      }
    }
    // Tracking data structure for random effects groups
    random_effects_tracker_ = std::make_unique<RandomEffectsTracker>(data_.rfx_group_ids_train, data_.n_train);
    // Container of random effects samples + label mapper. On continuation these already exist on the
    // samples object (from the prior run); preserve them so new draws append to the existing container.
    if (!continuation) {
      samples.rfx_container = std::make_unique<RandomEffectsContainer>(data_.rfx_basis_dim, data_.rfx_num_groups);
      // Mapping from RFX labels to 0-indexed group IDs for efficient lookup in the sampler; populated from the RFX dataset group labels
      samples.rfx_label_mapper = std::make_unique<LabelMapper>(random_effects_tracker_->GetLabelMap());
    }

    // Initialize random effects model object
    random_effects_model_ = std::make_unique<MultivariateRegressionRandomEffectsModel>(data_.rfx_basis_dim, data_.rfx_num_groups);

    // Handle "working" parameter prior mean
    Eigen::VectorXd working_parameter_prior_mean;
    if (!config_.rfx_working_parameter_mean_prior.empty()) {
      if ((int)config_.rfx_working_parameter_mean_prior.size() != data_.rfx_basis_dim) {
        Log::Fatal("rfx_working_parameter_mean_prior must have rfx_basis_dim = %d elements, but has %zu",
                   data_.rfx_basis_dim, config_.rfx_working_parameter_mean_prior.size());
      }
      // Column-major interpretation matches both R and Eigen (python must be reordered before passing to C++)
      working_parameter_prior_mean = Eigen::Map<const Eigen::VectorXd>(config_.rfx_working_parameter_mean_prior.data(), data_.rfx_basis_dim);
    } else {
      working_parameter_prior_mean = Eigen::VectorXd::Zero(data_.rfx_basis_dim);
    }
    random_effects_model_->SetWorkingParameter(working_parameter_prior_mean);

    // Handle "group" parameter prior mean
    Eigen::MatrixXd group_parameter_prior_mean;
    if (!config_.rfx_group_parameter_mean_prior.empty()) {
      if ((int)config_.rfx_group_parameter_mean_prior.size() != data_.rfx_basis_dim * data_.rfx_num_groups) {
        Log::Fatal("rfx_group_parameter_mean_prior must have rfx_basis_dim * rfx_num_groups = %d elements, but has %zu",
                   data_.rfx_basis_dim * data_.rfx_num_groups, config_.rfx_group_parameter_mean_prior.size());
      }
      // Column-major interpretation matches both R and Eigen (python must be reordered before passing to C++)
      group_parameter_prior_mean = Eigen::Map<const Eigen::MatrixXd>(config_.rfx_group_parameter_mean_prior.data(), data_.rfx_basis_dim, data_.rfx_num_groups);
    } else {
      group_parameter_prior_mean = Eigen::MatrixXd::Zero(data_.rfx_basis_dim, data_.rfx_num_groups);
    }
    random_effects_model_->SetGroupParameters(group_parameter_prior_mean);

    // Handle "working" parameter prior covariance
    Eigen::MatrixXd working_parameter_prior_cov;
    if (!config_.rfx_working_parameter_cov_prior.empty()) {
      if ((int)config_.rfx_working_parameter_cov_prior.size() != data_.rfx_basis_dim * data_.rfx_basis_dim) {
        Log::Fatal("rfx_working_parameter_cov_prior must have rfx_basis_dim * rfx_basis_dim = %d elements, but has %zu",
                   data_.rfx_basis_dim * data_.rfx_basis_dim, config_.rfx_working_parameter_cov_prior.size());
      }
      // Column-major interpretation matches both R and Eigen (python must be reordered before passing to C++)
      working_parameter_prior_cov = Eigen::Map<const Eigen::MatrixXd>(config_.rfx_working_parameter_cov_prior.data(), data_.rfx_basis_dim, data_.rfx_basis_dim);
    } else {
      working_parameter_prior_cov = Eigen::MatrixXd::Identity(data_.rfx_basis_dim, data_.rfx_basis_dim);
    }
    random_effects_model_->SetWorkingParameterCovariance(working_parameter_prior_cov);

    // Handle "group" parameter prior covariance
    Eigen::MatrixXd group_parameter_prior_cov;
    if (!config_.rfx_group_parameter_cov_prior.empty()) {
      if ((int)config_.rfx_group_parameter_cov_prior.size() != data_.rfx_basis_dim * data_.rfx_basis_dim) {
        Log::Fatal("rfx_group_parameter_cov_prior must have rfx_basis_dim * rfx_basis_dim = %d elements, but has %zu",
                   data_.rfx_basis_dim * data_.rfx_basis_dim, config_.rfx_group_parameter_cov_prior.size());
      }
      // Column-major interpretation matches both R and Eigen (python must be reordered before passing to C++)
      group_parameter_prior_cov = Eigen::Map<const Eigen::MatrixXd>(config_.rfx_group_parameter_cov_prior.data(), data_.rfx_basis_dim, data_.rfx_basis_dim);
    } else {
      group_parameter_prior_cov = Eigen::MatrixXd::Identity(data_.rfx_basis_dim, data_.rfx_basis_dim);
    }
    random_effects_model_->SetGroupParameterCovariance(group_parameter_prior_cov);

    // Handle variance model priors
    if (config_.rfx_variance_prior_shape <= 0.0) {
      config_.rfx_variance_prior_shape = 1.0;
    }
    if (config_.rfx_variance_prior_scale <= 0.0) {
      config_.rfx_variance_prior_scale = 1.0;
    }
    random_effects_model_->SetVariancePriorShape(config_.rfx_variance_prior_shape);
    random_effects_model_->SetVariancePriorScale(config_.rfx_variance_prior_scale);

    // Set has_random_effects_ flag to true so that the sampler will perform random effects updates at each iteration
    has_random_effects_ = true;

    // Continuation: warm-start the rfx model from the last retained sample. ResetFromSample restores
    // the SAMPLED state (working parameter alpha, group parameters xi, group-parameter covariance
    // sigma) from the persisted container; the FIXED priors were just set above from config. The
    // tracker ResetFromSample then swaps the last sample's rfx contribution into the residual.
    // Mirrors BART + BCF RestoreStateFromGFRSnapshot.
    if (continuation) {
      int last_rfx_idx = samples.rfx_container->NumSamples() - 1;
      random_effects_model_->ResetFromSample(*samples.rfx_container, last_rfx_idx);
      random_effects_tracker_->ResetFromSample(*random_effects_model_, *random_effects_dataset_, *residual_);
    }
  }

  // RNG
  rng_ = std::mt19937(config_.random_seed >= 0 ? config_.random_seed : std::random_device{}());

  // Other internal model state
  if (continuation) {
    // Warm-start the scalar state from the last retained sample. Continuation now appends in place
    // onto the model's samples object, whose global variance history is POST-PROCESSED (x y_std^2)
    // and whose tau_0 history is scaled by y_std; the sampler works in standardized space, so divide
    // those back out here. Leaf scales are stored standardized (postprocess does not touch them).
    const double y_std2 = samples.y_std * samples.y_std;
    if (sample_sigma2_global_ && !samples.global_error_variance_samples.empty()) {
      global_variance_ = samples.global_error_variance_samples.back() / y_std2;
    } else {
      global_variance_ = config_.sigma2_global_init;
    }
    if (sample_sigma2_leaf_mu_ && !samples.leaf_scale_mu_samples.empty()) {
      leaf_scale_mu_ = samples.leaf_scale_mu_samples.back();
    } else {
      leaf_scale_mu_ = config_.sigma2_mu_init;
    }
    if (sample_sigma2_leaf_tau_ && !samples.leaf_scale_tau_samples.empty()) {
      leaf_scale_tau_ = samples.leaf_scale_tau_samples.back();
    } else {
      leaf_scale_tau_ = config_.sigma2_tau_init;
    }
    leaf_scale_tau_multivariate_ = config_.sigma2_leaf_tau_matrix;
    // Sync the leaf models' scales with the warm-started leaf scales so the first continued
    // iteration samples each forest using the last retained scale (matching the one-shot run).
    mu_leaf_model_.SetScale(leaf_scale_mu_);
    std::visit(ScaleUpdateVisitor{*this, leaf_scale_tau_}, tau_leaf_model_);
    // Restore the treatment intercept and remove its contribution from the residual.
    // After the forest warm-starts above, residual_ = y_std - mu_last - Z*tau_last; the one-shot
    // sampler additionally carries -Z*tau_0_last at the start of the next iteration.
    if (sample_tau_0_ && !samples.tau_0_samples.empty()) {
      double* resid_ptr = residual_->GetData().data();
      if (data_.treatment_dim == 1) {
        tau_0_scalar_ = samples.tau_0_samples.back() / samples.y_std;  // stored x y_std -> standardized
        const double* basis = adaptive_coding_ ? tau_basis_vector_train_.data() : data_.treatment_train;
        for (int i = 0; i < data_.n_train; i++) {
          resid_ptr[i] -= tau_0_scalar_ * basis[i];
        }
      } else {
        // Multivariate treatment: the last sample is the final treatment_dim block (col-major),
        // stored x y_std. Adaptive coding is binary-only, so the basis is the raw treatment here.
        const int last = samples.num_samples - 1;
        tau_0_vector_.resize(data_.treatment_dim);
        for (int k = 0; k < data_.treatment_dim; k++) {
          tau_0_vector_[k] = samples.tau_0_samples[last * data_.treatment_dim + k] / samples.y_std;
        }
        for (int i = 0; i < data_.n_train; i++) {
          for (int k = 0; k < data_.treatment_dim; k++) {
            resid_ptr[i] -= data_.treatment_train[k * data_.n_train + i] * tau_0_vector_[k];
          }
        }
      }
    }
    // Probit continuation: the latent outcome is not persisted (re-drawn each iteration). Its
    // regeneration is deferred to RegenerateProbitLatent(), which the wrappers call AFTER SetRngState
    // so the draw comes from the resumed (or user-re-seeded) stream rather than this pre-seed RNG.
  } else {
    global_variance_ = config_.sigma2_global_init;
    leaf_scale_mu_ = config_.sigma2_mu_init;
    leaf_scale_tau_ = config_.sigma2_tau_init;
    leaf_scale_tau_multivariate_ = config_.sigma2_leaf_tau_matrix;
  }

  tau_raw_sum_preds_.assign(data_.n_train * data_.treatment_dim, 0.0);

  initialized_ = true;
}

void BCFSampler::RegenerateProbitLatent(BCFSamples& samples) {
  // No-op unless this is a probit model. Called by the continuation wrappers after SetRngState so the
  // fresh latent draw comes from the resumed (or user-re-seeded) stream. Regenerates the latent from
  // (y, warm-started model predictions mu + Z*tau + rfx + tau_0*Z), exactly as one MCMC iteration
  // does, placing the residual in a valid, stationary state before the first continued draw (which is
  // retained when num_burnin == 0).
  if (config_.link_function != LinkFunction::Probit) return;
  AddModelTermsForProbit(model_preds_.data(), mu_forest_tracker_.get(), tau_forest_tracker_.get(),
                         has_random_effects_ ? random_effects_tracker_.get() : nullptr, data_.n_train);
  if (sample_tau_0_) {
    if (data_.treatment_dim > 1) {
      for (int i = 0; i < data_.n_train; i++) {
        for (int k = 0; k < data_.treatment_dim; k++) {
          model_preds_[i] += data_.treatment_train[data_.n_train * k + i] * tau_0_vector_[k];
        }
      }
    } else {
      const double* treatment_ptr = adaptive_coding_ ? tau_basis_vector_train_.data() : data_.treatment_train;
      for (int i = 0; i < data_.n_train; i++) {
        model_preds_[i] += tau_0_scalar_ * treatment_ptr[i];
      }
    }
  }
  sample_probit_latent_outcome(rng_, outcome_raw_->GetData().data(), model_preds_.data(),
                               residual_->GetData().data(), samples.y_bar, data_.n_train);
}

void BCFSampler::run_gfr(BCFSamples& samples, int num_gfr, bool keep_gfr, int num_chains) {
  // Reserve space for GFR predictions if they are to be retained
  if (keep_gfr) {
    samples.mu_forest_predictions_train.reserve(data_.n_train * num_gfr);
    samples.tau_forest_predictions_train.reserve(data_.n_train * data_.treatment_dim * num_gfr);
    if (has_variance_forest_) {
      samples.variance_forest_predictions_train.reserve(data_.n_train * num_gfr);
    }
  }

  // NOTE: for serial sampling (which is all we currently support), chain 1 uses the live sampler state after
  // the GFR loop (i.e. the state after GFR iteration num_gfr-1). Chains 2..N each need their own earlier
  // GFR starting point so that the chains are initialized from distinct states.
  // We save exactly num_chains-1 snapshots, one per "extra" chain:
  //   gfr_snapshots_[k] = state after GFR iteration (num_gfr - num_chains + k), for k = 0..num_chains-2.
  // The last GFR iteration (i = num_gfr-1) is NOT snapshotted because chain 1 uses the live state.
  // Chain j (1-indexed, j >= 2) uses gfr_snapshots_[num_chains-j] = state after GFR[num_gfr-j].
  // If num_chains is 1, we keep no snapshots.
  int snapshot_start = (num_chains > 1) ? std::max(0, num_gfr - num_chains) : num_gfr;
  if (num_chains > 1) {
    gfr_snapshots_.clear();
    gfr_snapshots_.reserve(num_chains - 1);
  }

  if (config_.verbose && num_gfr > 0) {
    Log::Info("Running GFR sampler (%d iterations)", num_gfr);
  }
  const int gfr_report_every = std::max(1, num_gfr / 10);
  bool write_snapshot = false;
  for (int i = 0; i < num_gfr; i++) {
    // Do not snapshot the final GFR iteration: chain 1 uses the live sampler state directly.
    write_snapshot = (i >= snapshot_start) && (i < num_gfr - 1);
    RunOneIteration(samples, /*gfr=*/true, /*keep_sample=*/keep_gfr, /*write_snapshot=*/write_snapshot);
    if (config_.verbose && ((i + 1) % gfr_report_every == 0 || i + 1 == num_gfr)) {
      Log::Info("GFR: %d%% (%d/%d)", (100 * (i + 1)) / num_gfr, i + 1, num_gfr);
    }
  }
}

void BCFSampler::run_mcmc(BCFSamples& samples, int num_burnin, int keep_every, int num_mcmc) {
  // Reserve space for MCMC predictions if they are to be retained
  samples.mu_forest_predictions_train.reserve(data_.n_train * num_mcmc);
  samples.tau_forest_predictions_train.reserve(data_.n_train * data_.treatment_dim * num_mcmc);
  if (has_test_) {
    samples.mu_forest_predictions_test.reserve(data_.n_test * num_mcmc);
    samples.tau_forest_predictions_test.reserve(data_.n_test * data_.treatment_dim * num_mcmc);
  }
  if (has_variance_forest_) {
    samples.variance_forest_predictions_train.reserve(data_.n_train * num_mcmc);
    if (has_test_) {
      samples.variance_forest_predictions_test.reserve(data_.n_test * num_mcmc);
    }
  }

  // Create leaf models and pass them to the RunOneIteration function; these are updated in place and will reflect the current state of the leaf scale parameters (if they are being sampled)
  bool keep_forest = false;
  const int mcmc_total = num_burnin + keep_every * num_mcmc;
  const int mcmc_report_every = std::max(1, mcmc_total / 10);
  for (int i = 0; i < mcmc_total; i++) {
    if (i >= num_burnin && (i - num_burnin) % keep_every == 0)
      keep_forest = true;
    else
      keep_forest = false;
    RunOneIteration(samples, /*gfr=*/false, /*keep_sample=*/keep_forest, /*write_snapshot=*/false);
    if (config_.verbose && ((i + 1) % mcmc_report_every == 0 || i + 1 == mcmc_total)) {
      Log::Info("MCMC: %d%% (%d/%d)", (100 * (i + 1)) / mcmc_total, i + 1, mcmc_total);
    }
  }
}

void BCFSampler::run_mcmc_chains(BCFSamples& samples, int num_chains, int num_burnin, int keep_every, int num_mcmc) {
  for (int chain_idx = 0; chain_idx < num_chains; chain_idx++) {
    if (chain_idx > 0 && !gfr_snapshots_.empty()) {
      // Re-initialize the sampler state for each new chain.
      // gfr_snapshots_ holds num_chains-1 states (oldest-first): index 0 = GFR[num_gfr-num_chains],
      // index num_chains-2 = GFR[num_gfr-2].  Chain j (1-indexed, j>=2) uses index num_chains-j.
      // When num_gfr < num_chains we may not have enough distinct snapshots; in that case
      // fall back to running chains from whatever state is available (same behavior as R).
      int snapshot_idx = num_chains - 1 - chain_idx;
      if (snapshot_idx >= 0 && snapshot_idx < static_cast<int>(gfr_snapshots_.size())) {
        RestoreStateFromGFRSnapshot(samples, snapshot_idx);
      }
    }
    if (config_.verbose) {
      Log::Info("Running MCMC chain %d/%d (%d samples)", chain_idx + 1, num_chains, num_mcmc);
    }
    run_mcmc(samples, num_burnin, keep_every, num_mcmc);
  }
}

// `start_sample` is the first sample index to process: 0 for an initial run, or the pre-continuation
// sample count for a continuation, so only the newly-appended draws are scaled (the history is
// already on the original scale). Continuation supplies no test data (has_test_ == false), and the
// test block is additionally guarded on start_sample == 0.
void BCFSampler::postprocess_samples(BCFSamples& samples, int start_sample) {
  // Compute outcome predictions on the linear (link) scale: E[eta|X,Z] = mu(X) + Z*tau(X) + rfx
  // tau_forest_predictions stores raw tau(x) (no z multiplication), so we multiply by z here.
  // Callers that need probability-scale predictions (probit, cloglog) apply the inverse link themselves.
  samples.y_hat_train.resize(data_.n_train * samples.num_samples);
  double mu_term, tau_term, y_term;
  const int treatment_dim = data_.treatment_dim;
  for (int j = start_sample; j < samples.num_samples; j++) {
    for (int i = 0; i < data_.n_train; i++) {
      // Data index for the two terms that are guaranteed to be univariate - mu(x) and y_hat
      const int k = j * data_.n_train + i;
      mu_term = samples.mu_forest_predictions_train[k];
      if (treatment_dim > 1) {
        tau_term = 0;
        for (int treatment_idx = 0; treatment_idx < treatment_dim; treatment_idx++) {
          // Starting data index for multivariate treatment case, where tau(x) is col-major with dimensions (n_train, treatment_dim, num_samples)
          const int k_tau = j * data_.n_train * treatment_dim + data_.n_train * treatment_idx + i;
          tau_term += samples.tau_forest_predictions_train[k_tau] * data_.treatment_train[data_.n_train * treatment_idx + i];
        }
      } else {
        tau_term = samples.tau_forest_predictions_train[k] * data_.treatment_train[i];
      }
      y_term = mu_term + tau_term;
      if (has_random_effects_) y_term += samples.rfx_predictions_train[k];
      samples.y_hat_train[k] = y_term * samples.y_std + samples.y_bar;
    }
  }

  // Unpack test set predictions for mean and variance forest. Guarded on start_sample == 0: the
  // test block predicts from ALL retained forests and appends, so it only runs for an initial sample
  // (continuation supplies no test data and re-derives test predictions via predict()).
  if (has_test_) {
    // Recompute the FULL test-prediction trace (all retained forests) and assign (not append), so this
    // is correct on both an initial run and a continuation that re-supplies a (possibly new) test set.
    std::vector<double> mu_predictions = samples.mu_forests->Predict(*forest_dataset_test_);
    std::vector<double> tau_predictions = samples.tau_forests->PredictRaw(*forest_dataset_test_, /*row_major=*/false);
    // Add tau_0 to the treatment effect function predictions if it was sampled.
    // tau_0_samples layout: col-major (treatment dim k, sample j) -> j * treatment_dim + k.
    // For treatment_dim==1 this collapses to samples.tau_0_samples[j].
    // tau_predictions (from PredictRaw) is standardized. tau_0_samples is standardized for the NEW
    // draws (j >= start_sample; this call scales them to original below) but already original-scale
    // for the history (j < start_sample; scaled by a prior postprocess call). Divide the history
    // values by y_std so every tau_0 added here is in the standardized space of tau_predictions.
    if (sample_tau_0_) {
      const int treatment_dim = data_.treatment_dim;
      for (int j = 0; j < samples.num_samples; j++) {
        for (int k = 0; k < treatment_dim; k++) {
          double tau_0_val = samples.tau_0_samples[j * treatment_dim + k];
          if (j < start_sample) tau_0_val /= samples.y_std;
          for (int i = 0; i < data_.n_test; i++) {
            const int idx = j * data_.n_test * treatment_dim + data_.n_test * k + i;
            tau_predictions[idx] += tau_0_val;
          }
        }
      }
    }
    // Handle adaptive coding correctly:
    // When treatment is b_0 (1-Z) + b_1 Z, the conditional mean model:
    //      mu(x) + [tau_0 + tau(x)] * (b_0 * (1-Z) + b_1 * Z)
    // turns into
    //      [mu(x) + b_0 * (tau_0 + tau(x))] + (tau_0 + tau(x)) * (b_1 - b_0) * Z
    // So the treatment effect function that gets multiplied by Z is actually (b_1 - b_0) * (tau_0 + tau(x))
    // and the prognostic function has an added contribution of b_0 * (tau_0 + tau(x))
    if (adaptive_coding_) {
      for (int i = 0; i < samples.num_samples; i++) {
        double b_0 = samples.b0_samples[i];
        double b_1 = samples.b1_samples[i];
        for (int j = 0; j < data_.n_test; j++) {
          const int idx = i * data_.n_test + j;
          // Add b_0 * (tau_0 + tau(x)) to the prognostic function predictions
          mu_predictions[idx] += b_0 * tau_predictions[idx];
          // Scale tau_predictions by (b_1 - b_0)
          tau_predictions[idx] *= (b_1 - b_0);
        }
      }
    }
    samples.mu_forest_predictions_test = std::move(mu_predictions);
    samples.tau_forest_predictions_test = std::move(tau_predictions);
    if (has_variance_forest_) {
      samples.variance_forest_predictions_test = samples.variance_forests->Predict(*forest_dataset_test_);
    }
    if (has_random_effects_) {
      RandomEffectsDataset rfx_dataset_test;
      rfx_dataset_test.AddGroupLabels(data_.rfx_group_ids_test, data_.n_test);
      if (data_.rfx_basis_test != nullptr) {
        rfx_dataset_test.AddBasis(data_.rfx_basis_test, data_.n_test, data_.rfx_basis_dim, /*row_major=*/false);
      } else if (config_.rfx_model_spec == BCFRFXModelSpec::InterceptOnly) {
        std::vector<double> rfx_basis(data_.n_test, 1.0);
        rfx_dataset_test.AddBasis(rfx_basis.data(), data_.n_test, 1, /*row_major=*/false);
      } else if (config_.rfx_model_spec == BCFRFXModelSpec::InterceptPlusTreatment) {
        // Column-major rfx basis
        std::vector<double> rfx_basis(data_.n_test * (1 + data_.treatment_dim));
        for (int i = 0; i < data_.n_test; i++) {
          rfx_basis[i] = 1.0;
        }
        for (int j = 0; j < data_.treatment_dim; j++) {
          for (int i = 0; i < data_.n_test; i++) {
            rfx_basis[(j + 1) * data_.n_test + i] = data_.treatment_test[j * data_.n_test + i];
          }
        }
        rfx_dataset_test.AddBasis(rfx_basis.data(), data_.n_test, 1 + data_.treatment_dim, /*row_major=*/false);
      } else {
        Log::Fatal("BCF model random effects term was not sampled with intercept_only or intercept_plus_treatment specification, but not random effect basis was provided for prediction");
      }
      samples.rfx_predictions_test.resize(data_.n_test * samples.num_samples);
      samples.rfx_container->Predict(rfx_dataset_test, *samples.rfx_label_mapper, samples.rfx_predictions_test);
    }

    samples.y_hat_test.resize(data_.n_test * samples.num_samples);
    for (int j = 0; j < samples.num_samples; j++) {
      for (int i = 0; i < data_.n_test; i++) {
        // Data index for the two terms that are guaranteed to be univariate - mu(x) and y_hat
        const int k = j * data_.n_test + i;
        mu_term = samples.mu_forest_predictions_test[k];
        if (treatment_dim > 1) {
          tau_term = 0;
          for (int treatment_idx = 0; treatment_idx < treatment_dim; treatment_idx++) {
            // Starting data index for multivariate treatment case, where tau(x) is col-major with dimensions (n_test, treatment_dim, num_samples)
            const int k_tau = j * data_.n_test * treatment_dim + data_.n_test * treatment_idx + i;
            tau_term += samples.tau_forest_predictions_test[k_tau] * data_.treatment_test[data_.n_test * treatment_idx + i];
          }
        } else {
          tau_term = samples.tau_forest_predictions_test[k] * data_.treatment_test[i];
        }
        y_term = mu_term + tau_term;
        if (has_random_effects_) y_term += samples.rfx_predictions_test[k];
        samples.y_hat_test[k] = y_term * samples.y_std + samples.y_bar;
      }
    }
  } else if (!has_test_) {
    // No test set for this run (e.g. a continuation that did not re-supply a test set). Any test
    // predictions still on the samples object are from a prior run and are now stale (they cover
    // only the pre-continuation draws), so drop them. num_test was already reset in InitializeState.
    // (On an initial run without a test set these are already empty, so this is a no-op.)
    samples.mu_forest_predictions_test.clear();
    samples.tau_forest_predictions_test.clear();
    samples.variance_forest_predictions_test.clear();
    samples.rfx_predictions_test.clear();
    samples.y_hat_test.clear();
  }

  // Convert variance forest predictions and global error variance from
  // standardized space to original outcome scale.
  // - Train predictions come from ForestTracker::GetSumPredictions() (log-scale leaf sums),
  //   so apply exp() then multiply by y_std^2.
  // - Test predictions come from ForestContainer::Predict() with is_exponentiated_=true,
  //   which already applies exp() internally, so just multiply by y_std^2.
  // - Global error variance samples are in standardized space; multiply by y_std^2.
  // Train predictions + params are scaled only over the newly-appended range [start_sample, end);
  // test arrays are full-range (recomputed in full from all retained forests whenever a test set is
  // present, on both an initial run and a continuation).
  const size_t train_off = static_cast<size_t>(start_sample) * static_cast<size_t>(data_.n_train);
  const size_t tau_train_off = train_off * static_cast<size_t>(treatment_dim);
  if (has_variance_forest_) {
    double y_std2 = samples.y_std * samples.y_std;
    for (size_t i = train_off; i < samples.variance_forest_predictions_train.size(); i++)
      samples.variance_forest_predictions_train[i] = std::exp(samples.variance_forest_predictions_train[i]) * y_std2;
    for (double& v : samples.variance_forest_predictions_test) v *= y_std2;
  }
  if (sample_sigma2_global_) {
    double y_std2 = samples.y_std * samples.y_std;
    for (size_t i = static_cast<size_t>(start_sample); i < samples.global_error_variance_samples.size(); i++)
      samples.global_error_variance_samples[i] *= y_std2;
  }
  // Treatment intercept tau_0 is sampled in standardized space; scale it to the original outcome
  // scale for storage/readers, mirroring sigma2_global above and the other parametric terms (and the
  // R main-branch convention). predict() and the continuation warm-start convert it back to
  // standardized as needed. Runs after the test-prediction block above, which consumes the still-
  // standardized values. Layout is col-major (sample j, treatment dim k) -> j * treatment_dim + k.
  if (sample_tau_0_) {
    const size_t tau0_off = static_cast<size_t>(start_sample) * static_cast<size_t>(treatment_dim);
    for (size_t i = tau0_off; i < samples.tau_0_samples.size(); i++)
      samples.tau_0_samples[i] *= samples.y_std;
  }

  // Convert the cached prognostic / treatment-effect / random-effects predictions from standardized
  // space to the original outcome scale, matching the R main-branch cached attributes and predict():
  // the prognostic function mu(x) carries the location shift (y_bar); the treatment effect tau(x)
  // (already the full CATE tau_0 + tau(x), with any adaptive-coding (b1 - b0) factor folded in during
  // sampling) and the random effects carry only the scale factor (y_std). y_hat_train / y_hat_test
  // were already placed on the original scale above, computed from these caches while still standardized.
  for (size_t i = train_off; i < samples.mu_forest_predictions_train.size(); i++)
    samples.mu_forest_predictions_train[i] = samples.mu_forest_predictions_train[i] * samples.y_std + samples.y_bar;
  for (double& v : samples.mu_forest_predictions_test) v = v * samples.y_std + samples.y_bar;
  for (size_t i = tau_train_off; i < samples.tau_forest_predictions_train.size(); i++)
    samples.tau_forest_predictions_train[i] *= samples.y_std;
  for (double& v : samples.tau_forest_predictions_test) v *= samples.y_std;
  if (has_random_effects_) {
    for (size_t i = train_off; i < samples.rfx_predictions_train.size(); i++)
      samples.rfx_predictions_train[i] *= samples.y_std;
    for (double& v : samples.rfx_predictions_test) v *= samples.y_std;
  }
}

void BCFSampler::RunOneIteration(BCFSamples& samples, bool gfr, bool keep_sample, bool write_snapshot) {
  // Prognostic forest
  if (gfr) {
    GFRSampleOneIter<GaussianConstantLeafModel, GaussianConstantSuffStat>(
        *mu_forest_, *mu_forest_tracker_, *samples.mu_forests, mu_leaf_model_,
        *forest_dataset_, *residual_, *tree_prior_mu_, rng_,
        config_.var_weights_mu, config_.sweep_update_indices_mu, global_variance_, config_.feature_types,
        config_.cutpoint_grid_size, /*keep_forest=*/keep_sample,
        /*pre_initialized=*/true, /*backfitting=*/true,
        /*num_features_subsample=*/config_.num_features_subsample_mu, config_.num_threads);
  } else {
    MCMCSampleOneIter<GaussianConstantLeafModel, GaussianConstantSuffStat>(
        *mu_forest_, *mu_forest_tracker_, *samples.mu_forests, mu_leaf_model_,
        *forest_dataset_, *residual_, *tree_prior_mu_, rng_,
        config_.var_weights_mu, config_.sweep_update_indices_mu, global_variance_, /*keep_forest=*/keep_sample,
        /*pre_initialized=*/true, /*backfitting=*/true,
        /*num_threads=*/config_.num_threads);
  }

  // Parametric treatment intercept term
  if (sample_tau_0_) {
    SampleParametricTreatmentEffect();
  }

  // Treatment effect forest
  if (gfr) {
    std::visit(GFROneIterationVisitorTau{*this, samples, keep_sample}, tau_leaf_model_);
  } else {
    std::visit(MCMCOneIterationVisitorTau{*this, samples, keep_sample}, tau_leaf_model_);
  }

  // Update raw tau(x): sum leaf values across trees for each dimension of the tau leaf.
  // Uses node IDs already cached in the tracker — no tree traversal needed.
  // Stored col-major: tau_raw_sum_preds_[k * n_train + i] matches postprocess_samples indexing.
  const int tau_dim = data_.treatment_dim;
  const int data_dim = data_.n_train;
  double tau_0 = 0.0;
  for (int k = 0; k < tau_dim; k++) {
    if (sample_tau_0_) {
      if (data_.treatment_dim > 1) {
        tau_0 = tau_0_vector_[k];
      } else {
        tau_0 = tau_0_scalar_;
      }
    }
    for (int i = 0; i < data_dim; i++) {
      tau_raw_sum_preds_[k * data_dim + i] = tau_0;
      for (int j = 0; j < config_.num_trees_tau; j++) {
        data_size_t leaf = tau_forest_tracker_->GetNodeId(i, j);
        tau_raw_sum_preds_[k * data_dim + i] += tau_forest_->GetTree(j)->LeafValue(leaf, k);
      }
    }
  }

  // Adaptive coding parameters
  if (adaptive_coding_) {
    SampleAdaptiveCodingParameters();
  }

  // Variance forest term
  if (has_variance_forest_) {
    if (gfr) {
      GFRSampleOneIter<LogLinearVarianceLeafModel, LogLinearVarianceSuffStat>(
          *variance_forest_, *variance_forest_tracker_, *samples.variance_forests, variance_leaf_model_,
          *forest_dataset_, *residual_, *tree_prior_variance_, rng_,
          config_.var_weights_variance, config_.sweep_update_indices_variance, global_variance_, config_.feature_types,
          config_.cutpoint_grid_size, /*keep_forest=*/keep_sample,
          /*pre_initialized=*/true, /*backfitting=*/false,
          /*num_features_subsample=*/config_.num_features_subsample_variance, config_.num_threads);
    } else {
      MCMCSampleOneIter<LogLinearVarianceLeafModel, LogLinearVarianceSuffStat>(
          *variance_forest_, *variance_forest_tracker_, *samples.variance_forests, variance_leaf_model_,
          *forest_dataset_, *residual_, *tree_prior_variance_, rng_,
          config_.var_weights_variance, config_.sweep_update_indices_variance, global_variance_, /*keep_forest=*/keep_sample,
          /*pre_initialized=*/true, /*backfitting=*/false,
          /*num_threads=*/config_.num_threads);
    }
  }

  // Latent continuous outcome for probit link
  if (config_.link_function == LinkFunction::Probit) {
    // Add mu(x) + Z*tau(x) + rfx to model_preds_ for each training observation, then sample the latent outcome given the observed binary outcome and the model prediction
    AddModelTermsForProbit(model_preds_.data(), mu_forest_tracker_.get(), tau_forest_tracker_.get(), random_effects_tracker_.get(), data_.n_train);
    // If tau_0 is sampled, then add it to model_preds_ as well
    if (sample_tau_0_) {
      if (data_.treatment_dim > 1) {
        for (int i = 0; i < data_.n_train; i++) {
          for (int k = 0; k < data_.treatment_dim; k++) {
            model_preds_[i] += data_.treatment_train[data_.n_train * k + i] * tau_0_vector_[k];
          }
        }
      } else {
        const double* treatment_ptr = adaptive_coding_ ? tau_basis_vector_train_.data() : data_.treatment_train;
        for (int i = 0; i < data_.n_train; i++) {
          model_preds_[i] += tau_0_scalar_ * treatment_ptr[i];
        }
      }
    }
    // Sample latent outcome into outcome_raw_ (overwriting the previous iteration's raw predictions, which are not needed for the probit likelihood)
    sample_probit_latent_outcome(rng_, outcome_raw_->GetData().data(), model_preds_.data(),
                                 residual_->GetData().data(), samples.y_bar, data_.n_train);
  }

  // Global error scale
  if (sample_sigma2_global_) {
    global_variance_ = var_model_->SampleVarianceParameter(
        residual_->GetData(), config_.a_sigma2_global, config_.b_sigma2_global, rng_);
  }

  // Prognostic forest leaf scale
  if (sample_sigma2_leaf_mu_) {
    leaf_scale_mu_ = leaf_scale_model_mu_->SampleVarianceParameter(
        mu_forest_.get(), config_.a_sigma2_mu, config_.b_sigma2_mu, rng_);
    mu_leaf_model_.SetScale(leaf_scale_mu_);
  }

  // Treatment effect forest leaf scale
  if (sample_sigma2_leaf_tau_) {
    leaf_scale_tau_ = leaf_scale_model_tau_->SampleVarianceParameter(
        tau_forest_.get(), config_.a_sigma2_tau, config_.b_sigma2_tau, rng_);
    std::visit(ScaleUpdateVisitor{*this, leaf_scale_tau_}, tau_leaf_model_);
  }

  // Gibbs updates for random effects model
  if (has_random_effects_) {
    random_effects_model_->SampleRandomEffects(*random_effects_dataset_, *residual_, *random_effects_tracker_, global_variance_, rng_);
    // NOTE: we keep this code in the random effects sampling block (as opposed to the parameter / prediction storage block below) to mirror the way that forests are retained within a sampling step
    if (keep_sample) {
      samples.rfx_container->AddSample(*random_effects_model_);
    }
  }

  if (keep_sample) {
    // Add parameter and prediction samples
    samples.num_samples++;
    // Global error variance
    if (sample_sigma2_global_) samples.global_error_variance_samples.push_back(global_variance_);
    // Prognostic forest leaf scale
    if (sample_sigma2_leaf_mu_) samples.leaf_scale_mu_samples.push_back(leaf_scale_mu_);
    // Treatment effect forest leaf scale
    if (sample_sigma2_leaf_tau_) samples.leaf_scale_tau_samples.push_back(leaf_scale_tau_);
    // Treatment intercept
    if (sample_tau_0_) {
      if (data_.treatment_dim > 1) {
        samples.tau_0_samples.insert(samples.tau_0_samples.end(), tau_0_vector_.begin(), tau_0_vector_.end());
      } else {
        samples.tau_0_samples.push_back(tau_0_scalar_);
      }
    }
    // Adaptive coding parameters
    if (adaptive_coding_) {
      samples.b0_samples.push_back(b_0_);
      samples.b1_samples.push_back(b_1_);
    }
    // Prognostic and treatment forest predictions
    double* mu_forest_preds_train = mu_forest_tracker_->GetSumPredictions();
    if (adaptive_coding_) {
      // TODO: refactor this or at least cache to avoid unnecessary malloc
      std::vector<double> mu_adj(data_.n_train);
      std::vector<double> tau_adj(data_.n_train);
      for (int i = 0; i < data_.n_train; i++) {
        mu_adj[i] = mu_forest_preds_train[i] + b_0_ * tau_raw_sum_preds_[i];
        tau_adj[i] = tau_raw_sum_preds_[i] * (b_1_ - b_0_);
      }
      samples.mu_forest_predictions_train.insert(samples.mu_forest_predictions_train.end(),
                                                 mu_adj.begin(), mu_adj.end());
      samples.tau_forest_predictions_train.insert(samples.tau_forest_predictions_train.end(),
                                                  tau_adj.begin(), tau_adj.end());
    } else {
      samples.mu_forest_predictions_train.insert(samples.mu_forest_predictions_train.end(),
                                                 mu_forest_preds_train,
                                                 mu_forest_preds_train + samples.num_train);
      samples.tau_forest_predictions_train.insert(samples.tau_forest_predictions_train.end(),
                                                  tau_raw_sum_preds_.begin(), tau_raw_sum_preds_.end());
    }
    // Variance forest predictions
    if (has_variance_forest_) {
      double* variance_forest_preds_train = variance_forest_tracker_->GetSumPredictions();
      samples.variance_forest_predictions_train.insert(samples.variance_forest_predictions_train.end(),
                                                       variance_forest_preds_train,
                                                       variance_forest_preds_train + samples.num_train);
    }
    // Random effects predictions
    if (has_random_effects_) {
      for (int i = 0; i < data_.n_train; i++) {
        samples.rfx_predictions_train.push_back(random_effects_tracker_->GetPrediction(i));
      }
    }
  }

  if (write_snapshot) {
    GFRSnapshot snap;
    // Forests
    snap.mu_forest = std::make_unique<TreeEnsemble>(*mu_forest_);
    snap.tau_forest = std::make_unique<TreeEnsemble>(*tau_forest_);
    if (has_variance_forest_) snap.variance_forest = std::make_unique<TreeEnsemble>(*variance_forest_);
    // Scale parameters
    snap.sigma2 = global_variance_;
    snap.leaf_scale_mu = leaf_scale_mu_;
    if (config_.tau_leaf_model_type == MeanLeafModelType::GaussianMultivariateRegression) {
      snap.leaf_scale_tau_multivariate = leaf_scale_tau_multivariate_;
    } else if (config_.tau_leaf_model_type == MeanLeafModelType::GaussianUnivariateRegression) {
      snap.leaf_scale_tau = leaf_scale_tau_;
    }
    // Treatment intercept
    if (sample_tau_0_) {
      if (data_.treatment_dim > 1) {
        snap.tau_0_vector = tau_0_vector_;
      } else {
        snap.tau_0_scalar = tau_0_scalar_;
      }
    }
    // Adaptive coding
    if (adaptive_coding_) {
      snap.b_0 = b_0_;
      snap.b_1 = b_1_;
    }
    // Residual
    snap.residual.clear();
    snap.residual.resize(data_.n_train);
    snap.residual.assign(residual_->GetData().data(), residual_->GetData().data() + data_.n_train);
    // Variance weights (from variance forest)
    if (has_variance_forest_) {
      snap.variance_weights.clear();
      snap.variance_weights.resize(data_.n_train);
      snap.variance_weights.assign(forest_dataset_->GetVarWeights().data(), forest_dataset_->GetVarWeights().data() + data_.n_train);
    }
    // Random effects terms
    if (config_.has_random_effects) {
      snap.rfx_working_parameter = random_effects_model_->GetWorkingParameter();
      snap.rfx_group_parameters = random_effects_model_->GetGroupParameters();
      snap.rfx_group_parameter_covariance = random_effects_model_->GetGroupParameterCovariance();
      snap.rfx_working_parameter_covariance = random_effects_model_->GetWorkingParameterCovariance();
      snap.rfx_variance_prior_shape = random_effects_model_->GetVariancePriorShape();
      snap.rfx_variance_prior_scale = random_effects_model_->GetVariancePriorScale();
    }
    gfr_snapshots_.push_back(std::move(snap));
  }
}

void BCFSampler::RestoreStateFromGFRSnapshot(BCFSamples& samples, int snapshot_index) {
  GFRSnapshot& snap = gfr_snapshots_[snapshot_index];

  // Remove the contribution of tau_0 from the residual before all forest-based state restoration, which modifies the residual in-place
  if (sample_tau_0_) {
    double* resid_ptr = residual_->GetData().data();
    if (data_.treatment_dim == 1) {
      const double* previous_basis = adaptive_coding_ ? tau_basis_vector_train_.data() : data_.treatment_train;
      for (int i = 0; i < data_.n_train; i++) {
        resid_ptr[i] += tau_0_scalar_ * previous_basis[i];
      }
    } else {
      for (int i = 0; i < data_.n_train; i++) {
        for (int k = 0; k < data_.treatment_dim; k++) {
          resid_ptr[i] += data_.treatment_train[k * data_.n_train + i] * tau_0_vector_[k];
        }
      }
    }
  }

  // Restore mu and tau forest state
  // Prognostic forest
  mu_forest_->ReconstituteFromForest(*snap.mu_forest);
  mu_forest_tracker_->ReconstituteFromForest(*snap.mu_forest, *forest_dataset_, *residual_, true);
  mu_forest_tracker_->UpdatePredictions(mu_forest_.get(), *forest_dataset_.get());

  // Adaptive coding parameters and their implied basis
  if (adaptive_coding_) {
    b_0_ = snap.b_0;
    b_1_ = snap.b_1;
    for (int i = 0; i < data_.n_train; i++) {
      double z = data_.treatment_train[i];
      tau_basis_vector_train_[i] = b_0_ * (1 - z) + b_1_ * z;
    }
    forest_dataset_->UpdateBasis(tau_basis_vector_train_.data(), /*num_row=*/data_.n_train, /*num_col=*/1, /*row_major=*/false);
    if (has_test_ && data_.treatment_test != nullptr) {
      for (int i = 0; i < data_.n_test; i++) {
        double z = data_.treatment_test[i];
        tau_basis_vector_test_[i] = b_0_ * (1 - z) + b_1_ * z;
      }
      forest_dataset_test_->UpdateBasis(tau_basis_vector_test_.data(), /*num_row=*/data_.n_test, /*num_col=*/1, /*row_major=*/false);
    }
  }

  // Treatment intercept
  if (sample_tau_0_) {
    if (data_.treatment_dim > 1) {
      tau_0_vector_ = snap.tau_0_vector;
    } else {
      tau_0_scalar_ = snap.tau_0_scalar;
    }
  }

  // Remove tau_0 from residual
  if (sample_tau_0_) {
    double* resid_ptr = residual_->GetData().data();
    if (data_.treatment_dim == 1) {
      const double* current_basis = adaptive_coding_ ? tau_basis_vector_train_.data() : data_.treatment_train;
      for (int i = 0; i < data_.n_train; i++) {
        resid_ptr[i] -= tau_0_scalar_ * current_basis[i];
      }
    } else {
      for (int i = 0; i < data_.n_train; i++) {
        for (int k = 0; k < data_.treatment_dim; k++) {
          resid_ptr[i] -= data_.treatment_train[k * data_.n_train + i] * tau_0_vector_[k];
        }
      }
    }
  }

  // Treatment effect forest
  std::visit(TauForestResetVisitor{*this, samples, *snap.tau_forest}, tau_leaf_model_);

  // Initialize variance forest state (if present)
  if (config_.num_trees_variance > 0) {
    variance_leaf_model_ = LogLinearVarianceLeafModel(config_.shape_variance_forest, config_.scale_variance_forest);
    variance_forest_ = std::make_unique<TreeEnsemble>(config_.num_trees_variance, config_.leaf_dim_variance, config_.leaf_constant_variance, config_.exponentiated_leaf_variance);
    samples.variance_forests = std::make_unique<ForestContainer>(config_.num_trees_variance, config_.leaf_dim_variance, config_.leaf_constant_variance, config_.exponentiated_leaf_variance);
    variance_forest_tracker_ = std::make_unique<ForestTracker>(forest_dataset_->GetCovariates(), config_.feature_types, config_.num_trees_variance, data_.n_train);
    tree_prior_variance_ = std::make_unique<TreePrior>(config_.alpha_variance, config_.beta_variance, config_.min_samples_leaf_variance, config_.max_depth_variance);
    // Leaf values for the log-linear variance model are on the log scale; the ensemble sums
    // log(sigma^2_i) contributions, so each tree starts at log(init_val) / num_trees.
    variance_forest_->SetLeafValue(std::log(init_val_variance_) / config_.num_trees_variance);
    variance_forest_tracker_->UpdatePredictions(variance_forest_.get(), *forest_dataset_.get());
    // UpdateVarModelTree (called inside GFRSampleOneIter / MCMCSampleOneIter) unconditionally
    // reads and writes the dataset variance weight slot via VarWeightValue / SetVarWeightValue.
    // This slot tracks the cumulative per-observation variance prediction
    // (sigma^2_i = exp(sum of tree leaf values)) and is incompatible with case weights, which
    // would need to be reapplied after every per-tree update. The R/Python APIs enforce this
    // as a hard error; guard here for callers that use BARTSampler directly.
    if (forest_dataset_->HasVarWeights()) {
      Log::Fatal("observation_weights and a variance forest cannot be used together.");
    }
    std::vector<double> initial_variance_preds(data_.n_train, init_val_variance_);
    forest_dataset_->AddVarianceWeights(initial_variance_preds.data(), data_.n_train);
    has_variance_forest_ = true;
  }

  // Random effects model
  if (config_.has_random_effects) {
    // Restore "working" parameter prior mean
    random_effects_model_->SetWorkingParameter(snap.rfx_working_parameter);

    // Restore "group" parameter prior mean
    random_effects_model_->SetGroupParameters(snap.rfx_group_parameters);

    // Restore "working" parameter prior covariance
    random_effects_model_->SetWorkingParameterCovariance(snap.rfx_working_parameter_covariance);

    // Restore "group" parameter prior covariance
    random_effects_model_->SetGroupParameterCovariance(snap.rfx_group_parameter_covariance);

    // Restore variance model priors
    random_effects_model_->SetVariancePriorShape(snap.rfx_variance_prior_shape);
    random_effects_model_->SetVariancePriorScale(snap.rfx_variance_prior_scale);

    // Swap the chain-N RFX contribution out of the residual and the GFR-snapshot
    // contribution in, exactly as the R sampler does via resetRandomEffectsTracker.
    // At this point residual_ = y - f_gfr - rfx_chain_N (forest already swapped above),
    // so ResetFromSample produces: residual_ += rfx_chain_N - rfx_gfr = y - f_gfr - rfx_gfr.
    random_effects_tracker_->ResetFromSample(*random_effects_model_, *random_effects_dataset_, *residual_);
  }

  // Other internal model state
  global_variance_ = snap.sigma2;
  leaf_scale_mu_ = snap.leaf_scale_mu;
  if (config_.tau_leaf_model_type == MeanLeafModelType::GaussianMultivariateRegression) {
    leaf_scale_tau_multivariate_ = snap.leaf_scale_tau_multivariate;
  } else if (config_.tau_leaf_model_type == MeanLeafModelType::GaussianUnivariateRegression || config_.tau_leaf_model_type == MeanLeafModelType::GaussianConstant) {
    leaf_scale_tau_ = snap.leaf_scale_tau;
  }
}

void BCFSampler::SampleParametricTreatmentEffect() {
  // Determine whether treatment is univariate, bivariate, or multivariate
  const int tau_dim = data_.treatment_dim;
  const int n_train = data_.n_train;
  if (tau_dim == 1) {
    // Dispatch univariate specialization of regression sampler
    // Add tau_0 * Z to residual to get partial residual
    double* partial_resid_ptr = residual_->GetData().data();
    double* treatment_ptr = adaptive_coding_ ? tau_basis_vector_train_.data() : data_.treatment_train;
    for (int i = 0; i < n_train; i++) {
      partial_resid_ptr[i] += treatment_ptr[i] * tau_0_scalar_;
    }
    // Sample tau_0 via sample_univariate_gaussian_regression_coefficient
    double tau_0_update = sample_univariate_gaussian_regression_coefficient(partial_resid_ptr, treatment_ptr, global_variance_, config_.tau_0_prior_var_scalar, n_train, rng_);
    tau_0_scalar_ = tau_0_update;
    // Subtract tau_0 * Z from partial residual
    for (int i = 0; i < n_train; i++) {
      partial_resid_ptr[i] -= treatment_ptr[i] * tau_0_scalar_;
    }
  } else if (tau_dim == 2) {
    // Add tau_0 * Z to residual to get partial residual
    double* partial_resid_ptr = residual_->GetData().data();
    for (int i = 0; i < n_train; i++) {
      for (int k = 0; k < 2; k++) {
        partial_resid_ptr[i] += data_.treatment_train[k * n_train + i] * tau_0_vector_[k];
      }
    }
    // Dispatch bivariate specialization of regression sampler (with diagonal covariance)
    std::vector<double> tau_0_update(2, 0.0);
    sample_diagonal_bivariate_gaussian_regression_coefficients(tau_0_update.data(), partial_resid_ptr, data_.treatment_train, data_.treatment_train + n_train, global_variance_, config_.tau_0_prior_var_multivariate[0], config_.tau_0_prior_var_multivariate[1], n_train, rng_);
    // Push results back to tau_0_vector_
    tau_0_vector_[0] = tau_0_update[0];
    tau_0_vector_[1] = tau_0_update[1];
    // Subtract tau_0 * Z from partial residual
    for (int i = 0; i < n_train; i++) {
      for (int k = 0; k < 2; k++) {
        partial_resid_ptr[i] -= data_.treatment_train[k * n_train + i] * tau_0_vector_[k];
      }
    }
  } else {
    // Dispatch general-purpose multivariate regression sampler, which returns parameters as an Eigen::VectorXd
    // Add tau_0 * Z to residual to get partial residual
    double* partial_resid_ptr = residual_->GetData().data();
    for (int i = 0; i < n_train; i++) {
      for (int k = 0; k < tau_dim; k++) {
        partial_resid_ptr[i] += data_.treatment_train[k * n_train + i] * tau_0_vector_[k];
      }
    }
    // Wrap an Eigen map around the partial residual, treatment, and prior covariance for efficient vectorized operations
    Eigen::Map<Eigen::VectorXd> partial_resid(partial_resid_ptr, n_train);
    Eigen::Map<const Eigen::MatrixXd> treatment(data_.treatment_train, n_train, tau_dim);
    // Construct diagonal prior covariance matrix from config_.tau_0_prior_var_multivariate
    Eigen::Map<const Eigen::VectorXd> tau_0_prior_var_vec(config_.tau_0_prior_var_multivariate.data(), tau_dim);
    const Eigen::MatrixXd tau_0_prior_cov = tau_0_prior_var_vec.asDiagonal();
    // Sample tau_0 via sample_general_gaussian_regression_coefficients
    Eigen::VectorXd tau_0_update = sample_general_gaussian_regression_coefficients(partial_resid, treatment, global_variance_, tau_0_prior_cov, n_train, rng_);
    // Push results back to tau_0_vector_
    for (int k = 0; k < tau_dim; k++) {
      tau_0_vector_[k] = tau_0_update[k];
    }
    // Subtract tau_0 * Z from partial residual
    for (int i = 0; i < n_train; i++) {
      for (int k = 0; k < tau_dim; k++) {
        partial_resid_ptr[i] -= data_.treatment_train[k * n_train + i] * tau_0_vector_[k];
      }
    }
  }
}

void BCFSampler::SampleAdaptiveCodingParameters() {
  // Extract data dimensions and pointers
  const int n = data_.n_train;
  double* resid_ptr = residual_->GetData().data();
  double* treatment_ptr = data_.treatment_train;

  // Add [b_0 * (1-Z) + b_1 * Z] * tau(x) to residual to get partial residual
  std::vector<double> partial_resid(n, 0.0);
  for (int i = 0; i < n; i++) {
    partial_resid[i] = resid_ptr[i] + tau_raw_sum_preds_[i] * tau_basis_vector_train_[i];
  }

  // Compute sufficient statistics for b_0 and b_1
  double xtx_control = 0.0;    // sum of squared regression basis for control group: sum of (1-Z)^2 * tau(x)^2
  double xtx_treatment = 0.0;  // sum of squared regression basis for treatment group: sum of Z^2 * tau(x)^2
  double xty_control = 0.0;    // sum of (1-Z) * tau(x) * y
  double xty_treatment = 0.0;  // sum of Z * tau(x) * y
  for (int i = 0; i < n; i++) {
    double x_i = tau_raw_sum_preds_[i];
    double y_i = partial_resid[i];
    double z_i = treatment_ptr[i];
    if (z_i == 0.0) {
      xtx_control += x_i * x_i;
      xty_control += x_i * y_i;
    } else if (z_i == 1.0) {
      xtx_treatment += x_i * x_i;
      xty_treatment += x_i * y_i;
    }
  }

  // Perform regression Gibbs update for b_0 and b_1
  // We use a fixed prior of b_0, b_1 ~ N(0, 1/2) (independent across b_0 and b_1)
  const double prior_var = 0.5;
  double posterior_var_control = global_variance_ / (xtx_control + (global_variance_ / prior_var));
  double posterior_var_treatment = global_variance_ / (xtx_treatment + (global_variance_ / prior_var));
  double posterior_mean_control = xty_control / (xtx_control + (global_variance_ / prior_var));
  double posterior_mean_treatment = xty_treatment / (xtx_treatment + (global_variance_ / prior_var));
  b_0_ = sample_standard_normal(posterior_mean_control, std::sqrt(posterior_var_control), rng_);
  b_1_ = sample_standard_normal(posterior_mean_treatment, std::sqrt(posterior_var_treatment), rng_);

  // Update basis
  std::vector<double> prev_tau_basis = tau_basis_vector_train_;
  for (int i = 0; i < n; i++) {
    double z = treatment_ptr[i];
    tau_basis_vector_train_[i] = (b_0_ * (1.0 - z) + b_1_ * z);
  }
  forest_dataset_->UpdateBasis(tau_basis_vector_train_.data(), n, 1, false);
  if (has_test_ && data_.treatment_test != nullptr) {
    for (int i = 0; i < data_.n_test; i++) {
      double z = data_.treatment_test[i];
      tau_basis_vector_test_[i] = (b_0_ * (1.0 - z) + b_1_ * z);
    }
    forest_dataset_test_->UpdateBasis(tau_basis_vector_test_.data(), data_.n_test, 1, false);
  }

  // Propagate basis changes through to the trackers
  UpdateResidualNewBasis(*tau_forest_tracker_, *forest_dataset_, *residual_, tau_forest_.get());

  // If a tau_0 treatment intercept term is sampled, we must also subtract tau_0 * (new_basis - old_basis) from the residual
  if (sample_tau_0_) {
    for (int i = 0; i < n; i++) {
      resid_ptr[i] -= tau_0_scalar_ * (tau_basis_vector_train_[i] - prev_tau_basis[i]);
    }
  }
}

}  // namespace StochTree
