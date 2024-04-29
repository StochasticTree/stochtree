/*!
 * Copyright (c) 2024 stochtree authors.
 * 
 * High-level C++ API for BART and BCF
 */
#ifndef STOCHTREE_CPP_API_H_
#define STOCHTREE_CPP_API_H_

#include <Eigen/Dense>
#include <stochtree/container.h>
#include <stochtree/data.h>
#include <stochtree/leaf_model.h>
#include <stochtree/log.h>
#include <stochtree/meta.h>
#include <stochtree/tree_sampler.h>
#include <stochtree/variance_model.h>
#include <functional>
#include <type_traits>
#include <vector>

namespace StochTree {

struct BCFParameters {
  int cutpoint_grid_size;
  double sigma_leaf_mu;
  double sigma_leaf_tau;
  Eigen::MatrixXd sigma_leaf_tau_mat;
  double alpha_mu;
  double alpha_tau;
  double beta_mu;
  double beta_tau;
  int min_samples_leaf_mu;
  int min_samples_leaf_tau;
  double nu;
  double lamb;
  double a_leaf_mu;
  double a_leaf_tau;
  double b_leaf_mu;
  double b_leaf_tau;
  double sigma2;
  int num_trees_mu;
  int num_trees_tau;
  double b1;
  double b0;
  int num_gfr;
  int num_burnin;
  int num_mcmc;
  std::vector<FeatureType> feature_types_mu;
  std::vector<FeatureType> feature_types_tau;
  double leaf_init_mu;
  double leaf_init_tau;
  
  BCFParameters(int cutpoint_grid_size, double sigma_leaf_mu, double sigma_leaf_tau, 
                double alpha_mu, double alpha_tau, double beta_mu, double beta_tau, 
                int min_samples_leaf_mu, int min_samples_leaf_tau, double nu, double lamb, 
                double a_leaf_mu, double a_leaf_tau, double b_leaf_mu, double b_leaf_tau, 
                double sigma2, int num_trees_mu, int num_trees_tau, double b1, double b0, 
                std::vector<FeatureType>& feature_types_mu, std::vector<FeatureType>& feature_types_tau, 
                int num_gfr, int num_burnin, int num_mcmc, double leaf_init_mu, double leaf_init_tau) {
    cutpoint_grid_size = cutpoint_grid_size;
    sigma_leaf_mu = sigma_leaf_mu;
    sigma_leaf_tau = sigma_leaf_tau;
    alpha_mu = alpha_mu;
    alpha_tau = alpha_tau;
    beta_mu = beta_mu;
    beta_tau = beta_tau;
    min_samples_leaf_mu = min_samples_leaf_mu;
    min_samples_leaf_tau = min_samples_leaf_tau;
    nu = nu;
    lamb = lamb;
    a_leaf_mu = a_leaf_mu;
    a_leaf_tau = a_leaf_tau;
    b_leaf_mu = b_leaf_mu;
    b_leaf_tau = b_leaf_tau;
    sigma2 = sigma2;
    num_trees_mu = num_trees_mu;
    num_trees_tau = num_trees_tau;
    b1 = b1;
    b0 = b0;
    num_gfr = num_gfr;
    num_burnin = num_burnin;
    num_mcmc = num_mcmc;
    feature_types_mu = feature_types_mu;
    feature_types_tau = feature_types_tau;
    leaf_init_mu = leaf_init_mu;
    leaf_init_tau = leaf_init_tau;
  }
  BCFParameters(int cutpoint_grid_size, double sigma_leaf_mu, Eigen::MatrixXd& sigma_leaf_tau, 
                double alpha_mu, double alpha_tau, double beta_mu, double beta_tau, 
                int min_samples_leaf_mu, int min_samples_leaf_tau, double nu, double lamb, 
                double a_leaf_mu, double a_leaf_tau, double b_leaf_mu, double b_leaf_tau, 
                double sigma2, int num_trees_mu, int num_trees_tau, double b1, double b0, 
                std::vector<FeatureType>& feature_types_mu, std::vector<FeatureType>& feature_types_tau, 
                int num_gfr, int num_burnin, int num_mcmc, double leaf_init_mu, double leaf_init_tau) {
    cutpoint_grid_size = cutpoint_grid_size;
    sigma_leaf_mu = sigma_leaf_mu;
    sigma_leaf_tau_mat = sigma_leaf_tau;
    alpha_mu = alpha_mu;
    alpha_tau = alpha_tau;
    beta_mu = beta_mu;
    beta_tau = beta_tau;
    min_samples_leaf_mu = min_samples_leaf_mu;
    min_samples_leaf_tau = min_samples_leaf_tau;
    nu = nu;
    lamb = lamb;
    a_leaf_mu = a_leaf_mu;
    a_leaf_tau = a_leaf_tau;
    b_leaf_mu = b_leaf_mu;
    b_leaf_tau = b_leaf_tau;
    sigma2 = sigma2;
    num_trees_mu = num_trees_mu;
    num_trees_tau = num_trees_tau;
    b1 = b1;
    b0 = b0;
    num_gfr = num_gfr;
    num_burnin = num_burnin;
    num_mcmc = num_mcmc;
    feature_types_mu = feature_types_mu;
    feature_types_tau = feature_types_tau;
    leaf_init_mu = leaf_init_mu;
    leaf_init_tau = leaf_init_tau;
  }
};

/*! \brief Class that coordinates BCF sampler and returns results */
template <typename TauModelType>
class BCFModel {
 public:
  BCFModel(){}
  ~BCFModel(){}
  void SampleBCF(ForestContainer* forest_samples_mu, ForestContainer* forest_samples_tau, std::mt19937* rng, 
                 int cutpoint_grid_size, double sigma_leaf_mu, double sigma_leaf_tau, 
                 double alpha_mu, double alpha_tau, double beta_mu, double beta_tau, 
                 int min_samples_leaf_mu, int min_samples_leaf_tau, double nu, double lamb, 
                 double a_leaf_mu, double a_leaf_tau, double b_leaf_mu, double b_leaf_tau, 
                 double sigma2, int num_trees_mu, int num_trees_tau, double b1, double b0, 
                 std::vector<FeatureType>& feature_types_mu, std::vector<FeatureType>& feature_types_tau, 
                 int num_gfr, int num_burnin, int num_mcmc, double leaf_init_mu, double leaf_init_tau) {
    BCFParameters params(cutpoint_grid_size, sigma_leaf_mu, sigma_leaf_tau, alpha_mu, alpha_tau, beta_mu, beta_tau, 
                         min_samples_leaf_mu, min_samples_leaf_tau, nu, lamb, a_leaf_mu, a_leaf_tau, b_leaf_mu, b_leaf_tau, 
                         sigma2, num_trees_mu, num_trees_tau, b1, b0, feature_types_mu, feature_types_tau, 
                         num_gfr, num_burnin, num_mcmc, leaf_init_mu, leaf_init_tau);
    SampleBCFInternal(forest_samples_mu, forest_samples_tau, rng, params);
  }

  void SampleBCF(ForestContainer* forest_samples_mu, ForestContainer* forest_samples_tau, std::mt19937* rng, 
                 int cutpoint_grid_size, double sigma_leaf_mu, Eigen::MatrixXd& sigma_leaf_tau, 
                 double alpha_mu, double alpha_tau, double beta_mu, double beta_tau, 
                 int min_samples_leaf_mu, int min_samples_leaf_tau, double nu, double lamb, 
                 double a_leaf_mu, double a_leaf_tau, double b_leaf_mu, double b_leaf_tau, 
                 double sigma2, int num_trees_mu, int num_trees_tau, double b1, double b0, 
                 std::vector<FeatureType>& feature_types_mu, std::vector<FeatureType>& feature_types_tau, 
                 int num_gfr, int num_burnin, int num_mcmc, double leaf_init_mu, double leaf_init_tau) {
    BCFParameters params(cutpoint_grid_size, sigma_leaf_mu, sigma_leaf_tau, alpha_mu, alpha_tau, beta_mu, beta_tau, 
                         min_samples_leaf_mu, min_samples_leaf_tau, nu, lamb, a_leaf_mu, a_leaf_tau, b_leaf_mu, b_leaf_tau, 
                         sigma2, num_trees_mu, num_trees_tau, b1, b0, feature_types_mu, feature_types_tau, 
                         num_gfr, num_burnin, num_mcmc, leaf_init_mu, leaf_init_tau);
    SampleBCFInternal(forest_samples_mu, forest_samples_tau, rng, params);
  }

  void LoadTrain(double* residual_data_ptr, int num_rows, double* prognostic_covariate_data_ptr, int num_prognostic_covariates, 
                 double* treatment_covariate_data_ptr, int num_treatment_covariates, double* treatment_data_ptr, 
                 int num_treatment, bool treatment_binary) {
    // Residual
    residual_train_.LoadData(residual_data_ptr, num_rows);

    // Prognostic term training dataset
    forest_dataset_mu_train_.AddCovariates(prognostic_covariate_data_ptr, num_rows, num_prognostic_covariates, false);

    // Treatment term training dataset
    forest_dataset_tau_train_.AddCovariates(treatment_covariate_data_ptr, num_rows, num_treatment_covariates, false);
    forest_dataset_tau_train_.AddBasis(treatment_data_ptr, num_rows, num_treatment, false);
    treatment_dim_ = num_treatment;
    treatment_binary_ = treatment_binary;
  }
  
  void LoadTrain(double* residual_data_ptr, int num_rows, double* prognostic_covariate_data_ptr, int num_prognostic_covariates, 
                 double* treatment_covariate_data_ptr, int num_treatment_covariates, double* treatment_data_ptr, 
                 int num_treatment, bool treatment_binary, double* weights_data_ptr) {
    // Residual
    residual_train_.LoadData(residual_data_ptr, num_rows);

    // Prognostic term training dataset
    forest_dataset_mu_train_.AddCovariates(prognostic_covariate_data_ptr, num_rows, num_prognostic_covariates, false);
    forest_dataset_mu_train_.AddVarianceWeights(weights_data_ptr, num_rows);

    // Treatment term training dataset
    forest_dataset_tau_train_.AddCovariates(treatment_covariate_data_ptr, num_rows, num_treatment_covariates, false);
    forest_dataset_tau_train_.AddBasis(treatment_data_ptr, num_rows, num_treatment, false);
    forest_dataset_tau_train_.AddVarianceWeights(weights_data_ptr, num_rows);
    treatment_dim_ = num_treatment;
    treatment_binary_ = treatment_binary;

    has_weights_ = true;
  }
  
  void LoadTest(double* prognostic_covariate_data_ptr, int num_rows, int num_prognostic_covariates, 
                double* treatment_covariate_data_ptr, int num_treatment_covariates, double* treatment_data_ptr, int num_treatment) {
    // Prognostic term training dataset
    forest_dataset_mu_test_.AddCovariates(prognostic_covariate_data_ptr, num_rows, num_prognostic_covariates, false);

    // Treatment term training dataset
    forest_dataset_tau_test_.AddCovariates(treatment_covariate_data_ptr, num_rows, num_treatment_covariates, false);
    forest_dataset_tau_test_.AddBasis(treatment_data_ptr, num_rows, num_treatment, false);

    has_test_ = true;
  }
  
  void ResetGlobalVarSamples(double* data_ptr, int num_samples) {
    new (&global_var_samples_) MatrixMap(data_ptr, num_samples, 1);
    global_var_random_ = true;
  }

  void ResetPrognosticLeafVarSamples(double* data_ptr, int num_samples) {
    new (&prognostic_leaf_var_samples_) MatrixMap(data_ptr, num_samples, 1);
    prognostic_leaf_var_random_ = true;
  }
  
  void ResetTreatmentLeafVarSamples(double* data_ptr, int num_samples) {
    new (&treatment_leaf_var_samples_) MatrixMap(data_ptr, num_samples, 1);
    treatment_leaf_var_random_ = true;
  }
  
  void ResetTreatedCodingSamples(double* data_ptr, int num_samples) {
    new (&b1_samples_) MatrixMap(data_ptr, num_samples, 1);
  }
  
  void ResetControlCodingSamples(double* data_ptr, int num_samples) {
    new (&b0_samples_) MatrixMap(data_ptr, num_samples, 1);
  }
  
  void ResetTrainPredictionSamples(double* muhat_data_ptr, double* tauhat_data_ptr, double* yhat_data_ptr, int num_obs, int num_samples) {
    new (&muhat_train_samples_) MatrixMap(muhat_data_ptr, num_obs, num_samples);
    new (&tauhat_train_samples_) MatrixMap(tauhat_data_ptr, num_obs, num_samples);
    new (&yhat_train_samples_) MatrixMap(yhat_data_ptr, num_obs, num_samples);
  }
  
  void ResetTestPredictionSamples(double* muhat_data_ptr, double* tauhat_data_ptr, double* yhat_data_ptr, int num_obs, int num_samples) {
    new (&muhat_test_samples_) MatrixMap(muhat_data_ptr, num_obs, num_samples);
    new (&tauhat_test_samples_) MatrixMap(tauhat_data_ptr, num_obs, num_samples);
    new (&yhat_test_samples_) MatrixMap(yhat_data_ptr, num_obs, num_samples);
  }

 private:
  // Details of the model
  int treatment_dim_{1};
  bool adaptive_coding_{false};
  bool treatment_binary_{true};
  bool global_var_random_{false};
  bool prognostic_leaf_var_random_{false};
  bool treatment_leaf_var_random_{false};
  bool has_weights_{false};
  bool has_test_{false};

  // Train and test sets
  ColumnVector residual_train_;
  ForestDataset forest_dataset_mu_train_;
  ForestDataset forest_dataset_mu_test_;
  ForestDataset forest_dataset_tau_train_;
  ForestDataset forest_dataset_tau_test_;

  // There is no default initializer for Eigen::Map, so we initialize to 
  // NULL, 1, 1 and reset the map when necessary
  MatrixMap global_var_samples_{NULL,1,1};
  MatrixMap prognostic_leaf_var_samples_{NULL,1,1};
  MatrixMap treatment_leaf_var_samples_{NULL,1,1};
  MatrixMap b1_samples_{NULL,1,1};
  MatrixMap b0_samples_{NULL,1,1};
  MatrixMap muhat_train_samples_{NULL,1,1};
  MatrixMap tauhat_train_samples_{NULL,1,1};
  MatrixMap yhat_train_samples_{NULL,1,1};
  MatrixMap muhat_test_samples_{NULL,1,1};
  MatrixMap tauhat_test_samples_{NULL,1,1};
  MatrixMap yhat_test_samples_{NULL,1,1};

  TauModelType InitializeTauLeafModel(double sigma_leaf, Eigen::MatrixXd& sigma_leaf_mat) {
    if constexpr (std::is_same_v<TauModelType, GaussianMultivariateRegressionLeafModel>) {
      return TauModelType(sigma_leaf_mat);
    } else {
      return TauModelType(sigma_leaf);
    }
  }
  
  void SampleBCFInternal(ForestContainer* forest_samples_mu, ForestContainer* forest_samples_tau, std::mt19937* rng, BCFParameters& params) {
    // Initialize leaf models for mu and tau forests
    GaussianConstantLeafModel leaf_model_mu = GaussianConstantLeafModel(params.sigma_leaf_mu);
    TauModelType leaf_model_tau = InitializeTauLeafModel(params.sigma_leaf_tau, params.sigma_leaf_tau_mat);
    // TauModelType leaf_model_tau;
    // if constexpr (std::is_same_v<TauModelType, GaussianMultivariateRegressionLeafModel>) {
    //   leaf_model_tau = TauModelType(params.sigma_leaf_tau_mat);
    // } else {
    //   leaf_model_tau = TauModelType(params.sigma_leaf_tau);
    // }

    // Set variable weights
    double const_var_wt_mu = static_cast<double>(1/(forest_dataset_mu_train_.NumCovariates()));
    std::vector<double> variable_weights_mu(forest_dataset_mu_train_.NumCovariates(), const_var_wt_mu);
    double const_var_wt_tau = static_cast<double>(1/forest_dataset_tau_train_.NumCovariates());
    std::vector<double> variable_weights_tau(forest_dataset_tau_train_.NumCovariates(), const_var_wt_tau);

    // Initialize trackers and tree priors
    int n = forest_dataset_mu_train_.NumObservations();
    ForestTracker mu_tracker = ForestTracker(forest_dataset_mu_train_.GetCovariates(), params.feature_types_mu, params.num_trees_mu, n);
    ForestTracker tau_tracker = ForestTracker(forest_dataset_tau_train_.GetCovariates(), params.feature_types_tau, params.num_trees_tau, n);
    TreePrior tree_prior_mu = TreePrior(params.alpha_mu, params.beta_mu, params.min_samples_leaf_mu);
    TreePrior tree_prior_tau = TreePrior(params.alpha_tau, params.beta_tau, params.min_samples_leaf_tau);

    // Initialize leaf values
    // TODO: handle multivariate tau case
    forest_samples_mu->SetLeafValue(0, params.leaf_init_mu);
    forest_samples_tau->SetLeafValue(0, params.leaf_init_tau);
    UpdateResidualEntireForest(mu_tracker, forest_dataset_mu_train_, residual_train_, forest_samples_mu->GetEnsemble(0), false, std::minus<double>());
    UpdateResidualEntireForest(tau_tracker, forest_dataset_tau_train_, residual_train_, forest_samples_tau->GetEnsemble(0), true, std::minus<double>());

    // Variance models (if requested)
    GlobalHomoskedasticVarianceModel global_var_model;
    LeafNodeHomoskedasticVarianceModel prognostic_leaf_var_model;
    LeafNodeHomoskedasticVarianceModel treatment_leaf_var_model;
    if (global_var_random_) global_var_model = GlobalHomoskedasticVarianceModel();
    if (prognostic_leaf_var_random_) prognostic_leaf_var_model = LeafNodeHomoskedasticVarianceModel();
    if (treatment_leaf_var_random_) treatment_leaf_var_model = LeafNodeHomoskedasticVarianceModel();

    // Initial values of (potentially random) parameters
    double sigma2 = params.sigma2;
    double leaf_scale_mu = params.sigma_leaf_mu;
    double leaf_scale_tau = params.sigma_leaf_tau;
    Eigen::MatrixXd leaf_scale_tau_mat = params.sigma_leaf_tau_mat;

    if (params.num_gfr > 0) {
      // Initialize GFR sampler for mu and tau
      GFRForestSampler<GaussianConstantLeafModel> mu_sampler_gfr = GFRForestSampler<GaussianConstantLeafModel>(params.cutpoint_grid_size);
      GFRForestSampler<TauModelType> tau_sampler_gfr = GFRForestSampler<TauModelType>(params.cutpoint_grid_size);
    
      // Run the GFR sampler
      for (int i = 0; i < params.num_gfr; i++) {

        // Sample mu ensemble
        mu_sampler_gfr.SampleOneIter(mu_tracker, *forest_samples_mu, leaf_model_mu, forest_dataset_mu_train_, residual_train_, tree_prior_mu, *rng, variable_weights_mu, sigma2, params.feature_types_mu, true);

        // Sample leaf node variance
        if (prognostic_leaf_var_random_) {
          leaf_scale_mu = prognostic_leaf_var_model.SampleVarianceParameter(forest_samples_mu->GetEnsemble(i), params.a_leaf_mu, params.b_leaf_mu, *rng);
          prognostic_leaf_var_samples_(i) = leaf_scale_mu;
        }

        // Sample global variance
        if (global_var_random_) {
          sigma2 = global_var_model.SampleVarianceParameter(residual_train_.GetData(), params.nu, params.nu*params.lamb, *rng);
          global_var_samples_(i) = sigma2;
        }

        // Sample tau ensemble
        tau_sampler_gfr.SampleOneIter(tau_tracker, *forest_samples_tau, leaf_model_tau, forest_dataset_tau_train_, residual_train_, tree_prior_tau, *rng, variable_weights_tau, sigma2, params.feature_types_tau, true);

        // Sample leaf node variance
        if (treatment_leaf_var_random_) {
          leaf_scale_tau = treatment_leaf_var_model.SampleVarianceParameter(forest_samples_tau->GetEnsemble(i), params.a_leaf_tau, params.b_leaf_tau, *rng);
          treatment_leaf_var_samples_(i) = leaf_scale_tau;
        }

        // Sample global variance
        if (global_var_random_) {
          sigma2 = global_var_model.SampleVarianceParameter(residual_train_.GetData(), params.nu, params.nu*params.lamb, *rng);
          global_var_samples_(i) = sigma2;
        }
      }
    }

    if (params.num_burnin + params.num_mcmc > 0) {
      // Initialize GFR sampler for mu and tau
      MCMCForestSampler<GaussianConstantLeafModel> mu_sampler_mcmc = MCMCForestSampler<GaussianConstantLeafModel>();
      MCMCForestSampler<TauModelType> tau_sampler_mcmc = MCMCForestSampler<TauModelType>();
    
      // Run the GFR sampler
      for (int i = params.num_gfr; i < params.num_gfr + params.num_burnin + params.num_mcmc; i++) {

        // Sample mu ensemble
        mu_sampler_mcmc.SampleOneIter(mu_tracker, *forest_samples_mu, leaf_model_mu, forest_dataset_mu_train_, residual_train_, tree_prior_mu, *rng, variable_weights_mu, sigma2, true);

        // Sample leaf node variance
        if (prognostic_leaf_var_random_) {
          leaf_scale_mu = prognostic_leaf_var_model.SampleVarianceParameter(forest_samples_mu->GetEnsemble(i), params.a_leaf_mu, params.b_leaf_mu, *rng);
          prognostic_leaf_var_samples_(i) = leaf_scale_mu;
        }

        // Sample global variance
        if (global_var_random_) {
          sigma2 = global_var_model.SampleVarianceParameter(residual_train_.GetData(), params.nu, params.nu*params.lamb, *rng);
          global_var_samples_(i) = sigma2;
        }

        // Sample tau ensemble
        tau_sampler_mcmc.SampleOneIter(tau_tracker, *forest_samples_tau, leaf_model_tau, forest_dataset_tau_train_, residual_train_, tree_prior_tau, *rng, variable_weights_tau, sigma2, true);

        // Sample leaf node variance
        if (treatment_leaf_var_random_) {
          leaf_scale_tau = treatment_leaf_var_model.SampleVarianceParameter(forest_samples_tau->GetEnsemble(i), params.a_leaf_tau, params.b_leaf_tau, *rng);
          treatment_leaf_var_samples_(i) = leaf_scale_tau;
        }

        // Sample global variance
        if (global_var_random_) {
          sigma2 = global_var_model.SampleVarianceParameter(residual_train_.GetData(), params.nu, params.nu*params.lamb, *rng);
          global_var_samples_(i) = sigma2;
        }
      }
    }
  }  
};

} // namespace StochTree

#endif // STOCHTREE_CPP_API_H_
