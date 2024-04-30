/*!
 * Copyright (c) 2024 stochtree authors.
 * 
 * High-level C++ API for BART and BCF
 */
#ifndef STOCHTREE_CPP_API_H_
#define STOCHTREE_CPP_API_H_

#include <Eigen/Dense>
#include <stochtree/category_tracker.h>
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

class BCFParameters {
 public:
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
  
  BCFParameters(int cutpoint_grid_size_, double sigma_leaf_mu_, double sigma_leaf_tau_,
                double alpha_mu_, double alpha_tau_, double beta_mu_, double beta_tau_,
                int min_samples_leaf_mu_, int min_samples_leaf_tau_, double nu_, double lamb_, 
                double a_leaf_mu_, double a_leaf_tau_, double b_leaf_mu_, double b_leaf_tau_, 
                double sigma2_, int num_trees_mu_, int num_trees_tau_, double b1_, double b0_, 
                std::vector<FeatureType>& feature_types_mu_, std::vector<FeatureType>& feature_types_tau_, 
                int num_gfr_, int num_burnin_, int num_mcmc_, double leaf_init_mu_, double leaf_init_tau_) {
    cutpoint_grid_size = cutpoint_grid_size_;
    sigma_leaf_mu = sigma_leaf_mu_;
    sigma_leaf_tau = sigma_leaf_tau_;
    alpha_mu = alpha_mu_;
    alpha_tau = alpha_tau_;
    beta_mu = beta_mu_;
    beta_tau = beta_tau_;
    min_samples_leaf_mu = min_samples_leaf_mu_;
    min_samples_leaf_tau = min_samples_leaf_tau_;
    nu = nu_;
    lamb = lamb_;
    a_leaf_mu = a_leaf_mu_;
    a_leaf_tau = a_leaf_tau_;
    b_leaf_mu = b_leaf_mu_;
    b_leaf_tau = b_leaf_tau_;
    sigma2 = sigma2_;
    num_trees_mu = num_trees_mu_;
    num_trees_tau = num_trees_tau_;
    b1 = b1_;
    b0 = b0_;
    num_gfr = num_gfr_;
    num_burnin = num_burnin_;
    num_mcmc = num_mcmc_;
    feature_types_mu = feature_types_mu_;
    feature_types_tau = feature_types_tau_;
    leaf_init_mu = leaf_init_mu_;
    leaf_init_tau = leaf_init_tau_;
  }
  BCFParameters(int cutpoint_grid_size_, double sigma_leaf_mu_, Eigen::MatrixXd& sigma_leaf_tau_, 
                double alpha_mu_, double alpha_tau_, double beta_mu_, double beta_tau_, 
                int min_samples_leaf_mu_, int min_samples_leaf_tau_, double nu_, double lamb_, 
                double a_leaf_mu_, double a_leaf_tau_, double b_leaf_mu_, double b_leaf_tau_, 
                double sigma2_, int num_trees_mu_, int num_trees_tau_, double b1_, double b0_, 
                std::vector<FeatureType>& feature_types_mu_, std::vector<FeatureType>& feature_types_tau_, 
                int num_gfr_, int num_burnin_, int num_mcmc_, double leaf_init_mu_, double leaf_init_tau_) {
    cutpoint_grid_size = cutpoint_grid_size_;
    sigma_leaf_mu = sigma_leaf_mu_;
    sigma_leaf_tau_mat = sigma_leaf_tau_;
    alpha_mu = alpha_mu_;
    alpha_tau = alpha_tau_;
    beta_mu = beta_mu_;
    beta_tau = beta_tau_;
    min_samples_leaf_mu = min_samples_leaf_mu_;
    min_samples_leaf_tau = min_samples_leaf_tau_;
    nu = nu_;
    lamb = lamb_;
    a_leaf_mu = a_leaf_mu_;
    a_leaf_tau = a_leaf_tau_;
    b_leaf_mu = b_leaf_mu_;
    b_leaf_tau = b_leaf_tau_;
    sigma2 = sigma2_;
    num_trees_mu = num_trees_mu_;
    num_trees_tau = num_trees_tau_;
    b1 = b1_;
    b0 = b0_;
    num_gfr = num_gfr_;
    num_burnin = num_burnin_;
    num_mcmc = num_mcmc_;
    feature_types_mu = feature_types_mu_;
    feature_types_tau = feature_types_tau_;
    leaf_init_mu = leaf_init_mu_;
    leaf_init_tau = leaf_init_tau_;
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
    global_var_samples_mapped_ = true;
  }

  void ResetPrognosticLeafVarSamples(double* data_ptr, int num_samples) {
    new (&prognostic_leaf_var_samples_) MatrixMap(data_ptr, num_samples, 1);
    prognostic_leaf_var_random_ = true;
    prognostic_leaf_var_samples_mapped_ = true;
  }
  
  void ResetTreatmentLeafVarSamples(double* data_ptr, int num_samples) {
    new (&treatment_leaf_var_samples_) MatrixMap(data_ptr, num_samples, 1);
    treatment_leaf_var_random_ = true;
    treatment_leaf_var_samples_mapped_ = true;
  }
  
  void ResetTreatedCodingSamples(double* data_ptr, int num_samples) {
    new (&b1_samples_) MatrixMap(data_ptr, num_samples, 1);
    b1_samples_mapped_ = true;
  }
  
  void ResetControlCodingSamples(double* data_ptr, int num_samples) {
    new (&b0_samples_) MatrixMap(data_ptr, num_samples, 1);
    b0_samples_mapped_ = true;
  }
  
  void ResetTrainPredictionSamples(double* muhat_data_ptr, double* tauhat_data_ptr, double* yhat_data_ptr, int num_obs, int num_samples, int treatment_dim) {
    new (&muhat_train_samples_) MatrixMap(muhat_data_ptr, num_obs, num_samples);
    new (&tauhat_train_samples_) VectorMap(tauhat_data_ptr, num_obs*treatment_dim*num_samples);
    new (&yhat_train_samples_) MatrixMap(yhat_data_ptr, num_obs, num_samples);
    muhat_train_samples_mapped_ = true;
    tauhat_train_samples_mapped_ = true;
    yhat_train_samples_mapped_ = true;
  }
  
  void ResetTestPredictionSamples(double* muhat_data_ptr, double* tauhat_data_ptr, double* yhat_data_ptr, int num_obs, int num_samples, int treatment_dim) {
    new (&muhat_test_samples_) MatrixMap(muhat_data_ptr, num_obs, num_samples);
    new (&tauhat_test_samples_) VectorMap(tauhat_data_ptr, num_obs*treatment_dim*num_samples);
    new (&yhat_test_samples_) MatrixMap(yhat_data_ptr, num_obs, num_samples);
    muhat_test_samples_mapped_ = true;
    tauhat_test_samples_mapped_ = true;
    yhat_test_samples_mapped_ = true;
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
  
  // Internal details about whether a MatrixMap has been mapped to a data buffer
  bool global_var_samples_mapped_{false};
  bool prognostic_leaf_var_samples_mapped_{false};
  bool treatment_leaf_var_samples_mapped_{false};
  bool b1_samples_mapped_{false};
  bool b0_samples_mapped_{false};
  bool muhat_train_samples_mapped_{false};
  bool tauhat_train_samples_mapped_{false};
  bool yhat_train_samples_mapped_{false};
  bool muhat_test_samples_mapped_{false};
  bool tauhat_test_samples_mapped_{false};
  bool yhat_test_samples_mapped_{false};

  TauModelType InitializeTauLeafModel(double sigma_leaf, Eigen::MatrixXd& sigma_leaf_mat) {
    if constexpr (std::is_same_v<TauModelType, GaussianMultivariateRegressionLeafModel>) {
      return TauModelType(sigma_leaf_mat);
    } else {
      return TauModelType(sigma_leaf);
    }
  }
  
  void SampleBCFInternal(ForestContainer* forest_samples_mu, ForestContainer* forest_samples_tau, std::mt19937* rng, BCFParameters& params) {
    // Input checks
    CHECK(yhat_train_samples_mapped_);
    CHECK(muhat_train_samples_mapped_);
    CHECK(tauhat_train_samples_mapped_);
    
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

    // Test set details (if a test set is provided)
    int n_test;
    if (has_test_) {
      n_test = forest_dataset_mu_test_.NumObservations();
    }

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

    // If treatment is binary, update the basis of the tau regression to use (b1*Z + b0*(1-Z))
    double b1 = params.b1;
    double b0 = params.b0;
    MatrixObject Z_orig(n, treatment_dim_);
    MatrixObject Z_adj;
    std::vector<int32_t> Z_int(n * treatment_dim_);
    if (treatment_binary_) {
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < treatment_dim_; j++) {
          Z_orig(i,j) = forest_dataset_tau_train_.BasisValue(i,j);
        }
      }
      Z_adj = (
        (b1 * Z_orig).array() + 
        (b0 * (1 - Z_orig.array()))
      );
      forest_dataset_tau_train_.UpdateBasis(Z_adj.data(), n, treatment_dim_, false);
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < treatment_dim_; j++) {
          Z_int.at(i + n * j) = static_cast<int32_t>(forest_dataset_tau_train_.BasisValue(i,j));
        }
      }
    }

    // Do the same for test set
    MatrixObject Z_orig_test;
    MatrixObject Z_adj_test;
    if (has_test_) {
      Z_orig_test = MatrixObject(n_test, treatment_dim_);
      if (treatment_binary_) {
        for (int i = 0; i < n_test; i++) {
          for (int j = 0; j < treatment_dim_; j++) {
            Z_orig_test(i,j) = forest_dataset_tau_train_.BasisValue(i,j);
          }
        }
        Z_adj_test = (
          (b1 * Z_orig_test).array() + 
          (b0 * (1 - Z_orig_test.array()))
        );
        forest_dataset_tau_test_.UpdateBasis(Z_adj_test.data(), n_test, treatment_dim_, false);
      }
    }

    // Override adaptive coding if treatment is not binary or treatment is multivariate
    if (adaptive_coding_) {
      if (!treatment_binary_) {
        adaptive_coding_ = false;
      }
      if (treatment_dim_ > 1) {
        adaptive_coding_ = false;
      }
    }

    // Prepare to run the adaptive coding sampler (if requested)
    VectorObject initial_outcome;
    VectorObject outcome_minus_mu;
    VectorObject muhat;
    VectorObject tauhat;
    CategorySampleTracker treatment_category_sampler_tracker;
    UnivariateNormalSampler adaptive_coding_sampler = UnivariateNormalSampler();
    if (adaptive_coding_) {
      initial_outcome.resize(n);
      outcome_minus_mu.resize(n);
      muhat.resize(n);
      tauhat.resize(n * treatment_dim_);
      for (int i = 0; i < n; i++) initial_outcome(i) = residual_train_.GetElement(i);
      treatment_category_sampler_tracker = CategorySampleTracker(Z_int);
    }

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

        // Predict from the mu ensemble for the train set
        for (int obs_idx = 0; obs_idx < n; obs_idx++) {
          double mupred = 0.;
          for (int tree_idx = 0; tree_idx < forest_samples_mu->NumTrees(); tree_idx++) {
            mupred += forest_samples_mu->GetEnsemble(i)->GetTree(tree_idx)->LeafValue(mu_tracker.GetNodeId(obs_idx, tree_idx), 0);
          }
          muhat_train_samples_(obs_idx, i) = mupred;
          if (adaptive_coding_) muhat(obs_idx) = mupred;
        }

        // Predict from the mu ensemble for the test set (if provided)
        if (has_test_) {
          for (int obs_idx = 0; obs_idx < n_test; obs_idx++) {
            double mupred = 0.;
            for (int tree_idx = 0; tree_idx < forest_samples_mu->NumTrees(); tree_idx++) {
              auto &tree = *forest_samples_mu->GetEnsemble(i)->GetTree(tree_idx);
              int32_t nidx = EvaluateTree(tree, forest_dataset_mu_test_.GetCovariates(), i);
              mupred += tree.LeafValue(nidx, 0);
            }
            muhat_test_samples_(obs_idx, i) = mupred;
          }
        }

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

        // Sample adaptive coding parameter for tau (if applicable), before defining the tau(x) prediction for a sample
        if (adaptive_coding_) {
          for (int obs_idx = 0; obs_idx < n; obs_idx++) {
            for (int treatment_dim_idx = 0; treatment_dim_idx < treatment_dim_; treatment_dim_idx++) {
              double taupred = 0.;
              for (int tree_idx = 0; tree_idx < forest_samples_mu->NumTrees(); tree_idx++) {
                taupred += forest_samples_tau->GetEnsemble(i)->GetTree(tree_idx)->LeafValue(tau_tracker.GetNodeId(obs_idx, tree_idx), 0);
              }
              tauhat(obs_idx + n * treatment_dim_idx) = taupred;
            }
          }

          // Data checks and partial residualization
          CHECK_EQ(muhat.size(), tauhat.size());
          CHECK_EQ(muhat.size(), initial_outcome.size());
          outcome_minus_mu = initial_outcome.array() - muhat.array();

          // Compute sufficient statistics for control group observations
          std::vector<data_size_t>::iterator control_begin_iter = treatment_category_sampler_tracker.CategoryBeginIterator(0);
          std::vector<data_size_t>::iterator control_end_iter = treatment_category_sampler_tracker.CategoryEndIterator(0);
          double sum_xx_control = 0.;
          double sum_xy_control = 0.;
          for (auto obs_it = control_begin_iter; obs_it != control_end_iter; i++) {
            auto idx = *obs_it;
            sum_xx_control += tauhat(idx)*tauhat(idx);
            sum_xy_control += tauhat(idx)*outcome_minus_mu(idx);
          }

          // Compute sufficient statistics for treatment group observations
          std::vector<data_size_t>::iterator treated_begin_iter = treatment_category_sampler_tracker.CategoryBeginIterator(1);
          std::vector<data_size_t>::iterator treated_end_iter = treatment_category_sampler_tracker.CategoryEndIterator(1);
          double sum_xx_treated = 0.;
          double sum_xy_treated = 0.;
          for (auto obs_it = treated_begin_iter; obs_it != treated_end_iter; i++) {
            auto idx = *obs_it;
            sum_xx_treated += tauhat(idx)*tauhat(idx);
            sum_xy_treated += tauhat(idx)*outcome_minus_mu(idx);
          }

          // Compute the posterior mean and variance for treated and control coding parameters
          double post_mean_control = (sum_xy_control / (sum_xx_control + 2*sigma2));
          double post_var_control = (sigma2 / (sum_xx_control + 2*sigma2));
          double post_mean_treated = (sum_xy_treated / (sum_xx_treated + 2*sigma2));
          double post_var_treated = (sigma2 / (sum_xx_treated + 2*sigma2));

          // Sample new adaptive coding parameters
          b0 = adaptive_coding_sampler.Sample(post_mean_control, post_var_control, *rng);
          b1 = adaptive_coding_sampler.Sample(post_mean_treated, post_var_treated, *rng);

          // Update sample container for b1 and b0
          b0_samples_(i) = b0;
          b1_samples_(i) = b1;

          // Update basis used in the leaf regression in the next iteration
          Z_adj = (
            (b1 * Z_orig).array() + 
            (b0 * (1 - Z_orig.array()))
          );
          forest_dataset_tau_train_.UpdateBasis(Z_adj.data(), n, treatment_dim_, false);

          // Update basis used in the leaf regression in the next iteration
          if (has_test_) {
            Z_adj_test = (
              (b1 * Z_orig).array() + 
              (b0 * (1 - Z_orig.array()))
            );
            forest_dataset_tau_test_.UpdateBasis(Z_adj_test.data(), n_test, treatment_dim_, false);
          }

          // Update residual and tree predictions with the new basis
          for (int obs_idx = 0; obs_idx < n; obs_idx++) {
            // double outcome_pred_tau = 0.;
            for (int tree_idx = 0; tree_idx < forest_samples_tau->NumTrees(); tree_idx++) {
              // Retrieve the "old" prediction for tree_idx back and the residual value to be updated
              double prev_resid = residual_train_.GetElement(obs_idx);
              double prev_pred = tau_tracker.GetTreeSamplePrediction(obs_idx, tree_idx);
              // Compute the new prediction for tree_idx with updated basis
              double outcome_pred_tau_tree = 0.;
              for (int treatment_dim_idx = 0; treatment_dim_idx < treatment_dim_; treatment_dim_idx++) {
                double leaf_val = forest_samples_tau->GetEnsemble(i)->GetTree(tree_idx)->LeafValue(tau_tracker.GetNodeId(obs_idx, tree_idx), treatment_dim_idx);
                double basis_val = forest_dataset_tau_train_.BasisValue(obs_idx, treatment_dim_idx);
                outcome_pred_tau_tree += leaf_val * basis_val;
              }
              // Recalculate prediction in tau_tracker
              tau_tracker.SetTreeSamplePrediction(obs_idx, tree_idx, outcome_pred_tau_tree);
              // outcome_pred_tau += outcome_pred_tau_tree;
              // Update residual (adding back "old" prediction and subtracting new prediction)
              residual_train_.SetElement(obs_idx, prev_resid + prev_pred + outcome_pred_tau_tree);
            }
          }
        }
        
        // Predict from the tau ensemble for the train set
        for (int obs_idx = 0; obs_idx < n; obs_idx++) {
          double outcome_pred_tau = 0.;
          for (int treatment_dim_idx = 0; treatment_dim_idx < treatment_dim_; treatment_dim_idx++) {
            double taupred = 0.;
            for (int tree_idx = 0; tree_idx < forest_samples_tau->NumTrees(); tree_idx++) {
              taupred += forest_samples_tau->GetEnsemble(i)->GetTree(tree_idx)->LeafValue(tau_tracker.GetNodeId(obs_idx, tree_idx), treatment_dim_idx);
            }
            tauhat_train_samples_(obs_idx + treatment_dim_idx*n + i*n*treatment_dim_) = taupred * (b1 - b0);
            outcome_pred_tau += taupred * forest_dataset_tau_train_.BasisValue(obs_idx, treatment_dim_idx);
          }
          yhat_train_samples_(obs_idx, i) = outcome_pred_tau + muhat_train_samples_(obs_idx, i);
        }

        // Predict from the tau ensemble for the test set (if provided)
        if (has_test_) {
          for (int obs_idx = 0; obs_idx < n_test; obs_idx++) {
            double outcome_pred_tau = 0.;
            for (int treatment_dim_idx = 0; treatment_dim_idx < treatment_dim_; treatment_dim_idx++) {
              double taupred = 0.;
              for (int tree_idx = 0; tree_idx < forest_samples_tau->NumTrees(); tree_idx++) {
                auto &tree = *forest_samples_tau->GetEnsemble(i)->GetTree(tree_idx);
                int32_t nidx = EvaluateTree(tree, forest_dataset_tau_test_.GetCovariates(), i);
                taupred += tree.LeafValue(nidx, treatment_dim_idx);
              }
              tauhat_test_samples_(obs_idx + treatment_dim_idx*n + i*n*treatment_dim_) = taupred * (b1 - b0);
              outcome_pred_tau += taupred * forest_dataset_tau_test_.BasisValue(obs_idx, treatment_dim_idx);
            }
            yhat_test_samples_(obs_idx, i) = outcome_pred_tau + muhat_test_samples_(obs_idx, i);
          }
        }

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
      // Initialize MCMC sampler for mu and tau
      MCMCForestSampler<GaussianConstantLeafModel> mu_sampler_mcmc = MCMCForestSampler<GaussianConstantLeafModel>();
      MCMCForestSampler<TauModelType> tau_sampler_mcmc = MCMCForestSampler<TauModelType>();
    
      // Run the MCMC sampler
      for (int i = params.num_gfr; i < params.num_gfr + params.num_burnin + params.num_mcmc; i++) {

        // Sample mu ensemble
        mu_sampler_mcmc.SampleOneIter(mu_tracker, *forest_samples_mu, leaf_model_mu, forest_dataset_mu_train_, residual_train_, tree_prior_mu, *rng, variable_weights_mu, sigma2, true);

        // Predict from the mu ensemble for the train set
        for (int obs_idx = 0; obs_idx < n; obs_idx++) {
          double mupred = 0.;
          for (int tree_idx = 0; tree_idx < forest_samples_mu->NumTrees(); tree_idx++) {
            mupred += forest_samples_mu->GetEnsemble(i)->GetTree(tree_idx)->LeafValue(mu_tracker.GetNodeId(obs_idx, tree_idx), 0);
          }
          muhat_train_samples_(obs_idx, i) = mupred;
          if (adaptive_coding_) muhat(obs_idx) = mupred;
        }

        // Predict from the mu ensemble for the test set (if provided)
        if (has_test_) {
          for (int obs_idx = 0; obs_idx < n_test; obs_idx++) {
            double mupred = 0.;
            for (int tree_idx = 0; tree_idx < forest_samples_mu->NumTrees(); tree_idx++) {
              auto &tree = *forest_samples_mu->GetEnsemble(i)->GetTree(tree_idx);
              int32_t nidx = EvaluateTree(tree, forest_dataset_mu_test_.GetCovariates(), i);
              mupred += tree.LeafValue(nidx, 0);
            }
            muhat_test_samples_(obs_idx, i) = mupred;
          }
        }

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

        // Sample adaptive coding parameter for tau (if applicable), before defining the tau(x) prediction for a sample
        if (adaptive_coding_) {
          for (int obs_idx = 0; obs_idx < n; obs_idx++) {
            for (int treatment_dim_idx = 0; treatment_dim_idx < treatment_dim_; treatment_dim_idx++) {
              double taupred = 0.;
              for (int tree_idx = 0; tree_idx < forest_samples_mu->NumTrees(); tree_idx++) {
                taupred += forest_samples_tau->GetEnsemble(i)->GetTree(tree_idx)->LeafValue(tau_tracker.GetNodeId(obs_idx, tree_idx), 0);
              }
              tauhat(obs_idx + n * treatment_dim_idx) = taupred;
            }
          }

          // Data checks and partial residualization
          CHECK_EQ(muhat.size(), tauhat.size());
          CHECK_EQ(muhat.size(), initial_outcome.size());
          outcome_minus_mu = initial_outcome.array() - muhat.array();

          // Compute sufficient statistics for control group observations
          std::vector<data_size_t>::iterator control_begin_iter = treatment_category_sampler_tracker.CategoryBeginIterator(0);
          std::vector<data_size_t>::iterator control_end_iter = treatment_category_sampler_tracker.CategoryEndIterator(0);
          double sum_xx_control = 0.;
          double sum_xy_control = 0.;
          for (auto obs_it = control_begin_iter; obs_it != control_end_iter; i++) {
            auto idx = *obs_it;
            sum_xx_control += tauhat(idx)*tauhat(idx);
            sum_xy_control += tauhat(idx)*outcome_minus_mu(idx);
          }

          // Compute sufficient statistics for treatment group observations
          std::vector<data_size_t>::iterator treated_begin_iter = treatment_category_sampler_tracker.CategoryBeginIterator(1);
          std::vector<data_size_t>::iterator treated_end_iter = treatment_category_sampler_tracker.CategoryEndIterator(1);
          double sum_xx_treated = 0.;
          double sum_xy_treated = 0.;
          for (auto obs_it = treated_begin_iter; obs_it != treated_end_iter; i++) {
            auto idx = *obs_it;
            sum_xx_treated += tauhat(idx)*tauhat(idx);
            sum_xy_treated += tauhat(idx)*outcome_minus_mu(idx);
          }

          // Compute the posterior mean and variance for treated and control coding parameters
          double post_mean_control = (sum_xy_control / (sum_xx_control + 2*sigma2));
          double post_var_control = (sigma2 / (sum_xx_control + 2*sigma2));
          double post_mean_treated = (sum_xy_treated / (sum_xx_treated + 2*sigma2));
          double post_var_treated = (sigma2 / (sum_xx_treated + 2*sigma2));

          // Sample new adaptive coding parameters
          b0 = adaptive_coding_sampler.Sample(post_mean_control, post_var_control, *rng);
          b1 = adaptive_coding_sampler.Sample(post_mean_treated, post_var_treated, *rng);

          // Update sample container for b1 and b0
          b0_samples_(i) = b0;
          b1_samples_(i) = b1;

          // Update basis used in the leaf regression in the next iteration
          Z_adj = (
            (b1 * Z_orig).array() + 
            (b0 * (1 - Z_orig.array()))
          );
          forest_dataset_tau_train_.UpdateBasis(Z_adj.data(), n, treatment_dim_, false);

          // Update basis used in the leaf regression in the next iteration
          if (has_test_) {
            Z_adj_test = (
              (b1 * Z_orig).array() + 
              (b0 * (1 - Z_orig.array()))
            );
            forest_dataset_tau_test_.UpdateBasis(Z_adj_test.data(), n_test, treatment_dim_, false);
          }

          // Update residual and tree predictions with the new basis
          for (int obs_idx = 0; obs_idx < n; obs_idx++) {
            // double outcome_pred_tau = 0.;
            for (int tree_idx = 0; tree_idx < forest_samples_tau->NumTrees(); tree_idx++) {
              // Retrieve the "old" prediction for tree_idx back and the residual value to be updated
              double prev_resid = residual_train_.GetElement(obs_idx);
              double prev_pred = tau_tracker.GetTreeSamplePrediction(obs_idx, tree_idx);
              // Compute the new prediction for tree_idx with updated basis
              double outcome_pred_tau_tree = 0.;
              for (int treatment_dim_idx = 0; treatment_dim_idx < treatment_dim_; treatment_dim_idx++) {
                double leaf_val = forest_samples_tau->GetEnsemble(i)->GetTree(tree_idx)->LeafValue(tau_tracker.GetNodeId(obs_idx, tree_idx), treatment_dim_idx);
                double basis_val = forest_dataset_tau_train_.BasisValue(obs_idx, treatment_dim_idx);
                outcome_pred_tau_tree += leaf_val * basis_val;
              }
              // Recalculate prediction in tau_tracker
              tau_tracker.SetTreeSamplePrediction(obs_idx, tree_idx, outcome_pred_tau_tree);
              // outcome_pred_tau += outcome_pred_tau_tree;
              // Update residual (adding back "old" prediction and subtracting new prediction)
              residual_train_.SetElement(obs_idx, prev_resid + prev_pred + outcome_pred_tau_tree);
            }
          }
        }
        
        // Predict from the tau ensemble for the train set
        for (int obs_idx = 0; obs_idx < n; obs_idx++) {
          double outcome_pred_tau = 0.;
          for (int treatment_dim_idx = 0; treatment_dim_idx < treatment_dim_; treatment_dim_idx++) {
            double taupred = 0.;
            for (int tree_idx = 0; tree_idx < forest_samples_tau->NumTrees(); tree_idx++) {
              taupred += forest_samples_tau->GetEnsemble(i)->GetTree(tree_idx)->LeafValue(tau_tracker.GetNodeId(obs_idx, tree_idx), treatment_dim_idx);
            }
            tauhat_train_samples_(obs_idx + treatment_dim_idx*n + i*n*treatment_dim_) = taupred * (b1 - b0);
            outcome_pred_tau += taupred * forest_dataset_tau_train_.BasisValue(obs_idx, treatment_dim_idx);
          }
          yhat_train_samples_(obs_idx, i) = outcome_pred_tau + muhat_train_samples_(obs_idx, i);
        }

        // Predict from the tau ensemble for the test set (if provided)
        if (has_test_) {
          for (int obs_idx = 0; obs_idx < n_test; obs_idx++) {
            double outcome_pred_tau = 0.;
            for (int treatment_dim_idx = 0; treatment_dim_idx < treatment_dim_; treatment_dim_idx++) {
              double taupred = 0.;
              for (int tree_idx = 0; tree_idx < forest_samples_tau->NumTrees(); tree_idx++) {
                auto &tree = *forest_samples_tau->GetEnsemble(i)->GetTree(tree_idx);
                int32_t nidx = EvaluateTree(tree, forest_dataset_tau_test_.GetCovariates(), i);
                taupred += tree.LeafValue(nidx, treatment_dim_idx);
              }
              tauhat_test_samples_(obs_idx + treatment_dim_idx*n + i*n*treatment_dim_) = taupred * (b1 - b0);
              outcome_pred_tau += taupred * forest_dataset_tau_test_.BasisValue(obs_idx, treatment_dim_idx);
            }
            yhat_test_samples_(obs_idx, i) = outcome_pred_tau + muhat_test_samples_(obs_idx, i);
          }
        }

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
  
  // Predict from each forest
  
  }
};

} // namespace StochTree

#endif // STOCHTREE_CPP_API_H_
