/*! Copyright (c) 2024 stochtree authors. */
#ifndef STOCHTREE_BART_H_
#define STOCHTREE_BART_H_

#include <stochtree/container.h>
#include <stochtree/data.h>
#include <stochtree/io.h>
#include <nlohmann/json.hpp>
#include <stochtree/leaf_model.h>
#include <stochtree/log.h>
#include <stochtree/random_effects.h>
#include <stochtree/tree_sampler.h>
#include <stochtree/variance_model.h>

#include <memory>

namespace StochTree {

class BARTResult {
 public:
  BARTResult(int num_trees, int output_dimension = 1, bool is_leaf_constant = true) {
    forest_samples_ = std::make_unique<ForestContainer>(num_trees, output_dimension, is_leaf_constant);
  }
  ~BARTResult() {}
  ForestContainer* GetForests() {return forest_samples_.get();}
  ForestContainer* ReleaseForests() {return forest_samples_.release();}
  RandomEffectsContainer* GetRFXContainer() {return rfx_container_.get();}
  RandomEffectsContainer* ReleaseRFXContainer() {return rfx_container_.release();}
  LabelMapper* GetRFXLabelMapper() {return rfx_label_mapper_.get();}
  LabelMapper* ReleaseRFXLabelMapper() {return rfx_label_mapper_.release();}
  std::vector<double>& GetOutcomeTrainPreds() {return outcome_preds_train_;}
  std::vector<double>& GetOutcomeTestPreds() {return outcome_preds_test_;}
  std::vector<double>& GetRFXTrainPreds() {return rfx_preds_train_;}
  std::vector<double>& GetRFXTestPreds() {return rfx_preds_test_;}
  std::vector<double>& GetForestTrainPreds() {return forest_preds_train_;}
  std::vector<double>& GetForestTestPreds() {return forest_preds_test_;}
  std::vector<double>& GetGlobalVarianceSamples() {return sigma_samples_;}
  std::vector<double>& GetLeafVarianceSamples() {return tau_samples_;}
  int NumGFRSamples() {return num_gfr_;}
  int NumBurninSamples() {return num_burnin_;}
  int NumMCMCSamples() {return num_mcmc_;}
  int NumTrainObservations() {return num_train_;}
  int NumTestObservations() {return num_test_;}
  bool IsGlobalVarRandom() {return is_global_var_random_;}
  bool IsLeafVarRandom() {return is_leaf_var_random_;}
  bool HasTestSet() {return has_test_set_;}
  bool HasRFX() {return has_rfx_;}
 private:
  std::unique_ptr<ForestContainer> forest_samples_;
  std::unique_ptr<RandomEffectsContainer> rfx_container_;
  std::unique_ptr<LabelMapper> rfx_label_mapper_;
  std::vector<double> outcome_preds_train_;
  std::vector<double> outcome_preds_test_;
  std::vector<double> rfx_preds_train_;
  std::vector<double> rfx_preds_test_;
  std::vector<double> forest_preds_train_;
  std::vector<double> forest_preds_test_;
  std::vector<double> sigma_samples_;
  std::vector<double> tau_samples_;
  int num_gfr_{0};
  int num_burnin_{0};
  int num_mcmc_{0};
  int num_train_{0};
  int num_test_{0};
  bool is_global_var_random_{true};
  bool is_leaf_var_random_{false};
  bool has_test_set_{false};
  bool has_rfx_{false};
};

template <typename ModelType>
class BARTDispatcher {
 public:
  BARTDispatcher() {}
  ~BARTDispatcher() {}

  BARTResult CreateOutputObject(int num_trees, int output_dimension = 1, bool is_leaf_constant = true) {
    return BARTResult(num_trees, output_dimension, is_leaf_constant);
  }
  
  void AddDataset(double* covariates, data_size_t num_row, int num_col, bool is_row_major, bool train) {
    if (train) {
      train_dataset_ = ForestDataset();
      train_dataset_.AddCovariates(covariates, num_row, num_col, is_row_major);
      num_train_ = num_row;
    } else {
      test_dataset_ = ForestDataset();
      test_dataset_.AddCovariates(covariates, num_row, num_col, is_row_major);
      has_test_set_ = true;
      num_test_ = num_row;
    }
  }

  void AddDataset(double* covariates, double* basis, data_size_t num_row, int num_covariates, int num_basis, bool is_row_major, bool train) {
    if (train) {
      train_dataset_ = ForestDataset();
      train_dataset_.AddCovariates(covariates, num_row, num_covariates, is_row_major);
      train_dataset_.AddBasis(basis, num_row, num_basis, is_row_major);
      num_train_ = num_row;
    } else {
      test_dataset_ = ForestDataset();
      test_dataset_.AddCovariates(covariates, num_row, num_covariates, is_row_major);
      test_dataset_.AddBasis(basis, num_row, num_basis, is_row_major);
      has_test_set_ = true;
      num_test_ = num_row;
    }
  }

  void AddRFXTerm(double* rfx_basis, std::vector<int>& rfx_group_indices, data_size_t num_row, 
                  int num_groups, int num_basis, bool is_row_major, bool train, 
                  Eigen::VectorXd& alpha_init, Eigen::MatrixXd& xi_init, 
                  Eigen::MatrixXd& sigma_alpha_init, Eigen::MatrixXd& sigma_xi_init, 
                  double sigma_xi_shape, double sigma_xi_scale) {
    if (train) {
      rfx_train_dataset_ = RandomEffectsDataset();
      rfx_train_dataset_.AddBasis(rfx_basis, num_row, num_basis, is_row_major);
      rfx_train_dataset_.AddGroupLabels(rfx_group_indices);
      rfx_tracker_.Reset(rfx_group_indices);
      rfx_model_.Reset(num_basis, num_groups);
      num_rfx_groups_ = num_groups;
      num_rfx_basis_ = num_basis;
      has_rfx_ = true;
      rfx_model_.SetWorkingParameter(alpha_init);
      rfx_model_.SetGroupParameters(xi_init);
      rfx_model_.SetWorkingParameterCovariance(sigma_alpha_init);
      rfx_model_.SetGroupParameterCovariance(sigma_xi_init);
      rfx_model_.SetVariancePriorShape(sigma_xi_shape);
      rfx_model_.SetVariancePriorScale(sigma_xi_scale);
    } else {
      rfx_test_dataset_ = RandomEffectsDataset();
      rfx_test_dataset_.AddBasis(rfx_basis, num_row, num_basis, is_row_major);
      rfx_test_dataset_.AddGroupLabels(rfx_group_indices);
    }
  }

  void AddTrainOutcome(double* outcome, data_size_t num_row) {
    train_outcome_ = ColumnVector();
    train_outcome_.LoadData(outcome, num_row);
  }

  void RunSampler(
    BARTResult& output, std::vector<FeatureType>& feature_types, std::vector<double>& variable_weights, 
    int num_trees, int num_gfr, int num_burnin, int num_mcmc, double global_var_init, Eigen::MatrixXd& leaf_cov_init, 
    double alpha, double beta, double nu, double lamb, double a_leaf, double b_leaf, int min_samples_leaf, int cutpoint_grid_size, 
    bool sample_global_var, bool sample_leaf_var, int random_seed = -1, int max_depth = -1
  ) {
    // Unpack sampling details
    num_gfr_ = num_gfr;
    num_burnin_ = num_burnin;
    num_mcmc_ = num_mcmc;
    int num_samples = num_gfr + num_burnin + num_mcmc;

    // Random number generation
    std::mt19937 rng;
    if (random_seed == -1) {
      std::random_device rd;
      std::mt19937 rng(rd());
    }
    else {
      std::mt19937 rng(random_seed);
    }

    // Obtain references to forest / parameter samples and predictions in BARTResult
    ForestContainer* forest_samples = output.GetForests();
    RandomEffectsContainer* rfx_container = output.GetRFXContainer();
    LabelMapper* label_mapper = output.GetRFXLabelMapper();
    std::vector<double>& sigma2_samples = output.GetGlobalVarianceSamples();
    std::vector<double>& tau_samples = output.GetLeafVarianceSamples();
    std::vector<double>& forest_train_preds = output.GetForestTrainPreds();
    std::vector<double>& forest_test_preds = output.GetForestTestPreds();
    std::vector<double>& rfx_train_preds = output.GetRFXTrainPreds();
    std::vector<double>& rfx_test_preds = output.GetRFXTestPreds();
    std::vector<double>& outcome_train_preds = output.GetOutcomeTrainPreds();
    std::vector<double>& outcome_test_preds = output.GetOutcomeTestPreds();

    // Update RFX output containers
    if (has_rfx_) {
      rfx_container->Initialize(num_rfx_basis_, num_rfx_groups_);
      label_mapper->Initialize(rfx_tracker_.GetLabelMap());
    }

    // Clear and prepare vectors to store results
    forest_train_preds.clear();
    forest_train_preds.resize(num_samples*num_train_);
    outcome_train_preds.clear();
    outcome_train_preds.resize(num_samples*num_train_);
    if (has_test_set_) {
      forest_test_preds.clear();
      forest_test_preds.resize(num_samples*num_test_);
      outcome_test_preds.clear();
      outcome_test_preds.resize(num_samples*num_test_);
    }
    if (sample_global_var) {
      sigma2_samples.clear();
      sigma2_samples.resize(num_samples);
    }
    if (sample_leaf_var) {
      tau_samples.clear();
      tau_samples.resize(num_samples);
    }
    if (has_rfx_) {
      rfx_train_preds.clear();
      rfx_train_preds.resize(num_samples*num_train_);
      if (has_test_set_) {
        rfx_test_preds.clear();
        rfx_test_preds.resize(num_samples*num_test_);
      }
    }
    
    // Initialize tracker and tree prior
    ForestTracker tracker = ForestTracker(train_dataset_.GetCovariates(), feature_types, num_trees, num_train_);
    TreePrior tree_prior = TreePrior(alpha, beta, min_samples_leaf, max_depth);

    // Initialize global variance model
    GlobalHomoskedasticVarianceModel global_var_model = GlobalHomoskedasticVarianceModel();

    // Initialize leaf variance model
    LeafNodeHomoskedasticVarianceModel leaf_var_model = LeafNodeHomoskedasticVarianceModel();
    double leaf_var;
    if (sample_leaf_var) {
      CHECK_EQ(leaf_cov_init.rows(),1);
      CHECK_EQ(leaf_cov_init.cols(),1);
      leaf_var = leaf_cov_init(0,0);
    }

    // Initialize leaf model and samplers
    // TODO: add template specialization for GaussianMultivariateRegressionLeafModel which takes Eigen::MatrixXd&
    // as initialization parameter instead of double
    ModelType leaf_model = ModelType(leaf_cov_init);
    GFRForestSampler<ModelType> gfr_sampler = GFRForestSampler<ModelType>(cutpoint_grid_size);
    MCMCForestSampler<ModelType> mcmc_sampler = MCMCForestSampler<ModelType>();

    // Running variable for current sampled value of global outcome variance parameter
    double global_var = global_var_init;
    Eigen::MatrixXd leaf_cov = leaf_cov_init;

    // Run the XBART Gibbs sampler
    int iter = 0;
    if (num_gfr > 0) {
      for (int i = 0; i < num_gfr; i++) {
        // Sample the forests
        gfr_sampler.SampleOneIter(tracker, *forest_samples, leaf_model, train_dataset_, train_outcome_, tree_prior, 
                                  rng, variable_weights, global_var, feature_types, false);
        
        if (sample_global_var) {
          // Sample the global outcome variance
          global_var = global_var_model.SampleVarianceParameter(train_outcome_.GetData(), nu, lamb, rng);
          sigma2_samples.at(iter) = global_var;
        }
        
        if (sample_leaf_var) {
          // Sample the leaf model variance
          TreeEnsemble* ensemble = forest_samples->GetEnsemble(iter);
          leaf_var = leaf_var_model.SampleVarianceParameter(ensemble, a_leaf, b_leaf, rng);
          tau_samples.at(iter) = leaf_var;
          leaf_cov(0,0) = leaf_var;
        }

        // Increment sample counter
        iter++;
      }
    }

    // Run the MCMC sampler
    if (num_burnin + num_mcmc > 0) {
      for (int i = 0; i < num_burnin + num_mcmc; i++) {
        // Sample the forests
        mcmc_sampler.SampleOneIter(tracker, *forest_samples, leaf_model, train_dataset_, train_outcome_, tree_prior, 
                                  rng, variable_weights, global_var, true);
        
        if (sample_global_var) {
          // Sample the global outcome variance
          global_var = global_var_model.SampleVarianceParameter(train_outcome_.GetData(), nu, lamb, rng);
          sigma2_samples.at(iter) = global_var;
        }
        
        if (sample_leaf_var) {
          // Sample the leaf model variance
          TreeEnsemble* ensemble = forest_samples->GetEnsemble(iter);
          leaf_var = leaf_var_model.SampleVarianceParameter(ensemble, a_leaf, b_leaf, rng);
          tau_samples.at(iter) = leaf_var;
          leaf_cov(0,0) = leaf_var;
        }

        // Increment sample counter
        iter++;
      }
    }

    // Predict forests and rfx
    forest_samples->PredictInPlace(train_dataset_, forest_train_preds);
    if (has_test_set_) forest_samples->PredictInPlace(test_dataset_, forest_test_preds);
    if (has_rfx_) {
      rfx_container->Predict(rfx_train_dataset_, *label_mapper, rfx_train_preds);
      for (data_size_t ind = 0; ind < rfx_train_preds.size(); ind++) {
        outcome_train_preds.at(ind) = rfx_train_preds.at(ind) + forest_train_preds.at(ind);
      }
      if (has_test_set_) {
        rfx_container->Predict(rfx_test_dataset_, *label_mapper, rfx_test_preds);
        for (data_size_t ind = 0; ind < rfx_test_preds.size(); ind++) {
          outcome_test_preds.at(ind) = rfx_test_preds.at(ind) + forest_test_preds.at(ind);
        }
      }
    } else {
      forest_samples->PredictInPlace(train_dataset_, outcome_train_preds);
      if (has_test_set_) forest_samples->PredictInPlace(test_dataset_, outcome_test_preds);
    }
  }
 
 private:
  // "Core" BART / XBART sampling objects
  // Dimensions
  int num_gfr_{0};
  int num_burnin_{0};
  int num_mcmc_{0};
  int num_train_{0};
  int num_test_{0};
  bool has_test_set_{false};
  // Data objects
  ForestDataset train_dataset_;
  ForestDataset test_dataset_;
  ColumnVector train_outcome_;

  // (Optional) random effect sampling details
  // Dimensions
  int num_rfx_groups_{0};
  int num_rfx_basis_{0};
  bool has_rfx_{false};
  // Data objects
  RandomEffectsDataset rfx_train_dataset_;
  RandomEffectsDataset rfx_test_dataset_;
  RandomEffectsTracker rfx_tracker_;
  MultivariateRegressionRandomEffectsModel rfx_model_;
};

class BARTResultSimplified {
 public:
  BARTResultSimplified(int num_trees, int output_dimension = 1, bool is_leaf_constant = true) : 
    forests_samples_{num_trees, output_dimension, is_leaf_constant} {}
  ~BARTResultSimplified() {}
  ForestContainer& GetForests() {return forests_samples_;}
  std::vector<double>& GetTrainPreds() {return raw_preds_train_;}
  std::vector<double>& GetTestPreds() {return raw_preds_test_;}
  std::vector<double>& GetVarianceSamples() {return sigma_samples_;}
  int NumGFRSamples() {return num_gfr_;}
  int NumBurninSamples() {return num_burnin_;}
  int NumMCMCSamples() {return num_mcmc_;}
  int NumTrainObservations() {return num_train_;}
  int NumTestObservations() {return num_test_;}
  bool HasTestSet() {return has_test_set_;}
 private:
  ForestContainer forests_samples_;
  std::vector<double> raw_preds_train_;
  std::vector<double> raw_preds_test_;
  std::vector<double> sigma_samples_;
  int num_gfr_{0};
  int num_burnin_{0};
  int num_mcmc_{0};
  int num_train_{0};
  int num_test_{0};
  bool has_test_set_{false};
};

class BARTDispatcherSimplified {
 public:
  BARTDispatcherSimplified() {}
  ~BARTDispatcherSimplified() {}
  BARTResultSimplified CreateOutputObject(int num_trees, int output_dimension = 1, bool is_leaf_constant = true) {
    return BARTResultSimplified(num_trees, output_dimension, is_leaf_constant);
  }
  void RunSampler(
    BARTResultSimplified& output, std::vector<FeatureType>& feature_types, std::vector<double>& variable_weights, 
    int num_trees, int num_gfr, int num_burnin, int num_mcmc, double global_var_init, double leaf_var_init, 
    double alpha, double beta, double nu, double lamb, double a_leaf, double b_leaf, int min_samples_leaf, 
    int cutpoint_grid_size, int random_seed = -1, int max_depth = -1
  ) {
    // Unpack sampling details
    num_gfr_ = num_gfr;
    num_burnin_ = num_burnin;
    num_mcmc_ = num_mcmc;
    int num_samples = num_gfr + num_burnin + num_mcmc;

    // Random number generation
    std::mt19937 rng;
    if (random_seed == -1) {
      std::random_device rd;
      std::mt19937 rng(rd());
    }
    else {
      std::mt19937 rng(random_seed);
    }

    // Obtain references to forest / parameter samples and predictions in BARTResult
    ForestContainer& forest_samples = output.GetForests();
    std::vector<double>& sigma2_samples = output.GetVarianceSamples();
    std::vector<double>& train_preds = output.GetTrainPreds();
    std::vector<double>& test_preds = output.GetTestPreds();

    // Clear and prepare vectors to store results
    sigma2_samples.clear();
    train_preds.clear();
    test_preds.clear();
    sigma2_samples.resize(num_samples);
    train_preds.resize(num_samples*num_train_);
    if (has_test_set_) test_preds.resize(num_samples*num_test_);
    
    // Initialize tracker and tree prior
    ForestTracker tracker = ForestTracker(train_dataset_.GetCovariates(), feature_types, num_trees, num_train_);
    TreePrior tree_prior = TreePrior(alpha, beta, min_samples_leaf, max_depth);

    // Initialize variance model
    GlobalHomoskedasticVarianceModel global_var_model = GlobalHomoskedasticVarianceModel();

    // Initialize leaf model and samplers
    GaussianConstantLeafModel leaf_model = GaussianConstantLeafModel(leaf_var_init);
    GFRForestSampler<GaussianConstantLeafModel> gfr_sampler = GFRForestSampler<GaussianConstantLeafModel>(cutpoint_grid_size);
    MCMCForestSampler<GaussianConstantLeafModel> mcmc_sampler = MCMCForestSampler<GaussianConstantLeafModel>();

    // Running variable for current sampled value of global outcome variance parameter
    double global_var = global_var_init;

    // Run the XBART Gibbs sampler
    int iter = 0;
    if (num_gfr > 0) {
      for (int i = 0; i < num_gfr; i++) {
        // Sample the forests
        gfr_sampler.SampleOneIter(tracker, forest_samples, leaf_model, train_dataset_, train_outcome_, tree_prior, 
                                  rng, variable_weights, global_var, feature_types, false);
        
        // Sample the global outcome
        global_var = global_var_model.SampleVarianceParameter(train_outcome_.GetData(), nu, lamb, rng);
        sigma2_samples.at(iter) = global_var;

        // Increment sample counter
        iter++;
      }
    }

    // Run the MCMC sampler
    if (num_burnin + num_mcmc > 0) {
      for (int i = 0; i < num_burnin + num_mcmc; i++) {
        // Sample the forests
        mcmc_sampler.SampleOneIter(tracker, forest_samples, leaf_model, train_dataset_, train_outcome_, tree_prior, 
                                  rng, variable_weights, global_var, true);
        
        // Sample the global outcome
        global_var = global_var_model.SampleVarianceParameter(train_outcome_.GetData(), nu, lamb, rng);
        sigma2_samples.at(iter) = global_var;

        // Increment sample counter
        iter++;
      }
    }

    // Predict forests
    forest_samples.PredictInPlace(train_dataset_, train_preds);
    if (has_test_set_) forest_samples.PredictInPlace(test_dataset_, test_preds);
  }
  void AddDataset(double* covariates, data_size_t num_row, int num_col, bool is_row_major, bool train) {
    if (train) {
      train_dataset_ = ForestDataset();
      train_dataset_.AddCovariates(covariates, num_row, num_col, is_row_major);
      num_train_ = num_row;
    } else {
      test_dataset_ = ForestDataset();
      test_dataset_.AddCovariates(covariates, num_row, num_col, is_row_major);
      has_test_set_ = true;
      num_test_ = num_row;
    }
  }
  void AddTrainOutcome(double* outcome, data_size_t num_row) {
    train_outcome_ = ColumnVector();
    train_outcome_.LoadData(outcome, num_row);
  }
 private:
  // Sampling details
  int num_gfr_{0};
  int num_burnin_{0};
  int num_mcmc_{0};
  int num_train_{0};
  int num_test_{0};
  bool has_test_set_{false};

  // Sampling data objects
  ForestDataset train_dataset_;
  ForestDataset test_dataset_;
  ColumnVector train_outcome_;
};


} // namespace StochTree

#endif // STOCHTREE_SAMPLING_DISPATCH_H_
