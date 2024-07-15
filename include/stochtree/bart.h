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
  std::unique_ptr<ForestContainer> forest_samples_;
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

  void AddTrainOutcome(double* outcome, data_size_t num_row) {
    train_outcome_ = ColumnVector();
    train_outcome_.LoadData(outcome, num_row);
  }

  void RunSampler(
    BARTResult& output, std::vector<FeatureType>& feature_types, std::vector<double>& variable_weights, 
    int num_trees, int num_gfr, int num_burnin, int num_mcmc, double global_var_init, double leaf_var_init, 
    double alpha, double beta, double nu, double lamb, double a_leaf, double b_leaf, int min_samples_leaf, 
    int cutpoint_grid_size, int random_seed = -1
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
    TreePrior tree_prior = TreePrior(alpha, beta, min_samples_leaf);

    // Initialize variance model
    GlobalHomoskedasticVarianceModel global_var_model = GlobalHomoskedasticVarianceModel();

    // Initialize leaf model and samplers
    // TODO: add template specialization for GaussianMultivariateRegressionLeafModel which takes Eigen::MatrixXd&
    // as initialization parameter instead of double
    ModelType leaf_model = ModelType(leaf_var_init);
    GFRForestSampler<ModelType> gfr_sampler = GFRForestSampler<ModelType>(cutpoint_grid_size);
    MCMCForestSampler<ModelType> mcmc_sampler = MCMCForestSampler<ModelType>();

    // Running variable for current sampled value of global outcome variance parameter
    double global_var = global_var_init;

    // Run the XBART Gibbs sampler
    int iter = 0;
    if (num_gfr > 0) {
      for (int i = 0; i < num_gfr; i++) {
        // Sample the forests
        gfr_sampler.SampleOneIter(tracker, *forest_samples, leaf_model, train_dataset_, train_outcome_, tree_prior, 
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
        mcmc_sampler.SampleOneIter(tracker, *forest_samples, leaf_model, train_dataset_, train_outcome_, tree_prior, 
                                  rng, variable_weights, global_var, true);
        
        // Sample the global outcome
        global_var = global_var_model.SampleVarianceParameter(train_outcome_.GetData(), nu, lamb, rng);
        sigma2_samples.at(iter) = global_var;

        // Increment sample counter
        iter++;
      }
    }

    // Predict forests
    forest_samples->PredictInPlace(train_dataset_, train_preds);
    if (has_test_set_) forest_samples->PredictInPlace(test_dataset_, test_preds);
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
