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

namespace StochTree {

class BARTResult {
 public:
  BARTResult(int num_trees, int output_dimension = 1, bool is_leaf_constant = true) : 
    forests_samples_{num_trees, output_dimension, is_leaf_constant} {}
  ~BARTResult() {}
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

class BARTDispatcher {
 public:
  BARTDispatcher() {}
  ~BARTDispatcher() {}
  BARTResult CreateOutputObject(int num_trees, int output_dimension = 1, bool is_leaf_constant = true);
  void RunSampler(
    BARTResult& output, std::vector<FeatureType>& feature_types, std::vector<double>& variable_weights, 
    int num_trees, int num_gfr, int num_burnin, int num_mcmc, double global_var_init, double leaf_var_init, 
    double alpha, double beta, double nu, double lamb, double a_leaf, double b_leaf, int min_samples_leaf, 
    int cutpoint_grid_size, int random_seed = -1
  );
  void AddDataset(double* covariates, data_size_t num_row, int num_col, bool is_row_major, bool train);
  void AddTrainOutcome(double* outcome, data_size_t num_row);
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
