/*!
 * Test of the ensemble prediction method
 */
#include <gtest/gtest.h>
#include <testutils.h>
#include <stochtree/data.h>
#include <stochtree/ensemble.h>
#include <stochtree/interface.h>
#include <stochtree/tree.h>
#include <iostream>
#include <memory>

TEST(Ensemble, PredictUnivariate) {
  // Generate some in-memory data
  // Initially generated as five independent 
  // standard uniform covariates and the outcome
  // (sixth) column as X * (5, 10, 0, 0, 0) plus 
  // a small amount of normal noise 
  StochTree::data_size_t n = 10;
  int p = 5;
  std::vector<double> data_vector = {
    0.80195179, 0.56627934, 0.85066760, 0.55333717, 0.67683323, 9.665316, 
    0.62574001, 0.71743631, 0.92349582, 0.87779366, 0.68785497, 10.316500, 
    0.97678451, 0.96493876, 0.78560157, 0.05775895, 0.43964365, 14.541611, 
    0.68112895, 0.69137664, 0.09408876, 0.50819643, 0.15438995, 10.329346, 
    0.31852576, 0.17459775, 0.33655242, 0.09688789, 0.22522345, 3.340881, 
    0.97831176, 0.34125955, 0.70473897, 0.09524901, 0.57536714, 8.305549, 
    0.06442722, 0.50771215, 0.72807792, 0.51536037, 0.15120395, 5.392728, 
    0.88865773, 0.25730240, 0.95612361, 0.05282520, 0.39155233, 7.017450, 
    0.50874687, 0.39883808, 0.98027747, 0.02904932, 0.37047444, 6.527128, 
    0.65685255, 0.62172827, 0.19717188, 0.90977738, 0.92560747, 9.501689
  };
  
  // Declare unique pointer to training data
  std::unique_ptr<StochTree::Dataset> dataset;
  
  // Define any config parameters that aren't defaults
  const char* params = "header=true outcome_columns=5 treatment_columns=-1 num_trees=2";
  auto param = StochTree::Config::Str2Map(params);
  StochTree::Config config;
  config.Set(param);

  // Define data loader
  StochTree::DataLoader dataset_loader(config, 1, nullptr);

  // Load some test data
  dataset.reset(dataset_loader.ConstructFromMatrix(data_vector.data(), p + 1, n, true));
  
  // Create a small ensemble
  int output_dim = 1;
  StochTree::TreeEnsemble ensemble(config, output_dim);

  // Set the first tree with a simple split rule
  auto* tree = ensemble.GetTree(0);
  tree->ExpandNode(0, 0, 0.5, true, -5, 5);
  tree = ensemble.GetTree(1);
  tree->ExpandNode(0, 1, 0.5, true, -2.5, 2.5);

  std::vector<double> result(n*output_dim);
  std::vector<double> expected_pred = {7.5, 7.5, 7.5, 7.5, -7.5, 2.5, -2.5, 2.5, 2.5, 7.5};
  ensemble.PredictInplace(dataset.get(), result, 0);

  for (int i = 0; i < n; i++) {
    ASSERT_NEAR(expected_pred[i], result[i], 0.01);
  }
}

TEST(Ensemble, PredictMultivariate) {
  // Generate some in-memory data
  // Initially generated as five independent 
  // standard uniform covariates and the outcome
  // (sixth) column as X * (5, 10, 0, 0, 0) plus 
  // a small amount of normal noise 
  StochTree::data_size_t n = 10;
  int p = 5;
  std::vector<double> data_vector = {
    0.80195179, 0.56627934, 0.85066760, 0.55333717, 0.67683323, 9.665316, 
    0.62574001, 0.71743631, 0.92349582, 0.87779366, 0.68785497, 10.316500, 
    0.97678451, 0.96493876, 0.78560157, 0.05775895, 0.43964365, 14.541611, 
    0.68112895, 0.69137664, 0.09408876, 0.50819643, 0.15438995, 10.329346, 
    0.31852576, 0.17459775, 0.33655242, 0.09688789, 0.22522345, 3.340881, 
    0.97831176, 0.34125955, 0.70473897, 0.09524901, 0.57536714, 8.305549, 
    0.06442722, 0.50771215, 0.72807792, 0.51536037, 0.15120395, 5.392728, 
    0.88865773, 0.25730240, 0.95612361, 0.05282520, 0.39155233, 7.017450, 
    0.50874687, 0.39883808, 0.98027747, 0.02904932, 0.37047444, 6.527128, 
    0.65685255, 0.62172827, 0.19717188, 0.90977738, 0.92560747, 9.501689
  };
  
  // Declare unique pointer to training data
  std::unique_ptr<StochTree::Dataset> dataset;
  
  // Define any config parameters that aren't defaults
  const char* params = "header=true outcome_columns=5 treatment_columns=-1 num_trees=2";
  auto param = StochTree::Config::Str2Map(params);
  StochTree::Config config;
  config.Set(param);

  // Define data loader
  StochTree::DataLoader dataset_loader(config, 1, nullptr);

  // Load some test data
  dataset.reset(dataset_loader.ConstructFromMatrix(data_vector.data(), p + 1, n, true));
  
  // Create a small ensemble
  int output_dim = 2;
  StochTree::TreeEnsemble ensemble(config, output_dim);

  // Set the first tree with a simple split rule
  auto* tree = ensemble.GetTree(0);
  tree->ExpandNode(0, 0, 0.5, true, std::vector<double>{-5, -2.5}, std::vector<double>{5, 2.5});
  tree = ensemble.GetTree(1);
  tree->ExpandNode(0, 1, 0.5, true, std::vector<double>{-2.5, -1.25}, std::vector<double>{2.5, 1.25});

  std::vector<double> result(n*output_dim);
  std::vector<double> expected_pred = {7.5, 3.75, 7.5, 3.75, 7.5, 3.75, 7.5, 3.75, -7.5, -3.75, 2.5, 1.25, -2.5, -1.25, 2.5, 1.25, 2.5, 1.25, 7.5, 3.75};
  ensemble.PredictInplace(dataset.get(), result, 0);

  for (int i = 0; i < expected_pred.size(); i++) {
    ASSERT_NEAR(expected_pred[i], result[i], 0.01);
  }
}

