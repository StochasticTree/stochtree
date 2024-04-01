#include <gtest/gtest.h>
#include <testutils.h>
#include <stochtree/cutpoint_candidates.h>
#include <stochtree/leaf_model.h>
#include <stochtree/log.h>
#include <stochtree/partition_tracker.h>
#include <iostream>
#include <memory>
#include <vector>

TEST(LeafConstantModel, FullEnumeration) {
  // Load test data
  StochTree::TestUtils::TestDataset test_dataset;
  test_dataset = StochTree::TestUtils::LoadSmallDatasetUnivariateBasis();
  std::vector<StochTree::FeatureType> feature_types(test_dataset.x_cols, StochTree::FeatureType::kNumeric);

  // Construct datasets
  using data_size_t = StochTree::data_size_t;
  data_size_t n = test_dataset.n;
  int p = test_dataset.x_cols;
  StochTree::ForestDataset dataset = StochTree::ForestDataset();
  dataset.AddCovariates(test_dataset.covariates.data(), n, test_dataset.x_cols, test_dataset.row_major);
  StochTree::ColumnVector residual = StochTree::ColumnVector(test_dataset.outcome.data(), n);

  // Construct a ForestTracker
  int num_trees = 1;
  StochTree::ForestTracker tracker = StochTree::ForestTracker(dataset.GetCovariates(), feature_types, num_trees, n);

  // Set sampling parameters
  double alpha = 0.95;
  double beta = 1.25;
  int min_samples_leaf = 1;
  double global_variance = 1.;
  double tau = 1.;
  int cutpoint_grid_size = n;
  StochTree::TreePrior tree_prior = StochTree::TreePrior(alpha, beta, min_samples_leaf);

  // Construct temporary data structures needed to enumerate splits
  std::vector<double> log_cutpoint_evaluations;
  std::vector<int> cutpoint_features;
  std::vector<double> cutpoint_values;
  std::vector<StochTree::FeatureType> cutpoint_feature_types;
  StochTree::data_size_t valid_cutpoint_count = 0;
  StochTree::CutpointGridContainer cutpoint_grid_container(dataset.GetCovariates(), residual.GetData(), cutpoint_grid_size);

  // Initialize a leaf model
  StochTree::GaussianConstantLeafModel leaf_model = StochTree::GaussianConstantLeafModel(tau);

  // Evaluate all possible cutpoints
  leaf_model.EvaluateAllPossibleSplits(dataset, tracker, residual, tree_prior, global_variance, 0, 0, log_cutpoint_evaluations, cutpoint_features, 
                                       cutpoint_values, cutpoint_feature_types, valid_cutpoint_count, cutpoint_grid_container, 0, n, feature_types);

  // Check that there are (n - 2*min_samples_leaf + 1)*p + 1 cutpoints considered
  ASSERT_EQ(log_cutpoint_evaluations.size(), (n - 2*min_samples_leaf + 1)*p + 1);

  // Check the values of the cutpoint evaluations
  std::vector<double> expected_split_evals{3.773828, 3.349927, 3.001568, 3.085074, 2.989927, 3.101841, 2.980939, 3.068029, 3.822045, 3.663843, 3.710592, 3.354912, 3.135288,
                                           3.553728, 2.969388, 3.540838, 3.961885, 3.822045, 4.908861, 4.032006, 4.083473, 4.442268, 5.023573, 4.171735, 3.353457, 3.862124,
                                           3.323620, 3.998112, 3.425777, 3.096926, 3.131347, 2.947921, 2.935892, 3.224115, 3.144767, 3.213065, 3.863427, 3.792850, 3.146056,
                                           3.348693, 3.487161, 4.600861, 4.226219, 4.879161, 3.773828, 3.940111};
  for (int i = 0; i < log_cutpoint_evaluations.size(); i++) {
    ASSERT_NEAR(log_cutpoint_evaluations[i], expected_split_evals[i], 0.01);
  }
}

TEST(LeafConstantModel, CutpointThinning) {
  // Load test data
  StochTree::TestUtils::TestDataset test_dataset;
  test_dataset = StochTree::TestUtils::LoadSmallDatasetUnivariateBasis();
  std::vector<StochTree::FeatureType> feature_types(test_dataset.x_cols, StochTree::FeatureType::kNumeric);

  // Construct datasets
  using data_size_t = StochTree::data_size_t;
  data_size_t n = test_dataset.n;
  int p = test_dataset.x_cols;
  StochTree::ForestDataset dataset = StochTree::ForestDataset();
  dataset.AddCovariates(test_dataset.covariates.data(), n, test_dataset.x_cols, test_dataset.row_major);
  StochTree::ColumnVector residual = StochTree::ColumnVector(test_dataset.outcome.data(), n);

  // Construct a ForestTracker
  int num_trees = 1;
  StochTree::ForestTracker tracker = StochTree::ForestTracker(dataset.GetCovariates(), feature_types, num_trees, n);

  // Set sampling parameters
  double alpha = 0.95;
  double beta = 1.25;
  int min_samples_leaf = 1;
  double global_variance = 1.;
  double tau = 1.;
  int cutpoint_grid_size = 5;
  StochTree::TreePrior tree_prior = StochTree::TreePrior(alpha, beta, min_samples_leaf);

  // Construct temporary data structures needed to enumerate splits
  std::vector<double> log_cutpoint_evaluations;
  std::vector<int> cutpoint_features;
  std::vector<double> cutpoint_values;
  std::vector<StochTree::FeatureType> cutpoint_feature_types;
  StochTree::data_size_t valid_cutpoint_count = 0;
  StochTree::CutpointGridContainer cutpoint_grid_container(dataset.GetCovariates(), residual.GetData(), cutpoint_grid_size);

  // Initialize a leaf model
  StochTree::GaussianConstantLeafModel leaf_model = StochTree::GaussianConstantLeafModel(tau);

  // Evaluate all possible cutpoints
  leaf_model.EvaluateAllPossibleSplits(dataset, tracker, residual, tree_prior, global_variance, 0, 0, log_cutpoint_evaluations, cutpoint_features, 
                                       cutpoint_values, cutpoint_feature_types, valid_cutpoint_count, cutpoint_grid_container, 0, n, feature_types);

  // Check that there are (n - 2*min_samples_leaf + 1)*p + 1 cutpoints considered
  ASSERT_EQ(log_cutpoint_evaluations.size(), (cutpoint_grid_size - 1)*p + 1);

  // Check the values of the cutpoint evaluations
  std::vector<double> expected_split_evals{3.349927, 3.085074, 3.101841, 3.068029, 3.710592, 3.135288, 2.969388, 3.961885, 4.032006, 
                                           4.442268, 4.171735, 3.862124, 3.425777, 3.131347, 2.935892, 3.144767, 3.792850, 3.348693, 
                                           4.600861, 4.879161, 3.940111};
  for (int i = 0; i < log_cutpoint_evaluations.size(); i++) {
    ASSERT_NEAR(log_cutpoint_evaluations[i], expected_split_evals[i], 0.01);
  }
}

TEST(LeafUnivariateRegressionModel, FullEnumeration) {
  // Load test data
  StochTree::TestUtils::TestDataset test_dataset;
  test_dataset = StochTree::TestUtils::LoadSmallDatasetUnivariateBasis();
  std::vector<StochTree::FeatureType> feature_types(test_dataset.x_cols, StochTree::FeatureType::kNumeric);

  // Construct datasets
  using data_size_t = StochTree::data_size_t;
  data_size_t n = test_dataset.n;
  int p = test_dataset.x_cols;
  StochTree::ForestDataset dataset = StochTree::ForestDataset();
  dataset.AddCovariates(test_dataset.covariates.data(), n, test_dataset.x_cols, test_dataset.row_major);
  dataset.AddBasis(test_dataset.omega.data(), test_dataset.n, test_dataset.omega_cols, test_dataset.row_major);
  StochTree::ColumnVector residual = StochTree::ColumnVector(test_dataset.outcome.data(), n);

  // Construct a ForestTracker
  int num_trees = 1;
  StochTree::ForestTracker tracker = StochTree::ForestTracker(dataset.GetCovariates(), feature_types, num_trees, n);

  // Set sampling parameters
  double alpha = 0.95;
  double beta = 1.25;
  int min_samples_leaf = 1;
  double global_variance = 1.;
  double tau = 1.;
  int cutpoint_grid_size = n;
  StochTree::TreePrior tree_prior = StochTree::TreePrior(alpha, beta, min_samples_leaf);

  // Construct temporary data structures needed to enumerate splits
  std::vector<double> log_cutpoint_evaluations;
  std::vector<int> cutpoint_features;
  std::vector<double> cutpoint_values;
  std::vector<StochTree::FeatureType> cutpoint_feature_types;
  StochTree::data_size_t valid_cutpoint_count = 0;
  StochTree::CutpointGridContainer cutpoint_grid_container(dataset.GetCovariates(), residual.GetData(), cutpoint_grid_size);

  // Initialize a leaf model
  StochTree::GaussianUnivariateRegressionLeafModel leaf_model = StochTree::GaussianUnivariateRegressionLeafModel(tau);

  // Evaluate all possible cutpoints
  leaf_model.EvaluateAllPossibleSplits(dataset, tracker, residual, tree_prior, global_variance, 0, 0, log_cutpoint_evaluations, cutpoint_features, 
                                       cutpoint_values, cutpoint_feature_types, valid_cutpoint_count, cutpoint_grid_container, 0, n, feature_types);

  // Check that there are (n - 2*min_samples_leaf + 1)*p + 1 cutpoints considered
  ASSERT_EQ(log_cutpoint_evaluations.size(), (n - 2*min_samples_leaf + 1)*p + 1);

  // Check the values of the cutpoint evaluations
  std::vector<double> expected_split_evals{4.978556, 4.067172, 3.823266, 3.850415, 3.796388, 3.791759, 3.864699, 3.970411, 5.105565, 4.886562, 4.812292, 4.450645, 4.180200, 
                                           4.625754, 3.983956, 4.906961, 5.307099, 5.105565, 6.057032, 5.463854, 5.312733, 5.504701, 5.872222, 4.936127, 4.203568, 4.192258, 
                                           4.633795, 4.060248, 4.032323, 4.040458, 4.176712, 3.809356, 3.854872, 4.404108, 4.243114, 4.116230, 5.167773, 5.031023, 4.203335, 
                                           4.094302, 4.280394, 5.557678, 5.394644, 5.945185, 4.978556, 5.069763};
  for (int i = 0; i < log_cutpoint_evaluations.size(); i++) {
    ASSERT_NEAR(log_cutpoint_evaluations[i], expected_split_evals[i], 0.01);
  }
}

TEST(LeafUnivariateRegressionModel, CutpointThinning) {
  // Load test data
  StochTree::TestUtils::TestDataset test_dataset;
  test_dataset = StochTree::TestUtils::LoadSmallDatasetUnivariateBasis();
  std::vector<StochTree::FeatureType> feature_types(test_dataset.x_cols, StochTree::FeatureType::kNumeric);

  // Construct datasets
  using data_size_t = StochTree::data_size_t;
  data_size_t n = test_dataset.n;
  int p = test_dataset.x_cols;
  StochTree::ForestDataset dataset = StochTree::ForestDataset();
  dataset.AddCovariates(test_dataset.covariates.data(), n, test_dataset.x_cols, test_dataset.row_major);
  dataset.AddBasis(test_dataset.omega.data(), test_dataset.n, test_dataset.omega_cols, test_dataset.row_major);
  StochTree::ColumnVector residual = StochTree::ColumnVector(test_dataset.outcome.data(), n);

  // Construct a ForestTracker
  int num_trees = 1;
  StochTree::ForestTracker tracker = StochTree::ForestTracker(dataset.GetCovariates(), feature_types, num_trees, n);

  // Set sampling parameters
  double alpha = 0.95;
  double beta = 1.25;
  int min_samples_leaf = 1;
  double global_variance = 1.;
  double tau = 1.;
  int cutpoint_grid_size = 5;
  StochTree::TreePrior tree_prior = StochTree::TreePrior(alpha, beta, min_samples_leaf);

  // Construct temporary data structures needed to enumerate splits
  std::vector<double> log_cutpoint_evaluations;
  std::vector<int> cutpoint_features;
  std::vector<double> cutpoint_values;
  std::vector<StochTree::FeatureType> cutpoint_feature_types;
  StochTree::data_size_t valid_cutpoint_count = 0;
  StochTree::CutpointGridContainer cutpoint_grid_container(dataset.GetCovariates(), residual.GetData(), cutpoint_grid_size);

  // Initialize a leaf model
  StochTree::GaussianUnivariateRegressionLeafModel leaf_model = StochTree::GaussianUnivariateRegressionLeafModel(tau);

  // Evaluate all possible cutpoints
  leaf_model.EvaluateAllPossibleSplits(dataset, tracker, residual, tree_prior, global_variance, 0, 0, log_cutpoint_evaluations, cutpoint_features, 
                                       cutpoint_values, cutpoint_feature_types, valid_cutpoint_count, cutpoint_grid_container, 0, n, feature_types);

  // Check that there are (n - 2*min_samples_leaf + 1)*p + 1 cutpoints considered
  ASSERT_EQ(log_cutpoint_evaluations.size(), (cutpoint_grid_size - 1)*p + 1);

  // Check the values of the cutpoint evaluations
  std::vector<double> expected_split_evals{4.067172, 3.850415, 3.791759, 3.970411, 4.812292, 4.180200, 3.983956, 5.307099, 5.463854, 
                                           5.504701, 4.936127, 4.192258, 4.032323, 4.176712, 3.854872, 4.243114, 5.031023, 4.094302, 
                                           5.557678, 5.945185, 5.069763};
  for (int i = 0; i < log_cutpoint_evaluations.size(); i++) {
    ASSERT_NEAR(log_cutpoint_evaluations[i], expected_split_evals[i], 0.01);
  }
}
