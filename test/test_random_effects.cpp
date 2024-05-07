/*!
 * Derived from xgboost tree unit test code:
 * https://github.com/dmlc/xgboost/blob/master/tests/cpp/tree/test_tree_model.cc
 */
#include <gtest/gtest.h>
#include <testutils.h>
#include <stochtree/data.h>
#include <stochtree/log.h>
#include <stochtree/random_effects.h>
#include <stochtree/tree.h>
#include <iostream>
#include <memory>

TEST(RandomEffects, Setup) {
  // Load test data
  StochTree::TestUtils::TestDataset test_dataset;
  test_dataset = StochTree::TestUtils::LoadSmallDatasetUnivariateBasis();
  std::vector<StochTree::FeatureType> feature_types(test_dataset.x_cols, StochTree::FeatureType::kNumeric);

  // Construct dataset
  int n = test_dataset.n;
  StochTree::RandomEffectsDataset dataset = StochTree::RandomEffectsDataset();
  dataset.AddBasis(test_dataset.rfx_basis.data(), test_dataset.n, test_dataset.rfx_basis_cols, test_dataset.row_major);
  dataset.AddGroupLabels(test_dataset.rfx_groups);
  
  // Construct tracker, model state, and container
  StochTree::RandomEffectsTracker tracker = StochTree::RandomEffectsTracker(test_dataset.rfx_groups);
  StochTree::MultivariateRegressionRandomEffectsModel model = StochTree::MultivariateRegressionRandomEffectsModel(test_dataset.rfx_basis_cols, test_dataset.rfx_num_groups);
  StochTree::RandomEffectsContainer container = StochTree::RandomEffectsContainer(test_dataset.rfx_basis_cols, test_dataset.rfx_num_groups);

  // Check the internal label map of the RandomEffectsTracker
  std::map<int32_t, int32_t> label_map = tracker.GetLabelMap();
  std::map<int32_t, int32_t> expected_label_map {{1, 0}, {2, 1}};
  ASSERT_EQ(label_map, expected_label_map);
}

TEST(RandomEffects, Construction) {
  // Load test data
  StochTree::TestUtils::TestDataset test_dataset;
  test_dataset = StochTree::TestUtils::LoadSmallDatasetUnivariateBasis();
  std::vector<StochTree::FeatureType> feature_types(test_dataset.x_cols, StochTree::FeatureType::kNumeric);

  // Construct dataset
  int n = test_dataset.n;
  StochTree::RandomEffectsDataset dataset = StochTree::RandomEffectsDataset();
  dataset.AddBasis(test_dataset.rfx_basis.data(), test_dataset.n, test_dataset.rfx_basis_cols, test_dataset.row_major);
  dataset.AddGroupLabels(test_dataset.rfx_groups);
  
  // Construct tracker, model state, and container
  StochTree::RandomEffectsTracker tracker = StochTree::RandomEffectsTracker(test_dataset.rfx_groups);
  StochTree::MultivariateRegressionRandomEffectsModel model = StochTree::MultivariateRegressionRandomEffectsModel(test_dataset.rfx_basis_cols, test_dataset.rfx_num_groups);
  StochTree::RandomEffectsContainer container = StochTree::RandomEffectsContainer(test_dataset.rfx_basis_cols, test_dataset.rfx_num_groups);
  StochTree::LabelMapper label_mapper = StochTree::LabelMapper(tracker.GetLabelMap());

  // Set the values of alpha, xi and sigma in the model state (rather than simulating)
  Eigen::VectorXd alpha(test_dataset.rfx_basis_cols);
  Eigen::MatrixXd xi(test_dataset.rfx_basis_cols, test_dataset.rfx_num_groups);
  Eigen::MatrixXd sigma(test_dataset.rfx_basis_cols, test_dataset.rfx_basis_cols);
  alpha << 1.5;
  xi << 2, 4;
  Eigen::VectorXd xi0 = xi(Eigen::all, 0);
  Eigen::VectorXd xi1 = xi(Eigen::all, 1);
  sigma << 1;
  model.SetWorkingParameter(alpha);
  model.SetGroupParameter(xi0, 0);
  model.SetGroupParameter(xi1, 1);
  model.SetGroupParameterCovariance(sigma);
  
  // Push to the container
  container.AddSample(model);

  // Change values and push a second "sample" to the container
  alpha << 2.0;
  xi << 1, 3;
  xi0 = xi(Eigen::all, 0);
  xi1 = xi(Eigen::all, 1);
  sigma << 1;
  model.SetWorkingParameter(alpha);
  model.SetGroupParameter(xi0, 0);
  model.SetGroupParameter(xi1, 1);
  model.SetGroupParameterCovariance(sigma);
  container.AddSample(model);

  // Check data in the container
  std::vector<double> alpha_retrieved = container.GetAlpha();
  std::vector<double> alpha_expected {1.5, 2.0};
  for (int i = 0; i < alpha_expected.size(); i++) {
    ASSERT_EQ(alpha_retrieved[i], alpha_expected[i]);
  }
  std::vector<double> xi_retrieved = container.GetXi();
  std::vector<double> xi_expected {2, 4, 1, 3};
  for (int i = 0; i < xi_expected.size(); i++) {
    ASSERT_EQ(xi_retrieved[i], xi_expected[i]);
  }
  std::vector<double> beta_retrieved = container.GetBeta();
  std::vector<double> beta_expected {3, 6, 2, 6};
  for (int i = 0; i < beta_expected.size(); i++) {
    ASSERT_EQ(beta_retrieved[i], beta_expected[i]);
  }
}

TEST(RandomEffects, Predict) {
  // Load test data
  StochTree::TestUtils::TestDataset test_dataset;
  test_dataset = StochTree::TestUtils::LoadSmallDatasetUnivariateBasis();
  std::vector<StochTree::FeatureType> feature_types(test_dataset.x_cols, StochTree::FeatureType::kNumeric);

  // Construct dataset
  int n = test_dataset.n;
  StochTree::RandomEffectsDataset dataset = StochTree::RandomEffectsDataset();
  dataset.AddBasis(test_dataset.rfx_basis.data(), test_dataset.n, test_dataset.rfx_basis_cols, test_dataset.row_major);
  dataset.AddGroupLabels(test_dataset.rfx_groups);
  
  // Construct tracker, model state, and container
  StochTree::RandomEffectsTracker tracker = StochTree::RandomEffectsTracker(test_dataset.rfx_groups);
  StochTree::MultivariateRegressionRandomEffectsModel model = StochTree::MultivariateRegressionRandomEffectsModel(test_dataset.rfx_basis_cols, test_dataset.rfx_num_groups);
  StochTree::RandomEffectsContainer container = StochTree::RandomEffectsContainer(test_dataset.rfx_basis_cols, test_dataset.rfx_num_groups);
  StochTree::LabelMapper label_mapper = StochTree::LabelMapper(tracker.GetLabelMap());

  // Set the values of alpha, xi and sigma in the model state (rather than simulating)
  Eigen::VectorXd alpha(test_dataset.rfx_basis_cols);
  Eigen::MatrixXd xi(test_dataset.rfx_basis_cols, test_dataset.rfx_num_groups);
  Eigen::MatrixXd sigma(test_dataset.rfx_basis_cols, test_dataset.rfx_basis_cols);
  alpha << 1.5;
  xi << 2, 4;
  Eigen::VectorXd xi0 = xi(Eigen::all, 0);
  Eigen::VectorXd xi1 = xi(Eigen::all, 1);
  sigma << 1;
  model.SetWorkingParameter(alpha);
  model.SetGroupParameter(xi0, 0);
  model.SetGroupParameter(xi1, 1);
  model.SetGroupParameterCovariance(sigma);
  
  // Push to the container
  container.AddSample(model);

  // Change values and push a second "sample" to the container
  alpha << 2.0;
  xi << 1, 3;
  xi0 = xi(Eigen::all, 0);
  xi1 = xi(Eigen::all, 1);
  sigma << 1;
  model.SetWorkingParameter(alpha);
  model.SetGroupParameter(xi0, 0);
  model.SetGroupParameter(xi1, 1);
  model.SetGroupParameterCovariance(sigma);
  container.AddSample(model);

  // Predict from the container
  int num_samples = 2;
  std::vector<double> output(n*num_samples);
  container.Predict(dataset, label_mapper, output);

  // Check predictions
  std::vector<double> output_expected {
    3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 
    2, 6, 2, 6, 2, 6, 2, 6, 2, 6, 
  };
  for (int i = 0; i < output.size(); i++) {
    ASSERT_EQ(output[i], output_expected[i]);
  }
}

TEST(RandomEffects, Serialization) {
  // Load test data
  StochTree::TestUtils::TestDataset test_dataset;
  test_dataset = StochTree::TestUtils::LoadSmallDatasetUnivariateBasis();
  std::vector<StochTree::FeatureType> feature_types(test_dataset.x_cols, StochTree::FeatureType::kNumeric);

  // Construct dataset
  int n = test_dataset.n;
  StochTree::RandomEffectsDataset dataset = StochTree::RandomEffectsDataset();
  dataset.AddBasis(test_dataset.rfx_basis.data(), test_dataset.n, test_dataset.rfx_basis_cols, test_dataset.row_major);
  dataset.AddGroupLabels(test_dataset.rfx_groups);
  
  // Construct tracker, model state, and container
  StochTree::RandomEffectsTracker tracker = StochTree::RandomEffectsTracker(test_dataset.rfx_groups);
  StochTree::MultivariateRegressionRandomEffectsModel model = StochTree::MultivariateRegressionRandomEffectsModel(test_dataset.rfx_basis_cols, test_dataset.rfx_num_groups);
  StochTree::RandomEffectsContainer container = StochTree::RandomEffectsContainer(test_dataset.rfx_basis_cols, test_dataset.rfx_num_groups);
  StochTree::LabelMapper label_mapper = StochTree::LabelMapper(tracker.GetLabelMap());

  // Set the values of alpha, xi and sigma in the model state (rather than simulating)
  Eigen::VectorXd alpha(test_dataset.rfx_basis_cols);
  Eigen::MatrixXd xi(test_dataset.rfx_basis_cols, test_dataset.rfx_num_groups);
  Eigen::MatrixXd sigma(test_dataset.rfx_basis_cols, test_dataset.rfx_basis_cols);
  alpha << 1.5;
  xi << 2, 4;
  Eigen::VectorXd xi0 = xi(Eigen::all, 0);
  Eigen::VectorXd xi1 = xi(Eigen::all, 1);
  sigma << 1;
  model.SetWorkingParameter(alpha);
  model.SetGroupParameter(xi0, 0);
  model.SetGroupParameter(xi1, 1);
  model.SetGroupParameterCovariance(sigma);
  
  // Push to the container
  container.AddSample(model);

  // Change values and push a second "sample" to the container
  alpha << 2.0;
  xi << 1, 3;
  xi0 = xi(Eigen::all, 0);
  xi1 = xi(Eigen::all, 1);
  sigma << 1;
  model.SetWorkingParameter(alpha);
  model.SetGroupParameter(xi0, 0);
  model.SetGroupParameter(xi1, 1);
  model.SetGroupParameterCovariance(sigma);
  container.AddSample(model);

  // Json round trip
  nlohmann::json container_json = container.to_json();
  StochTree::RandomEffectsContainer container_deserialized = StochTree::RandomEffectsContainer();
  container_deserialized.from_json(container_json);

  // Check data in the container
  std::vector<double> alpha_original = container.GetAlpha();
  std::vector<double> alpha_deserialized = container_deserialized.GetAlpha();
  for (int i = 0; i < alpha_deserialized.size(); i++) {
    ASSERT_EQ(alpha_original[i], alpha_deserialized[i]);
  }
  std::vector<double> xi_original = container.GetXi();
  std::vector<double> xi_deserialized = container_deserialized.GetXi();
  for (int i = 0; i < xi_deserialized.size(); i++) {
    ASSERT_EQ(xi_original[i], xi_deserialized[i]);
  }
  std::vector<double> beta_original = container.GetBeta();
  std::vector<double> beta_deserialized = container_deserialized.GetBeta();
  for (int i = 0; i < beta_deserialized.size(); i++) {
    ASSERT_EQ(beta_original[i], beta_deserialized[i]);
  }
}
