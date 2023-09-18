/*!
 * Copyright (c) 2022 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>
#include <testutils.h>
#include <stochtree/log.h>
#include <stochtree/random.h>
#include <stochtree/train_data.h>
#include <iostream>
#include <memory>

TEST(TrainingData, ReadFromFileNoTreatment) {
  // Declare config, training data, and training data loader
  std::unique_ptr<StochTree::TrainData> dataset;
  
  // Define any config parameters that aren't defaults
  const char* params = "header=true label_column=0";

  // Load data from file and check that it has the right dimensions
  StochTree::TestUtils::LoadDatasetFromDemos("xbart_train/test.csv", params, dataset);
  EXPECT_EQ(40, dataset->num_data()) << "Data size: " << dataset->num_data();
  EXPECT_EQ(3, dataset->num_variables()) << "Number of features: " << dataset->num_variables();
  
  // Compute average value for each feature, compared to their known values
  std::vector<double> total;
  std::vector<double> average;
  total.resize(dataset->num_variables(), 0.);
  average.resize(dataset->num_variables(), 0.);
  for (int i = 0; i < dataset->num_data(); i++) {
    for (int j = 0; j < dataset->num_variables(); j++) {
      total[j] += dataset->get_feature_value(i, j);
    }
  }
  for (int j = 0; j < dataset->num_variables(); j++) {
    average[j] = total[j] / dataset->num_data();
  }
  EXPECT_NEAR(0.458435078, average[0], 0.0001);
  EXPECT_NEAR(0.428693069, average[1], 0.0001);
  EXPECT_NEAR(0.393319613, average[2], 0.0001);
}

TEST(TrainingData, ReadFromFileWithTreatment) {
  // Declare unique pointer to training data
  std::unique_ptr<StochTree::TrainData> dataset;
  
  // Define any config parameters that aren't defaults
  const char* params = "header=true label_column=0 treatment_column=1";

  // Load data from file and check that it has the right dimensions
  StochTree::TestUtils::LoadDatasetFromDemos("xbart_train/test.csv", params, dataset);
  EXPECT_EQ(40, dataset->num_data()) << "Data size: " << dataset->num_data();
  EXPECT_EQ(2, dataset->num_variables()) << "Number of features: " << dataset->num_variables();
  
  // Compute average value for each feature, compared to their known values
  std::vector<double> total;
  std::vector<double> average;
  total.resize(dataset->num_variables(), 0.);
  average.resize(dataset->num_variables(), 0.);
  for (int i = 0; i < dataset->num_data(); i++) {
    for (int j = 0; j < dataset->num_variables(); j++) {
      total[j] += dataset->get_feature_value(i, j);
    }
  }
  for (int j = 0; j < dataset->num_variables(); j++) {
    average[j] = total[j] / dataset->num_data();
  }
  EXPECT_NEAR(0.428693069, average[0], 0.0001);
  EXPECT_NEAR(0.393319613, average[1], 0.0001);
}

TEST(TrainingData, ReadFromRowMajor) {
  // Generate some in-memory data
  int n = 10;
  int p = 5;
  std::mt19937 gen;
  std::uniform_real_distribution<> d{0.0,1.0};
  std::unique_ptr<double[]> matrix_data(new double[n*p]);
  for (int i = 0; i < n*p; i++){
    matrix_data[i] = d(gen);
  }

  // Declare unique pointer to training data
  std::unique_ptr<StochTree::TrainData> dataset;
  
  // Define any config parameters that aren't defaults
  const char* params = "header=true label_column=0 treatment_column=1";
  auto param = StochTree::Config::Str2Map(params);
  StochTree::Config config;
  config.Set(param);

  // Define data loader
  StochTree::TrainDataLoader dataset_loader(config, 1, nullptr);

  // Load some test data
  dataset.reset(dataset_loader.ConstructFromMatrix(matrix_data.get(), p, n, true));
  
  // Check data dimensions  
  EXPECT_EQ(10, dataset->num_data()) << "Data size: " << dataset->num_data();
  EXPECT_EQ(3, dataset->num_variables()) << "Number of features: " << dataset->num_variables();

  // Compute average value for each feature, compared to their known values
  std::vector<double> total;
  std::vector<double> average;
  total.resize(dataset->num_variables(), 0.);
  average.resize(dataset->num_variables(), 0.);
  for (int i = 0; i < dataset->num_data(); i++) {
    for (int j = 0; j < dataset->num_variables(); j++) {
      total[j] += dataset->get_feature_value(i, j);
    }
  }
  for (int j = 0; j < dataset->num_variables(); j++) {
    average[j] = total[j] / dataset->num_data();
  }
  EXPECT_NEAR(0.5826760, average[0], 0.0001);
  EXPECT_NEAR(0.5346280, average[1], 0.0001);
  EXPECT_NEAR(0.565114,  average[2], 0.0001);
}
