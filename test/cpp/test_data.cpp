/*!
 * Copyright (c) 2022 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include "stochtree/meta.h"
#include <gtest/gtest.h>
#include <testutils.h>
#include <stochtree/log.h>
#include <stochtree/random.h>
#include <stochtree/data.h>
#include <iostream>
#include <memory>

TEST(Data, ReadFromSmallDatasetRowMajor) {
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
  
  // Compute average value for each feature, compared to their known values
  std::vector<double> total;
  std::vector<double> average;
  total.resize(p, 0.);
  average.resize(p, 0.);
  for (int j = 0; j < p; j++) {
    for (data_size_t i = 0; i < n; i++) {  
      total[j] += dataset.CovariateValue(i, j);
    }
  }
  for (int j = 0; j < p; j++) {
    average[j] = total[j] / static_cast<double>(n);
  }
  EXPECT_NEAR(0.6051545, average[0], 0.0001);
  EXPECT_NEAR(0.5343037, average[1], 0.0001);
  EXPECT_NEAR(0.5894948, average[2], 0.0001);
  EXPECT_NEAR(0.5405563, average[3], 0.0001);
  EXPECT_NEAR(0.2526782, average[4], 0.0001);
}

TEST(Data, ReadFromMediumDatasetRowMajor) {
  // Load test data
  StochTree::TestUtils::TestDataset test_dataset;
  test_dataset = StochTree::TestUtils::LoadMediumDatasetUnivariateBasis();
  std::vector<StochTree::FeatureType> feature_types(test_dataset.x_cols, StochTree::FeatureType::kNumeric);

  // Construct datasets
  using data_size_t = StochTree::data_size_t;
  data_size_t n = test_dataset.n;
  int p = test_dataset.x_cols;
  StochTree::ForestDataset dataset = StochTree::ForestDataset();
  dataset.AddCovariates(test_dataset.covariates.data(), n, test_dataset.x_cols, test_dataset.row_major);
  dataset.AddBasis(test_dataset.omega.data(), test_dataset.n, test_dataset.omega_cols, test_dataset.row_major);
  StochTree::ColumnVector residual = StochTree::ColumnVector(test_dataset.outcome.data(), n);
  
  // Compute average value for each feature, compared to their known values
  std::vector<double> total;
  std::vector<double> average;
  total.resize(p, 0.);
  average.resize(p, 0.);
  for (int j = 0; j < p; j++) {
    for (data_size_t i = 0; i < n; i++) {  
      total[j] += dataset.CovariateValue(i, j);
    }
  }
  for (int j = 0; j < p; j++) {
    average[j] = total[j] / static_cast<double>(n);
  }
  EXPECT_NEAR(0.5317846, average[0], 0.0001);
  EXPECT_NEAR(0.5241688, average[1], 0.0001);
  EXPECT_NEAR(0.4983934, average[2], 0.0001);
  EXPECT_NEAR(0.4863596, average[3], 0.0001);
  EXPECT_NEAR(0.4413101, average[4], 0.0001);
}

