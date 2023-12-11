#include <gtest/gtest.h>
#include <testutils.h>
#include <stochtree/cutpoint_candidates.h>
#include <stochtree/data.h>
#include <stochtree/log.h>
#include <stochtree/partition_tracker.h>
#include <iostream>
#include <memory>
#include <vector>

TEST(CutpointGrid, NumericFeaturesGrid) {
  // Generate some in-memory data
  StochTree::data_size_t n = 20;
  int p = 2;
  std::vector<double> data_vector = {
    0.96670851, 0.5170531, 0.43164016, 
    0.88927847, 0.875529, 1.73105073, 
    0.26387458, 0.6568786, 1.71923726, 
    0.12020009, 0.48823, 1.35202919, 
    0.35117133, 0.7267357, 1.72840826, 
    0.03440321, 0.992761, 2.86722066, 
    0.93066095, 0.3518555, 0.15004401, 
    0.53183342, 0.7565854, 1.71501758, 
    0.7792386, 0.5842398, 1.03309461, 
    0.55593617, 0.8112128, 1.75616277, 
    0.87592489, 0.4641341, 0.35003031, 
    0.42742281, 0.6252176, 1.33252932, 
    0.91417973, 0.7831089, 1.52772027, 
    0.83594354, 0.9656884, 1.88203102, 
    0.95951289, 0.66017, 0.99754716, 
    0.95430127, 0.2653695, -0.23894334, 
    0.69898676, 0.2949846, 0.03323651, 
    0.56475759, 0.726991, 1.57430907, 
    0.58828617, 0.5107459, 1.02735085, 
    0.03144349, 0.9995112, 3.07706032
  };

  // Declare unique pointer to training data
  std::unique_ptr<StochTree::Dataset> dataset;
  
  // Define any config parameters that aren't defaults
  const char* params = "header=true outcome_columns=2 treatment_columns=-1 num_trees=1 cutpoint_grid_size=5";
  auto param = StochTree::Config::Str2Map(params);
  StochTree::Config config;
  config.Set(param);
  
  // Define data loader
  StochTree::DataLoader dataset_loader(config, 1, nullptr);

  // Load some test data
  dataset.reset(dataset_loader.ConstructFromMatrix(data_vector.data(), p + 1, n, true));

  // Unique pointer to cutpoint grid tracker
  std::unique_ptr<StochTree::CutpointGridContainer> cutpoint_grid_container;
  cutpoint_grid_container.reset(new StochTree::CutpointGridContainer(dataset.get(), config));

  // Construct a container of presorted feature indices
  std::unique_ptr<StochTree::FeaturePresortRootContainer> presort_container = std::make_unique<StochTree::FeaturePresortRootContainer>(dataset.get());

  // Construct a NodeSampleTracker
  std::unique_ptr<StochTree::SortedNodeSampleTracker> sorted_node_sample_tracker = std::make_unique<StochTree::SortedNodeSampleTracker>(presort_container.get(), dataset.get());
  
  // Construct a SampleNodeMapper
  int num_trees = 1;
  std::unique_ptr<StochTree::SampleNodeMapper> sample_node_mapper = std::make_unique<StochTree::SampleNodeMapper>(num_trees, n);
  sample_node_mapper->AssignAllSamplesToRoot(0);
  for (StochTree::data_size_t i = 0; i < n; i++) {
    ASSERT_EQ(sample_node_mapper->GetNodeId(i, 0), 0);
  }

  // Check that indices are correctly sorted for feature 0 in the root node (i.e. entire dataset)
  ASSERT_EQ(sorted_node_sample_tracker->NodeBegin(0, 0), 0);
  ASSERT_EQ(sorted_node_sample_tracker->NodeEnd(0, 0), n);
  std::vector<StochTree::data_size_t> expected_result{19, 5, 3, 2, 4, 11, 7, 9, 17, 18, 16, 8, 13, 10, 1, 12, 6, 15, 14, 0};
  ASSERT_EQ(sorted_node_sample_tracker->NodeIndices(0, 0), expected_result);

  // Check that indices are correctly sorted for feature 1 in the root node (i.e. entire dataset)
  ASSERT_EQ(sorted_node_sample_tracker->NodeBegin(0, 1), 0);
  ASSERT_EQ(sorted_node_sample_tracker->NodeEnd(0, 1), n);
  expected_result = {15, 16, 6, 10, 3, 18, 0, 8, 11, 2, 14, 4, 17, 7, 12, 9, 1, 13, 5, 19};
  ASSERT_EQ(sorted_node_sample_tracker->NodeIndices(0, 1), expected_result);

  // Enumerate cutpoint strides for each feature
  for (int j = 0; j < dataset->NumCovariates(); j++) {
    cutpoint_grid_container->CalculateStrides(dataset.get(), sorted_node_sample_tracker.get(), 0, 0, n, j);
  }

  // Check cutpoint strides for feature 0
  double kDelta = 0.0001;
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(0, 0), 0);
  ASSERT_EQ(cutpoint_grid_container->BinLength(0, 0), 4);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(0, 0), 0.2638746, kDelta);
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(1, 0), 4);
  ASSERT_EQ(cutpoint_grid_container->BinLength(1, 0), 4);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(1, 0), 0.5559362, kDelta);
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(2, 0), 8);
  ASSERT_EQ(cutpoint_grid_container->BinLength(2, 0), 4);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(2, 0), 0.7792386, kDelta);
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(3, 0), 12);
  ASSERT_EQ(cutpoint_grid_container->BinLength(3, 0), 4);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(3, 0), 0.9141797, kDelta);
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(4, 0), 16);
  ASSERT_EQ(cutpoint_grid_container->BinLength(4, 0), 4);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(4, 0), 0.9667085, kDelta);

  // Check cutpoint strides for feature 1
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(0, 1), 0);
  ASSERT_EQ(cutpoint_grid_container->BinLength(0, 1), 4);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(0, 1), 0.4641341, kDelta);
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(1, 1), 4);
  ASSERT_EQ(cutpoint_grid_container->BinLength(1, 1), 4);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(1, 1), 0.5842398, kDelta);
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(2, 1), 8);
  ASSERT_EQ(cutpoint_grid_container->BinLength(2, 1), 4);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(2, 1), 0.7267357, kDelta);
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(3, 1), 12);
  ASSERT_EQ(cutpoint_grid_container->BinLength(3, 1), 4);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(3, 1), 0.8112128, kDelta);
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(4, 1), 16);
  ASSERT_EQ(cutpoint_grid_container->BinLength(4, 1), 4);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(4, 1), 0.9995112, kDelta);
}

TEST(CutpointGrid, NumericFeaturesFullyEnumerated) {
  // Generate some in-memory data
  StochTree::data_size_t n = 20;
  int p = 2;
  std::vector<double> data_vector = {
    0.96670851, 0.5170531, 0.43164016, 
    0.88927847, 0.875529, 1.73105073, 
    0.26387458, 0.6568786, 1.71923726, 
    0.12020009, 0.48823, 1.35202919, 
    0.35117133, 0.7267357, 1.72840826, 
    0.03440321, 0.992761, 2.86722066, 
    0.93066095, 0.3518555, 0.15004401, 
    0.53183342, 0.7565854, 1.71501758, 
    0.7792386, 0.5842398, 1.03309461, 
    0.55593617, 0.8112128, 1.75616277, 
    0.87592489, 0.4641341, 0.35003031, 
    0.42742281, 0.6252176, 1.33252932, 
    0.91417973, 0.7831089, 1.52772027, 
    0.83594354, 0.9656884, 1.88203102, 
    0.95951289, 0.66017, 0.99754716, 
    0.95430127, 0.2653695, -0.23894334, 
    0.69898676, 0.2949846, 0.03323651, 
    0.56475759, 0.726991, 1.57430907, 
    0.58828617, 0.5107459, 1.02735085, 
    0.03144349, 0.9995112, 3.07706032
  };

  // Declare unique pointer to training data
  std::unique_ptr<StochTree::Dataset> dataset;
  
  // Define any config parameters that aren't defaults
  const char* params = "header=true outcome_columns=2 treatment_columns=-1 num_trees=1 cutpoint_grid_size=20";
  auto param = StochTree::Config::Str2Map(params);
  StochTree::Config config;
  config.Set(param);
  
  // Define data loader
  StochTree::DataLoader dataset_loader(config, 1, nullptr);

  // Load some test data
  dataset.reset(dataset_loader.ConstructFromMatrix(data_vector.data(), p + 1, n, true));

  // Unique pointer to cutpoint grid tracker
  std::unique_ptr<StochTree::CutpointGridContainer> cutpoint_grid_container;
  cutpoint_grid_container.reset(new StochTree::CutpointGridContainer(dataset.get(), config));

  // Construct a container of presorted feature indices
  std::unique_ptr<StochTree::FeaturePresortRootContainer> presort_container = std::make_unique<StochTree::FeaturePresortRootContainer>(dataset.get());

  // Construct a NodeSampleTracker
  std::unique_ptr<StochTree::SortedNodeSampleTracker> sorted_node_sample_tracker = std::make_unique<StochTree::SortedNodeSampleTracker>(presort_container.get(), dataset.get());
  
  // Construct a SampleNodeMapper
  int num_trees = 1;
  std::unique_ptr<StochTree::SampleNodeMapper> sample_node_mapper = std::make_unique<StochTree::SampleNodeMapper>(num_trees, n);
  sample_node_mapper->AssignAllSamplesToRoot(0);
  for (StochTree::data_size_t i = 0; i < n; i++) {
    ASSERT_EQ(sample_node_mapper->GetNodeId(i, 0), 0);
  }

  // Check that indices are correctly sorted for feature 0 in the root node (i.e. entire dataset)
  ASSERT_EQ(sorted_node_sample_tracker->NodeBegin(0, 0), 0);
  ASSERT_EQ(sorted_node_sample_tracker->NodeEnd(0, 0), n);
  std::vector<StochTree::data_size_t> expected_result{19, 5, 3, 2, 4, 11, 7, 9, 17, 18, 16, 8, 13, 10, 1, 12, 6, 15, 14, 0};
  ASSERT_EQ(sorted_node_sample_tracker->NodeIndices(0, 0), expected_result);

  // Check that indices are correctly sorted for feature 1 in the root node (i.e. entire dataset)
  ASSERT_EQ(sorted_node_sample_tracker->NodeBegin(0, 1), 0);
  ASSERT_EQ(sorted_node_sample_tracker->NodeEnd(0, 1), n);
  expected_result = {15, 16, 6, 10, 3, 18, 0, 8, 11, 2, 14, 4, 17, 7, 12, 9, 1, 13, 5, 19};
  ASSERT_EQ(sorted_node_sample_tracker->NodeIndices(0, 1), expected_result);

  // Enumerate cutpoint strides for each feature
  for (int j = 0; j < dataset->NumCovariates(); j++) {
    cutpoint_grid_container->CalculateStrides(dataset.get(), sorted_node_sample_tracker.get(), 0, 0, n, j);
  }

  // Check cutpoint strides for feature 0
  double kDelta = 0.0001;
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(0, 0), 0);
  ASSERT_EQ(cutpoint_grid_container->BinLength(0, 0), 1);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(0, 0), 0.03144349, kDelta);
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(1, 0), 1);
  ASSERT_EQ(cutpoint_grid_container->BinLength(1, 0), 1);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(1, 0), 0.03440321, kDelta);
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(2, 0), 2);
  ASSERT_EQ(cutpoint_grid_container->BinLength(2, 0), 1);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(2, 0), 0.12020009, kDelta);
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(3, 0), 3);
  ASSERT_EQ(cutpoint_grid_container->BinLength(3, 0), 1);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(3, 0), 0.26387458, kDelta);
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(4, 0), 4);
  ASSERT_EQ(cutpoint_grid_container->BinLength(4, 0), 1);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(4, 0), 0.35117133, kDelta);
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(5, 0), 5);
  ASSERT_EQ(cutpoint_grid_container->BinLength(5, 0), 1);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(5, 0), 0.42742281, kDelta);
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(19, 0), 19);
  ASSERT_EQ(cutpoint_grid_container->BinLength(19, 0), 1);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(19, 0), 0.96670851, kDelta);

  // Check cutpoint strides for feature 1
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(0, 1), 0);
  ASSERT_EQ(cutpoint_grid_container->BinLength(0, 1), 1);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(0, 1), 0.2653695, kDelta);
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(1, 1), 1);
  ASSERT_EQ(cutpoint_grid_container->BinLength(1, 1), 1);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(1, 1), 0.2949846, kDelta);
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(2, 1), 2);
  ASSERT_EQ(cutpoint_grid_container->BinLength(2, 1), 1);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(2, 1), 0.3518555, kDelta);
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(3, 1), 3);
  ASSERT_EQ(cutpoint_grid_container->BinLength(3, 1), 1);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(3, 1), 0.4641341, kDelta);
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(4, 1), 4);
  ASSERT_EQ(cutpoint_grid_container->BinLength(4, 1), 1);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(4, 1), 0.4882300, kDelta);
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(5, 1), 5);
  ASSERT_EQ(cutpoint_grid_container->BinLength(5, 1), 1);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(5, 1), 0.5107459, kDelta);
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(19, 1), 19);
  ASSERT_EQ(cutpoint_grid_container->BinLength(19, 1), 1);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(19, 1), 0.9995112, kDelta);
}

TEST(CutpointGrid, OrderedCategoricalFeatures) {
  // Generate some in-memory data
  StochTree::data_size_t n = 20;
  int p = 2;
  std::vector<double> data_vector = {
    1, 2, 0.43164016, 
    3, 1, 1.73105073, 
    3, 2, 1.71923726, 
    2, 1, 1.35202919, 
    1, 3, 1.72840826, 
    1, 1, 2.86722066, 
    2, 3, 0.15004401, 
    3, 1, 1.71501758, 
    2, 1, 1.03309461, 
    1, 2, 1.75616277, 
    2, 3, 0.35003031, 
    3, 3, 1.33252932, 
    2, 1, 1.52772027, 
    2, 2, 1.88203102, 
    1, 2, 0.99754716, 
    1, 2, -0.23894334, 
    3, 1, 0.03323651, 
    3, 2, 1.57430907, 
    1, 1, 1.02735085, 
    1, 2, 3.07706032
  };

  // Declare unique pointer to training data
  std::unique_ptr<StochTree::Dataset> dataset;
  
  // Define any config parameters that aren't defaults
  const char* params = "header=true outcome_columns=2 treatment_columns=-1 num_trees=1 ordered_categoricals=0,1";
  auto param = StochTree::Config::Str2Map(params);
  StochTree::Config config;
  config.Set(param);
  
  // Define data loader
  StochTree::DataLoader dataset_loader(config, 1, nullptr);

  // Load some test data
  dataset.reset(dataset_loader.ConstructFromMatrix(data_vector.data(), p + 1, n, true));

  // Unique pointer to cutpoint grid tracker
  std::unique_ptr<StochTree::CutpointGridContainer> cutpoint_grid_container;
  cutpoint_grid_container.reset(new StochTree::CutpointGridContainer(dataset.get(), config));

  // Construct a container of presorted feature indices
  std::unique_ptr<StochTree::FeaturePresortRootContainer> presort_container = std::make_unique<StochTree::FeaturePresortRootContainer>(dataset.get());

  // Construct a NodeSampleTracker
  std::unique_ptr<StochTree::SortedNodeSampleTracker> sorted_node_sample_tracker = std::make_unique<StochTree::SortedNodeSampleTracker>(presort_container.get(), dataset.get());
  
  // Construct a SampleNodeMapper
  int num_trees = 1;
  std::unique_ptr<StochTree::SampleNodeMapper> sample_node_mapper = std::make_unique<StochTree::SampleNodeMapper>(num_trees, n);
  sample_node_mapper->AssignAllSamplesToRoot(0);
  for (StochTree::data_size_t i = 0; i < n; i++) {
    ASSERT_EQ(sample_node_mapper->GetNodeId(i, 0), 0);
  }

  // Check that indices are correctly sorted for feature 0 in the root node (i.e. entire dataset)
  ASSERT_EQ(sorted_node_sample_tracker->NodeBegin(0, 0), 0);
  ASSERT_EQ(sorted_node_sample_tracker->NodeEnd(0, 0), n);
  std::vector<StochTree::data_size_t> expected_result{0, 4, 5, 9, 14, 15, 18, 19, 3, 6, 8, 10, 12, 13, 1, 2, 7, 11, 16, 17};
  ASSERT_EQ(sorted_node_sample_tracker->NodeIndices(0, 0), expected_result);

  // Check that indices are correctly sorted for feature 1 in the root node (i.e. entire dataset)
  ASSERT_EQ(sorted_node_sample_tracker->NodeBegin(0, 1), 0);
  ASSERT_EQ(sorted_node_sample_tracker->NodeEnd(0, 1), n);
  expected_result = {1, 3, 5, 7, 8, 12, 16, 18, 0, 2, 9, 13, 14, 15, 17, 19, 4, 6, 10, 11};
  ASSERT_EQ(sorted_node_sample_tracker->NodeIndices(0, 1), expected_result);

  // Enumerate cutpoint strides for each feature
  for (int j = 0; j < dataset->NumCovariates(); j++) {
    cutpoint_grid_container->CalculateStrides(dataset.get(), sorted_node_sample_tracker.get(), 0, 0, n, j);
  }

  // Check cutpoint strides for feature 0
  double kDelta = 0.0001;
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(0, 0), 0);
  ASSERT_EQ(cutpoint_grid_container->BinLength(0, 0), 8);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(0, 0), 1, kDelta);
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(1, 0), 8);
  ASSERT_EQ(cutpoint_grid_container->BinLength(1, 0), 6);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(1, 0), 2, kDelta);
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(2, 0), 14);
  ASSERT_EQ(cutpoint_grid_container->BinLength(2, 0), 6);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(2, 0), 3, kDelta);

  // Check cutpoint strides for feature 1
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(0, 1), 0);
  ASSERT_EQ(cutpoint_grid_container->BinLength(0, 1), 8);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(0, 1), 1, kDelta);
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(1, 1), 8);
  ASSERT_EQ(cutpoint_grid_container->BinLength(1, 1), 8);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(1, 1), 2, kDelta);
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(2, 1), 16);
  ASSERT_EQ(cutpoint_grid_container->BinLength(2, 1), 4);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(2, 1), 3, kDelta);
}

TEST(CutpointGrid, UnorderedCategoricalFeatures) {
  // Generate some in-memory data
  StochTree::data_size_t n = 20;
  int p = 2;
  std::vector<double> data_vector = {
    1, 2, 0.43164016, 
    3, 1, 1.73105073, 
    3, 2, 1.71923726, 
    2, 1, 1.35202919, 
    1, 3, 1.72840826, 
    1, 1, 2.86722066, 
    2, 3, 0.15004401, 
    3, 1, 1.71501758, 
    2, 1, 1.03309461, 
    1, 2, 1.75616277, 
    2, 3, 0.35003031, 
    3, 3, 1.33252932, 
    2, 1, 1.52772027, 
    2, 2, 1.88203102, 
    1, 2, 0.99754716, 
    1, 2, -0.23894334, 
    3, 1, 0.03323651, 
    3, 2, 1.57430907, 
    1, 1, 1.02735085, 
    1, 2, 3.07706032
  };

  // Declare unique pointer to training data
  std::unique_ptr<StochTree::Dataset> dataset;
  
  // Define any config parameters that aren't defaults
  const char* params = "header=true outcome_columns=2 treatment_columns=-1 num_trees=1 unordered_categoricals=0,1";
  auto param = StochTree::Config::Str2Map(params);
  StochTree::Config config;
  config.Set(param);
  
  // Define data loader
  StochTree::DataLoader dataset_loader(config, 1, nullptr);

  // Load some test data
  dataset.reset(dataset_loader.ConstructFromMatrix(data_vector.data(), p + 1, n, true));

  // Unique pointer to cutpoint grid tracker
  std::unique_ptr<StochTree::CutpointGridContainer> cutpoint_grid_container;
  cutpoint_grid_container.reset(new StochTree::CutpointGridContainer(dataset.get(), config));

  // Construct a container of presorted feature indices
  std::unique_ptr<StochTree::FeaturePresortRootContainer> presort_container = std::make_unique<StochTree::FeaturePresortRootContainer>(dataset.get());

  // Construct a NodeSampleTracker
  std::unique_ptr<StochTree::SortedNodeSampleTracker> sorted_node_sample_tracker = std::make_unique<StochTree::SortedNodeSampleTracker>(presort_container.get(), dataset.get());
  
  // Construct a SampleNodeMapper
  int num_trees = 1;
  std::unique_ptr<StochTree::SampleNodeMapper> sample_node_mapper = std::make_unique<StochTree::SampleNodeMapper>(num_trees, n);
  sample_node_mapper->AssignAllSamplesToRoot(0);
  for (StochTree::data_size_t i = 0; i < n; i++) {
    ASSERT_EQ(sample_node_mapper->GetNodeId(i, 0), 0);
  }

  // Check that indices are correctly sorted for feature 0 in the root node (i.e. entire dataset)
  ASSERT_EQ(sorted_node_sample_tracker->NodeBegin(0, 0), 0);
  ASSERT_EQ(sorted_node_sample_tracker->NodeEnd(0, 0), n);
  std::vector<StochTree::data_size_t> expected_result{0, 4, 5, 9, 14, 15, 18, 19, 3, 6, 8, 10, 12, 13, 1, 2, 7, 11, 16, 17};
  ASSERT_EQ(sorted_node_sample_tracker->NodeIndices(0, 0), expected_result);

  // Check that indices are correctly sorted for feature 1 in the root node (i.e. entire dataset)
  ASSERT_EQ(sorted_node_sample_tracker->NodeBegin(0, 1), 0);
  ASSERT_EQ(sorted_node_sample_tracker->NodeEnd(0, 1), n);
  expected_result = {1, 3, 5, 7, 8, 12, 16, 18, 0, 2, 9, 13, 14, 15, 17, 19, 4, 6, 10, 11};
  ASSERT_EQ(sorted_node_sample_tracker->NodeIndices(0, 1), expected_result);

  // Enumerate cutpoint strides for each feature
  for (int j = 0; j < dataset->NumCovariates(); j++) {
    cutpoint_grid_container->CalculateStrides(dataset.get(), sorted_node_sample_tracker.get(), 0, 0, n, j);
  }

  // Check cutpoint strides for feature 0
  double kDelta = 0.0001;
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(0, 0), 8);
  ASSERT_EQ(cutpoint_grid_container->BinLength(0, 0), 6);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(0, 0), 2, kDelta);
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(1, 0), 14);
  ASSERT_EQ(cutpoint_grid_container->BinLength(1, 0), 6);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(1, 0), 3, kDelta);
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(2, 0), 0);
  ASSERT_EQ(cutpoint_grid_container->BinLength(2, 0), 8);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(2, 0), 1, kDelta);


  // Check cutpoint strides for feature 1
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(0, 1), 16);
  ASSERT_EQ(cutpoint_grid_container->BinLength(0, 1), 4);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(0, 1), 3, kDelta);
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(1, 1), 8);
  ASSERT_EQ(cutpoint_grid_container->BinLength(1, 1), 8);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(1, 1), 2, kDelta);
  ASSERT_EQ(cutpoint_grid_container->BinStartIndex(2, 1), 0);
  ASSERT_EQ(cutpoint_grid_container->BinLength(2, 1), 8);
  ASSERT_NEAR(cutpoint_grid_container->CutpointValue(2, 1), 1, kDelta);
}
