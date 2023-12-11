#include <gtest/gtest.h>
#include <testutils.h>
#include <stochtree/data.h>
#include <stochtree/log.h>
#include <stochtree/partition_tracker.h>
#include <iostream>
#include <memory>

TEST(UnsortedNodeSampleTracker, BasicOperations) {
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

  // Construct a NodeSampleTracker
  int num_trees = 1;
  StochTree::UnsortedNodeSampleTracker node_sample_tracker(n, num_trees);

  // Construct a SampleNodeMapper
  std::unique_ptr<StochTree::SampleNodeMapper> sample_node_mapper = std::make_unique<StochTree::SampleNodeMapper>(num_trees, n);
  sample_node_mapper->AssignAllSamplesToRoot(0);
  for (StochTree::data_size_t i = 0; i < n; i++) {
    ASSERT_EQ(sample_node_mapper->GetNodeId(i, 0), 0);
  }

  // Check leaf node begin and node end
  ASSERT_EQ(node_sample_tracker.NodeBegin(0, 0), 0);
  ASSERT_EQ(node_sample_tracker.NodeEnd(0, 0), n);

  // Partition based on X[,0] <= 0.5
  node_sample_tracker.PartitionTreeNode(dataset.get(), 0, 0, 1, 2, 0, 0.5);

  // Update the SampleNodeMapper based on the split
  node_sample_tracker.UpdateObservationMapping(1, 0, sample_node_mapper.get());
  node_sample_tracker.UpdateObservationMapping(2, 0, sample_node_mapper.get());
  ASSERT_EQ(sample_node_mapper->GetNodeId(0, 0), 2);
  ASSERT_EQ(sample_node_mapper->GetNodeId(1, 0), 2);
  ASSERT_EQ(sample_node_mapper->GetNodeId(2, 0), 2);
  ASSERT_EQ(sample_node_mapper->GetNodeId(3, 0), 2);
  ASSERT_EQ(sample_node_mapper->GetNodeId(4, 0), 1);
  ASSERT_EQ(sample_node_mapper->GetNodeId(5, 0), 2);
  ASSERT_EQ(sample_node_mapper->GetNodeId(6, 0), 1);
  ASSERT_EQ(sample_node_mapper->GetNodeId(7, 0), 2);
  ASSERT_EQ(sample_node_mapper->GetNodeId(8, 0), 2);
  ASSERT_EQ(sample_node_mapper->GetNodeId(9, 0), 2);
  
  // Check that node begin and node end haven't changed for root node, but that the indices have been sifted
  ASSERT_EQ(node_sample_tracker.NodeBegin(0, 0), 0);
  ASSERT_EQ(node_sample_tracker.NodeEnd(0, 0), n);
  std::vector<StochTree::data_size_t> expected_result{4, 6, 0, 1, 2, 3, 5, 7, 8, 9};
  ASSERT_EQ(node_sample_tracker.TreeNodeIndices(0, 0), expected_result);
  
  // Check node begin and node end for left node
  ASSERT_EQ(node_sample_tracker.NodeBegin(0, 1), 0);
  ASSERT_EQ(node_sample_tracker.NodeEnd(0, 1), 2);
  expected_result = {4, 6};
  ASSERT_EQ(node_sample_tracker.TreeNodeIndices(0, 1), expected_result);
  
  // Check node begin and node end for right node
  ASSERT_EQ(node_sample_tracker.NodeBegin(0, 2), 2);
  ASSERT_EQ(node_sample_tracker.NodeEnd(0, 2), n);
  expected_result = {0, 1, 2, 3, 5, 7, 8, 9};
  ASSERT_EQ(node_sample_tracker.TreeNodeIndices(0, 2), expected_result);

  // Partition right node based on X[,1] <= 0.5
  node_sample_tracker.PartitionTreeNode(dataset.get(), 0, 2, 3, 4, 1, 0.5);

  // Check that node begin and node end haven't changed for old right node, but that the indices have been sifted
  ASSERT_EQ(node_sample_tracker.NodeBegin(0, 2), 2);
  ASSERT_EQ(node_sample_tracker.NodeEnd(0, 2), n);
  expected_result = {5, 7, 8, 0, 1, 2, 3, 9};
  ASSERT_EQ(node_sample_tracker.TreeNodeIndices(0, 2), expected_result);
  
  // Check node begin and node end for new left node
  ASSERT_EQ(node_sample_tracker.NodeBegin(0, 3), 2);
  ASSERT_EQ(node_sample_tracker.NodeEnd(0, 3), 5);
  expected_result = {5, 7, 8};
  ASSERT_EQ(node_sample_tracker.TreeNodeIndices(0, 3), expected_result);
  
  // Check node begin and node end for new right node
  ASSERT_EQ(node_sample_tracker.NodeBegin(0, 4), 5);
  ASSERT_EQ(node_sample_tracker.NodeEnd(0, 4), n);
  expected_result = {0, 1, 2, 3, 9};
  ASSERT_EQ(node_sample_tracker.TreeNodeIndices(0, 4), expected_result);
}
