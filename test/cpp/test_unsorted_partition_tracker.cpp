#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <testutils.h>
#include <stochtree/data.h>
#include <stochtree/ensemble.h>
#include <stochtree/log.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/tree.h>
#include <map>
#include <set>
#include <vector>

TEST(UnsortedNodeSampleTracker, BasicOperations) {
  // Load test data
  StochTree::TestUtils::TestDataset test_dataset;
  test_dataset = StochTree::TestUtils::LoadSmallDatasetUnivariateBasis();

  // Construct datasets
  int n = test_dataset.n;
  StochTree::ForestDataset dataset = StochTree::ForestDataset();
  dataset.AddCovariates(test_dataset.covariates.data(), n, test_dataset.x_cols, test_dataset.row_major);
  dataset.AddBasis(test_dataset.omega.data(), test_dataset.n, test_dataset.omega_cols, test_dataset.row_major);
  StochTree::ColumnVector residual = StochTree::ColumnVector(test_dataset.outcome.data(), n);

  // Construct a NodeSampleTracker
  int num_trees = 1;
  StochTree::UnsortedNodeSampleTracker node_sample_tracker(n, num_trees);

  // Construct a SampleNodeMapper
  StochTree::SampleNodeMapper sample_node_mapper = StochTree::SampleNodeMapper(num_trees, n);
  sample_node_mapper.AssignAllSamplesToRoot(0);
  for (StochTree::data_size_t i = 0; i < n; i++) {
    ASSERT_EQ(sample_node_mapper.GetNodeId(i, 0), 0);
  }

  // Check leaf node begin and node end
  ASSERT_EQ(node_sample_tracker.NodeBegin(0, 0), 0);
  ASSERT_EQ(node_sample_tracker.NodeEnd(0, 0), n);

  // Partition based on X[,0] <= 0.5
  StochTree::TreeSplit tree_split = StochTree::TreeSplit(0.5);
  node_sample_tracker.PartitionTreeNode(dataset.GetCovariates(), 0, 0, 1, 2, 0, tree_split);
  sample_node_mapper.AddSplit(dataset.GetCovariates(), tree_split, 0, 0, 0, 1, 2);

  // Check that node begin and node end haven't changed for root node, but that the indices have been sifted
  ASSERT_EQ(node_sample_tracker.NodeBegin(0, 0), 0);
  ASSERT_EQ(node_sample_tracker.NodeEnd(0, 0), n);
  std::vector<StochTree::data_size_t> expected_result{2, 8, 9, 0, 1, 3, 4, 5, 6, 7};
  ASSERT_EQ(node_sample_tracker.TreeNodeIndices(0, 0), expected_result);

  // Check that terminal nodes are updated for for every observation
  ASSERT_EQ(sample_node_mapper.GetNodeId(0, 0), 2);
  ASSERT_EQ(sample_node_mapper.GetNodeId(1, 0), 2);
  ASSERT_EQ(sample_node_mapper.GetNodeId(2, 0), 1);
  ASSERT_EQ(sample_node_mapper.GetNodeId(3, 0), 2);
  ASSERT_EQ(sample_node_mapper.GetNodeId(4, 0), 2);
  ASSERT_EQ(sample_node_mapper.GetNodeId(5, 0), 2);
  ASSERT_EQ(sample_node_mapper.GetNodeId(6, 0), 2);
  ASSERT_EQ(sample_node_mapper.GetNodeId(7, 0), 2);
  ASSERT_EQ(sample_node_mapper.GetNodeId(8, 0), 1);
  ASSERT_EQ(sample_node_mapper.GetNodeId(9, 0), 1);

  // Check node begin and node end for left node
  ASSERT_EQ(node_sample_tracker.NodeBegin(0, 1), 0);
  ASSERT_EQ(node_sample_tracker.NodeEnd(0, 1), 3);
  expected_result = {2, 8, 9};
  ASSERT_EQ(node_sample_tracker.TreeNodeIndices(0, 1), expected_result);

  // Check node begin and node end for right node
  ASSERT_EQ(node_sample_tracker.NodeBegin(0, 2), 3);
  ASSERT_EQ(node_sample_tracker.NodeEnd(0, 2), n);
  expected_result = {0, 1, 3, 4, 5, 6, 7};
  ASSERT_EQ(node_sample_tracker.TreeNodeIndices(0, 2), expected_result);

  // Partition right node based on X[,1] <= 0.5
  tree_split = StochTree::TreeSplit(0.5);
  node_sample_tracker.PartitionTreeNode(dataset.GetCovariates(), 0, 2, 3, 4, 1, tree_split);
  sample_node_mapper.AddSplit(dataset.GetCovariates(), tree_split, 1, 0, 2, 3, 4);

  // Check that node begin and node end haven't changed for old right node, but that the indices have been sifted
  ASSERT_EQ(node_sample_tracker.NodeBegin(0, 2), 3);
  ASSERT_EQ(node_sample_tracker.NodeEnd(0, 2), n);
  expected_result = {1, 6, 0, 3, 4, 5, 7};
  ASSERT_EQ(node_sample_tracker.TreeNodeIndices(0, 2), expected_result);

  // Check node begin and node end for new left node
  ASSERT_EQ(node_sample_tracker.NodeBegin(0, 3), 3);
  ASSERT_EQ(node_sample_tracker.NodeEnd(0, 3), 5);
  expected_result = {1, 6};
  ASSERT_EQ(node_sample_tracker.TreeNodeIndices(0, 3), expected_result);

  // Check node begin and node end for new right node
  ASSERT_EQ(node_sample_tracker.NodeBegin(0, 4), 5);
  ASSERT_EQ(node_sample_tracker.NodeEnd(0, 4), n);
  expected_result = {0, 3, 4, 5, 7};
  ASSERT_EQ(node_sample_tracker.TreeNodeIndices(0, 4), expected_result);

  // Prune the node 2 split
  node_sample_tracker.PruneTreeNodeToLeaf(0, 2);
  ASSERT_TRUE(node_sample_tracker.IsLeaf(0, 2));
  ASSERT_FALSE(node_sample_tracker.IsValidNode(0, 3));
  ASSERT_FALSE(node_sample_tracker.IsValidNode(0, 4));
}

// Regression test for a bug in FeatureUnsortedPartition::ReconstituteFromTree.
//
// The reconstitution used to iterate nodes by numeric id, assuming every parent has a
// SMALLER id than its children (so the parent always set a node's index bounds before the
// node itself was processed). That holds for grow-only trees, but the Tree free-list
// RECYCLES deleted node ids after prune+regrow, so a split node can end up with a parent
// of *higher* id. Iterating by id then partitioned that node from stale bounds and
// mis-assigned observations. The fix traverses top-down (BFS from root). This test builds
// a tree whose ids are recycled into a non-topological layout and asserts the
// reconstituted unsorted partition matches a direct evaluation of the tree.
TEST(UnsortedNodeSampleTracker, ReconstituteWithRecycledNodeIds) {
  int n = 12;
  int p = 2;
  // Row-major covariates (n x p), chosen so the splits below route observations to
  // several different leaves.
  std::vector<double> covariates_raw = {
      0.10, 0.20, 0.30, 0.80, 0.40, 0.40, 0.20, 0.90, 0.60, 0.10, 0.70, 0.60,
      0.80, 0.30, 0.90, 0.70, 0.55, 0.95, 0.65, 0.05, 0.85, 0.50, 0.95, 0.25};
  StochTree::ForestDataset dataset;
  dataset.AddCovariates(covariates_raw.data(), n, p, /*row_major=*/true);
  std::vector<double> residual_raw(n, 0.0);
  StochTree::ColumnVector residual(residual_raw.data(), n);
  std::vector<StochTree::FeatureType> feature_types(p, StochTree::FeatureType::kNumeric);

  // Build a single-tree forest, then grow/prune/regrow so node ids get recycled into a
  // non-topological layout (a split node whose parent has a larger id).
  StochTree::TreeEnsemble forest(/*num_trees=*/1, /*output_dim=*/1, /*is_leaf_constant=*/true);
  StochTree::Tree* tree = forest.GetTree(0);
  tree->ExpandNode(0, 0, 0.5, 0.0, 0.0);   // 0 -> {1, 2}
  tree->ExpandNode(1, 0, 0.25, 0.0, 0.0);  // 1 -> {3, 4}
  tree->ExpandNode(2, 0, 0.75, 0.0, 0.0);  // 2 -> {5, 6}
  tree->CollapseToLeaf(1, 0.0);            // prune node 1: frees ids 3, 4
  tree->ExpandNode(5, 1, 0.5, 0.0, 0.0);   // 5 -> recycled {3, 4}
  tree->ExpandNode(3, 0, 0.6, 0.0, 0.0);   // 3 -> new {7, 8}; node 3 is a split whose parent (5) has a LARGER id

  // Guard: confirm the setup actually produced the non-topological condition under test
  // (some internal/split node has a parent with a larger id). If this ever fails, the
  // Tree id-recycling policy changed and the test setup needs to be revisited.
  bool has_parent_with_larger_id = false;
  for (int nid = 0; nid < tree->NumNodes(); nid++) {
    if (tree->IsDeleted(nid) || tree->IsLeaf(nid)) continue;
    int parent = tree->Parent(nid);
    if (parent != StochTree::Tree::kInvalidNodeId && parent > nid) {
      has_parent_with_larger_id = true;
    }
  }
  ASSERT_TRUE(has_parent_with_larger_id)
      << "Test setup did not produce a node whose parent has a larger id";

  // Reconstitute the tracker from the forest (this exercises ReconstituteFromTree).
  StochTree::ForestTracker tracker(dataset.GetCovariates(), feature_types, /*num_trees=*/1, n);
  tracker.ReconstituteFromForest(forest, dataset, residual, /*is_mean_model=*/true);

  // Ground truth membership: evaluate the tree directly for every observation.
  Eigen::MatrixXd& covariates = dataset.GetCovariates();
  std::map<int, std::set<int>> expected_membership;
  for (int i = 0; i < n; i++) {
    int leaf = StochTree::EvaluateTree(*tree, covariates, i);
    expected_membership[leaf].insert(i);
  }

  // Every leaf's reconstituted membership must match the ground truth exactly.
  int total_assigned = 0;
  for (int leaf : tree->GetLeaves()) {
    std::set<int> reconstituted;
    auto it_begin = tracker.UnsortedNodeBeginIterator(0, leaf);
    auto it_end = tracker.UnsortedNodeEndIterator(0, leaf);
    for (auto it = it_begin; it != it_end; ++it) {
      reconstituted.insert(static_cast<int>(*it));
    }
    total_assigned += static_cast<int>(reconstituted.size());
    EXPECT_EQ(reconstituted, expected_membership[leaf])
        << "Reconstituted membership mismatch for leaf node " << leaf;
  }
  // Every observation must be assigned to exactly one leaf.
  EXPECT_EQ(total_assigned, n);
}
