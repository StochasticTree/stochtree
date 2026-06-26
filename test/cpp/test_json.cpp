/*!
 * Derived from xgboost tree unit test code:
 * https://github.com/dmlc/xgboost/blob/master/tests/cpp/tree/test_tree_model.cc
 */
#include <gtest/gtest.h>
#include <testutils.h>
#include <stochtree/bart.h>
#include <stochtree/bcf.h>
#include <stochtree/ensemble.h>
#include <stochtree/log.h>
#include <stochtree/tree.h>
#include <nlohmann/json.hpp>
#include <memory>

TEST(Json, TreeUnivariateLeaf) {
  // Initialize tree
  StochTree::Tree tree;
  StochTree::TreeSplit split;
  tree.Init(1);

  // Perform three splits
  split = StochTree::TreeSplit(0.5);
  tree.ExpandNode(0, 0, split, 0., 0.);
  split = StochTree::TreeSplit(0.75);
  tree.ExpandNode(1, 1, split, 0., 0.);
  split = StochTree::TreeSplit(0.6);
  tree.ExpandNode(3, 2, split, 0., 0.);

  // Prune node 3 to a leaf
  tree.CollapseToLeaf(3, 0.);

  // Write to json
  nlohmann::json tree_json = tree.to_json();

  // Convert back to a tree
  StochTree::Tree tree_parsed;
  tree_parsed.from_json(tree_json);

  // Check that trees are the same
  ASSERT_EQ(tree, tree_parsed);
}

TEST(Json, TreeUnivariateLeafCategoricalSplit) {
  // Initialize tree
  StochTree::Tree tree;
  StochTree::TreeSplit split;
  tree.Init(1);

  // Perform three splits
  std::vector<uint32_t> split_categories_1{1, 3, 5, 7};
  split = StochTree::TreeSplit(split_categories_1);
  tree.ExpandNode(0, 0, split, 0., 0.);
  std::vector<uint32_t> split_categories_2{2, 3, 5};
  split = StochTree::TreeSplit(split_categories_2);
  tree.ExpandNode(1, 1, split, 0., 0.);
  split = StochTree::TreeSplit(0.6);
  tree.ExpandNode(3, 2, split, 0., 0.);

  // Prune node 3 to a leaf
  tree.CollapseToLeaf(3, 0.);

  // Write to json
  nlohmann::json tree_json = tree.to_json();

  // Convert back to a tree
  StochTree::Tree tree_parsed;
  tree_parsed.from_json(tree_json);

  // Check that trees are the same
  ASSERT_EQ(tree, tree_parsed);
}

TEST(Json, TreeMultivariateLeaf) {
  // Initialize tree
  StochTree::Tree tree;
  StochTree::TreeSplit split;
  int tree_dim = 2;
  std::vector<double> leaf_values1(tree_dim, 0.);
  std::vector<double> leaf_values2(tree_dim, 1.5);
  std::vector<double> leaf_values3(tree_dim, -0.75);
  std::vector<double> leaf_values4(tree_dim, 0.33);
  std::vector<double> leaf_values5(tree_dim, 345235636.4);
  std::vector<double> leaf_values6(tree_dim, 10023.1);
  tree.Init(tree_dim);

  // Perform three splits
  split = StochTree::TreeSplit(0.5);
  tree.ExpandNode(0, 0, split, leaf_values1, leaf_values2);
  split = StochTree::TreeSplit(0.75);
  tree.ExpandNode(1, 1, split, leaf_values3, leaf_values4);
  split = StochTree::TreeSplit(0.6);
  tree.ExpandNode(1, 1, split, leaf_values5, leaf_values6);

  // Prune node 3 to a leaf
  tree.CollapseToLeaf(3, leaf_values3);

  // Write to json
  nlohmann::json tree_json = tree.to_json();

  // Convert back to a tree
  StochTree::Tree tree_parsed;
  tree_parsed.Init(tree_dim);
  tree_parsed.from_json(tree_json);

  // Check that trees are the same
  ASSERT_EQ(tree, tree_parsed);
}

TEST(Json, BARTSamplesRoundTrip) {
  // Build a minimal BARTSamples: a 2-sample mean forest container + parameter traces + scalars.
  StochTree::BARTSamples samples;
  samples.mean_forests = std::make_unique<StochTree::ForestContainer>(
      /*num_trees=*/1, /*output_dimension=*/1, /*is_leaf_constant=*/true, /*is_exponentiated=*/false);
  // Build two retained samples via AddSample (InitializeRoot does NOT count as a retained sample --
  // it sets num_samples_ back to 0 -- so two AddSample calls give a consistent 2-sample container).
  StochTree::TreeEnsemble ens0(/*num_trees=*/1, /*output_dimension=*/1, /*is_leaf_constant=*/true);
  ens0.SetLeafValue(0.5);
  samples.mean_forests->AddSample(ens0);  // sample 0
  StochTree::TreeEnsemble ens1(/*num_trees=*/1, /*output_dimension=*/1, /*is_leaf_constant=*/true);
  ens1.SetLeafValue(-0.25);
  samples.mean_forests->AddSample(ens1);  // sample 1
  samples.global_error_variance_samples = {1.1, 2.2};
  samples.leaf_scale_samples = {0.3, 0.4};
  samples.y_bar = 3.14;
  samples.y_std = 2.71;
  samples.num_samples = 2;

  // Round-trip through the samples-owned subtree
  nlohmann::json obj = samples.ToJson();
  StochTree::BARTSamples restored;
  restored.FromJson(obj);

  // Round-trip must reproduce the original samples exactly (compare against the source object,
  // not re-typed literals -- ToJson is const, so `samples` is untouched and is the source of truth).
  EXPECT_EQ(restored.num_samples, samples.num_samples);
  EXPECT_DOUBLE_EQ(restored.y_bar, samples.y_bar);
  EXPECT_DOUBLE_EQ(restored.y_std, samples.y_std);
  // Every per-draw parameter trace should carry exactly one entry per retained sample.
  ASSERT_EQ(restored.global_error_variance_samples.size(), static_cast<size_t>(samples.num_samples));
  ASSERT_EQ(restored.leaf_scale_samples.size(), static_cast<size_t>(samples.num_samples));
  EXPECT_EQ(restored.global_error_variance_samples, samples.global_error_variance_samples);
  EXPECT_EQ(restored.leaf_scale_samples, samples.leaf_scale_samples);

  // Mean forest survives (byte-level: re-serialized JSON identical); variance forest stays absent
  ASSERT_NE(restored.mean_forests, nullptr);
  EXPECT_EQ(restored.mean_forests->NumSamples(), samples.mean_forests->NumSamples());
  EXPECT_EQ(samples.mean_forests->to_json(), restored.mean_forests->to_json());
  EXPECT_EQ(restored.variance_forests, nullptr);

  // num_forests counter matches the number of forests actually written (mean only -> 1)
  EXPECT_EQ(obj.at("num_forests").get<int>(), 1);
}

TEST(Json, BCFSamplesRoundTrip) {
  // Build a minimal BCFSamples: 2-sample prognostic (mu) + treatment (tau) forests, the full set of
  // BCF parameter traces (univariate treatment), and scalars including treatment_dim.
  StochTree::BCFSamples samples;
  samples.mu_forests = std::make_unique<StochTree::ForestContainer>(1, 1, true, false);
  samples.tau_forests = std::make_unique<StochTree::ForestContainer>(1, 1, false, false);
  StochTree::TreeEnsemble mu0(1, 1, true), mu1(1, 1, true);
  mu0.SetLeafValue(0.5);
  mu1.SetLeafValue(-0.25);
  samples.mu_forests->AddSample(mu0);
  samples.mu_forests->AddSample(mu1);
  StochTree::TreeEnsemble tau0(1, 1, false), tau1(1, 1, false);
  tau0.SetLeafValue(1.0);
  tau1.SetLeafValue(0.8);
  samples.tau_forests->AddSample(tau0);
  samples.tau_forests->AddSample(tau1);

  samples.global_error_variance_samples = {1.1, 2.2};
  samples.leaf_scale_mu_samples = {0.3, 0.4};
  samples.leaf_scale_tau_samples = {0.05, 0.06};
  samples.b0_samples = {-0.5, -0.4};
  samples.b1_samples = {0.5, 0.6};
  samples.tau_0_samples = {0.1, 0.2};  // univariate: treatment_dim x num_samples = 1 x 2
  samples.y_bar = 3.14;
  samples.y_std = 2.71;
  samples.num_samples = 2;
  samples.treatment_dim = 1;

  // Round-trip through the samples-owned subtree
  nlohmann::json obj = samples.ToJson();
  StochTree::BCFSamples restored;
  restored.FromJson(obj);

  // Scalars survive exactly (compared against the source object, not re-typed literals)
  EXPECT_EQ(restored.num_samples, samples.num_samples);
  EXPECT_EQ(restored.treatment_dim, samples.treatment_dim);
  EXPECT_DOUBLE_EQ(restored.y_bar, samples.y_bar);
  EXPECT_DOUBLE_EQ(restored.y_std, samples.y_std);

  // Every per-draw parameter trace carries one entry per retained sample, and round-trips exactly
  ASSERT_EQ(restored.global_error_variance_samples.size(), static_cast<size_t>(samples.num_samples));
  ASSERT_EQ(restored.leaf_scale_mu_samples.size(), static_cast<size_t>(samples.num_samples));
  ASSERT_EQ(restored.leaf_scale_tau_samples.size(), static_cast<size_t>(samples.num_samples));
  ASSERT_EQ(restored.b0_samples.size(), static_cast<size_t>(samples.num_samples));
  ASSERT_EQ(restored.b1_samples.size(), static_cast<size_t>(samples.num_samples));
  ASSERT_EQ(restored.tau_0_samples.size(),
            static_cast<size_t>(samples.num_samples * samples.treatment_dim));
  EXPECT_EQ(restored.global_error_variance_samples, samples.global_error_variance_samples);
  EXPECT_EQ(restored.leaf_scale_mu_samples, samples.leaf_scale_mu_samples);
  EXPECT_EQ(restored.leaf_scale_tau_samples, samples.leaf_scale_tau_samples);
  EXPECT_EQ(restored.b0_samples, samples.b0_samples);
  EXPECT_EQ(restored.b1_samples, samples.b1_samples);
  EXPECT_EQ(restored.tau_0_samples, samples.tau_0_samples);

  // Both forests survive (byte-level: re-serialized JSON identical); no variance forest present
  ASSERT_NE(restored.mu_forests, nullptr);
  ASSERT_NE(restored.tau_forests, nullptr);
  EXPECT_EQ(restored.mu_forests->NumSamples(), samples.mu_forests->NumSamples());
  EXPECT_EQ(restored.tau_forests->NumSamples(), samples.tau_forests->NumSamples());
  EXPECT_EQ(samples.mu_forests->to_json(), restored.mu_forests->to_json());
  EXPECT_EQ(samples.tau_forests->to_json(), restored.tau_forests->to_json());
  EXPECT_EQ(restored.variance_forests, nullptr);

  // num_forests counter matches the number written (prognostic + treatment -> 2)
  EXPECT_EQ(obj.at("num_forests").get<int>(), 2);
}

TEST(Json, TreeMultivariateLeafCategoricalSplit) {
  // Initialize tree
  StochTree::Tree tree;
  StochTree::TreeSplit split;
  int tree_dim = 2;
  std::vector<double> leaf_values1(tree_dim, 0.);
  std::vector<double> leaf_values2(tree_dim, 1.5);
  std::vector<double> leaf_values3(tree_dim, -0.75);
  std::vector<double> leaf_values4(tree_dim, 0.33);
  std::vector<double> leaf_values5(tree_dim, 345235636.4);
  std::vector<double> leaf_values6(tree_dim, 10023.1);
  tree.Init(tree_dim);

  // Perform three splits
  std::vector<uint32_t> split_categories_1{1, 3, 5, 7};
  split = StochTree::TreeSplit(split_categories_1);
  tree.ExpandNode(0, 0, split, leaf_values1, leaf_values2);
  std::vector<uint32_t> split_categories_2{2, 3, 5};
  split = StochTree::TreeSplit(split_categories_2);
  tree.ExpandNode(1, 1, split, leaf_values3, leaf_values4);
  split = StochTree::TreeSplit(0.6);
  tree.ExpandNode(1, 1, split, leaf_values5, leaf_values6);

  // Prune node 3 to a leaf
  tree.CollapseToLeaf(3, leaf_values3);

  // Write to json
  nlohmann::json tree_json = tree.to_json();

  // Convert back to a tree
  StochTree::Tree tree_parsed;
  tree_parsed.Init(tree_dim);
  tree_parsed.from_json(tree_json);

  // Check that trees are the same
  ASSERT_EQ(tree, tree_parsed);
}
