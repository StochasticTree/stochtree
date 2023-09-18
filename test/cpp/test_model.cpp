/*!
 * End-to-end test of the "grow-from-root" procedure, removing stochastic aspects:
 *   1. Data is fixed so that log-likelihoods can be computed deterministically
 *   2. Variance parameters are fixed
 *   3. Leaf node parameters are set to the posterior mean, rather than sampled
 */
#include <gtest/gtest.h>
#include <testutils.h>
#include <stochtree/ensemble.h>
#include <stochtree/log.h>
#include <stochtree/meta.h>
#include <stochtree/model.h>
#include <stochtree/model_draw.h>
#include <stochtree/train_data.h>
#include <stochtree/tree.h>
#include <iostream>
#include <memory>
#include <vector>

class ModelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Define a common dataset to be used in multiple tests
    n = 40;
    p = 3;
    std::vector<double> data_vector = {
      554.5587, 0.113703411, 0.55333359, 0.92640048, 
      652.06658, 0.622299405, 0.64640609, 0.47190972, 
      314.83592, 0.609274733, 0.31182431, 0.14261534, 
      624.26646, 0.623379442, 0.6218192, 0.54426976, 
      334.06715, 0.860915384, 0.32977018, 0.19617465, 
      506.97611, 0.640310605, 0.50199747, 0.89858049, 
      676.0034, 0.009495756, 0.67709453, 0.38949978, 
      487.52182, 0.232550506, 0.48499124, 0.31087078, 
      248.58881, 0.666083758, 0.24392883, 0.16002866, 
      768.36752, 0.514251141, 0.76545979, 0.89618585, 
      77.25473, 0.693591292, 0.07377988, 0.16639378, 
      311.95601, 0.544974836, 0.3096866, 0.9004246, 
      718.31889, 0.282733584, 0.71727174, 0.1340782, 
      509.81137, 0.923433484, 0.50454591, 0.13161413, 
      156.53081, 0.29231584, 0.15299896, 0.1052875, 
      507.96657, 0.837295628, 0.50393349, 0.51158358, 
      494.00134, 0.286223285, 0.49396092, 0.30019905, 
      751.81072, 0.26682078, 0.7512002, 0.0267169, 
      175.8417, 0.18672279, 0.17464982, 0.30964743, 
      849.23648, 0.232225911, 0.84839241, 0.74211966, 
      866.2391, 0.316612455, 0.86483383, 0.03545673, 
      43.20075, 0.302693371, 0.04185728, 0.56507611, 
      316.60508, 0.159046003, 0.31718216, 0.28025778, 
      13.77613, 0.039995918, 0.01374994, 0.20419632, 
      240.96996, 0.218799541, 0.23902573, 0.1337389, 
      711.24522, 0.810598552, 0.70649462, 0.32568192, 
      311.27324, 0.525697547, 0.30809476, 0.15506197, 
      512.71812, 0.914658166, 0.50854757, 0.12996214, 
      55.61175, 0.831345047, 0.05164662, 0.43553106, 
      563.60416, 0.045770263, 0.56456984, 0.03864265, 
      123.70749, 0.456091482, 0.12148019, 0.71330156, 
      894.41751, 0.265186672, 0.89283638, 0.10076904, 
      17.85658, 0.304672203, 0.01462726, 0.95030494, 
      786.65915, 0.50730687, 0.7831211, 0.12181776, 
      90.37123, 0.181096208, 0.08996133, 0.21965662, 
      523.34388, 0.759670635, 0.51918998, 0.91308777, 
      384.13832, 0.201248038, 0.38426669, 0.94585312, 
      72.22475, 0.258809819, 0.0700525, 0.27915622, 
      326.57809, 0.992150418, 0.32064442, 0.12347109, 
      674.65328, 0.80735234, 0.6684954, 0.79716046, 
    };
    y_bar= 431.229372;
    
    // Define any config parameters that aren't defaults
    const char* params = "label_column=0 num_trees=2 min_data_in_leaf=1 alpha=0.95 beta=1.25";
    auto param = StochTree::Config::Str2Map(params);
    config.Set(param);

    // Define data loader
    StochTree::TrainDataLoader dataset_loader(config, 1, nullptr);

    // Load some test data
    dataset.reset(dataset_loader.ConstructFromMatrix(data_vector.data(), p + 1, n, true));
  }

  // void TearDown() override {}
  std::unique_ptr<StochTree::TrainData> dataset;
  std::vector<std::vector<StochTree::data_size_t>> tree_observation_indices;
  StochTree::Config config;
  double y_bar;
  int p;
  StochTree::data_size_t n;
};

TEST_F(ModelTest, GrowFromRoot) {
  // Initialize model and ensemble
  std::unique_ptr<StochTree::XBARTGaussianRegressionModel> model;
  std::unique_ptr<StochTree::XBARTGaussianRegressionModelDraw> model_draw;
  model.reset(new StochTree::XBARTGaussianRegressionModel(config));
  model_draw.reset(new StochTree::XBARTGaussianRegressionModelDraw(config));

  int num_trees = config.num_trees;
  EXPECT_EQ(num_trees, 2);

  // Initialize the vector of vectors of leaf indices for each tree
  tree_observation_indices.resize(num_trees);
  for (int j = 0; j < num_trees; j++) {
    tree_observation_indices[j].resize(n);
  }

  // "Initialize" the ensemble by setting all trees to a root node predicting mean(y) / num_trees
  StochTree::Tree* tree;
  for (int j = 0; j < num_trees; j++) {
    tree = (model_draw->GetEnsemble())->GetTree(j);
    (*tree)[0].SetLeaf(y_bar / num_trees);
    for (size_t k = 0; k < n; k++) {
      tree_observation_indices[j][k] = 0;
    }
  }

  // Compute sufficient statistics for the root node
  StochTree::XBARTGaussianRegressionSuffStat root_suff_stat = model->ComputeNodeSuffStat(dataset.get(), 0, n, 0);
  EXPECT_EQ(n, 40);
  EXPECT_EQ(n, root_suff_stat.sample_size_);
  EXPECT_NEAR(17249.17488, root_suff_stat.outcome_sum_, 0.01);

  // Fix both sigma^2 and tau
  model->SetGlobalParameter("sigma_sq", 10);
  model->SetGlobalParameter("tau", 20);

  // Residualize the model
  dataset->ResidualReset();
  for (int j = 0; j < num_trees; j++) {
    tree = (model_draw->GetEnsemble())->GetTree(j);
    dataset->ResidualSubtract(tree->PredictFromNodes(tree_observation_indices[j]));
  }
  root_suff_stat = model->ComputeNodeSuffStat(dataset.get(), 0, n, 0);
  EXPECT_NEAR(-2.27374E-12, root_suff_stat.outcome_sum_, 0.01);

  // Add back in predictions from the first tree
  dataset->ResidualAdd(tree->PredictFromNodes(tree_observation_indices[0]));

  /**************************************************************************/
  /*  Split 1: Leaf 0                                                       */
  /**************************************************************************/

  // Enumerate possible model cutpoints for the root node in the first tree
  tree = (model_draw->GetEnsemble())->GetTree(0);
  std::vector<double> log_cutpoint_evaluations;
  std::vector<int> cutpoint_features;
  std::vector<double> cutpoint_values;
  StochTree::data_size_t valid_cutpoint_count;
  valid_cutpoint_count = 0;
  model->Cutpoints(dataset.get(), tree, 0, 0, n, log_cutpoint_evaluations, cutpoint_features, cutpoint_values, valid_cutpoint_count);

  std::vector<double> expected_logliks = {
    -139088.45, -138292.46, -139552.08, -139814.94, -139686.88, -137996.03, -135860.32, 
    -135773.04, -133978.60, -138216.40, -138654.86, -136206.08, -139237.40, -139815.35, 
    -139365.06, -139148.31, -139772.06, -139402.62, -137435.68, -139492.33, -138284.08, 
    -139620.52, -139712.82, -139811.21, -139777.70, -139609.63, -139810.49, -139516.18, 
    -139238.30, -139704.37, -139492.43, -139708.87, -139679.45, -138246.40, -139794.82, 
    -139617.17, -139796.81, -139504.25, -138366.51, -130553.15, -121571.83, -113082.32, 
    -104646.73, -96447.56, -87986.69, -79597.59, -72182.00, -65738.29, -59665.06, 
    -56035.69, -52251.23, -50775.33, -48833.25, -46540.60, -43852.71, -41126.36, 
    -38242.39, -37102.81, -40080.97, -42828.71, -45650.92, -48047.04, -50070.81, 
    -51773.73, -53483.44, -56081.65, -58615.22, -63310.97, -68744.26, -74686.10, 
    -80241.74, -86839.89, -93299.84, -100710.46, -108361.25, -116307.43, -126062.69, 
    -135248.59, -138129.75, -130510.78, -129854.49, -119779.02, -129366.60, -122950.06, 
    -126884.87, -126378.48, -125817.39, -129901.86, -125543.76, -128119.35, -130325.17, 
    -132959.66, -136620.30, -137380.20, -139422.43, -139800.62, -138999.20, -138506.87, 
    -138825.70, -137402.00, -137783.33, -139182.77, -139760.19, -138619.09, -139477.29, 
    -139657.85, -139808.79, -139031.53, -136901.54, -139426.37, -139811.98, -138616.03, 
    -137752.91, -138462.67, -136973.23, -133251.23, -130698.66, -138692.2
  };

  for (int i = 0; i < log_cutpoint_evaluations.size(); i++) {
    EXPECT_NEAR(log_cutpoint_evaluations[i], expected_logliks[i], 0.01);
  }

  // Convert log marginal likelihood to marginal likelihood, normalizing by the largest value
  double largest_mll = *std::max_element(log_cutpoint_evaluations.begin(), log_cutpoint_evaluations.end());
  std::vector<double> cutpoint_evaluations(log_cutpoint_evaluations.size());
  for (StochTree::data_size_t i = 0; i < log_cutpoint_evaluations.size(); i++){
    cutpoint_evaluations[i] = std::exp(log_cutpoint_evaluations[i] - largest_mll);
  }

  std::vector<double> expected_likelihoods = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  };

  for (int i = 0; i < cutpoint_evaluations.size(); i++) {
    EXPECT_NEAR(cutpoint_evaluations[i], expected_likelihoods[i], 0.01);
  }

  // Split at the only nonzero likelihood
  std::unordered_map<StochTree::node_t, std::pair<StochTree::data_size_t, StochTree::data_size_t>> node_index_map;
  node_index_map.insert({0, std::make_pair(0, n)});
  std::deque<StochTree::node_t> split_queue;
  StochTree::data_size_t split_chosen = 57;
  int split_feature = cutpoint_features[split_chosen];
  double split_value = cutpoint_values[split_chosen];
  EXPECT_EQ(split_feature, 1);
  EXPECT_NEAR(split_value, 0.3842667, 0.001);
  model->AddSplitToModel(dataset.get(), tree, 0, 0, n, split_feature, split_value, 
                         split_queue, tree_observation_indices, 0);
  
  // Check that split queue now has 1 and 2
  EXPECT_EQ(split_queue.size(), 2);
  EXPECT_EQ(split_queue.front(), 1);
  EXPECT_EQ(split_queue.back(), 2);

  // Check that the training dataset was correctly partitioned
  std::vector<StochTree::data_size_t> sort_inds_feature_0 = {
    23, 22, 34, 18, 36, 24, 37, 14, 21, 32, 30, 26, 11, 2, 
    8, 10, 28, 4, 38, 6, 29, 0, 19, 7, 31, 17, 12, 16, 20, 
    33, 9, 1, 3, 5, 35, 39, 25, 15, 27, 13
  };
  std::vector<StochTree::data_size_t> sort_inds_feature_1 = {
    23, 32, 21, 28, 37, 10, 34, 30, 14, 18, 24, 8, 26, 11, 
    2, 22, 38, 4, 36, 7, 16, 5, 15, 13, 27, 35, 0, 
    29, 3, 1, 39, 6, 25, 12, 17, 9, 33, 19, 20, 31
  };
  std::vector<StochTree::data_size_t> sort_inds_feature_2 = {
    14, 38, 24, 2, 26, 8, 10, 4, 23, 34, 37, 22, 18, 28, 21, 
    30, 11, 36, 32, 17, 20, 29, 31, 33, 27, 13, 12, 16, 
    7, 25, 6, 1, 15, 3, 19, 39, 9, 5, 35, 0
  };

  for (int i = 0; i < n; i++) {
    EXPECT_EQ(dataset->get_feature_sort_index(i, 0), sort_inds_feature_0[i]);
    EXPECT_EQ(dataset->get_feature_sort_index(i, 1), sort_inds_feature_1[i]);
    EXPECT_EQ(dataset->get_feature_sort_index(i, 2), sort_inds_feature_2[i]);
  }

  // Check that leaf node statistics are computed correctly
  StochTree::data_size_t left_cutoff = 19;
  StochTree::XBARTGaussianRegressionSuffStat node_suff_stat, left_suff_stat, right_suff_stat;
  node_suff_stat = model->ComputeNodeSuffStat(dataset.get(), 0, n, 0);
  model->AccumulateSplitRule(dataset.get(), left_suff_stat, split_feature, split_value, 0, n);
  right_suff_stat = model->SubtractSuffStat(node_suff_stat, left_suff_stat);
  EXPECT_EQ(left_suff_stat.sample_size_, left_cutoff);
  EXPECT_NEAR(left_suff_stat.outcome_sum_, -481.2905, 0.01);
  EXPECT_EQ(right_suff_stat.sample_size_, n-left_cutoff);
  EXPECT_NEAR(right_suff_stat.outcome_sum_, 9105.878, 0.01);

  // Check that tree leaf indices are updated correctly
  for (int i = 0; i < n; i++) {
    if (i < left_cutoff) {
      EXPECT_EQ(tree_observation_indices[0][sort_inds_feature_0[i]], 1);
      EXPECT_EQ(tree_observation_indices[0][sort_inds_feature_1[i]], 1);
      EXPECT_EQ(tree_observation_indices[0][sort_inds_feature_2[i]], 1);
    } else {
      EXPECT_EQ(tree_observation_indices[0][sort_inds_feature_0[i]], 2);
      EXPECT_EQ(tree_observation_indices[0][sort_inds_feature_1[i]], 2);
      EXPECT_EQ(tree_observation_indices[0][sort_inds_feature_2[i]], 2);
    }
  }

  /**************************************************************************/
  /*  Split 2: Leaf 1                                                       */
  /**************************************************************************/

  // Enumerate possible model cutpoints for the leaf 1 in the first tree
  StochTree::data_size_t leaf_node = 1;
  log_cutpoint_evaluations.clear();
  cutpoint_features.clear();
  cutpoint_values.clear();
  valid_cutpoint_count = 0;
  model->Cutpoints(dataset.get(), tree, leaf_node, 0, left_cutoff, log_cutpoint_evaluations, cutpoint_features, cutpoint_values, valid_cutpoint_count);

  expected_logliks = {
    -13406.944, -14359.255, -14048.732, -14041.346, -14382.021, -14316.199, -14384.489, 
    -14347.495, -13919.936, -12857.376, -12234.703, -13163.874, -13847.539, -14280.911, 
    -14375.135, -13988.490, -12480.409, -13617.525, -13406.944, -11763.577, -10277.164, 
    -8819.827, -7474.244, -6020.932, -4597.302, -3620.755, -3103.892, -2767.128, 
    -3593.270, -4348.064, -6146.148, -7705.995, -9111.069, -10371.546, -11642.708, 
    -12888.055, -14377.824, -14110.164, -13945.812, -13234.882, -12346.039, -11970.208, 
    -13111.537, -11943.939, -13421.903, -13952.124, -14311.937, -13897.475, -13937.957, 
    -14344.645, -14297.390, -14000.668, -14375.231, -13455.087, -14377.1
  };

  EXPECT_EQ(expected_logliks.size(), log_cutpoint_evaluations.size());
  for (int i = 0; i < log_cutpoint_evaluations.size(); i++) {
    EXPECT_NEAR(log_cutpoint_evaluations[i], expected_logliks[i], 0.01);
  }

  // Convert log marginal likelihood to marginal likelihood, normalizing by the largest value
  largest_mll = *std::max_element(log_cutpoint_evaluations.begin(), log_cutpoint_evaluations.end());
  cutpoint_evaluations.clear();
  cutpoint_evaluations.resize(log_cutpoint_evaluations.size());
  for (StochTree::data_size_t i = 0; i < log_cutpoint_evaluations.size(); i++){
    cutpoint_evaluations[i] = std::exp(log_cutpoint_evaluations[i] - largest_mll);
  }

  expected_likelihoods = {
    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 
    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 
    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 
    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 
    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 
    0.000000e+00, 5.562753e-147, 1.000000e+00, 0.000000e+00, 0.000000e+00, 
    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 
    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 
    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 
    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 
    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00 
  };

  for (int i = 0; i < cutpoint_evaluations.size(); i++) {
    EXPECT_NEAR(cutpoint_evaluations[i], expected_likelihoods[i], 0.01);
  }

  // Split at the only nonzero likelihood
  split_queue.pop_front();
  split_chosen = 27;
  split_feature = cutpoint_features[split_chosen];
  split_value = cutpoint_values[split_chosen];
  EXPECT_EQ(split_feature, 1);
  EXPECT_NEAR(split_value, 0.1746498, 0.001);
  model->AddSplitToModel(dataset.get(), tree, leaf_node, 0, left_cutoff, split_feature, split_value, 
                         split_queue, tree_observation_indices, 0);
  
  // Check that split queue now has 3, 4, and 2
  EXPECT_EQ(split_queue.size(), 3);
  EXPECT_EQ(split_queue.front(), 3);
  EXPECT_EQ(split_queue.back(), 2);

  // Check that the training dataset was correctly partitioned
  sort_inds_feature_0.clear();
  sort_inds_feature_0 = {
    23, 34, 18, 37, 14, 21, 32, 30, 10, 28, 22, 36, 24, 26, 11, 2, 8, 4, 38
  };
  sort_inds_feature_1.clear();
  sort_inds_feature_1 = {
    23, 32, 21, 28, 37, 10, 34, 30, 14, 18, 24, 8, 26, 11, 2, 22, 38, 4, 36
  };
  sort_inds_feature_2.clear();
  sort_inds_feature_2 = {
    14, 10, 23, 34, 37, 18, 28, 21, 30, 32, 38, 24, 2, 26, 8, 4, 22, 11, 36
  };

  for (int i = 0; i < left_cutoff; i++) {
    EXPECT_EQ(dataset->get_feature_sort_index(i, 0), sort_inds_feature_0[i]);
    EXPECT_EQ(dataset->get_feature_sort_index(i, 1), sort_inds_feature_1[i]);
    EXPECT_EQ(dataset->get_feature_sort_index(i, 2), sort_inds_feature_2[i]);
  }

  // Check that leaf node statistics are computed correctly
  StochTree::data_size_t new_left_cutoff = 10;
  model->ResetSuffStat(left_suff_stat);
  node_suff_stat = model->ComputeNodeSuffStat(dataset.get(), 0, left_cutoff, 0);
  model->AccumulateSplitRule(dataset.get(), left_suff_stat, split_feature, split_value, 0, left_cutoff);
  right_suff_stat = model->SubtractSuffStat(node_suff_stat, left_suff_stat);
  EXPECT_EQ(node_suff_stat.sample_size_, left_cutoff);
  EXPECT_NEAR(node_suff_stat.outcome_sum_, -481.2905, 0.01);
  EXPECT_EQ(left_suff_stat.sample_size_, new_left_cutoff);
  EXPECT_NEAR(left_suff_stat.outcome_sum_, -1329.771, 0.01);
  EXPECT_EQ(right_suff_stat.sample_size_, left_cutoff - new_left_cutoff);
  EXPECT_NEAR(right_suff_stat.outcome_sum_, 848.4804, 0.01);

  // Check that tree leaf indices are updated correctly
  for (int i = 0; i < left_cutoff; i++) {
    if (i < new_left_cutoff) {
      EXPECT_EQ(tree_observation_indices[0][sort_inds_feature_0[i]], 3);
      EXPECT_EQ(tree_observation_indices[0][sort_inds_feature_1[i]], 3);
      EXPECT_EQ(tree_observation_indices[0][sort_inds_feature_2[i]], 3);
    } else {
      EXPECT_EQ(tree_observation_indices[0][sort_inds_feature_0[i]], 4);
      EXPECT_EQ(tree_observation_indices[0][sort_inds_feature_1[i]], 4);
      EXPECT_EQ(tree_observation_indices[0][sort_inds_feature_2[i]], 4);
    }
  }

  /**************************************************************************/
  /*  Split 3: Leaf 4                                                       */
  /**************************************************************************/

  // Enumerate possible model cutpoints for the leaf 4 in the first tree
  leaf_node = 4;
  log_cutpoint_evaluations.clear();
  cutpoint_features.clear();
  cutpoint_values.clear();
  valid_cutpoint_count = 0;
  model->Cutpoints(dataset.get(), tree, leaf_node, new_left_cutoff, left_cutoff, log_cutpoint_evaluations, cutpoint_features, cutpoint_values, valid_cutpoint_count);

  expected_logliks = {
    -1133.1645, -1072.6029, -1160.4445, -1159.0502, -1152.3361, -1134.2633, -1152.4674, -1149.8215, -752.9172, 
    -529.7358, -711.2372, -811.3146, -883.5001, -936.4706, -1011.0201, -1093.5187, -1149.8215, -1007.2218, 
    -1077.6088, -1108.8227, -948.3706, -1056.2457, -1086.2574, -1093.5187, -965.9783
  };

  EXPECT_EQ(expected_logliks.size(), log_cutpoint_evaluations.size());
  for (int i = 0; i < log_cutpoint_evaluations.size(); i++) {
    EXPECT_NEAR(log_cutpoint_evaluations[i], expected_logliks[i], 0.01);
  }

  // Convert log marginal likelihood to marginal likelihood, normalizing by the largest value
  largest_mll = *std::max_element(log_cutpoint_evaluations.begin(), log_cutpoint_evaluations.end());
  cutpoint_evaluations.clear();
  cutpoint_evaluations.resize(log_cutpoint_evaluations.size());
  for (StochTree::data_size_t i = 0; i < log_cutpoint_evaluations.size(); i++){
    cutpoint_evaluations[i] = std::exp(log_cutpoint_evaluations[i] - largest_mll);
  }

  expected_likelihoods = {
    8.594875e-263, 1.721050e-236, 1.220931e-274, 4.922740e-274, 4.055971e-271, 2.864374e-263, 3.556896e-271, 
    5.014124e-270, 1.184530e-97, 1.000000e+00, 1.495956e-79, 5.150809e-123, 2.301904e-154, 2.276659e-177, 
    9.569368e-210, 1.419602e-245, 5.014124e-270, 4.270358e-208, 1.152820e-238, 3.204212e-252, 1.545982e-182, 
    2.185892e-229, 2.021721e-242, 1.419602e-245, 3.485443e-190
  };

  for (int i = 0; i < cutpoint_evaluations.size(); i++) {
    EXPECT_NEAR(cutpoint_evaluations[i], expected_likelihoods[i], 0.01);
  }

  // Split at the only nonzero likelihood
  split_chosen = 9;
  split_feature = cutpoint_features[split_chosen];
  split_value = cutpoint_values[split_chosen];
  StochTree::data_size_t leaf_begin = new_left_cutoff;
  StochTree::data_size_t leaf_end = left_cutoff;
  EXPECT_EQ(split_feature, 1);
  EXPECT_NEAR(split_value, 0.2439288, 0.001);
  model->AddSplitToModel(dataset.get(), tree, leaf_node, leaf_begin, leaf_end, split_feature, split_value, 
                         split_queue, tree_observation_indices, 0);

  // Check that the training dataset was correctly partitioned
  sort_inds_feature_0.clear();
  sort_inds_feature_0 = {
    24, 8, 22, 36, 26, 11, 2, 4, 38
  };
  sort_inds_feature_1.clear();
  sort_inds_feature_1 = {
    24, 8, 26, 11, 2, 22, 38, 4, 36
  };
  sort_inds_feature_2.clear();
  sort_inds_feature_2 = {
    24, 8, 38, 2, 26, 4, 22, 11, 36
  };

  StochTree::data_size_t idx = 0;
  for (int i = leaf_begin; i < leaf_end; i++) {
    EXPECT_EQ(dataset->get_feature_sort_index(i, 0), sort_inds_feature_0[idx]);
    EXPECT_EQ(dataset->get_feature_sort_index(i, 1), sort_inds_feature_1[idx]);
    EXPECT_EQ(dataset->get_feature_sort_index(i, 2), sort_inds_feature_2[idx]);
    idx++;
  }

  // Check that leaf node statistics are computed correctly
  new_left_cutoff = 2;
  model->ResetSuffStat(left_suff_stat);
  node_suff_stat = model->ComputeNodeSuffStat(dataset.get(), leaf_begin, leaf_end, 0);
  model->AccumulateSplitRule(dataset.get(), left_suff_stat, split_feature, split_value, leaf_begin, leaf_end);
  right_suff_stat = model->SubtractSuffStat(node_suff_stat, left_suff_stat);
  EXPECT_EQ(node_suff_stat.sample_size_, leaf_end - leaf_begin);
  EXPECT_NEAR(node_suff_stat.outcome_sum_, 848.4804, 0.01);
  EXPECT_EQ(left_suff_stat.sample_size_, new_left_cutoff);
  EXPECT_NEAR(left_suff_stat.outcome_sum_, 58.3294, 0.01);
  EXPECT_EQ(right_suff_stat.sample_size_, leaf_end - leaf_begin - new_left_cutoff);
  EXPECT_NEAR(right_suff_stat.outcome_sum_, 790.151, 0.01);

  // Check that tree leaf indices are updated correctly
  for (int i = 0; i < leaf_end - leaf_begin; i++) {
    if (i < new_left_cutoff) {
      EXPECT_EQ(tree_observation_indices[0][sort_inds_feature_0[i]], 5);
      EXPECT_EQ(tree_observation_indices[0][sort_inds_feature_1[i]], 5);
      EXPECT_EQ(tree_observation_indices[0][sort_inds_feature_2[i]], 5);
    } else {
      EXPECT_EQ(tree_observation_indices[0][sort_inds_feature_0[i]], 6);
      EXPECT_EQ(tree_observation_indices[0][sort_inds_feature_1[i]], 6);
      EXPECT_EQ(tree_observation_indices[0][sort_inds_feature_2[i]], 6);
    }
  }

  /**************************************************************************/
  /*  Split 4: Leaf 2                                                       */
  /**************************************************************************/

  // Enumerate possible model cutpoints for leaf 2 in the first tree
  leaf_node = 2;
  leaf_begin = 19;
  leaf_end = 40;
  log_cutpoint_evaluations.clear();
  cutpoint_features.clear();
  cutpoint_values.clear();
  valid_cutpoint_count = 0;
  model->Cutpoints(dataset.get(), tree, leaf_node, leaf_begin, leaf_end, log_cutpoint_evaluations, cutpoint_features, cutpoint_values, valid_cutpoint_count);

  expected_logliks = {
    -26183.22, -25948.20, -25562.07, -27011.75, -26498.76, -27086.33, -26834.34, -26494.44, -26978.52, -25966.15, 
    -24728.90, -23196.53, -22774.59, -22606.77, -23870.04, -24777.93, -23842.35, -21681.39, -22667.21, -23287.22, 
    -22748.53, -21808.63, -20845.20, -19690.49, -18383.01, -16936.33, -15497.52, -14511.09, -13495.40, -13604.60, 
    -14100.08, -14861.26, -15438.35, -16551.03, -17607.57, -19144.75, -20805.84, -22609.69, -25145.34, -26984.22, 
    -26847.95, -26582.83, -26978.64, -25383.45, -23974.52, -25432.45, -26358.39, -25859.32, -26636.22, -27042.32, 
    -26857.93, -26691.91, -26581.32, -26977.69, -26980.11, -25811.60, -25174.13, -22675.14, -23706.18, -24261.37, -22716.81
  };

  EXPECT_EQ(expected_logliks.size(), log_cutpoint_evaluations.size());
  for (int i = 0; i < log_cutpoint_evaluations.size(); i++) {
    EXPECT_NEAR(log_cutpoint_evaluations[i], expected_logliks[i], 0.01);
  }

  // Convert log marginal likelihood to marginal likelihood, normalizing by the largest value
  largest_mll = *std::max_element(log_cutpoint_evaluations.begin(), log_cutpoint_evaluations.end());
  cutpoint_evaluations.clear();
  cutpoint_evaluations.resize(log_cutpoint_evaluations.size());
  for (StochTree::data_size_t i = 0; i < log_cutpoint_evaluations.size(); i++){
    cutpoint_evaluations[i] = std::exp(log_cutpoint_evaluations[i] - largest_mll);
  }

  expected_likelihoods = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  };

  for (int i = 0; i < cutpoint_evaluations.size(); i++) {
    EXPECT_NEAR(cutpoint_evaluations[i], expected_likelihoods[i], 0.01);
  }

  // Split at the only nonzero likelihood
  split_chosen = 28;
  split_feature = cutpoint_features[split_chosen];
  split_value = cutpoint_values[split_chosen];
  EXPECT_EQ(split_feature, 1);
  EXPECT_NEAR(split_value, 0.5645698, 0.001);
  model->AddSplitToModel(dataset.get(), tree, leaf_node, leaf_begin, leaf_end, split_feature, split_value, 
                         split_queue, tree_observation_indices, 0);

  // Check that the training dataset was correctly partitioned
  sort_inds_feature_0.clear();
  sort_inds_feature_0 = {
    29, 0, 7, 16, 5, 35, 15, 27, 13, 6, 19, 31, 17, 12, 20, 33, 9, 1, 3, 39, 25
  };
  sort_inds_feature_1.clear();
  sort_inds_feature_1 = {
    7, 16, 5, 15, 13, 27, 35, 0, 29, 3, 1, 39, 6, 25, 12, 17, 9, 33, 19, 20, 31
  };
  sort_inds_feature_2.clear();
  sort_inds_feature_2 = {
    29, 27, 13, 16, 7, 15, 5, 35, 0, 17, 20, 31, 33, 12, 25, 6, 1, 3, 19, 39, 9
  };

  idx = 0;
  for (int i = leaf_begin; i < leaf_end; i++) {
    EXPECT_EQ(dataset->get_feature_sort_index(i, 0), sort_inds_feature_0[idx]);
    EXPECT_EQ(dataset->get_feature_sort_index(i, 1), sort_inds_feature_1[idx]);
    EXPECT_EQ(dataset->get_feature_sort_index(i, 2), sort_inds_feature_2[idx]);
    idx++;
  }

  // Check that leaf node statistics are computed correctly
  new_left_cutoff = 9;
  model->ResetSuffStat(left_suff_stat);
  node_suff_stat = model->ComputeNodeSuffStat(dataset.get(), leaf_begin, leaf_end, 0);
  model->AccumulateSplitRule(dataset.get(), left_suff_stat, split_feature, split_value, leaf_begin, leaf_end);
  right_suff_stat = model->SubtractSuffStat(node_suff_stat, left_suff_stat);
  EXPECT_EQ(node_suff_stat.sample_size_, leaf_end - leaf_begin);
  EXPECT_NEAR(node_suff_stat.outcome_sum_, 9105.878, 0.01);
  EXPECT_EQ(left_suff_stat.sample_size_, new_left_cutoff);
  EXPECT_NEAR(left_suff_stat.outcome_sum_, 2719.97, 0.01);
  EXPECT_EQ(right_suff_stat.sample_size_, leaf_end - leaf_begin - new_left_cutoff);
  EXPECT_NEAR(right_suff_stat.outcome_sum_, 6385.908, 0.01);

  // Check that tree leaf indices are updated correctly
  for (int i = 0; i < leaf_end - leaf_begin; i++) {
    if (i < new_left_cutoff) {
      EXPECT_EQ(tree_observation_indices[0][sort_inds_feature_0[i]], 7);
      EXPECT_EQ(tree_observation_indices[0][sort_inds_feature_1[i]], 7);
      EXPECT_EQ(tree_observation_indices[0][sort_inds_feature_2[i]], 7);
    } else {
      EXPECT_EQ(tree_observation_indices[0][sort_inds_feature_0[i]], 8);
      EXPECT_EQ(tree_observation_indices[0][sort_inds_feature_1[i]], 8);
      EXPECT_EQ(tree_observation_indices[0][sort_inds_feature_2[i]], 8);
    }
  }

  /**************************************************************************/
  /*  Stop growing tree, set leaf parameters                                */
  /**************************************************************************/
  
  std::vector<StochTree::node_t> tree_leaves = tree->GetLeaves();
  std::vector<StochTree::XBARTGaussianRegressionSuffStat> leaf_suff_stats;
  std::normal_distribution<double> leaf_node_dist(0.,1.);
  // Populate a vector with sufficient statistics of each leaf
  for (int i = 0; i < tree_leaves.size(); i++) {
    leaf_suff_stats.push_back(model->LeafSuffStat(dataset.get(), tree_leaves[i]));
  }

  std::vector<int> leaf_sample_sizes = {
    10, 2, 7, 9, 12
  };
  std::vector<double> leaf_sum_resid = {
    -1329.771, 58.3294, 790.151, 2719.97, 6385.908
  };
  std::vector<double> leaf_post_means = {
    -126.64486, 23.33176, 105.35347, 286.31263, 510.87264
  };
  std::vector<double> leaf_post_stddevs = {
    0.9759001, 2.0000000, 1.1547005, 1.0259784, 0.8944272
  };
  std::vector<int> leaf_indices = {
    3, 5, 6, 7, 8
  };

  // Sample the mean parameter for each leaf
  double node_mean;
  double node_stddev;
  double node_mu;
  for (int i = 0; i < tree_leaves.size(); i++) {
    // Compute mean and variance parameter
    node_mean = model->LeafPosteriorMean(leaf_suff_stats[i]);
    node_stddev = model->LeafPosteriorStddev(leaf_suff_stats[i]);

    EXPECT_EQ(leaf_suff_stats[i].sample_size_, leaf_sample_sizes[i]);
    EXPECT_NEAR(leaf_suff_stats[i].outcome_sum_, leaf_sum_resid[i], 0.01);
    EXPECT_EQ(tree_leaves[i], leaf_indices[i]);
    EXPECT_NEAR(node_mean, leaf_post_means[i], 0.01);
    EXPECT_NEAR(node_stddev, leaf_post_stddevs[i], 0.01);

    // Set leaf value to the posterior mean
    (*tree)[tree_leaves[i]].SetLeaf(node_mean);
  }

  // Update predictions
  std::vector<double> expected_predictions = {
    286.31263, 510.87264, 105.35347, 510.87264, 105.35347, 286.31263, 510.87264, 286.31263, 23.33176, 
    510.87264, -126.64486, 105.35347, 510.87264, 286.31263, -126.64486, 286.31263, 286.31263, 510.87264, 
    -126.64486, 510.87264, 510.87264, -126.64486, 105.35347, -126.64486, 23.33176, 510.87264, 105.35347, 
    286.31263, -126.64486, 286.31263, -126.64486, 510.87264, -126.64486, 510.87264, -126.64486, 286.31263, 
    105.35347, -126.64486, 105.35347, 510.87264
  };
  std::vector<double> actual_predictions = tree->PredictFromNodes(tree_observation_indices[0]);
  for (int i = 0; i < actual_predictions.size(); i++) {
    EXPECT_NEAR(expected_predictions[i], actual_predictions[i], 0.01);
  }

  // Update residuals
  dataset->ResidualSubtract(tree->PredictFromNodes(tree_observation_indices[0]));
  std::vector<double> expected_residuals = {
    52.631379, -74.420744, -6.132232, -102.220864, 13.098995,  5.048793, -50.483928, -14.405499,  9.642365, 
    41.880190, -11.715099, -9.012145, -8.168439, 7.884049,  67.560980,  6.039250, -7.925979,  25.323393, 
    86.871871, 122.749155, 139.751778, -45.769081, -4.363069, -75.193697,  2.023511, -15.242107, -9.694910, 
    10.790807, -33.358078, 61.676846, 34.737657, 167.930185, -71.113248,  60.171825,  1.401402,  21.416567, 
    63.170167, -16.745079, 5.609938, -51.834050
  };
  for (int i = 0; i < expected_residuals.size(); i++) {
    EXPECT_NEAR(expected_residuals[i], dataset->get_residual_value(i), 0.01);
  }

  // Retrieve pointer to second tree
  int tree_num = 1;
  tree = (model_draw->GetEnsemble())->GetTree(tree_num);
  dataset->ResidualAdd(tree->PredictFromNodes(tree_observation_indices[tree_num]));
  expected_residuals = {
    268.2461, 141.1939, 209.4825, 113.3938, 228.7137, 220.6635, 165.1308, 201.2092, 225.2571, 257.4949, 203.8996, 206.6025, 
    207.4462, 223.4987, 283.1757, 221.6539, 207.6887, 240.9381, 302.4866, 338.3638, 355.3665, 169.8456, 211.2516, 140.4210, 
    217.6382, 200.3726, 205.9198, 226.4055, 182.2566, 277.2915, 250.3523, 383.5449, 144.5014, 275.7865, 217.0161, 237.0313, 
    278.7849, 198.8696, 221.2246, 163.7806
  };
  for (int i = 0; i < expected_residuals.size(); i++) {
    EXPECT_NEAR(expected_residuals[i], dataset->get_residual_value(i), 0.01);
  }

  // Reset the dataset sort indices
  dataset->ResetToRaw();

  // Reset the node index map in the model
  model->NodeIndexMapReset(n);

  /**************************************************************************/
  /*  Split 1: Leaf 0                                                       */
  /**************************************************************************/

  // Enumerate possible model cutpoints for the root node in the first tree
  leaf_node = 0;
  log_cutpoint_evaluations.clear();
  cutpoint_features.clear();
  cutpoint_values.clear();
  valid_cutpoint_count = 0;
  model->Cutpoints(dataset.get(), tree, 0, 0, n, log_cutpoint_evaluations, cutpoint_features, cutpoint_values, valid_cutpoint_count);

  expected_logliks = {
    -7892.695, -7559.791, -8286.120, -8592.993, -8604.845, -8626.294, -8796.880, -8834.910, -8834.623, -8764.441, 
    -8790.409, -8811.182, -8546.589, -8503.194, -8544.418, -8579.982, -8423.260, -8558.207, -8707.807, -8405.274, 
    -8311.877, -8113.272, -7954.714, -7998.905, -8035.782, -8056.768, -8328.103, -8613.574, -8601.079, -8573.949, 
    -8598.639, -8529.991, -8660.185, -8688.342, -8752.225, -8732.939, -8681.523, -8597.955, -8424.382, -7589.402, 
    -7335.691, -7279.701, -7276.340, -7361.647, -7438.925, -7569.307, -7854.035, -8216.573, -8530.848, -8541.418, 
    -8568.430, -8544.380, -8520.192, -8501.610, -8485.877, -8494.776, -8520.702, -8651.303, -8614.188, -8587.023, 
    -8586.471, -8586.902, -8590.221, -8598.801, -8629.089, -8713.704, -8787.605, -8598.316, -8344.072, -8073.938, 
    -7715.979, -7476.279, -7216.639, -7155.676, -7194.440, -7364.037, -8077.124, -8736.290, -8559.530, -8789.999, 
    -8668.849, -7947.805, -7693.845, -7477.494, -7616.427, -7690.354, -7760.063, -7843.095, -7957.984, -8045.832, 
    -8134.695, -8138.693, -8217.505, -8200.751, -8461.055, -8473.853, -8530.542, -8552.466, -8580.298, -8359.497, 
    -8410.502, -8459.609, -8590.684, -8663.809, -8779.877, -8775.689, -8835.670, -8816.578, -8831.618, -8770.445, 
    -8824.202, -8784.387, -8775.829, -8787.611, -8738.267, -8491.798, -7642.399, -7606.485
  };

  for (int i = 0; i < log_cutpoint_evaluations.size(); i++) {
    EXPECT_NEAR(log_cutpoint_evaluations[i], expected_logliks[i], 0.01);
  }

  // Convert log marginal likelihood to marginal likelihood, normalizing by the largest value
  largest_mll = *std::max_element(log_cutpoint_evaluations.begin(), log_cutpoint_evaluations.end());
  cutpoint_evaluations.clear();
  cutpoint_evaluations.resize(log_cutpoint_evaluations.size());
  // double eval_sum = 0.;
  for (StochTree::data_size_t i = 0; i < log_cutpoint_evaluations.size(); i++){
    cutpoint_evaluations[i] = std::exp(log_cutpoint_evaluations[i] - largest_mll);
  }

  expected_likelihoods = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  };

  for (int i = 0; i < cutpoint_evaluations.size(); i++) {
    EXPECT_NEAR(cutpoint_evaluations[i], expected_likelihoods[i], 0.01);
  }

  // Split at the only nonzero likelihood
  node_index_map.clear();
  node_index_map.insert({0, std::make_pair(0, n)});
  split_queue.clear();
  split_chosen = 73;
  split_feature = cutpoint_features[split_chosen];
  split_value = cutpoint_values[split_chosen];
  EXPECT_EQ(split_feature, 1);
  EXPECT_NEAR(split_value, 0.7512002, 0.001);
  model->AddSplitToModel(dataset.get(), tree, leaf_node, 0, n, split_feature, split_value, 
                         split_queue, tree_observation_indices, tree_num);
  
  // Check that split queue now has 1 and 2
  EXPECT_EQ(split_queue.size(), 2);
  EXPECT_EQ(split_queue.front(), 1);
  EXPECT_EQ(split_queue.back(), 2);

  // Check that the training dataset was correctly partitioned
  sort_inds_feature_0 = {
    6, 23, 29, 0, 22, 34, 18, 36, 24, 7, 37, 17, 12, 16,  
    14, 21, 32, 30, 26, 11, 2, 1, 3, 5, 8, 10, 35, 
    39, 25, 28, 15, 4, 27, 13, 38, 19, 31, 20, 33, 9
  };
  sort_inds_feature_1 = {
    23, 32, 21, 28, 37, 10, 34, 30, 14, 18, 24, 8, 26, 11, 
    2, 22, 38, 4, 36, 7, 16, 5, 15, 13, 27, 35, 0, 29, 
    3, 1, 39, 6, 25, 12, 17, 9, 33, 19, 20, 31
  };
  sort_inds_feature_2 = {
    17, 29, 14, 38, 27, 13, 24, 12, 2, 26, 8, 10, 4, 23, 34, 
    37, 22, 16, 18, 7, 25, 6, 28, 1, 15, 3, 21, 30, 39, 
    5, 11, 35, 0, 36, 32, 20, 31, 33, 19, 9
  };

  for (int i = 0; i < n; i++) {
    EXPECT_EQ(dataset->get_feature_sort_index(i, 0), sort_inds_feature_0[i]);
    EXPECT_EQ(dataset->get_feature_sort_index(i, 1), sort_inds_feature_1[i]);
    EXPECT_EQ(dataset->get_feature_sort_index(i, 2), sort_inds_feature_2[i]);
  }

  // Check that leaf node statistics are computed correctly
  left_cutoff = 35;
  model->ResetSuffStat(left_suff_stat);
  node_suff_stat = model->ComputeNodeSuffStat(dataset.get(), 0, n, 0);
  model->AccumulateSplitRule(dataset.get(), left_suff_stat, split_feature, split_value, 0, n);
  right_suff_stat = model->SubtractSuffStat(node_suff_stat, left_suff_stat);
  EXPECT_EQ(left_suff_stat.sample_size_, left_cutoff);
  EXPECT_NEAR(left_suff_stat.outcome_sum_, 7413.644, 0.01);
  EXPECT_EQ(right_suff_stat.sample_size_, n-left_cutoff);
  EXPECT_NEAR(right_suff_stat.outcome_sum_, 1610.557, 0.01);

  // Check that tree leaf indices are updated correctly
  for (int i = 0; i < n; i++) {
    if (i < left_cutoff) {
      EXPECT_EQ(tree_observation_indices[tree_num][sort_inds_feature_0[i]], 1);
      EXPECT_EQ(tree_observation_indices[tree_num][sort_inds_feature_1[i]], 1);
      EXPECT_EQ(tree_observation_indices[tree_num][sort_inds_feature_2[i]], 1);
    } else {
      EXPECT_EQ(tree_observation_indices[tree_num][sort_inds_feature_0[i]], 2);
      EXPECT_EQ(tree_observation_indices[tree_num][sort_inds_feature_1[i]], 2);
      EXPECT_EQ(tree_observation_indices[tree_num][sort_inds_feature_2[i]], 2);
    }
  }

  /**************************************************************************/
  /*  Split 2: Leaf 1                                                       */
  /**************************************************************************/

  // Enumerate possible model cutpoints for leaf 1 in the second tree
  leaf_node = 1;
  leaf_begin = 0;
  leaf_end = 35;
  log_cutpoint_evaluations.clear();
  cutpoint_features.clear();
  cutpoint_values.clear();
  valid_cutpoint_count = 0;
  model->Cutpoints(dataset.get(), tree, leaf_node, leaf_begin, leaf_end, log_cutpoint_evaluations, cutpoint_features, cutpoint_values, valid_cutpoint_count);

  expected_logliks = {
    -4574.099, -4351.926, -4992.414, -5220.957, -5240.619, -5260.796, -5282.031, -5200.324, -5187.625, -5201.598, -5216.467, -5164.913, -5166.564, -5166.858, -5005.632, 
    -5093.035, -5202.761, -5128.528, -5127.525, -5124.115, -5113.670, -5219.656, -5291.616, -5286.455, -5274.657, -5274.931, -5241.354, -5280.036, -5282.332, -5292.315, 
    -5284.400, -5254.458, -5187.647, -5025.676, -4305.728, -4157.196, -4173.774, -4227.977, -4348.763, -4455.718, -4596.752, -4838.051, -5096.731, -5261.420, -5272.575, 
    -5284.863, -5285.172, -5285.704, -5287.318, -5289.287, -5292.940, -5293.514, -5256.379, -5260.162, -5257.919, -5241.880, -5220.519, -5190.837, -5147.645, -5065.821, 
    -4846.945, -4472.700, -4840.561, -5035.948, -5131.940, -5211.389, -5198.392, -5132.396, -5132.396, -5293.053, -5190.566, -5176.719, -5148.276, -5125.259, -5115.093, 
    -5127.017, -5131.423, -5141.190, -5109.209, -5120.990, -5078.271, -5203.364, -5188.662, -5200.476, -5193.256, -5190.613, -4990.517, -4999.033, -5007.824, -5098.064, 
    -5140.701, -5238.186, -5213.847, -5291.888, -5290.880, -5288.548, -5290.920, -5293.579, -5293.520, -5279.926, -5113.163, -4352.973, -4212.333
  };

  EXPECT_EQ(expected_logliks.size(), log_cutpoint_evaluations.size());
  for (int i = 0; i < log_cutpoint_evaluations.size(); i++) {
    EXPECT_NEAR(log_cutpoint_evaluations[i], expected_logliks[i], 0.01);
  }

  // Convert log marginal likelihood to marginal likelihood, normalizing by the largest value
  largest_mll = *std::max_element(log_cutpoint_evaluations.begin(), log_cutpoint_evaluations.end());
  cutpoint_evaluations.clear();
  cutpoint_evaluations.resize(log_cutpoint_evaluations.size());
  for (StochTree::data_size_t i = 0; i < log_cutpoint_evaluations.size(); i++){
    cutpoint_evaluations[i] = std::exp(log_cutpoint_evaluations[i] - largest_mll);
  }

  expected_likelihoods = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  };

  for (int i = 0; i < cutpoint_evaluations.size(); i++) {
    EXPECT_NEAR(cutpoint_evaluations[i], expected_likelihoods[i], 0.01);
  }

  // Split at the only nonzero likelihood
  split_chosen = 35;
  split_feature = cutpoint_features[split_chosen];
  split_value = cutpoint_values[split_chosen];
  EXPECT_EQ(split_feature, 1);
  EXPECT_NEAR(split_value, 0.01462726, 0.001);
  model->AddSplitToModel(dataset.get(), tree, leaf_node, leaf_begin, leaf_end, split_feature, split_value, 
                         split_queue, tree_observation_indices, tree_num);

  // Check that the training dataset was correctly partitioned
  sort_inds_feature_0.clear();
  sort_inds_feature_0 = {
    23, 32, 6, 29, 0, 22, 34, 18, 36, 24, 7, 37, 17, 12, 16, 14, 21, 
    30, 26, 11, 2, 1, 3, 5, 8, 10, 35, 39, 25, 28, 15, 4, 27, 13, 38
  };
  sort_inds_feature_1.clear();
  sort_inds_feature_1 = {
    23, 32, 21, 28, 37, 10, 34, 30, 14, 18, 24, 8, 26, 11, 2, 22, 38, 
    4, 36, 7, 16, 5, 15, 13, 27, 35, 0, 29, 3, 1, 39, 6, 25, 12, 17
  };
  sort_inds_feature_2.clear();
  sort_inds_feature_2 = {
    23, 32, 17, 29, 14, 38, 27, 13, 24, 12, 2, 26, 8, 10, 4, 34, 37, 
    22, 16, 18, 7, 25, 6, 28, 1, 15, 3, 21, 30, 39, 5, 11, 35, 0, 36
  };

  idx = 0;
  for (int i = leaf_begin; i < leaf_end; i++) {
    EXPECT_EQ(dataset->get_feature_sort_index(i, 0), sort_inds_feature_0[idx]);
    EXPECT_EQ(dataset->get_feature_sort_index(i, 1), sort_inds_feature_1[idx]);
    EXPECT_EQ(dataset->get_feature_sort_index(i, 2), sort_inds_feature_2[idx]);
    idx++;
  }

  // Check that leaf node statistics are computed correctly
  new_left_cutoff = 2;
  model->ResetSuffStat(left_suff_stat);
  node_suff_stat = model->ComputeNodeSuffStat(dataset.get(), leaf_begin, leaf_end, 0);
  model->AccumulateSplitRule(dataset.get(), left_suff_stat, split_feature, split_value, leaf_begin, leaf_end);
  right_suff_stat = model->SubtractSuffStat(node_suff_stat, left_suff_stat);
  EXPECT_EQ(node_suff_stat.sample_size_, leaf_end - leaf_begin);
  EXPECT_NEAR(node_suff_stat.outcome_sum_, 7413.644, 0.01);
  EXPECT_EQ(left_suff_stat.sample_size_, new_left_cutoff);
  EXPECT_NEAR(left_suff_stat.outcome_sum_, 284.9224, 0.01);
  EXPECT_EQ(right_suff_stat.sample_size_, leaf_end - leaf_begin - new_left_cutoff);
  EXPECT_NEAR(right_suff_stat.outcome_sum_, 7128.721, 0.01);

  // Check that tree leaf indices are updated correctly
  for (int i = 0; i < leaf_end - leaf_begin; i++) {
    if (i < new_left_cutoff) {
      EXPECT_EQ(tree_observation_indices[tree_num][sort_inds_feature_0[i]], 3);
      EXPECT_EQ(tree_observation_indices[tree_num][sort_inds_feature_1[i]], 3);
      EXPECT_EQ(tree_observation_indices[tree_num][sort_inds_feature_2[i]], 3);
    } else {
      EXPECT_EQ(tree_observation_indices[tree_num][sort_inds_feature_0[i]], 4);
      EXPECT_EQ(tree_observation_indices[tree_num][sort_inds_feature_1[i]], 4);
      EXPECT_EQ(tree_observation_indices[tree_num][sort_inds_feature_2[i]], 4);
    }
  }

  /**************************************************************************/
  /*  Split 3: Leaf 2                                                       */
  /**************************************************************************/

  // Enumerate possible model cutpoints for leaf 2 in the second tree
  leaf_node = 2;
  leaf_begin = 35;
  leaf_end = 40;
  log_cutpoint_evaluations.clear();
  cutpoint_features.clear();
  cutpoint_values.clear();
  valid_cutpoint_count = 0;
  model->Cutpoints(dataset.get(), tree, leaf_node, leaf_begin, leaf_end, log_cutpoint_evaluations, cutpoint_features, cutpoint_values, valid_cutpoint_count);

  expected_logliks = {
    -4720.215, -4815.328, -4253.065, -3967.524, -3967.524, -4253.065, 
    -4746.138, -4887.625, -4804.508, -4746.138, -4710.038, -3967.524, -2935.417
  };

  EXPECT_EQ(expected_logliks.size(), log_cutpoint_evaluations.size());
  for (int i = 0; i < log_cutpoint_evaluations.size(); i++) {
    EXPECT_NEAR(log_cutpoint_evaluations[i], expected_logliks[i], 0.01);
  }

  // Convert log marginal likelihood to marginal likelihood, normalizing by the largest value
  largest_mll = *std::max_element(log_cutpoint_evaluations.begin(), log_cutpoint_evaluations.end());
  cutpoint_evaluations.clear();
  cutpoint_evaluations.resize(log_cutpoint_evaluations.size());
  for (StochTree::data_size_t i = 0; i < log_cutpoint_evaluations.size(); i++){
    cutpoint_evaluations[i] = std::exp(log_cutpoint_evaluations[i] - largest_mll);
  }

  expected_likelihoods = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
  };

  for (int i = 0; i < cutpoint_evaluations.size(); i++) {
    EXPECT_NEAR(cutpoint_evaluations[i], expected_likelihoods[i], 0.01);
  }

  // Split at the only nonzero likelihood
  split_chosen = 12;
  split_feature = cutpoint_features[split_chosen];
  split_value = cutpoint_values[split_chosen];
  EXPECT_EQ(split_feature, -1);
  EXPECT_NEAR(split_value, std::numeric_limits<double>::max(), 0.001);
  // model->AddSplitToModel(dataset.get(), tree, leaf_node, leaf_begin, leaf_end, split_feature, split_value, 
  //                        split_queue, tree_observation_indices, tree_num);

  // Check that the training dataset was correctly partitioned
  sort_inds_feature_0.clear();
  sort_inds_feature_0 = {
    19, 31, 20, 33, 9
  };
  sort_inds_feature_1.clear();
  sort_inds_feature_1 = {
    9, 33, 19, 20, 31
  };
  sort_inds_feature_2.clear();
  sort_inds_feature_2 = {
    20, 31, 33, 19, 9
  };

  idx = 0;
  for (int i = leaf_begin; i < leaf_end; i++) {
    EXPECT_EQ(dataset->get_feature_sort_index(i, 0), sort_inds_feature_0[idx]);
    EXPECT_EQ(dataset->get_feature_sort_index(i, 1), sort_inds_feature_1[idx]);
    EXPECT_EQ(dataset->get_feature_sort_index(i, 2), sort_inds_feature_2[idx]);
    idx++;
  }

  // Check that tree leaf indices are still correct
  for (int i = 0; i < leaf_end - leaf_begin; i++) {
    EXPECT_EQ(tree_observation_indices[tree_num][sort_inds_feature_0[i]], 2);
    EXPECT_EQ(tree_observation_indices[tree_num][sort_inds_feature_1[i]], 2);
    EXPECT_EQ(tree_observation_indices[tree_num][sort_inds_feature_2[i]], 2);
  }

  /**************************************************************************/
  /*  Stop growing tree, set leaf parameters                                */
  /**************************************************************************/
  
  tree_leaves = tree->GetLeaves();
  leaf_suff_stats.clear();
  leaf_node_dist = std::normal_distribution<double>(0., 1.);
  // Populate a vector with sufficient statistics of each leaf
  for (int i = 0; i < tree_leaves.size(); i++) {
    leaf_suff_stats.push_back(model->LeafSuffStat(dataset.get(), tree_leaves[i]));
  }

  leaf_sample_sizes = {
    5, 2, 33
  };
  leaf_sum_resid = {
    1610.557, 284.9224, 7128.721
  };
  leaf_post_means = {
    292.8285, 113.9690, 212.7976
  };
  leaf_post_stddevs = {
    1.3483997, 2.0000000, 0.5463584
  };
  leaf_indices = {
    2, 3, 4
  };

  // Sample the mean parameter for each leaf
  for (int i = 0; i < tree_leaves.size(); i++) {
    // Compute mean and variance parameter
    node_mean = model->LeafPosteriorMean(leaf_suff_stats[i]);
    node_stddev = model->LeafPosteriorStddev(leaf_suff_stats[i]);

    EXPECT_EQ(leaf_suff_stats[i].sample_size_, leaf_sample_sizes[i]);
    EXPECT_NEAR(leaf_suff_stats[i].outcome_sum_, leaf_sum_resid[i], 0.01);
    EXPECT_EQ(tree_leaves[i], leaf_indices[i]);
    EXPECT_NEAR(node_mean, leaf_post_means[i], 0.01);
    EXPECT_NEAR(node_stddev, leaf_post_stddevs[i], 0.01);

    // Set leaf value to the posterior mean
    (*tree)[tree_leaves[i]].SetLeaf(node_mean);
  }

  // Update predictions
  expected_predictions = {
    212.7976, 212.7976, 212.7976, 212.7976, 212.7976, 212.7976, 
    212.7976, 212.7976, 212.7976, 292.8285, 212.7976, 212.7976, 
    212.7976, 212.7976, 212.7976, 212.7976, 212.7976, 212.7976, 
    212.7976, 292.8285, 292.8285, 212.7976, 212.7976, 113.969, 
    212.7976, 212.7976, 212.7976, 212.7976, 212.7976, 212.7976, 
    212.7976, 292.8285, 113.969, 292.8285, 212.7976, 212.7976, 
    212.7976, 212.7976, 212.7976, 212.7976
  };
  actual_predictions = tree->PredictFromNodes(tree_observation_indices[tree_num]);
  for (int i = 0; i < actual_predictions.size(); i++) {
    EXPECT_NEAR(expected_predictions[i], actual_predictions[i], 0.01);
  }

  // Update residuals
  dataset->ResidualSubtract(tree->PredictFromNodes(tree_observation_indices[tree_num]));
  expected_residuals = {
    55.448423, -71.6037, -3.315188, -99.40382, 15.916039, 7.865837, 
    -47.666883, -11.588455, 12.459409, -35.333669, -8.898055, -6.195101, 
    -5.351394, 10.701093, 70.378024, 8.856295, -5.108935, 28.140437, 
    89.688915, 45.535295, 62.537919, -42.952037, -1.546025, 26.452029, 
    4.840555, -12.425063, -6.877866, 13.607851, -30.541034, 64.49389, 
    37.554701, 90.716326, 30.532478, -17.042034, 4.218446, 24.233611, 
    65.987211, -13.928035, 8.426983, -49.017006
  };
  for (int i = 0; i < expected_residuals.size(); i++) {
    EXPECT_NEAR(expected_residuals[i], dataset->get_residual_value(i), 0.01);
  }

  // Retrieve pointer to first tree
  tree_num = 0;
  tree = (model_draw->GetEnsemble())->GetTree(tree_num);
  dataset->ResidualAdd(tree->PredictFromNodes(tree_observation_indices[tree_num]));
  expected_residuals = {
    341.76105, 439.26894, 102.03828, 411.46882, 121.26951, 294.17847, 463.20576, 
    274.72418, 35.79117, 475.53897, -135.54291, 99.15837, 505.52125, 297.01372, 
    -56.26683, 295.16893, 281.2037, 539.01308, -36.95594, 556.40794, 573.41056, 
    -169.59689, 103.80744, -100.19283, 28.17231, 498.44758, 98.4756, 299.92048, 
    -157.18589, 350.80652, -89.09016, 601.58897, -96.11238, 493.83061, -122.42641, 
    310.54624, 171.34068, -140.57289, 113.78045, 461.85563
  };
  for (int i = 0; i < expected_residuals.size(); i++) {
    EXPECT_NEAR(expected_residuals[i], dataset->get_residual_value(i), 0.01);
  }

  // Reset the dataset sort indices
  dataset->ResetToRaw();

  // Reset the node index map in the model
  model->NodeIndexMapReset(n);

}
