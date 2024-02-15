/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include <stochtree/data.h>
#include <stochtree/ensemble.h>
#include <stochtree/prior.h>
#include <stochtree/sampler.h>

namespace StochTree {

// void MCMCTreeSampler::SampleTree(Tree* tree, UnivariateResidual* residual, ConstantLeafForestDataset* data, LeafConstantGaussianPrior* leaf_prior, TreePrior* tree_prior, double outcome_variance, std::mt19937& gen, int32_t tree_num) {

//   // Determine whether it is possible to grow any of the leaves
//   bool grow_possible = false;
//   std::vector<int> leaves = tree->GetLeaves();
//   for (auto& leaf: leaves) {
//     if (unsorted_node_sample_tracker_->NodeSize(tree_num, leaf) > 2 * tree_prior->GetMinSamplesLeaf()) {
//       grow_possible = true;
//       break;
//     }
//   }

//   // Determine whether it is possible to prune the tree
//   bool prune_possible = false;
//   if (tree->NumValidNodes() > 1) {
//     prune_possible = true;
//   }

//   // Determine the relative probability of grow vs prune (0 = grow, 1 = prune)
//   double prob_grow;
//   std::vector<double> step_probs(2);
//   if (grow_possible && prune_possible) {
//     step_probs = {0.5, 0.5};
//     prob_grow = 0.5;
//   } else if (!grow_possible && prune_possible) {
//     step_probs = {0.0, 1.0};
//     prob_grow = 0.0;
//   } else if (grow_possible && !prune_possible) {
//     step_probs = {1.0, 0.0};
//     prob_grow = 1.0;
//   } else {
//     Log::Fatal("In this tree, neither grow nor prune is possible");
//   }
//   std::discrete_distribution<> step_dist(step_probs.begin(), step_probs.end());

//   // Draw a split rule at random
//   data_size_t step_chosen = step_dist(gen);
//   bool accept;
  
//   if (step_chosen == 0) {
//     GrowMCMC(tree, residual, data, leaf_prior, tree_prior, outcome_variance, gen, tree_num, prob_grow);
//   } else {
//     PruneMCMC(tree, residual, data, leaf_prior, tree_prior, outcome_variance, gen, tree_num);
//   }
// }

// void MCMCTreeSampler::GrowMCMC(Tree* tree, UnivariateResidual* residual, ConstantLeafForestDataset* data, LeafConstantGaussianPrior* leaf_prior, TreePrior* tree_prior, double outcome_variance, std::mt19937& gen, int32_t tree_num, double prob_grow_old) {
//   // Extract dataset information
//   data_size_t n = data->covariates.rows();
//   int basis_dim = 1;

//   // Choose a leaf node at random
//   int num_leaves = tree->NumLeaves();
//   std::vector<int> leaves = tree->GetLeaves();
//   std::vector<double> leaf_weights(num_leaves);
//   std::fill(leaf_weights.begin(), leaf_weights.end(), 1.0/num_leaves);
//   std::discrete_distribution<> leaf_dist(leaf_weights.begin(), leaf_weights.end());
//   int leaf_chosen = leaves[leaf_dist(gen)];
//   int leaf_depth = tree->GetDepth(leaf_chosen);

//   // Select a split variable at random
//   int p = data->covariates.cols();
//   std::vector<double> var_weights(p);
//   std::fill(var_weights.begin(), var_weights.end(), 1.0/p);
//   std::discrete_distribution<> var_dist(var_weights.begin(), var_weights.end());
//   int var_chosen = var_dist(gen);

//   // Determine the range of possible cutpoints
//   double var_min, var_max;
//   VarSplitRange(data, tree, unsorted_node_sample_tracker_.get(), leaf_chosen, var_chosen, var_min, var_max, tree_num);
//   if (var_max <= var_min) {
//     return;
//   }
//   // Split based on var_min to var_max in a given node
//   std::uniform_real_distribution<double> split_point_dist(var_min, var_max);
//   double split_point_chosen = split_point_dist(gen);

//   // Compute the marginal likelihood of split and no split, given the leaf prior
//   std::tuple<double, double, int32_t, int32_t> split_eval = EvaluateSplitAndNoSplit(tree, residual, data, leaf_prior, tree_prior, outcome_variance, tree_num, leaf_chosen, var_chosen, split_point_chosen);
//   double split_log_marginal_likelihood = std::get<0>(split_eval);
//   double no_split_log_marginal_likelihood = std::get<1>(split_eval);
//   int32_t left_n = std::get<2>(split_eval);
//   int32_t right_n = std::get<3>(split_eval);
  
//   // Determine probability of growing the split node and its two new left and right nodes
//   double pg = tree_prior->GetAlpha() * std::pow(1+leaf_depth, -tree_prior->GetBeta());
//   double pgl = tree_prior->GetAlpha() * std::pow(1+leaf_depth+1, -tree_prior->GetBeta());
//   double pgr = tree_prior->GetAlpha() * std::pow(1+leaf_depth+1, -tree_prior->GetBeta());

//   // Determine whether a "grow" move is possible from the newly formed tree
//   // in order to compute the probability of choosing "prune" from the new tree
//   // (which is always possible by construction)
//   bool left_non_constant = NodesNonConstantAfterSplit(data, tree, leaf_chosen, var_chosen, split_point_chosen, tree_num, true);
//   bool right_non_constant = NodesNonConstantAfterSplit(data, tree, leaf_chosen, var_chosen, split_point_chosen, tree_num, false);
//   bool min_samples_left_check = left_n >= 2*tree_prior->GetMinSamplesLeaf();
//   bool min_samples_right_check = right_n >= 2*tree_prior->GetMinSamplesLeaf();
//   double prob_prune_new;
//   if ((left_non_constant && min_samples_left_check) || (right_non_constant && min_samples_right_check)) {
//     prob_prune_new = 0.5;
//   } else {
//     prob_prune_new = 1.0;
//   }

//   // Determine the number of leaves in the current tree and leaf parents in the proposed tree
//   int num_leaf_parents = tree->NumLeafParents();
//   double p_leaf = 1/num_leaves;
//   double p_leaf_parent = 1/(num_leaf_parents+1);

//   // Compute the final MH ratio
//   double log_mh_ratio = (
//     std::log(pg) + std::log(1-pgl) + std::log(1-pgr) - std::log(1-pg) + std::log(prob_prune_new) +
//     std::log(p_leaf_parent) - std::log(prob_grow_old) - std::log(p_leaf) + no_split_log_marginal_likelihood - split_log_marginal_likelihood
//   );
//   // Threshold at 0
//   if (log_mh_ratio > 1) {
//     log_mh_ratio = 1;
//   }

//   // Draw a uniform random variable and accept/reject the proposal on this basis
//   bool accept;
//   std::uniform_real_distribution<double> mh_accept(0.0, 1.0);
//   double log_acceptance_prob = std::log(mh_accept(gen));
//   if (log_acceptance_prob <= log_mh_ratio) {
//     accept = true;
//     AddSplitToModel(data, tree, unsorted_node_sample_tracker_.get(), sample_node_mapper_.get(), leaf_chosen, var_chosen, split_point_chosen, tree_num);
//   } else {
//     accept = false;
//   }
// }

// void MCMCTreeSampler::PruneMCMC(Tree* tree, UnivariateResidual* residual, ConstantLeafForestDataset* data, LeafConstantGaussianPrior* leaf_prior, TreePrior* tree_prior, double outcome_variance, std::mt19937& gen, int32_t tree_num) {
//   // Choose a "leaf parent" node at random
//   int num_leaves = tree->NumLeaves();
//   int num_leaf_parents = tree->NumLeafParents();
//   std::vector<int> leaf_parents = tree->GetLeafParents();
//   std::vector<double> leaf_parent_weights(num_leaf_parents);
//   std::fill(leaf_parent_weights.begin(), leaf_parent_weights.end(), 1.0/num_leaf_parents);
//   std::discrete_distribution<> leaf_parent_dist(leaf_parent_weights.begin(), leaf_parent_weights.end());
//   int leaf_parent_chosen = leaf_parents[leaf_parent_dist(gen)];
//   int leaf_parent_depth = tree->GetDepth(leaf_parent_chosen);
//   int left_node = tree->LeftChild(leaf_parent_chosen);
//   int right_node = tree->RightChild(leaf_parent_chosen);
//   int feature_split = tree->SplitIndex(leaf_parent_chosen);
//   double split_value = tree->Threshold(leaf_parent_chosen);

//   // Compute the marginal likelihood for the leaf parent and its left and right nodes
//   std::tuple<double, double, int32_t, int32_t> split_eval = EvaluateSplitAndNoSplit(tree, residual, data, leaf_prior, tree_prior, outcome_variance, tree_num, leaf_parent_chosen, left_node, right_node);
//   double split_log_marginal_likelihood = std::get<0>(split_eval);
//   double no_split_log_marginal_likelihood = std::get<1>(split_eval);
//   int32_t left_n = std::get<2>(split_eval);
//   int32_t right_n = std::get<3>(split_eval);
  
//   // Determine probability of growing the split node and its two new left and right nodes
//   double pg = tree_prior->GetAlpha() * std::pow(1+leaf_parent_depth, -tree_prior->GetBeta());
//   double pgl = tree_prior->GetAlpha() * std::pow(1+leaf_parent_depth+1, -tree_prior->GetBeta());
//   double pgr = tree_prior->GetAlpha() * std::pow(1+leaf_parent_depth+1, -tree_prior->GetBeta());

//   // Determine whether a "prune" move is possible from the new tree,
//   // in order to compute the probability of choosing "grow" from the new tree
//   // (which is always possible by construction)
//   bool non_root_tree = tree->NumNodes() > 1;
//   double prob_grow_new;
//   if (non_root_tree) {
//     prob_grow_new = 0.5;
//   } else {
//     prob_grow_new = 1.0;
//   }

//   // Determine whether a "grow" move was possible from the old tree,
//   // in order to compute the probability of choosing "prune" from the old tree
//   bool non_constant_left = NodeNonConstant(data, tree, unsorted_node_sample_tracker_.get(), left_node, tree_num);
//   bool non_constant_right = NodeNonConstant(data, tree, unsorted_node_sample_tracker_.get(), right_node, tree_num);
//   double prob_prune_old;
//   if (non_constant_left && non_constant_right) {
//     prob_prune_old = 0.5;
//   } else {
//     prob_prune_old = 1.0;
//   }

//   // Determine the number of leaves in the current tree and leaf parents in the proposed tree
//   double p_leaf = 1/(num_leaves-1);
//   double p_leaf_parent = 1/(num_leaf_parents);

//   // Compute the final MH ratio
//   double log_mh_ratio = (
//     std::log(1-pg) - std::log(pg) - std::log(1-pgl) - std::log(1-pgr) + std::log(prob_prune_old) +
//     std::log(p_leaf) - std::log(prob_grow_new) - std::log(p_leaf_parent) + no_split_log_marginal_likelihood - split_log_marginal_likelihood
//   );
//   // Threshold at 0
//   if (log_mh_ratio > 0) {
//     log_mh_ratio = 0;
//   }

//   // Draw a uniform random variable and accept/reject the proposal on this basis
//   bool accept;
//   std::uniform_real_distribution<double> mh_accept(0.0, 1.0);
//   double log_acceptance_prob = std::log(mh_accept(gen));
//   if (log_acceptance_prob <= log_mh_ratio) {
//     accept = true;
//     RemoveSplitFromModel(data, tree, unsorted_node_sample_tracker_.get(), sample_node_mapper_.get(), leaf_parent_chosen, left_node, right_node, feature_split, split_value, tree_num);
//   } else {
//     accept = false;
//   }
// }


// std::tuple<double, double, int32_t, int32_t> MCMCTreeSampler::EvaluateSplitAndNoSplit(Tree* tree, UnivariateResidual* residual, ConstantLeafForestDataset* data, LeafConstantGaussianPrior* leaf_prior, TreePrior* tree_prior, double outcome_variance, int32_t tree_num, int leaf_split, int feature_split, double split_value) {
//   // Unpack shifted iterators to observations in a given node
//   auto tree_node_tracker = unsorted_node_sample_tracker_->GetFeaturePartition(tree_num);
//   data_size_t node_begin = tree_node_tracker->NodeBegin(leaf_split);
//   data_size_t node_end = tree_node_tracker->NodeEnd(leaf_split);
//   data_size_t idx;
//   auto node_begin_iter = tree_node_tracker->indices_.begin() + node_begin;
//   auto node_end_iter = tree_node_tracker->indices_.begin() + node_end;

//   // Initialize all sufficient statistics
//   LeafConstantGaussianSuffStat root_suff_stat = LeafConstantGaussianSuffStat();
//   LeafConstantGaussianSuffStat left_suff_stat = LeafConstantGaussianSuffStat();
//   LeafConstantGaussianSuffStat right_suff_stat = LeafConstantGaussianSuffStat();
  
//   // Iterate through every observation in the node
//   double feature_value;
//   for (auto i = node_begin_iter; i != node_end_iter; i++) {
//     idx = *i;
//     feature_value = data->covariates(idx, feature_split);
//     // Increment sufficient statistics for the split node, regardless of covariate value
//     root_suff_stat.IncrementSuffStat(data->covariates, residual->residual, idx);
//     if (SplitTrueNumeric(feature_value, split_value)) {
//       // Increment new left node sufficient statistic if split is true
//       left_suff_stat.IncrementSuffStat(data->covariates, residual->residual, idx);
//     } else {
//       // Increment new left node sufficient statistic if split is false
//       right_suff_stat.IncrementSuffStat(data->covariates, residual->residual, idx);
//     }
//   }

//   double split_log_ml = SplitLogMarginalLikelihood(left_suff_stat, right_suff_stat, *leaf_prior, outcome_variance);
//   double no_split_log_ml = NoSplitLogMarginalLikelihood(root_suff_stat, *leaf_prior, outcome_variance);

//   return std::tuple<double, double, int, int>(split_log_ml, no_split_log_ml, left_suff_stat.n, right_suff_stat.n);
// }

// std::tuple<double, double, int32_t, int32_t> MCMCTreeSampler::EvaluateSplitAndNoSplit(Tree* tree, UnivariateResidual* residual, ConstantLeafForestDataset* data, LeafConstantGaussianPrior* leaf_prior, TreePrior* tree_prior, double outcome_variance, int32_t tree_num, int leaf_split, int feature_split, std::vector<std::uint32_t>& split_categories) {
//   // Unpack shifted iterators to observations in a given node
//   auto tree_node_tracker = unsorted_node_sample_tracker_->GetFeaturePartition(tree_num);
//   data_size_t node_begin = tree_node_tracker->NodeBegin(leaf_split);
//   data_size_t node_end = tree_node_tracker->NodeEnd(leaf_split);
//   data_size_t idx;
//   auto node_begin_iter = tree_node_tracker->indices_.begin() + node_begin;
//   auto node_end_iter = tree_node_tracker->indices_.begin() + node_end;

//   // Initialize all sufficient statistics
//   LeafConstantGaussianSuffStat root_suff_stat = LeafConstantGaussianSuffStat();
//   LeafConstantGaussianSuffStat left_suff_stat = LeafConstantGaussianSuffStat();
//   LeafConstantGaussianSuffStat right_suff_stat = LeafConstantGaussianSuffStat();
  
//   // Iterate through every observation in the node
//   double feature_value;
//   for (auto i = node_begin_iter; i != node_end_iter; i++) {
//     idx = *i;
//     feature_value = data->covariates(idx, feature_split);
//     // Increment sufficient statistics for the split node, regardless of covariate value
//     root_suff_stat.IncrementSuffStat(data->covariates, residual->residual, idx);
//     if (SplitTrueCategorical(feature_value, split_categories)) {
//       // Increment new left node sufficient statistic if split is true
//       left_suff_stat.IncrementSuffStat(data->covariates, residual->residual, idx);
//     } else {
//       // Increment new left node sufficient statistic if split is false
//       right_suff_stat.IncrementSuffStat(data->covariates, residual->residual, idx);
//     }
//   }

//   double split_log_ml = SplitLogMarginalLikelihood(left_suff_stat, right_suff_stat, *leaf_prior, outcome_variance);
//   double no_split_log_ml = NoSplitLogMarginalLikelihood(root_suff_stat, *leaf_prior, outcome_variance);

//   return std::tuple<double, double, int, int>(split_log_ml, no_split_log_ml, left_suff_stat.n, right_suff_stat.n);
// }

// bool MCMCTreeSampler::NodesNonConstantAfterSplit(ConstantLeafForestDataset* data, Tree* tree, int leaf_split, int feature_split, double split_value, int tree_num, bool left) {
//   int p = data->covariates.cols();
//   data_size_t idx;
//   double feature_value;
//   double split_feature_value;
//   double var_max_left;
//   double var_min_left;
//   double var_max_right;
//   double var_min_right;
//   auto tree_node_tracker = unsorted_node_sample_tracker_->GetFeaturePartition(tree_num);
//   data_size_t node_begin = tree_node_tracker->NodeBegin(leaf_split);
//   data_size_t node_end = tree_node_tracker->NodeEnd(leaf_split);

//   for (int j = 0; j < p; j++) {
//     var_max_left = std::numeric_limits<double>::min();
//     var_min_left = std::numeric_limits<double>::max();
//     var_max_right = std::numeric_limits<double>::min();
//     var_min_right = std::numeric_limits<double>::max();
//     auto node_begin_iter = tree_node_tracker->indices_.begin() + node_begin;
//     auto node_end_iter = tree_node_tracker->indices_.begin() + node_end;
//     for (auto i = node_begin_iter; i != node_end_iter; i++) {
//       idx = *i;
//       feature_value = data->covariates(idx, j);
//       split_feature_value = data->covariates(idx, feature_split);
//       if (SplitTrueNumeric(split_feature_value, split_value)) {
//         if (left) {
//           if (var_max_left < feature_value) {
//             var_max_left = feature_value;
//           } else if (var_min_left > feature_value) {
//             var_max_left = feature_value;
//           }
//         }
//       } else {
//         if (!left) {
//           if (var_max_right < feature_value) {
//             var_max_right = feature_value;
//           } else if (var_min_right > feature_value) {
//             var_max_right = feature_value;
//           }
//         }
//       }
//     }
//     if ((var_max_left > var_min_left) && (var_max_right > var_min_right)) {
//       return true;
//     }
//   }
//   return false;
// }

// bool MCMCTreeSampler::NodesNonConstantAfterSplit(ConstantLeafForestDataset* data, Tree* tree, int leaf_split, int feature_split, std::vector<std::uint32_t> split_categories, int tree_num, bool left) {
//   int p = data->covariates.cols();
//   data_size_t idx;
//   double feature_value;
//   double split_feature_value;
//   double var_max_left;
//   double var_min_left;
//   double var_max_right;
//   double var_min_right;
//   auto tree_node_tracker = unsorted_node_sample_tracker_->GetFeaturePartition(tree_num);
//   data_size_t node_begin = tree_node_tracker->NodeBegin(leaf_split);
//   data_size_t node_end = tree_node_tracker->NodeEnd(leaf_split);

//   for (int j = 0; j < p; j++) {
//     var_max_left = std::numeric_limits<double>::min();
//     var_min_left = std::numeric_limits<double>::max();
//     var_max_right = std::numeric_limits<double>::min();
//     var_min_right = std::numeric_limits<double>::max();
//     auto node_begin_iter = tree_node_tracker->indices_.begin() + node_begin;
//     auto node_end_iter = tree_node_tracker->indices_.begin() + node_end;
//     for (auto i = node_begin_iter; i != node_end_iter; i++) {
//       idx = *i;
//       feature_value = data->covariates(idx, j);
//       split_feature_value = data->covariates(idx, feature_split);
//       if (SplitTrueCategorical(split_feature_value, split_categories)) {
//         if (left) {
//           if (var_max_left < feature_value) {
//             var_max_left = feature_value;
//           } else if (var_min_left > feature_value) {
//             var_max_left = feature_value;
//           }
//         }
//       } else {
//         if (!left) {
//           if (var_max_right < feature_value) {
//             var_max_right = feature_value;
//           } else if (var_min_right > feature_value) {
//             var_max_right = feature_value;
//           }
//         }
//       }
//     }
//     if ((var_max_left > var_min_left) && (var_max_right > var_min_right)) {
//       return true;
//     }
//   }
//   return false;
// }

// std::tuple<double, double, int32_t, int32_t> MCMCTreeSampler::EvaluateSplitAndNoSplit(Tree* tree, UnivariateResidual* residual, ConstantLeafForestDataset* data, LeafConstantGaussianPrior* leaf_prior, TreePrior* tree_prior, double outcome_variance, int32_t tree_num, int parent_node, int left_node, int right_node) {
//   // Unpack shifted iterators to observations in a given node
//   auto tree_node_tracker = unsorted_node_sample_tracker_->GetFeaturePartition(tree_num);
//   data_size_t left_node_begin = tree_node_tracker->NodeBegin(left_node);
//   data_size_t left_node_end = tree_node_tracker->NodeEnd(left_node);
//   data_size_t right_node_begin = tree_node_tracker->NodeBegin(right_node);
//   data_size_t right_node_end = tree_node_tracker->NodeEnd(right_node);
//   data_size_t idx;
//   auto left_node_begin_iter = tree_node_tracker->indices_.begin() + left_node_begin;
//   auto left_node_end_iter = tree_node_tracker->indices_.begin() + left_node_end;
//   auto right_node_begin_iter = tree_node_tracker->indices_.begin() + right_node_begin;
//   auto right_node_end_iter = tree_node_tracker->indices_.begin() + right_node_end;

//   // Initialize all sufficient statistics
//   LeafConstantGaussianSuffStat root_suff_stat = LeafConstantGaussianSuffStat();
//   LeafConstantGaussianSuffStat left_suff_stat = LeafConstantGaussianSuffStat();
//   LeafConstantGaussianSuffStat right_suff_stat = LeafConstantGaussianSuffStat();
  
//   double feature_value;
//   // Update left node sufficient statistics
//   for (auto i = left_node_begin_iter; i != left_node_end_iter; i++) {
//     idx = *i;
//     // Increment sufficient statistics for the parent and left nodes
//     root_suff_stat.IncrementSuffStat(data->covariates, residual->residual, idx);
//     left_suff_stat.IncrementSuffStat(data->covariates, residual->residual, idx);
//   }
//   // Update right node sufficient statistics
//   for (auto i = right_node_begin_iter; i != right_node_end_iter; i++) {
//     idx = *i;
//     // Increment sufficient statistics for the parent and left nodes
//     root_suff_stat.IncrementSuffStat(data->covariates, residual->residual, idx);
//     right_suff_stat.IncrementSuffStat(data->covariates, residual->residual, idx);
//   }

//   double split_log_ml = SplitLogMarginalLikelihood(left_suff_stat, right_suff_stat, *leaf_prior, outcome_variance);
//   double no_split_log_ml = NoSplitLogMarginalLikelihood(root_suff_stat, *leaf_prior, outcome_variance);

//   return std::tuple<double, double, int, int>(split_log_ml, no_split_log_ml, left_suff_stat.n, right_suff_stat.n);
// }

void MCMCTreeSampler::AssignAllSamplesToRoot(int tree_num) {
  sample_node_mapper_->AssignAllSamplesToRoot(tree_num);
}

// void GFRTreeSampler::AssignAllSamplesToRoot(int tree_num) {
//   sample_node_mapper_->AssignAllSamplesToRoot(tree_num);
// }

data_size_t MCMCTreeSampler::GetNodeId(int observation_num, int tree_num) {
  return sample_node_mapper_->GetNodeId(observation_num, tree_num);
}

// data_size_t GFRTreeSampler::GetNodeId(int observation_num, int tree_num) {
//   return sample_node_mapper_->GetNodeId(observation_num, tree_num);
// }

// void MCMCTreeSampler::Reset(TreeEnsembleContainer* container, ConstantLeafForestDataset* data, std::vector<FeatureType>& feature_types, int tree_num, int sample_num, int prev_sample_num) {
//   Tree* prev_tree = container->GetEnsemble(prev_sample_num)->GetTree(tree_num);
//   container->GetEnsemble(sample_num)->ResetTree(tree_num);
//   container->GetEnsemble(sample_num)->CloneFromExistingTree(tree_num, prev_tree);
// }
// void MCMCTreeSampler::Reset(TreeEnsembleContainer* container, RegressionLeafForestDataset* data, std::vector<FeatureType>& feature_types, int tree_num, int sample_num, int prev_sample_num) {
//   Tree* prev_tree = container->GetEnsemble(prev_sample_num)->GetTree(tree_num);
//   container->GetEnsemble(sample_num)->ResetTree(tree_num);
//   container->GetEnsemble(sample_num)->CloneFromExistingTree(tree_num, prev_tree);
// }

// void GFRTreeSampler::Reset(TreeEnsembleContainer* container, ConstantLeafForestDataset* data, std::vector<FeatureType>& feature_types, int tree_num, int sample_num, int prev_sample_num) {
//   container->GetEnsemble(sample_num)->ResetInitTree(tree_num);
//   sorted_node_sample_tracker_.reset(new SortedNodeSampleTracker(presort_container_.get(), data->covariates, feature_types));
//   sample_node_mapper_->AssignAllSamplesToRoot(tree_num);
// }
// void GFRTreeSampler::Reset(TreeEnsembleContainer* container, RegressionLeafForestDataset* data, std::vector<FeatureType>& feature_types, int tree_num, int sample_num, int prev_sample_num) {
//   container->GetEnsemble(sample_num)->ResetInitTree(tree_num);
//   sorted_node_sample_tracker_.reset(new SortedNodeSampleTracker(presort_container_.get(), data->covariates, feature_types));
//   sample_node_mapper_->AssignAllSamplesToRoot(tree_num);
// }

// void GFRTreeSampler::Initialize(ConstantLeafForestDataset* data, int num_trees, int num_observations, std::vector<FeatureType>& feature_types) {
//   sample_node_mapper_ = std::make_unique<SampleNodeMapper>(num_trees, num_observations);
//   presort_container_ = std::make_unique<FeaturePresortRootContainer>(data->covariates, feature_types);
//   sorted_node_sample_tracker_ = std::make_unique<SortedNodeSampleTracker>(presort_container_.get(), data->covariates, feature_types);
// }

// void GFRTreeSampler::Initialize(RegressionLeafForestDataset* data, int num_trees, int num_observations, std::vector<FeatureType>& feature_types) {
//   sample_node_mapper_ = std::make_unique<SampleNodeMapper>(num_trees, num_observations);
//   presort_container_ = std::make_unique<FeaturePresortRootContainer>(data->covariates, feature_types);
//   sorted_node_sample_tracker_ = std::make_unique<SortedNodeSampleTracker>(presort_container_.get(), data->covariates, feature_types);
// }

}  // namespace StochTree
