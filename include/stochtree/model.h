/*!
 * Copyright (c) 2023 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 * 
 * Inspired by the design of the Learner, GBTreeModel, and GBTree classes in xgboost, 
 * released under the Apache license with the following copyright:
 * 
 * Copyright 2015-2023 by XGBoost Contributors
 */
#ifndef STOCHTREE_MODEL_H_
#define STOCHTREE_MODEL_H_

#include <stochtree/config.h>
#include <stochtree/ensemble.h>
#include <stochtree/log.h>
#include <stochtree/train_data.h>
#include <stochtree/tree.h>
#include "../../dependencies/boost_math/include/boost/math/special_functions/gamma.hpp"

#include <cmath>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace StochTree {

class NodeSampleTracker;

class Model {
 public:
  Model() = default;
  Model(const Config& config) {}
  virtual ~Model() = default;
  virtual void InitializeGlobalParameters(TrainData* train_data) {}
  virtual void SampleTree(TrainData* train_data, Tree* tree, std::vector<std::vector<data_size_t>>& tree_observation_indices, int tree_num) {}
  virtual void SampleTree(TrainData* train_data, Tree* tree, NodeSampleTracker* node_tracker, std::vector<std::vector<data_size_t>>& tree_observation_indices, int tree_num) {}
  virtual void SampleLeafParameters(TrainData* train_data, Tree* tree) {}
  virtual void SampleGlobalParameters(TrainData* train_data, TreeEnsemble* tree_ensemble, std::set<std::string> update_params) {}
  virtual double GetGlobalParameter(std::string param_name) {}
  virtual void SetGlobalParameter(std::string param_name, double param_value) {}
};

struct XBARTGaussianRegressionSuffStat {
  data_size_t sample_size_;
  double outcome_sum_;
  double outcome_sum_sq_;
};

class XBARTGaussianRegressionModel : public Model {
 public:
  XBARTGaussianRegressionModel();
  XBARTGaussianRegressionModel(const Config& config);
  ~XBARTGaussianRegressionModel() {}

  /*! \brief Initialize hyperparameters for the global variance parameters (using the training data) */
  void InitializeGlobalParameters(TrainData* train_data);
  
  /*! \brief Sample a tree via the "grow from root" algorithm (He and Hahn 2021) */
  void SampleTree(TrainData* train_data, Tree* tree, std::vector<std::vector<data_size_t>>& tree_observation_indices, int tree_num);
  
  /*! \brief Sample leaf parameters, which are normal conditional on the data, tree structure, and variance parameters */
  void SampleLeafParameters(TrainData* train_data, Tree* tree);
  
  /*! \brief Compute sufficient statistics for node leaf_id */
  XBARTGaussianRegressionSuffStat LeafSuffStat(TrainData* train_data, node_t leaf_id);
  
  /*! \brief Sample residual variance and leaf node variance parameters, both IG */
  void SampleGlobalParameters(TrainData* train_data, TreeEnsemble* tree_ensemble, std::set<std::string> update_params);
  
  /*! \brief A way of calling the recursive "grow from root" procedure from a given 
   *  leaf node without actual recursive calls to a function. 
   *  The full grow-from-root procedure is initiated by calling `GrowFromLeaf(tree, 0);`
   */
  void SampleSplitRule(TrainData* train_data, Tree* tree, node_t leaf_node, 
                       data_size_t node_begin, data_size_t node_end, std::deque<node_t>& split_queue, 
                       std::vector<std::vector<data_size_t>>& tree_observation_indices, int tree_num);
  
  /*! \brief Compute the marginal log-likelihood of a proposed split, using sufficient statistics 
   *  accumulated for the potential left and right nodes */
  double SplitLogMarginalLikelihood(const XBARTGaussianRegressionSuffStat& left_stat, 
                                    const XBARTGaussianRegressionSuffStat& right_stat);

  /*! \brief Compute the marginal likelihood of a proposed split, using sufficient statistics 
   *  accumulated for the potential left and right nodes */
  double SplitMarginalLikelihood(const XBARTGaussianRegressionSuffStat& left_stat, 
                                 const XBARTGaussianRegressionSuffStat& right_stat);
  
  /*! \brief Compute the marginal log-likelihood of not splitting a node */
  double NoSplitLogMarginalLikelihood(const XBARTGaussianRegressionSuffStat& node_stat, 
                                      int node_depth, data_size_t num_split_candidates);
  
  /*! \brief Compute the marginal llikelihood of not splitting a node */
  double NoSplitMarginalLikelihood(const XBARTGaussianRegressionSuffStat& node_stat, 
                                   int node_depth, data_size_t num_split_candidates);
  
  /*! \brief Determine the number of possible cutpoints for a given node of a given tree */
  void Cutpoints(TrainData* train_data, Tree* tree, node_t leaf_node, 
                 data_size_t node_begin, data_size_t node_end, 
                 std::vector<double>& log_cutpoint_evaluations, 
                 std::vector<int>& cutpoint_feature, 
                 std::vector<double>& cutpoint_values, 
                 data_size_t& valid_cutpoint_count);
  
  /*! \brief Add a split to the model by growing the tree and modifying the training tracker */
  void AddSplitToModel(TrainData* train_data, Tree* tree, node_t leaf_node, 
                       data_size_t node_begin, data_size_t node_end, 
                       int feature_split, double split_value, std::deque<node_t>& split_queue, 
                       std::vector<std::vector<data_size_t>>& tree_observation_indices, int tree_num);

  /*! \brief Compute sufficient statistics for a given tree node */
  XBARTGaussianRegressionSuffStat ComputeNodeSuffStat(TrainData* train_data, data_size_t node_begin, 
                                                      data_size_t node_end, int feature_idx);

  /*! \brief Subtract the second sufficient statistic's entries from the first's sufficient statistics entries */
  XBARTGaussianRegressionSuffStat SubtractSuffStat(const XBARTGaussianRegressionSuffStat& first_node_suff_stat, 
                                                   const XBARTGaussianRegressionSuffStat& second_node_suff_stat);
  
  /*! \brief Accumulate data from a given (pre-sorted) row for a given column to a sufficient statistic */
  void AccumulateRowSuffStat(TrainData* train_data, XBARTGaussianRegressionSuffStat& suff_stat, 
                           data_size_t row, int col, data_size_t& node_row_iter);

  /*! \brief Accumulate sufficient statistics based on a proposed split to a node, 
   *  by default assumes we are accumulating statistics for the new left node, which 
   *  can be changed by is_left */
  void AccumulateSplitRule(TrainData* train_data, XBARTGaussianRegressionSuffStat& suff_stat, 
                         int split_col, double split_value, data_size_t node_begin, 
                         data_size_t node_end, bool is_left = true);

  /*! \brief Reset sufficient statistic to specific values (default 0) */
  void ResetSuffStat(XBARTGaussianRegressionSuffStat& suff_stat, data_size_t sample_size = 0, double outcome_sum = 0., double outcome_sum_sq = 0.);
  
  /*! \brief Add node to the node tracking table */
  inline void AddNode(node_t node_id, data_size_t node_begin, data_size_t node_end) {
    node_index_map_.insert({node_id, std::make_pair(node_begin, node_end)});
  }
  /*! \brief Compute the posterior mean of a leaf node, given the other model parameters and the samples that reach that node */
  inline double LeafPosteriorMean(XBARTGaussianRegressionSuffStat& leaf_suff_stat) {
    return leaf_suff_stat.outcome_sum_ / sigma_sq_ / ((1./tau_) + (leaf_suff_stat.sample_size_/sigma_sq_));
  }
  /*! \brief Compute the posterior standard deviation of a leaf node, given the other model parameters and the samples that reach that node */
  inline double LeafPosteriorStddev(XBARTGaussianRegressionSuffStat& leaf_suff_stat) {
    return std::sqrt(1. / ((1./tau_) + (leaf_suff_stat.sample_size_/sigma_sq_)));
  }
  /*! \brief Compute the posterior shape parameter for tau, the variance terms in the prior on leaf node parameters */
  inline double TauPosteriorShape(TreeEnsemble* tree_ensemble) {
    int num_leaves = 0;
    for (int i = 0; i < config_.num_trees; i++) {
      for (auto& leaf: tree_ensemble->GetTree(i)->GetLeaves()) {
        num_leaves += 1;
      }
    }
    return (a_tau_ + num_leaves)/2.;
  }
  /*! \brief Compute the posterior scale parameter for tau, the variance terms in the prior on leaf node parameters */
  inline double TauPosteriorScale(TreeEnsemble* tree_ensemble) {
    double sum_sq_leaf_vals = 0;
    for (int i = 0; i < config_.num_trees; i++) {
      for (auto& leaf: tree_ensemble->GetTree(i)->GetLeaves()) {
        sum_sq_leaf_vals += std::pow((*tree_ensemble->GetTree(i))[leaf].LeafValue(), 2.);
      }
    }
    return (b_tau_ + sum_sq_leaf_vals)/2.;
  }
  /*! \brief Compute the posterior shape parameter for sigma^2, the global residual error variance */
  inline double SigmaPosteriorShape(TrainData* train_data) {
    data_size_t n = train_data->num_data();
    return (a_sigma_ + n)/2.;
  }
  /*! \brief Compute the posterior scale parameter for sigma^2, the global residual error variance */
  inline double SigmaPosteriorScale(TrainData* train_data) {
    data_size_t n = train_data->num_data();
    double sum_sq_resid = 0.;
    for (data_size_t i = 0; i < n; i++) {
      sum_sq_resid += std::pow(train_data->get_residual_value(i), 2);
    }
    return (b_sigma_ + sum_sq_resid)/2.;
  }
  inline double GetGlobalParameter(std::string param_name) {
    if (param_name == "tau") {
      return tau_;
    } else if (param_name == "sigma_sq") {
      return sigma_sq_;
    } else if (param_name == "ybar_offset") {
      return ybar_offset_;
    } else if (param_name == "sd_scale") {
      return sd_scale_;
    }
  }
  inline void SetGlobalParameter(std::string param_name, double param_value) {
    if (param_name == "tau") {
      tau_ = param_value;
    } else if (param_name == "sigma_sq") {
      sigma_sq_ = param_value;
    } else if (param_name == "ybar_offset") {
      ybar_offset_ = param_value;
    } else if (param_name == "sd_scale") {
      sd_scale_ = param_value;
    }
  }
  inline void NodeIndexMapReset(data_size_t n) {
    node_index_map_.clear();
    AddNode(0, 0, n);
  }

 private:
  Config config_;
  double alpha_;
  double beta_;
  double a_sigma_;
  double a_tau_;
  double b_sigma_;
  double b_tau_;
  double sigma_sq_;
  double tau_;
  double ybar_offset_;
  double sd_scale_;
  std::unordered_map<node_t, std::pair<data_size_t, data_size_t>> node_index_map_;
  std::mt19937 gen;
  std::normal_distribution<double> leaf_node_dist_;
  std::gamma_distribution<double> residual_variance_dist_;
  std::gamma_distribution<double> leaf_node_variance_dist_;
};

struct BARTGaussianRegressionSuffStat {
  data_size_t sample_size_;
  double outcome_sum_;
  double outcome_sum_sq_;
};

/*! \brief Mapping nodes to the indices they contain */
class NodeSampleTracker {
 public:
  NodeSampleTracker(data_size_t n);

  /*! \brief Partition a node based on a new split rule */
  void PartitionNode(TrainData* train_data, int node_id, int left_node_id, int right_node_id, int feature_split, double split_value);

  /*! \brief Convert a (currently split) node to a leaf */
  void PruneNodeToLeaf(int node_id);

  /*! \brief Whether node_id is a leaf */
  bool IsLeaf(int node_id);

  /*! \brief Whether node_id's left child is a leaf */
  bool LeftNodeIsLeaf(int node_id);

  /*! \brief Whether node_id's right child is a leaf */
  bool RightNodeIsLeaf(int node_id);

  /*! \brief First index of data points contained in node_id */
  data_size_t NodeBegin(int node_id);

  /*! \brief One past the last index of data points contained in node_id */
  data_size_t NodeEnd(int node_id);

  /*! \brief Parent node_id */
  int Parent(int node_id);

  /*! \brief Left child of node_id */
  int LeftNode(int node_id);

  /*! \brief Right child of node_id */
  int RightNode(int node_id);

  std::vector<data_size_t> indices_;

 private:
  std::vector<data_size_t> node_begin_;
  std::vector<data_size_t> node_length_;
  std::vector<int32_t> parent_nodes_;
  std::vector<int32_t> left_nodes_;
  std::vector<int32_t> right_nodes_;
  int num_nodes_, num_deleted_nodes_;
  std::vector<int> deleted_nodes_;
  void ExpandNodeTrackingVectors(int node_id, int left_node_id, int right_node_id, data_size_t node_start_idx, data_size_t num_left, data_size_t num_right);
  void ConvertLeafParentToLeaf(int node_id);
};

class BARTGaussianRegressionModel : public Model {
 public:
  BARTGaussianRegressionModel();
  BARTGaussianRegressionModel(const Config& config);
  ~BARTGaussianRegressionModel() {}

  /*! \brief Initialize hyperparameters for the global variance parameters (using the training data) */
  void InitializeGlobalParameters(TrainData* train_data);
  
  /*! \brief Sample the tree */
  void SampleTree(TrainData* train_data, Tree* tree, NodeSampleTracker* node_tracker, std::vector<std::vector<data_size_t>>& tree_observation_indices, int tree_num);
  
  /*! \brief Perform one MCMC grow-prune step */
  void MCMCTreeStep(TrainData* train_data, Tree* tree, NodeSampleTracker* node_tracker);
  void GrowMCMC(TrainData* train_data, Tree* tree, NodeSampleTracker* node_tracker, bool& accept);
  void PruneMCMC(TrainData* train_data, Tree* tree, NodeSampleTracker* node_tracker, bool& accept);
  
  /*! \brief Compute sufficient statistics for node indexed by node_id in tree */
  BARTGaussianRegressionSuffStat ComputeNodeSufficientStatistics(TrainData* train_data, Tree* tree, NodeSampleTracker* node_tracker, int node_id);
  
  /*! \brief Sample leaf parameters, which are normal conditional on the data, tree structure, and variance parameters */
  void SampleLeafParameters(TrainData* train_data, Tree* tree);
  
  /*! \brief Sample residual variance and leaf node variance parameters, both IG */
  void SampleGlobalParameters(TrainData* train_data, TreeEnsemble* tree_ensemble, std::set<std::string> update_params);
  
  /*! \brief Compute the marginal log-likelihood of a proposed split, using sufficient statistics 
   *  accumulated for the potential left and right nodes */
  double SplitLogMarginalLikelihood(const BARTGaussianRegressionSuffStat& left_stat, 
                                    const BARTGaussianRegressionSuffStat& right_stat);
  
  /*! \brief Compute the marginal likelihood of a proposed split, using sufficient statistics 
   *  accumulated for the potential left and right nodes */
  double SplitMarginalLikelihood(const BARTGaussianRegressionSuffStat& left_stat, 
                                 const BARTGaussianRegressionSuffStat& right_stat);
  
  /*! \brief Compute the marginal log-likelihood of not splitting a node */
  double NoSplitLogMarginalLikelihood(const BARTGaussianRegressionSuffStat& node_stat);
  
  /*! \brief Compute the marginal likelihood of not splitting a node */
  double NoSplitMarginalLikelihood(const BARTGaussianRegressionSuffStat& node_stat);
  
  /*! \brief Add a split to the model by growing the tree and modifying the training tracker */
  void AddSplitToModel(TrainData* train_data, Tree* tree, NodeSampleTracker* node_tracker, node_t leaf_node, 
                       int feature_split, double split_value, BARTGaussianRegressionSuffStat& node_suff_stat, 
                       BARTGaussianRegressionSuffStat& left_suff_stat, BARTGaussianRegressionSuffStat& right_suff_stat);
  
  /*! \brief Remove a split from the model by growing the tree and modifying the training tracker */
  void RemoveSplitFromModel(TrainData* train_data, Tree* tree, NodeSampleTracker* node_tracker, node_t leaf_node, node_t left_node, 
                            node_t right_node, int feature_split, double split_value, BARTGaussianRegressionSuffStat& node_suff_stat, 
                            BARTGaussianRegressionSuffStat& left_suff_stat, BARTGaussianRegressionSuffStat& right_suff_stat);

  /*! \brief Compute sufficient statistic for a leaf */
  void LeafSuffStat(TrainData* train_data, Tree* tree, std::vector<std::vector<data_size_t>>& tree_observation_indices, 
                    int tree_num, int leaf_split, BARTGaussianRegressionSuffStat& node_suff_stat);
  
  /*! \brief Convert a split node to a leaf */
  void ConvertSplitToLeaf(TrainData* train_data, Tree* tree, node_t leaf_node, 
                          data_size_t node_begin, data_size_t node_end, 
                          int feature_split, double split_value, std::deque<node_t> split_queue, 
                          std::vector<std::vector<data_size_t>>& tree_observation_indices, int tree_num);
  
  /*! \brief Determine whether newly proposed left and right nodes have non-constant features (i.e. are able to split) */
  bool NodesNonConstantAfterSplit(TrainData* train_data, Tree* tree, NodeSampleTracker* node_tracker, 
                                  int leaf_split, int feature_split, double split_value);
  
  /*! \brief Determine whether a node has non-constant features */
  bool NodeNonConstant(TrainData* train_data, Tree* tree, NodeSampleTracker* node_tracker, int node_id);

  /*! \brief Compute sufficient statistics for a chosen leaf node and its new left and right nodes */
  void ComputeLeafNodeSuffStats(TrainData* train_data, Tree* tree, std::vector<std::vector<data_size_t>>& tree_observation_indices, 
                                int tree_num, int leaf_split, BARTGaussianRegressionSuffStat& node_suff_stat, 
                                BARTGaussianRegressionSuffStat& left_suff_stat, BARTGaussianRegressionSuffStat& right_suff_stat, 
                                int feature_split, double split_value);

  /*! \brief Sufficient statistics for the left and right nodes produces by a proposed split */
  void ComputeSplitSuffStats(TrainData* train_data, Tree* tree, NodeSampleTracker* node_tracker, 
                             int leaf_split, BARTGaussianRegressionSuffStat& left_suff_stat, 
                             BARTGaussianRegressionSuffStat& right_suff_stat, 
                             int feature_split, double split_value);
  
  /*! \brief Determine the minimum and maximum values available to split for a given feature in a given leaf */
  void VarSplitRange(TrainData* train_data, Tree* tree, NodeSampleTracker* node_tracker, 
                     int leaf_split, int feature_split, double& var_min, double& var_max);

  /*! \brief Reset sufficient statistic to specific values (default 0) */
  void ResetSuffStat(BARTGaussianRegressionSuffStat& suff_stat, data_size_t sample_size = 0, double outcome_sum = 0., double outcome_sum_sq = 0.);
  
  /*! \brief Compute the posterior mean of a leaf node, given the other model parameters and the samples that reach that node */
  inline double LeafPosteriorMean(BARTGaussianRegressionSuffStat& leaf_suff_stat) {
    data_size_t n_leaf = leaf_suff_stat.sample_size_;
    double y_bar = leaf_suff_stat.outcome_sum_ / ((double) n_leaf);
    double prior_contrib = (sigma_sq_/(sigma_sq_ + n_leaf*mu_sigma_))*mu_mean_;
    double data_contrib = ((n_leaf*mu_sigma_)/(sigma_sq_ + n_leaf*mu_sigma_))*y_bar;
    return prior_contrib + data_contrib;
  }
  /*! \brief Compute the posterior standard deviation of a leaf node, given the other model parameters and the samples that reach that node */
  inline double LeafPosteriorStddev(BARTGaussianRegressionSuffStat& leaf_suff_stat) {
    data_size_t n_leaf = leaf_suff_stat.sample_size_;
    return ((sigma_sq_*mu_sigma_)/(sigma_sq_ + n_leaf*mu_sigma_));
  }
  /*! \brief Compute the posterior shape parameter for sigma^2, the global residual error variance */
  inline double SigmaPosteriorShape(TrainData* train_data) {
    data_size_t n = train_data->num_data();
    return (nu_/2.0) + n;
  }
  /*! \brief Compute the posterior scale parameter for sigma^2, the global residual error variance */
  inline double SigmaPosteriorScale(TrainData* train_data) {
    data_size_t n = train_data->num_data();
    double nu_lambda_ = nu_*lambda_;
    double sum_sq_resid = 0.;
    for (data_size_t i = 0; i < n; i++) {
      sum_sq_resid += std::pow(train_data->get_residual_value(i), 2);
    }
    return (nu_lambda_/2.0) + sum_sq_resid;
  }
  inline double GetGlobalParameter(std::string param_name) {
    if (param_name == "sigma_sq") {
      return sigma_sq_;
    } else if (param_name == "ybar_offset") {
      return ybar_offset_;
    } else if (param_name == "sd_scale") {
      return sd_scale_;
    }
  }
  inline void SetGlobalParameter(std::string param_name, double param_value) {
    if (param_name == "sigma_sq") {
      sigma_sq_ = param_value;
    } else if (param_name == "ybar_offset") {
      ybar_offset_ = param_value;
    } else if (param_name == "sd_scale") {
      sd_scale_ = param_value;
    }
  }

 private:
  Config config_;
  double alpha_;
  double beta_;
  double nu_;
  double lambda_;
  double mu_mean_;
  double mu_sigma_;
  double sigma_sq_;
  double ybar_offset_;
  double sd_scale_;
  // std::vector<std::vector<data_size_t>>& tree_observation_indices_;
  std::vector<BARTGaussianRegressionSuffStat> node_suff_stats_;
  std::mt19937 gen;
  std::normal_distribution<double> leaf_node_dist_;
  std::gamma_distribution<double> residual_variance_dist_;
};

} // namespace StochTree

#endif // STOCHTREE_MODEL_H_
