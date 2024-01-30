/*!
 * Copyright (c) 2023 by randtree authors. 
 * 
 * Inspired by the C API of both lightgbm and xgboost, which carry the 
 * following respective copyrights:
 * 
 * LightGBM
 * ========
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 * 
 * xgboost
 * =======
 * Copyright 2015~2023 by XGBoost Contributors
 */
#ifndef STOCHTREE_DISPATCHER_H_
#define STOCHTREE_DISPATCHER_H_

#include <stochtree/model_draw.h>
#include <stochtree/outcome_model.h>
#include <stochtree/sampler.h>
#include <stochtree/tree_prior.h>
#include <stochtree/variance_model.h>

#include <Eigen/Dense>

#include <memory>

namespace StochTree {

class MCMCDispatcher {
 public:
  MCMCDispatcher(int num_samples, int num_burnin, int num_trees, int random_seed = -1);
  ~MCMCDispatcher();

  void Initialize();
  bool TrainDataConsistent();  
  bool PredictionDataConsistent();
  
  template <typename ModelType, typename TreePriorType>
  void SampleModel(double* covariate_data_ptr, int num_covariate, double* basis_data_ptr, int num_basis, double* outcome_data_ptr, int num_outcome, data_size_t num_row, bool is_row_major, bool non_constant_basis, double nu, double lambda, ModelType& model, TreePriorType& tree_prior, GlobalHomoskedasticVarianceModel& variance_model) {
    // Load the data
    LoadData(covariate_data_ptr, num_row, num_covariate, is_row_major, covariates_);
    LoadData(basis_data_ptr, num_row, num_basis, is_row_major, basis_);
    LoadData(outcome_data_ptr, num_row, num_outcome, is_row_major, outcome_);
    LoadData(outcome_data_ptr, num_row, num_outcome, is_row_major, residual_);

    // Check that a non-empty, consistently-sized dataset has been loaded
    CHECK(TrainDataConsistent());

    // Extract data dimensions
    num_observations_ = residual_.rows();
    num_covariates_ = covariates_.cols();
    num_basis_ = basis_.cols();
    has_basis_ = non_constant_basis;

    // Center and scale the residual
    double ybar_offset;
    double sd_scale;
    OutcomeCenterScale(residual_, ybar_offset, sd_scale);
    residual_ = residual_.array() - ybar_offset;
    residual_ /= sd_scale;

    // Compute the mean outcome for the model
    double mean_outcome = residual_.sum() / num_observations_;

    // Compute the implied leaf value initialization for each root node
    double initial_leaf_value;
    std::vector<double> initial_leaf_values;
    if (!has_basis_) {
      initial_leaf_value = mean_outcome / num_trees_;
    } else if (has_basis_ && (num_basis_ == 1)) {
      initial_leaf_value = (mean_outcome / num_trees_) / (basis_.array().sum());
    } else if (has_basis_ && (num_basis_ > 1)) {
      // TODO: find a heuristic initialization that yields mean_outcome as a prediction
      Eigen::MatrixXd leaf_reg_solution = (basis_.transpose() * basis_).inverse() * basis_.transpose() * residual_;
      initial_leaf_values.resize(num_basis_);
      for (int i = 0; i < num_basis_; i++) {
        initial_leaf_values[i] = leaf_reg_solution(i,0) / num_trees_;
      }
    }

    // Initialize the vector of vectors of leaf indices for each tree
    std::unique_ptr<SampleNodeMapper> sample_node_mapper = std::make_unique<SampleNodeMapper>(num_trees_, num_observations_);

    // Reset training data so that features are pre-sorted based on the entire dataset
    std::unique_ptr<UnsortedNodeSampleTracker> unsorted_node_sample_tracker = std::make_unique<UnsortedNodeSampleTracker>(num_observations_, num_trees_);

    // Placeholder declaration for unpacked prediction value
    double prediction_val;

    int model_iter = 0;
    int prev_model_iter = 0;
    for (int i = 0; i < num_samples_ + num_burnin_; i++) {
      // The way we handle "burn-in" samples is to write them to the first 
      // element of the model draw vector until we begin retaining samples.
      // Thus, the two conditions in which we reset an entry in the model 
      // draw vector are:
      //   1. The very first iteration of the sampler (i = 0)
      //   2. The model_iter variable tracking retained samples has advanced past 0
      if ((i == 0) || (model_iter > prev_model_iter)) {
        model_draws_[model_iter].reset(new ModelDraw(num_trees_, num_basis_, !has_basis_));
        model_draws_[model_iter]->SetGlobalParameters(ybar_offset, "ybar_offset");
        model_draws_[model_iter]->SetGlobalParameters(sd_scale, "sd_scale");
      }

      if (i == 0) {
        // Initialize the ensemble by setting all trees to a root node predicting mean(y) / num_trees
        for (int j = 0; j < num_trees_; j++) {
          Tree* tree = (model_draws_[model_iter]->GetEnsemble())->GetTree(j);
          if (num_basis_ == 1) {
            tree->SetLeaf(0, initial_leaf_value);
          } else {
            tree->SetLeafVector(0, initial_leaf_values);
          }
          sample_node_mapper->AssignAllSamplesToRoot(j);
        }

        // Subtract the predictions of the (constant) trees from the outcome to obtain initial residuals
        // train_data_->ResidualReset();
        for (int j = 0; j < num_trees_; j++) {
          Tree* tree = (model_draws_[model_iter]->GetEnsemble())->GetTree(j);
          for (data_size_t i = 0; i < num_observations_; i++) {
            if (!has_basis_) {
              prediction_val = tree->PredictFromNode(sample_node_mapper->GetNodeId(i, j));
            } else {
              prediction_val = tree->PredictFromNode(sample_node_mapper->GetNodeId(i, j), basis_, i);
            }
            // TODO: update to handle vector-valued residuals
            residual_(i,0) -= prediction_val;
          }
        }
      }

      // Sample the ensemble
      for (int j = 0; j < num_trees_; j++) {
        // Add the predictions from tree j in the previous sweep back to the residual
        // NOTE: in the first sweep, we treat each constant (ybar / num_trees) root tree 
        // as the result of the "previous sweep" which is why we use a special prev_model_iter
        // variable to track this
        // 
        // Similarly, we do not "store" any of the burnin draws, we just continue to overwrite 
        // draws in the first sweep, so we don't begin incrementing model_iter at an offset of 
        // 1 from prev_model_iter until burn-in is complete
        
        // Retrieve pointer to tree j from the previous draw of the model
        Tree* tree = (model_draws_[prev_model_iter]->GetEnsemble())->GetTree(j);

        // Add its prediction back to the residual to obtain a "partial" residual for fitting tree j
        for (data_size_t k = 0; k < num_observations_; k++) {
          if (!has_basis_) {
            prediction_val = tree->PredictFromNode(sample_node_mapper->GetNodeId(k, j));
          } else {
            prediction_val = tree->PredictFromNode(sample_node_mapper->GetNodeId(k, j), basis_, k);
          }
          // TODO: update to handle vector-valued residuals
          residual_(k, 0) += prediction_val;
        }

        // If model_iter is different from prev_model_iter, copy tree j from prev_model_iter to model_iter
        if (model_iter > prev_model_iter) {
          (model_draws_[model_iter]->GetEnsemble())->ResetTree(j);
          (model_draws_[model_iter]->GetEnsemble())->CloneFromExistingTree(j, tree);
        }
        
        // Retrieve pointer to tree j (which might be a new tree if we copied it)
        tree = (model_draws_[model_iter]->GetEnsemble())->GetTree(j);
        
        // Conduct one MCMC step of the grow/prune process
        BirthDeathMCMC<ModelType, TreePriorType>(covariates_, basis_, residual_, tree, unsorted_node_sample_tracker.get(), sample_node_mapper.get(), j, gen_, model, tree_prior);

        // Sample leaf node parameters
        SampleLeafParameters<ModelType, TreePriorType>(covariates_, basis_, residual_, tree, unsorted_node_sample_tracker.get(), sample_node_mapper.get(), j, gen_, model, tree_prior);
        
        // Subtract tree j's predictions back out of the residual
        for (data_size_t k = 0; k < num_observations_; k++) {
          if (!has_basis_) {
            prediction_val = tree->PredictFromNode(sample_node_mapper->GetNodeId(k, j));
          } else {
            prediction_val = tree->PredictFromNode(sample_node_mapper->GetNodeId(k, j), basis_, k);
          }
          // TODO: update to handle vector-valued residuals
          residual_(k, 0) -= prediction_val;
        }
      }

      // Sample sigma^2
      model.SetGlobalParameter(variance_model.SampleVarianceParameter(residual_, nu, lambda, gen_), GlobalParamName::GlobalVariance);
      
      // Determine whether to advance the model_iter variable
      if (i >= num_burnin_) {
        prev_model_iter = model_iter;
        model_iter += 1;
      }
    }
  }

  
  std::vector<double> PredictSamples(double* covariate_data_ptr, int num_covariate, double* basis_data_ptr, int num_basis, data_size_t num_row, bool is_row_major);
  std::vector<double> PredictSamples(double* covariate_data_ptr, int num_covariate, data_size_t num_row, bool is_row_major);

 private:
  /*! \brief Outcome, residual, covariates, and basis matrix for training */
  Eigen::MatrixXd outcome_;
  Eigen::MatrixXd residual_;
  Eigen::MatrixXd covariates_;
  Eigen::MatrixXd basis_;
  int num_observations_;
  int num_covariates_;
  int num_basis_;
  bool has_basis_{false};
  
  /*! \brief Covariates and basis matrix for prediction */
  Eigen::MatrixXd prediction_covariates_;
  Eigen::MatrixXd prediction_basis_;
  int num_pred_covariates_;
  int num_pred_basis_;

  /*! \brief Variance prior parameters */
  double nu_;
  double lambda_;

  /*! \brief Sampling parameters */
  int num_trees_;
  int num_samples_;
  int num_burnin_;
  int min_data_in_leaf_;
  double alpha_;
  double beta_;

  /*! \brief Random number generator */
  std::mt19937 gen_;

  /*! \brief Pointer to draws of the model */
  std::vector<std::unique_ptr<ModelDraw>> model_draws_;
  /*! \brief Compute ybar and scale parameter */
  void OutcomeCenterScale(Eigen::MatrixXd& outcome, double& ybar_offset, double& sd_scale);
  /*! \brief Load data from pointer to eigen matrix */
  void LoadData(double* data_ptr, int num_row, int num_col, bool is_row_major, Eigen::MatrixXd& data_matrix);
};

class GFRDispatcher {
 public:
  GFRDispatcher(int num_samples, int num_burnin, int num_trees, int random_seed = -1);
  ~GFRDispatcher();

  void Initialize();
  bool TrainDataConsistent();  
  bool PredictionDataConsistent();
  
  template <typename ModelType, typename TreePriorType>
  void SampleModel(double* covariate_data_ptr, int num_covariate, double* basis_data_ptr, int num_basis, double* outcome_data_ptr, int num_outcome, data_size_t num_row, bool is_row_major, bool non_constant_basis, double nu, double lambda, ModelType& model, TreePriorType& tree_prior, GlobalHomoskedasticVarianceModel& variance_model, std::vector<FeatureType>& feature_types, int cutpoint_grid_size) {
    // Load the data
    LoadData(covariate_data_ptr, num_row, num_covariate, is_row_major, covariates_);
    LoadData(basis_data_ptr, num_row, num_basis, is_row_major, basis_);
    LoadData(outcome_data_ptr, num_row, num_outcome, is_row_major, outcome_);
    LoadData(outcome_data_ptr, num_row, num_outcome, is_row_major, residual_);

    // Check that a non-empty, consistently-sized dataset has been loaded
    CHECK(TrainDataConsistent());

    // Extract data dimensions
    num_observations_ = residual_.rows();
    num_covariates_ = covariates_.cols();
    num_basis_ = basis_.cols();
    has_basis_ = non_constant_basis;

    // Center and scale the residual
    double ybar_offset;
    double sd_scale;
    OutcomeCenterScale(residual_, ybar_offset, sd_scale);
    residual_ = residual_.array() - ybar_offset;
    residual_ /= sd_scale;

    // Compute the mean outcome for the model
    double mean_outcome = residual_.sum() / num_observations_;

    // Compute the implied leaf value initialization for each root node
    double initial_leaf_value;
    std::vector<double> initial_leaf_values;
    if (!has_basis_) {
      initial_leaf_value = mean_outcome / num_trees_;
    } else if (has_basis_ && (num_basis_ == 1)) {
      initial_leaf_value = (mean_outcome / num_trees_) / (basis_.array().sum());
    } else if (has_basis_ && (num_basis_ > 1)) {
      // TODO: find a heuristic initialization that yields mean_outcome as a prediction
      Eigen::MatrixXd leaf_reg_solution = (basis_.transpose() * basis_).inverse() * basis_.transpose() * residual_;
      initial_leaf_values.resize(num_basis_);
      for (int i = 0; i < num_basis_; i++) {
        initial_leaf_values[i] = leaf_reg_solution(i,0) / num_trees_;
      }
    }

    // Initialize the vector of vectors of leaf indices for each tree
    std::unique_ptr<SampleNodeMapper> sample_node_mapper = std::make_unique<SampleNodeMapper>(num_trees_, num_observations_);

    // Initialize a FeaturePresortRootContainer unique pointer
    std::unique_ptr<FeaturePresortRootContainer> presort_container = std::make_unique<FeaturePresortRootContainer>(covariates_, feature_types);
    
    // Initialize a SortedNodeSampleTracker unique pointer (which will be reset with each sweep of the algorithm)
    std::unique_ptr<SortedNodeSampleTracker> sorted_node_sample_tracker = std::make_unique<SortedNodeSampleTracker>(presort_container.get(), covariates_, feature_types);

    // Placeholder declaration for unpacked prediction value
    double prediction_val;

    int model_iter = 0;
    int prev_model_iter = 0;
    for (int i = 0; i < num_samples_ + num_burnin_; i++) {
      // The way we handle "burn-in" samples is to write them to the first 
      // element of the model draw vector until we begin retaining samples.
      // Thus, the two conditions in which we reset an entry in the model 
      // draw vector are:
      //   1. The very first iteration of the sampler (i = 0)
      //   2. The model_iter variable tracking retained samples has advanced past 0
      if ((i == 0) || (model_iter > prev_model_iter)) {
        model_draws_[model_iter].reset(new ModelDraw(num_trees_, num_basis_, !has_basis_));
        model_draws_[model_iter]->SetGlobalParameters(ybar_offset, "ybar_offset");
        model_draws_[model_iter]->SetGlobalParameters(sd_scale, "sd_scale");
      }

      if (i == 0) {
        // Initialize the ensemble by setting all trees to a root node predicting mean(y) / num_trees
        for (int j = 0; j < num_trees_; j++) {
          Tree* tree = (model_draws_[model_iter]->GetEnsemble())->GetTree(j);
          if (num_basis_ == 1) {
            tree->SetLeaf(0, initial_leaf_value);
          } else {
            tree->SetLeafVector(0, initial_leaf_values);
          }
          sample_node_mapper->AssignAllSamplesToRoot(j);
        }

        // Subtract the predictions of the (constant) trees from the outcome to obtain initial residuals
        for (int j = 0; j < num_trees_; j++) {
          Tree* tree = (model_draws_[model_iter]->GetEnsemble())->GetTree(j);
          for (data_size_t i = 0; i < num_observations_; i++) {
            if (!has_basis_) {
              prediction_val = tree->PredictFromNode(sample_node_mapper->GetNodeId(i, j));
            } else {
              prediction_val = tree->PredictFromNode(sample_node_mapper->GetNodeId(i, j), basis_, i);
            }
            // TODO: update to handle vector-valued residuals
            residual_(i,0) -= prediction_val;
          }
        }
      }

      // Sample the ensemble
      for (int j = 0; j < num_trees_; j++) {
        // Add the predictions from tree j in the previous sweep back to the residual
        // NOTE: in the first sweep, we treat each constant (ybar / num_trees) root tree 
        // as the result of the "previous sweep" which is why we use a special prev_model_iter
        // variable to track this
        // 
        // Similarly, we do not "store" any of the burnin draws, we just continue to overwrite 
        // draws in the first sweep, so we don't begin incrementing model_iter at an offset of 
        // 1 from prev_model_iter until burn-in is complete
        
        // Retrieve pointer to tree j from the previous draw of the model
        Tree* tree = (model_draws_[prev_model_iter]->GetEnsemble())->GetTree(j);

        // Add its prediction back to the residual to obtain a "partial" residual for fitting tree j
        for (data_size_t k = 0; k < num_observations_; k++) {
          if (!has_basis_) {
            prediction_val = tree->PredictFromNode(sample_node_mapper->GetNodeId(k, j));
          } else {
            prediction_val = tree->PredictFromNode(sample_node_mapper->GetNodeId(k, j), basis_, k);
          }
          // TODO: update to handle vector-valued residuals
          residual_(k, 0) += prediction_val;
        }

        // Reset training data so that features are pre-sorted based on the entire dataset
        sorted_node_sample_tracker.reset(new SortedNodeSampleTracker(presort_container.get(), covariates_, feature_types));

        // Reset tree j to a constant root node
        (model_draws_[model_iter]->GetEnsemble())->ResetInitTree(j);

        // Reset the observation indices to point to node 0
        sample_node_mapper->AssignAllSamplesToRoot(j);
        
        // Retrieve pointer to the newly-reallocated tree j
        tree = (model_draws_[model_iter]->GetEnsemble())->GetTree(j);
        
        // Run the GFR algorithm
        TreeGrowFromRoot<ModelType, TreePriorType>(covariates_, basis_, residual_, tree, sorted_node_sample_tracker.get(), sample_node_mapper.get(), j, gen_, model, tree_prior, feature_types, cutpoint_grid_size);
        
        // Sample leaf node parameters
        SampleLeafParameters<ModelType, TreePriorType>(covariates_, basis_, residual_, tree, sorted_node_sample_tracker.get(), sample_node_mapper.get(), j, gen_, model, tree_prior);
        
        // Subtract tree j's predictions back out of the residual
        for (data_size_t k = 0; k < num_observations_; k++) {
          if (!has_basis_) {
            prediction_val = tree->PredictFromNode(sample_node_mapper->GetNodeId(k, j));
          } else {
            prediction_val = tree->PredictFromNode(sample_node_mapper->GetNodeId(k, j), basis_, k);
          }
          // TODO: update to handle vector-valued residuals
          residual_(k, 0) -= prediction_val;
        }
      }

      // Sample sigma^2
      model.SetGlobalParameter(variance_model.SampleVarianceParameter(residual_, nu, lambda, gen_), GlobalParamName::GlobalVariance);
      // TODO: figure out storage container for other global parameters
      
      // Determine whether to advance the model_iter variable
      if (i >= num_burnin_) {
        prev_model_iter = model_iter;
        model_iter += 1;
      }
    }
  }
  
  std::vector<double> PredictSamples(double* covariate_data_ptr, int num_covariate, double* basis_data_ptr, int num_basis, data_size_t num_row, bool is_row_major);
  std::vector<double> PredictSamples(double* covariate_data_ptr, int num_covariate, data_size_t num_row, bool is_row_major);

 private:
  /*! \brief Outcome, residual, covariates, and basis matrix for training */
  Eigen::MatrixXd outcome_;
  Eigen::MatrixXd residual_;
  Eigen::MatrixXd covariates_;
  Eigen::MatrixXd basis_;
  int num_observations_;
  int num_covariates_;
  int num_basis_;
  bool has_basis_{false};
  
  /*! \brief Covariates and basis matrix for prediction */
  Eigen::MatrixXd prediction_covariates_;
  Eigen::MatrixXd prediction_basis_;
  int num_pred_covariates_;
  int num_pred_basis_;

  /*! \brief Variance prior parameters */
  double nu_;
  double lambda_;

  /*! \brief Sampling parameters */
  int num_trees_;
  int num_samples_;
  int num_burnin_;
  int min_data_in_leaf_;
  double alpha_;
  double beta_;

  /*! \brief Random number generator */
  std::mt19937 gen_;

  /*! \brief Pointer to draws of the model */
  std::vector<std::unique_ptr<ModelDraw>> model_draws_;
  /*! \brief Compute ybar and scale parameter */
  void OutcomeCenterScale(Eigen::MatrixXd& outcome, double& ybar_offset, double& sd_scale);
  /*! \brief Load data from pointer to eigen matrix */
  void LoadData(double* data_ptr, int num_row, int num_col, bool is_row_major, Eigen::MatrixXd& data_matrix);
};

} // namespace StochTree

#endif  // STOCHTREE_DISPATCHER_H_
