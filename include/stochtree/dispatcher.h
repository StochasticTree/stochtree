/*!
 * Copyright (c) 2024 by randtree authors. 
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

#include <stochtree/data.h>
#include <stochtree/ensemble.h>
#include <stochtree/meta.h>
#include <stochtree/prior.h>
#include <stochtree/random_effects.h>
#include <stochtree/sampler.h>

#include <Eigen/Dense>

#include <memory>
#include <vector>

namespace StochTree {

class Dispatcher {
 public:
  Dispatcher() {
    num_forests_ = 0; num_rfx_ = 0; total_draws_ = 0; num_samples_ = 0; num_burnin_ = 0; 
    std::random_device rd; std::mt19937 gen_(rd());
  }
  Dispatcher(int random_seed) {
    num_forests_ = 0; num_rfx_ = 0; total_draws_ = 0; num_samples_ = 0; num_burnin_ = 0;
    std::mt19937 gen_(random_seed);
  }
  ~Dispatcher() {}
  
  // Adding model terms
  void AddOutcome(double* outcome_data_ptr, data_size_t num_row);
  void AddConstantLeafForest(double* covariate_data_ptr, int num_covariate, data_size_t num_row, bool is_row_major, int num_trees, double mu_bar, double tau, double alpha, double beta, int min_samples_in_leaf, ForestSampler forest_sampler, std::vector<FeatureType> feature_types, int cutpoint_grid_size, bool leaf_variance_random, double a_leaf = 1., double b_leaf = 1.);
  void AddUnivariateRegressionLeafForest(double* covariate_data_ptr, int num_covariate, double* basis_data_ptr, int num_basis, data_size_t num_row, bool is_row_major, int num_trees, double beta_bar, double tau, double alpha, double beta, int min_samples_in_leaf, ForestSampler forest_sampler, std::vector<FeatureType> feature_types, int cutpoint_grid_size, bool leaf_variance_random, double a_leaf = 1., double b_leaf = 1.);
  void AddMultivariateRegressionLeafForest(double* covariate_data_ptr, int num_covariate, double* basis_data_ptr, int num_basis, data_size_t num_row, bool is_row_major, int num_trees, Eigen::VectorXd& Beta, Eigen::MatrixXd& Sigma, double alpha, double beta, int min_samples_in_leaf, ForestSampler forest_sampler, std::vector<FeatureType> feature_types, int cutpoint_grid_size, bool leaf_variance_random);
  void AddRandomEffectRegression(double* basis_data_ptr, int num_basis, data_size_t num_row, bool is_row_major, std::vector<int32_t> group_indices, double a, double b, int num_components, int num_groups);
  void AddGlobalVarianceTerm(double a, double b, double global_variance_init);

  // Sampling model terms
  // void SampleModel(int num_samples, int num_burnin);
  template <typename ForestDatasetType, typename ForestLeafPriorType, typename ForestLeafSuffStatType, typename LeafSamplerType, typename ForestSamplerType, typename NodeSampleTrackerType>
  void SampleModel(int num_samples, int num_burnin) {
    // Unpack sample size
    num_burnin_ = num_burnin;
    num_samples_ = num_samples;
    total_draws_ = num_burnin + num_samples;

    // Resize forest containers
    for (int i = 0; i < num_forests_; i++) {
      forest_sample_containers_[i]->AddSamples(total_draws_);
      if (forest_leaf_variance_type_[i] == ForestLeafVarianceType::kStochastic) {
        forest_leaf_variance_sample_containers_[i].resize(total_draws_);
      }
    }
    // Resize rfx containers
    for (int i = 0; i < num_rfx_; i++) {
      rfx_sample_containers_[i]->AddSamples(total_draws_);
    }
    // Resize global variance container
    global_variance_sample_container_.resize(total_draws_);

    // Center and scale outcome
    CenterScaleOutcome();

    // Initialize forests with default values
    double mean_residual = MeanOutcome();
    int sample_num = 0;
    for (int i = 0; i < num_forests_; i++) {
      if (i == 0) {
        InitializeForest(i, sample_num, mean_residual);
      } else {
        InitializeForest(i, sample_num, 0.);
      }
    }
    
    // Initialize random effects sampler
    for (int i = 0; i < num_rfx_; i++) {
      rfx_samplers_[i]->InitializeParameters(rfx_datasets_[i].get(), residual_.get());
    }

    // Subtract prediction of each tree in each forest from the outcome to obtain the residual
    int n;
    for (int i = 0; i < num_forests_; i++) {
      ForestDatasetType* dataset = dynamic_cast<ForestDatasetType*>(forest_datasets_[i].get());
      n = dataset->covariates.rows();
      TreeEnsemble* ensemble = forest_sample_containers_[i]->GetEnsemble(sample_num);
      for (int j = 0; j < ensemble->NumTrees(); j++) {
        for (int k = 0; k < n; k++) {
          residual_->residual(k) -= TreePredictFromNodeSampleTracker<ForestDatasetType>(i, k, j, sample_num);
        }
      }
    }

    // We don't compute initial values of random effects, just assume 0 and don't residualize.
    // We will sample them in the first pass after sampling the forests.
    
    // Run the MCMC sampler
    double sigma_sq;
    for (int model_draw = 0; model_draw < total_draws_; model_draw++) {
      // Sample each forest
      for (int forest_num = 0; forest_num < num_forests_; forest_num++) {
        ForestDatasetType* dataset = dynamic_cast<ForestDatasetType*>(forest_datasets_[forest_num].get());
        ForestLeafPriorType* leaf_prior = dynamic_cast<ForestLeafPriorType*>(forest_leaf_priors_[forest_num].get());
        ForestSamplerType* tree_sampler = dynamic_cast<ForestSamplerType*>(forest_samplers_[forest_num].get());
        LeafSamplerType* leaf_sampler = dynamic_cast<LeafSamplerType*>(forest_leaf_mean_samplers_[forest_num].get());
        NodeSampleTrackerType* node_sample_tracker = tree_sampler->GetNodeSampleTracker();
        TreeEnsemble* ensemble = forest_sample_containers_[forest_num]->GetEnsemble(model_draw);
        int32_t cutpoint_grid_size = forest_cutpoint_grid_sizes_[forest_num];
        n = dataset->covariates.rows();
        for (int tree_num = 0; tree_num < ensemble->NumTrees(); tree_num++) {
          // Reset the tree and tracking info relevant to the sampler
          if (model_draw == 0) {
            sigma_sq = global_variance_init_;
          } else {
            sigma_sq = global_variance_sample_container_[model_draw - 1];
            tree_sampler->template Reset<ForestDatasetType, ForestLeafPriorType, ForestLeafSuffStatType>(forest_sample_containers_[forest_num].get(), dataset, forest_feature_types_[forest_num], tree_num, model_draw, model_draw - 1);
          }
          
          // Add tree's prediction back to the residual to obtain a "partial" residual for fitting tree j
          for (int k = 0; k < n; k++) {
            residual_->residual(k) += TreePredictFromNodeSampleTracker<ForestDatasetType>(forest_num, k, tree_num, model_draw);
          }
          
          // Obtain a pointer to the current tree
          Tree* tree = ensemble->GetTree(tree_num);

          // Sample the tree
          tree_sampler->template SampleTree<ForestDatasetType, ForestLeafPriorType, ForestLeafSuffStatType>(tree, residual_.get(), dataset, leaf_prior, forest_tree_priors_[forest_num].get(), sigma_sq, gen_, tree_num, cutpoint_grid_size, forest_feature_types_[forest_num]);
          
          // Obtain (potentially updated, in the case of GFR) pointer to node sample tracker
          node_sample_tracker = tree_sampler->GetNodeSampleTracker();
          
          // Sample the leaf node parameters
          leaf_sampler->SampleLeafParameters(leaf_prior, dataset, residual_.get(), tree, node_sample_tracker, tree_num, gen_, sigma_sq);
          
          // Subtract tree's prediction back out of the residual
          for (int k = 0; k < n; k++) {
            residual_->residual(k) -= TreePredictFromNodeSampleTracker<ForestDatasetType>(forest_num, k, tree_num, model_draw);
          }
        }
      }

      // Sample each random effect term
      for (int rfx_num = 0; rfx_num < num_rfx_; rfx_num++) {
        RegressionRandomEffectsDataset* rfx_dataset = rfx_datasets_[rfx_num].get();
        RandomEffectsRegressionGaussianPrior* rfx_prior = dynamic_cast<RandomEffectsRegressionGaussianPrior*>(rfx_priors_[rfx_num].get());
        RandomEffectsSampler* rfx_sampler = rfx_samplers_[rfx_num].get();
        
        // Add rfx predictions back to residual
        if (model_draw > 0) {
          residual_->residual += rfx_sampler->PredictRandomEffects(rfx_dataset->basis, rfx_dataset->group_indices);
        }
        
        // Sample the random effects
        rfx_sampler->SampleRandomEffects(rfx_prior, rfx_dataset, residual_.get(), gen_);

        // Subtract rfx predictions back out of residual
        residual_->residual -= rfx_sampler->PredictRandomEffects(rfx_dataset->basis, rfx_dataset->group_indices);

        // Store the random effects
        rfx_sample_containers_[rfx_num]->ResetSample(rfx_sampler, model_draw);
      }

      // Sample global variance
      global_variance_sample_container_[model_draw] = global_variance_sampler_->SampleVarianceParameter(residual_.get(), global_variance_prior_.get(), gen_);

      // Sample leaf node scale variances
      for (int forest_num = 0; forest_num < num_forests_; forest_num++) {
        bool leaf_scale_random = forest_leaf_variance_type_[forest_num] == ForestLeafVarianceType::kStochastic;
        if (leaf_scale_random) {
          LeafNodeHomoskedasticVarianceSampler* leaf_variance_sampler = dynamic_cast<LeafNodeHomoskedasticVarianceSampler*>(forest_leaf_variance_samplers_[forest_num].get());
          ForestLeafPriorType* leaf_prior = dynamic_cast<ForestLeafPriorType*>(forest_leaf_priors_[forest_num].get());
          TreeEnsemble* ensemble = forest_sample_containers_[forest_num]->GetEnsemble(model_draw);
          double forest_scale_update = leaf_variance_sampler->SampleVarianceParameter(ensemble, forest_leaf_variance_priors_[forest_num].get(), gen_);
          leaf_prior->SetPriorScale(forest_scale_update);
          forest_leaf_variance_sample_containers_[forest_num].push_back(forest_scale_update);
        }
      }
    }  
  }

  // Predicting model terms
  std::vector<double> PredictForest(int forest_num, double* covariate_data_ptr, int num_covariate, data_size_t num_row, bool is_row_major);
  std::vector<double> PredictForest(int forest_num, double* covariate_data_ptr, int num_covariate, double* basis_data_ptr, int num_basis, data_size_t num_row, bool is_row_major);
  std::vector<double> PredictRandomEffect(int rfx_num, double* basis_data_ptr, int num_basis, data_size_t num_row, bool is_row_major, std::vector<int32_t> group_indices);

 private:
  // Mean model components
  /*! \brief Forest storage, prior info, training data, and samplers */
  std::vector<std::unique_ptr<ForestDataset>> forest_datasets_;
  std::vector<std::unique_ptr<TreeEnsembleContainer>> forest_sample_containers_;
  std::vector<std::unique_ptr<LeafGaussianPrior>> forest_leaf_priors_;
  std::vector<std::unique_ptr<TreePrior>> forest_tree_priors_;
  std::vector<std::unique_ptr<IGVariancePrior>> forest_leaf_variance_priors_;
  std::vector<std::unique_ptr<TreeSampler>> forest_samplers_;
  std::vector<std::unique_ptr<LeafGaussianSampler>> forest_leaf_mean_samplers_;
  std::vector<std::unique_ptr<LeafNodeHomoskedasticVarianceSampler>> forest_leaf_variance_samplers_;
  std::vector<ForestLeafVarianceType> forest_leaf_variance_type_;
  std::vector<ForestLeafPriorType> forest_leaf_prior_type_;
  std::vector<ForestSampler> forest_sampler_type_;
  std::vector<std::vector<double>> forest_leaf_variance_sample_containers_;
  std::vector<std::vector<FeatureType>> forest_feature_types_;
  std::vector<ForestType> forest_types_;
  std::vector<int32_t> forest_cutpoint_grid_sizes_;
  int num_forests_;
  
  /*! \brief Random effects storage, prior info, training data, and samplers */
  std::vector<std::unique_ptr<RegressionRandomEffectsDataset>> rfx_datasets_;
  std::vector<std::unique_ptr<RandomEffectsContainer>> rfx_sample_containers_;
  std::vector<std::unique_ptr<RandomEffectsRegressionGaussianPrior>> rfx_priors_;
  std::vector<std::unique_ptr<RandomEffectsSampler>> rfx_samplers_;
  std::vector<RandomEffectsType> rfx_types_;
  int num_rfx_;
  
  // Variance model components
  /*! \brief Global outcome variance storage, prior info, and sampler */
  double global_variance_init_;
  std::unique_ptr<IGVariancePrior> global_variance_prior_;
  std::unique_ptr<GlobalHomoskedasticVarianceSampler> global_variance_sampler_;
  std::vector<double> global_variance_sample_container_;
  
  // Outcome
  /*! \brief Outcome for training */
  std::unique_ptr<UnivariateResidual> residual_;
  double outcome_offset_;
  double outcome_scale_;
  
  // Global state
  /*! \brief Random number generator */
  std::mt19937 gen_;
  int total_draws_;
  int num_samples_;
  int num_burnin_;

  // Private helper functions
  void CenterScaleOutcome();
  double MeanOutcome();
//  double TreePredictFromNode(int forest_num, int observation_num, int tree_num, int sample_num);
  void InitializeForest(int forest_num, int sample_num, double initial_forest_pred);
  void InitializeRandomEffects(int rfx_num, double initial_rfx_pred);
  
  template<typename ForestDatasetType>
  double TreePredictFromNodeSampleTracker(int forest_num, int observation_num, int tree_num, int sample_num) {
    ForestDatasetType* dataset = dynamic_cast<ForestDatasetType*>(forest_datasets_[forest_num].get());
    Tree* tree = forest_sample_containers_[forest_num]->GetEnsemble(sample_num)->GetTree(tree_num);
    data_size_t node_id = forest_samplers_[forest_num]->GetNodeId(observation_num, tree_num);
    if (std::is_same_v<RegressionLeafForestDataset*, ForestDatasetType*>){
      return tree->PredictFromNode(node_id, dataset->basis, observation_num);
    } else {
      return tree->PredictFromNode(node_id);
    }
  }
};

} // namespace StochTree

#endif  // STOCHTREE_DISPATCHER_H_
