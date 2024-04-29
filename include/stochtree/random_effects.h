/*!
 * Copyright (c) 2024 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_RANDOM_EFFECTS_H_
#define STOCHTREE_RANDOM_EFFECTS_H_

#include <stochtree/category_tracker.h>
#include <stochtree/cutpoint_candidates.h>
#include <stochtree/data.h>
#include <stochtree/ensemble.h>
#include <stochtree/ig_sampler.h>
#include <stochtree/log.h>
#include <stochtree/normal_sampler.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/prior.h>
#include <Eigen/Dense>

#include <cmath>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

namespace StochTree {

/*! \brief Wrapper around data structures for random effects sampling algorithms */
class RandomEffectsTracker {
 public:
  RandomEffectsTracker(std::vector<int32_t>& group_indices);
  ~RandomEffectsTracker() {}
  inline data_size_t GetCategoryId(int observation_num) {return sample_category_mapper_->GetCategoryId(observation_num);}
  inline data_size_t CategoryBegin(int category_id) {return category_sample_tracker_->CategoryBegin(category_id);}
  inline data_size_t CategoryEnd(int category_id) {return category_sample_tracker_->CategoryEnd(category_id);}
  inline data_size_t CategorySize(int category_id) {return category_sample_tracker_->CategorySize(category_id);}
  inline int32_t NumCategories() {return num_categories_;}
  SampleCategoryMapper* GetSampleCategoryMapper() {return sample_category_mapper_.get();}
  CategorySampleTracker* GetCategorySampleTracker() {return category_sample_tracker_.get();}
  std::vector<data_size_t>::iterator UnsortedNodeBeginIterator(int category_id);
  std::vector<data_size_t>::iterator UnsortedNodeEndIterator(int category_id);
  std::map<int32_t, int32_t>& GetLabelMap() {return category_sample_tracker_->GetLabelMap();}
  std::vector<data_size_t>& NodeIndices(int category_id) {return category_sample_tracker_->NodeIndices(category_id);}

 private:
  /*! \brief Mapper from observations to category indices */
  std::unique_ptr<SampleCategoryMapper> sample_category_mapper_;
  /*! \brief Data structure tracking / updating observations available in each category in a dataset */
  std::unique_ptr<CategorySampleTracker> category_sample_tracker_;
  /*! \brief Some high-level details of the random effects structure */
  int num_categories_;
};

/*! \brief Forward declaration */
class RandomEffectsTerm;

/*! \brief Posterior computation and sampling for random effect model with a group-level multivariate basis regression */
class MultivariateRegressionRandomEffectsModel {
 public:
  MultivariateRegressionRandomEffectsModel() {
    normal_sampler_ = MultivariateNormalSampler(); 
    ig_sampler_ = InverseGammaSampler();
  }
  ~MultivariateRegressionRandomEffectsModel() {}
  void SampleRandomEffects(RandomEffectsTerm* rfx, RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, double global_variance, std::mt19937& gen);
  void SampleWorkingParameter(RandomEffectsTerm* rfx, RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, double global_variance, std::mt19937& gen);
  void SampleGroupParameters(RandomEffectsTerm* rfx, RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, double global_variance, std::mt19937& gen);
  void SampleVarianceComponents(RandomEffectsTerm* rfx, RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, double global_variance, std::mt19937& gen);
 private:
  /*! \brief Compute the posterior mean of the working parameter, conditional on the group parameters and the variance components */
  Eigen::VectorXd WorkingParameterMean(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, RandomEffectsTerm& rfx_term, double global_variance);
  /*! \brief Compute the posterior covariance of the working parameter, conditional on the group parameters and the variance components */
  Eigen::MatrixXd WorkingParameterVariance(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, RandomEffectsTerm& rfx_term, double global_variance);
  /*! \brief Compute the posterior mean of a group parameter, conditional on the working parameter and the variance components */
  Eigen::VectorXd GroupParameterMean(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, RandomEffectsTerm& rfx_term, double global_variance, int32_t group_id);
  /*! \brief Compute the posterior covariance of a group parameter, conditional on the working parameter and the variance components */
  Eigen::MatrixXd GroupParameterVariance(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, RandomEffectsTerm& rfx_term, double global_variance, int32_t group_id);
  /*! \brief Compute the posterior shape of the group variance component, conditional on the working and group parameters */
  double VarianceComponentShape(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, RandomEffectsTerm& rfx_term, double global_variance, int32_t component_id);
  /*! \brief Compute the posterior scale of the group variance component, conditional on the working and group parameters */
  double VarianceComponentScale(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, RandomEffectsTerm& rfx_term, double global_variance, int32_t component_id);
  
  /*! \brief Samplers */
  MultivariateNormalSampler normal_sampler_;
  InverseGammaSampler ig_sampler_;
};

class RandomEffectsTerm {
 public: 
  RandomEffectsTerm(RandomEffectsDataset& rfx_dataset, RandomEffectsTracker& rfx_tracker) {
    num_components_ = rfx_dataset.GetBasis().cols();
    num_groups_ = rfx_tracker.NumCategories();
    label_map_ = rfx_tracker.GetLabelMap();
  }
  RandomEffectsTerm(RandomEffectsTerm& rfx_term) {
    num_components_ = rfx_term.num_components_;
    num_groups_ = rfx_term.num_groups_;
    label_map_ = rfx_term.label_map_;
    working_parameter_ = rfx_term.working_parameter_;
    group_parameters_ = rfx_term.group_parameters_;
    group_parameter_covariance_ = rfx_term.group_parameter_covariance_;
    working_parameter_covariance_ = rfx_term.working_parameter_covariance_;
    variance_prior_shape_ = rfx_term.variance_prior_shape_;
    variance_prior_scale_ = rfx_term.variance_prior_scale_;
  }
  ~RandomEffectsTerm() {}

  /*! \brief Setters */
  void SetWorkingParameter(Eigen::VectorXd& working_parameter) {
    working_parameter_ = working_parameter;
  }
  void SetGroupParameters(Eigen::MatrixXd& group_parameters) {
    group_parameters_ = group_parameters;
  }
  void SetGroupParameter(Eigen::VectorXd& group_parameter, int32_t group_id) {
    group_parameters_(Eigen::all, group_id) = group_parameter;
  }
  void SetWorkingParameterCovariance(Eigen::MatrixXd& working_parameter_covariance) {
    working_parameter_covariance_ = working_parameter_covariance;
  }
  void SetGroupParameterCovariance(Eigen::MatrixXd& group_parameter_covariance) {
    group_parameter_covariance_ = group_parameter_covariance;
  }
  void SetGroupParameterVarianceComponent(double value, int32_t component_id) {
    group_parameter_covariance_(component_id, component_id) = value;
  }
  void SetVariancePriorShape(double value) {
    variance_prior_shape_ = value;
  }
  void SetVariancePriorScale(double value) {
    variance_prior_scale_ = value;
  }

  /*! \brief Getters */
  Eigen::VectorXd& GetWorkingParameter() {
    return working_parameter_;
  }
  Eigen::MatrixXd& GetGroupParameters() {
    return group_parameters_;
  }
  Eigen::MatrixXd& GetWorkingParameterCovariance() {
    return working_parameter_covariance_;
  }
  Eigen::MatrixXd& GetGroupParameterCovariance() {
    return group_parameter_covariance_;
  }
  double GetVariancePriorShape() {
    return variance_prior_shape_;
  }
  double GetVariancePriorScale() {
    return variance_prior_scale_;
  }
  inline int32_t NumComponents() {return num_components_;}
  inline int32_t NumGroups() {return num_groups_;}
  
  std::vector<double> Predict(RandomEffectsDataset& rfx_dataset) {
    MatrixMap X = rfx_dataset.GetBasis();
    std::vector<int32_t> group_labels = rfx_dataset.GetGroupLabels();
    CHECK_EQ(X.rows(), group_labels.size());
    int n = X.rows();
    std::vector<double> output(n);
    Eigen::MatrixXd alpha_diag = working_parameter_.asDiagonal().toDenseMatrix();
    std::int32_t group_ind;
    for (int i = 0; i < n; i++) {
      group_ind = label_map_[group_labels[i]];
      output[i] = X(i, Eigen::all) * alpha_diag * group_parameters_(Eigen::all, group_ind);
    }
    return output;
  }

 private:
  /*! \brief Random effects structure details */
  bool default_rfx_;
  int num_components_;
  int num_groups_;
  std::map<int32_t, int32_t> label_map_;
  
  /*! \brief Group mean parameters, decomposed into "working parameter" and individual parameters
   *  under the "redundant" parameterization of Gelman et al (2008)
   */
  Eigen::VectorXd working_parameter_;
  Eigen::MatrixXd group_parameters_;

  /*! \brief Variance components for the group parameters */
  Eigen::MatrixXd group_parameter_covariance_;
  
  /*! \brief Variance components for the working parameter */
  Eigen::MatrixXd working_parameter_covariance_;

  /*! \brief Prior parameters */
  double variance_prior_shape_;
  double variance_prior_scale_;
};

class RandomEffectsContainer {
 public:
  RandomEffectsContainer() {
    rfx_ = std::vector<std::unique_ptr<RandomEffectsTerm>>(0);
    num_samples_ = 0;
  }
  RandomEffectsContainer(int num_samples) {
    rfx_ = std::vector<std::unique_ptr<RandomEffectsTerm>>(num_samples);
    num_samples_ = num_samples;
  }
  ~RandomEffectsContainer() {}

  void AddSamples(RandomEffectsDataset& rfx_dataset, RandomEffectsTracker& rfx_tracker, 
                  Eigen::VectorXd& working_parameter, Eigen::MatrixXd& group_parameters, 
                  Eigen::MatrixXd& working_parameter_covariance, Eigen::MatrixXd& group_parameter_covariance, 
                  double group_parameter_variance_prior_shape, double group_parameter_variance_prior_scale, 
                  int num_new_samples) {
    int total_new_samples = num_new_samples + num_samples_;
    rfx_.resize(total_new_samples);
    
    if (num_samples_ == 0) {
      // Initialize random effects terms from scratch
      for (int i = num_samples_; i < total_new_samples; i++) {
        rfx_[i].reset(new RandomEffectsTerm(rfx_dataset, rfx_tracker));
        rfx_[i]->SetWorkingParameter(working_parameter);
        rfx_[i]->SetGroupParameters(group_parameters);
        rfx_[i]->SetWorkingParameterCovariance(working_parameter_covariance);
        rfx_[i]->SetGroupParameterCovariance(group_parameter_covariance);
        rfx_[i]->SetVariancePriorShape(group_parameter_variance_prior_shape);
        rfx_[i]->SetVariancePriorScale(group_parameter_variance_prior_scale);
      }
    } else {
      for (int i = num_samples_; i < total_new_samples; i++) {
        rfx_[i].reset(new RandomEffectsTerm(*(rfx_[i-1].get())));
      }
    }   
    num_samples_ = total_new_samples;
  }

  inline int32_t NumSamples() {
    return num_samples_;
  }

  void ResetSample(RandomEffectsDataset& rfx_dataset, RandomEffectsTracker& rfx_tracker, int sample_num) {
    rfx_[sample_num].reset(new RandomEffectsTerm(rfx_dataset, rfx_tracker));
  }
  
  std::vector<double> Predict(int i, RandomEffectsDataset& rfx_dataset) {
    return rfx_[i]->Predict(rfx_dataset);
  }

  RandomEffectsTerm* GetRandomEffectsTerm(int32_t sample_id) {return rfx_[sample_id].get();}

 private:
  std::vector<std::unique_ptr<RandomEffectsTerm>> rfx_;
  int num_samples_;
};

} // namespace StochTree

#endif // STOCHTREE_RANDOM_EFFECTS_H_
