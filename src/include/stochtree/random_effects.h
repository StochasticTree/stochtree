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
#include <nlohmann/json.hpp>
#include <Eigen/Dense>

#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace StochTree {

/*! \brief Forward declarations */
class LabelMapper;
class MultivariateRegressionRandomEffectsModel;
class RandomEffectsContainer;

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
  inline int32_t CategoryNumber(int32_t category_id) {return category_sample_tracker_->CategoryNumber(category_id);}
  SampleCategoryMapper* GetSampleCategoryMapper() {return sample_category_mapper_.get();}
  CategorySampleTracker* GetCategorySampleTracker() {return category_sample_tracker_.get();}
  std::vector<data_size_t>::iterator UnsortedNodeBeginIterator(int category_id);
  std::vector<data_size_t>::iterator UnsortedNodeEndIterator(int category_id);
  std::map<int32_t, int32_t>& GetLabelMap() {return category_sample_tracker_->GetLabelMap();}
  std::vector<int32_t>& GetUniqueGroupIds() {return category_sample_tracker_->GetUniqueGroupIds();}
  std::vector<data_size_t>& NodeIndices(int category_id) {return category_sample_tracker_->NodeIndices(category_id);}
  std::vector<data_size_t>& NodeIndicesInternalIndex(int internal_category_id) {return category_sample_tracker_->NodeIndicesInternalIndex(internal_category_id);}
  double GetPrediction(data_size_t observation_num) {return rfx_predictions_.at(observation_num);}
  void SetPrediction(data_size_t observation_num, double pred) {rfx_predictions_.at(observation_num) = pred;}
  /*! \brief Resets RFX tracker based on a specific sample. Assumes tracker already exists in main memory. */
  void ResetFromSample(MultivariateRegressionRandomEffectsModel& rfx_model, 
                       RandomEffectsDataset& rfx_dataset, ColumnVector& residual);
  /*! \brief Resets RFX tracker to initial default. Assumes tracker already exists in main memory. 
   *         Assumes that the initial "clean slate" prediction of a random effects model is 0.
   */
  void RootReset(MultivariateRegressionRandomEffectsModel& rfx_model, 
                 RandomEffectsDataset& rfx_dataset, ColumnVector& residual);

 private:
  /*! \brief Mapper from observations to category indices */
  std::unique_ptr<SampleCategoryMapper> sample_category_mapper_;
  /*! \brief Data structure tracking / updating observations available in each category in a dataset */
  std::unique_ptr<CategorySampleTracker> category_sample_tracker_;
  /*! \brief Vector of random effect predictions */
  std::vector<double> rfx_predictions_;
  /*! \brief Some high-level details of the random effects structure */
  int num_categories_;
  int num_observations_;
};

/*! \brief Standalone container for the map from category IDs to 0-based indices */
class LabelMapper {
 public:
  LabelMapper() {}
  LabelMapper(std::map<int32_t, int32_t> label_map) {
    label_map_ = label_map;
    for (const auto& [key, value] : label_map) keys_.push_back(key);
  }
  ~LabelMapper() {}
  void LoadFromLabelMap(std::map<int32_t, int32_t> label_map) {
    label_map_ = label_map;
    for (const auto& [key, value] : label_map) keys_.push_back(key);
  }
  bool ContainsLabel(int32_t category_id) {
    auto pos = label_map_.find(category_id);
    return pos != label_map_.end();
  }
  int32_t CategoryNumber(int32_t category_id) {
    return label_map_[category_id];
  }
  void SaveToJsonFile(std::string filename) {
    nlohmann::json model_json = this->to_json();
    std::ofstream output_file(filename);
    output_file << model_json << std::endl;
  }
  void LoadFromJsonFile(std::string filename) {
    std::ifstream f(filename);
    nlohmann::json rfx_label_mapper_json = nlohmann::json::parse(f);
    this->Reset();
    this->from_json(rfx_label_mapper_json);
  }
  std::string DumpJsonString() {
    nlohmann::json model_json = this->to_json();
    return model_json.dump();
  }
  void LoadFromJsonString(std::string& json_string) {
    nlohmann::json rfx_label_mapper_json = nlohmann::json::parse(json_string);
    this->Reset();
    this->from_json(rfx_label_mapper_json);
  }
  std::vector<int32_t>& Keys() {return keys_;}
  std::map<int32_t, int32_t>& Map() {return label_map_;}
  void Reset() {label_map_.clear(); keys_.clear();}
  nlohmann::json to_json();
  void from_json(const nlohmann::json& rfx_label_mapper_json);
 private:
  std::map<int32_t, int32_t> label_map_;
  std::vector<int32_t> keys_;
};

/*! \brief Posterior computation and sampling and state storage for random effects model with a group-level multivariate basis regression */
class MultivariateRegressionRandomEffectsModel {
 public:
  MultivariateRegressionRandomEffectsModel(int num_components, int num_groups) {
    normal_sampler_ = MultivariateNormalSampler();
    ig_sampler_ = InverseGammaSampler();
    num_components_ = num_components;
    num_groups_ = num_groups;
    working_parameter_ = Eigen::VectorXd(num_components_);
    group_parameters_ = Eigen::MatrixXd(num_components_, num_groups_);
    group_parameter_covariance_ = Eigen::MatrixXd(num_components_, num_components_);
    working_parameter_covariance_ = Eigen::MatrixXd(num_components_, num_components_);
  }
  ~MultivariateRegressionRandomEffectsModel() {}

  /*! \brief Reconstruction from serialized model parameter samples */
  void ResetFromSample(RandomEffectsContainer& rfx_container, int sample_num);
  
  /*! \brief Samplers */
  void SampleRandomEffects(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& tracker, double global_variance, std::mt19937& gen);
  void SampleWorkingParameter(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& tracker, double global_variance, std::mt19937& gen);
  void SampleGroupParameters(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& tracker, double global_variance, std::mt19937& gen);
  void SampleVarianceComponents(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& tracker, double global_variance, std::mt19937& gen);

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
  
  std::vector<double> Predict(RandomEffectsDataset& dataset, RandomEffectsTracker& tracker) {
    std::vector<double> output(dataset.NumObservations());
    PredictInplace(dataset, tracker, output);
    return output;
  }

  void PredictInplace(RandomEffectsDataset& dataset, RandomEffectsTracker& tracker, std::vector<double>& output) {
    Eigen::MatrixXd X = dataset.GetBasis();
    std::vector<int32_t> group_labels = dataset.GetGroupLabels();
    CHECK_EQ(X.rows(), group_labels.size());
    int n = X.rows();
    CHECK_EQ(n, output.size());
    Eigen::MatrixXd alpha_diag = working_parameter_.asDiagonal().toDenseMatrix();
    std::int32_t group_ind;
    for (int i = 0; i < n; i++) {
      group_ind = tracker.CategoryNumber(group_labels[i]);
      output[i] = X(i, Eigen::all) * alpha_diag * group_parameters_(Eigen::all, group_ind);
    }
  }

  void AddCurrentPredictionToResidual(RandomEffectsDataset& dataset, RandomEffectsTracker& tracker, ColumnVector& residual) {
    data_size_t n = dataset.NumObservations();
    CHECK_EQ(n, residual.NumRows());
    double current_pred;
    double new_resid;
    for (data_size_t i = 0; i < n; i++) {
      current_pred = tracker.GetPrediction(i);
      new_resid = residual.GetElement(i) + current_pred;
      residual.SetElement(i, new_resid);
    }
  }

  void SubtractNewPredictionFromResidual(RandomEffectsDataset& dataset, RandomEffectsTracker& tracker, ColumnVector& residual) {
    Eigen::MatrixXd X = dataset.GetBasis();
    std::vector<int32_t> group_labels = dataset.GetGroupLabels();
    CHECK_EQ(X.rows(), group_labels.size());
    int n = X.rows();
    double new_pred;
    double new_resid;
    Eigen::MatrixXd alpha_diag = working_parameter_.asDiagonal().toDenseMatrix();
    std::int32_t group_ind;
    for (int i = 0; i < n; i++) {
      group_ind = tracker.CategoryNumber(group_labels[i]);
      new_pred = X(i, Eigen::all) * alpha_diag * group_parameters_(Eigen::all, group_ind);
      new_resid = residual.GetElement(i) - new_pred;
      residual.SetElement(i, new_resid);
      tracker.SetPrediction(i, new_pred);
    }
  }

  /*! \brief Compute the posterior mean of the working parameter, conditional on the group parameters and the variance components */
  Eigen::VectorXd WorkingParameterMean(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, double global_variance);
  /*! \brief Compute the posterior covariance of the working parameter, conditional on the group parameters and the variance components */
  Eigen::MatrixXd WorkingParameterVariance(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, double global_variance);
  /*! \brief Compute the posterior mean of a group parameter, conditional on the working parameter and the variance components */
  Eigen::VectorXd GroupParameterMean(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, double global_variance, int32_t group_id);
  /*! \brief Compute the posterior covariance of a group parameter, conditional on the working parameter and the variance components */
  Eigen::MatrixXd GroupParameterVariance(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, double global_variance, int32_t group_id);
  /*! \brief Compute the posterior shape of the group variance component, conditional on the working and group parameters */
  double VarianceComponentShape(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, double global_variance, int32_t component_id);
  /*! \brief Compute the posterior scale of the group variance component, conditional on the working and group parameters */
  double VarianceComponentScale(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, double global_variance, int32_t component_id);

 private:
  /*! \brief Samplers */
  MultivariateNormalSampler normal_sampler_;
  InverseGammaSampler ig_sampler_;

  /*! \brief Random effects structure details */
  int num_components_;
  int num_groups_;
  
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
  RandomEffectsContainer(int num_components, int num_groups) {
    num_components_ = num_components;
    num_groups_ = num_groups;
    num_samples_ = 0;
  }
  RandomEffectsContainer() {
    num_components_ = 0;
    num_groups_ = 0;
    num_samples_ = 0;
  }
  ~RandomEffectsContainer() {}
  void SaveToJsonFile(std::string filename) {
    nlohmann::json model_json = this->to_json();
    std::ofstream output_file(filename);
    output_file << model_json << std::endl;
  }
  void LoadFromJsonFile(std::string filename) {
    std::ifstream f(filename);
    nlohmann::json rfx_container_json = nlohmann::json::parse(f);
    this->Reset();
    this->from_json(rfx_container_json);
  }
  std::string DumpJsonString() {
    nlohmann::json model_json = this->to_json();
    return model_json.dump();
  }
  void LoadFromJsonString(std::string& json_string) {
    nlohmann::json rfx_container_json = nlohmann::json::parse(json_string);
    this->Reset();
    this->from_json(rfx_container_json);
  }
  void AddSample(MultivariateRegressionRandomEffectsModel& model);
  void DeleteSample(int sample_num);
  void Predict(RandomEffectsDataset& dataset, LabelMapper& label_mapper, std::vector<double>& output);
  inline int NumSamples() {return num_samples_;}
  inline int NumComponents() {return num_components_;}
  inline int NumGroups() {return num_groups_;}
  inline void SetNumSamples(int num_samples) {num_samples_ = num_samples;}
  inline void SetNumComponents(int num_components) {num_components_ = num_components;}
  inline void SetNumGroups(int num_groups) {num_groups_ = num_groups;}
  void Reset() {
    num_samples_ = 0;
    num_components_ = 0;
    num_groups_ = 0;
    beta_.clear();
    alpha_.clear();
    xi_.clear();
    sigma_xi_.clear();
  }
  std::vector<double>& GetBeta() {return beta_;}
  std::vector<double>& GetAlpha() {return alpha_;}
  std::vector<double>& GetXi() {return xi_;}
  std::vector<double>& GetSigma() {return sigma_xi_;}
  nlohmann::json to_json();
  void from_json(const nlohmann::json& rfx_container_json);
  void append_from_json(const nlohmann::json& rfx_container_json);
 private:
  int num_samples_;
  int num_components_;
  int num_groups_;
  std::vector<double> beta_;
  std::vector<double> alpha_;
  std::vector<double> xi_;
  std::vector<double> sigma_xi_;
  void AddAlpha(MultivariateRegressionRandomEffectsModel& model);
  void AddXi(MultivariateRegressionRandomEffectsModel& model);
  void AddSigma(MultivariateRegressionRandomEffectsModel& model);
};

} // namespace StochTree

#endif // STOCHTREE_RANDOM_EFFECTS_H_
