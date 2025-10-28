/*! Copyright (c) 2024 stochtree authors */
#include <stochtree/random_effects.h>

namespace StochTree {

RandomEffectsTracker::RandomEffectsTracker(std::vector<int32_t>& group_indices) {
  sample_category_mapper_ = std::make_unique<SampleCategoryMapper>(group_indices);
  category_sample_tracker_ = std::make_unique<CategorySampleTracker>(group_indices);
  num_categories_ = category_sample_tracker_->NumCategories();
  num_observations_ = group_indices.size();
  rfx_predictions_.resize(num_observations_, 0.);
}

nlohmann::json LabelMapper::to_json() {
  json output_obj;
  // Initialize a map with names of the node vectors and empty json arrays
  std::map<std::string, json> label_map_arrays;
  label_map_arrays.emplace(std::pair("keys", json::array()));
  label_map_arrays.emplace(std::pair("values", json::array()));
  for (const auto& [key, value] : label_map_) {
    label_map_arrays["keys"].emplace_back(key);
    label_map_arrays["values"].emplace_back(value);
  }
  for (auto& pair : label_map_arrays) {
    output_obj.emplace(pair);
  }
  return output_obj;
}

void LabelMapper::from_json(const nlohmann::json& rfx_label_mapper_json) {
  int num_keys = rfx_label_mapper_json.at("keys").size();
  int num_values = rfx_label_mapper_json.at("values").size();
  CHECK_EQ(num_keys, num_values);
  for (int i = 0; i < num_keys; i++) {
    int32_t key = rfx_label_mapper_json.at("keys").at(i);
    int32_t value = rfx_label_mapper_json.at("values").at(i);
    keys_.push_back(key);
    label_map_.insert({key, value});
  }
}

void RandomEffectsTracker::ResetFromSample(MultivariateRegressionRandomEffectsModel& rfx_model, 
                                           RandomEffectsDataset& rfx_dataset, ColumnVector& residual) {
  Eigen::MatrixXd X = rfx_dataset.GetBasis();
  std::vector<int32_t> group_labels = rfx_dataset.GetGroupLabels();
  CHECK_EQ(X.rows(), group_labels.size());
  int n = X.rows();
  double prev_pred;
  double new_pred;
  double new_resid;
  Eigen::MatrixXd alpha_diag = rfx_model.GetWorkingParameter().asDiagonal().toDenseMatrix();
  Eigen::MatrixXd xi = rfx_model.GetGroupParameters();
  std::int32_t group_ind;
  for (int i = 0; i < n; i++) {
    group_ind = CategoryNumber(group_labels[i]);
    prev_pred = GetPrediction(i);
    new_pred = X(i, Eigen::all) * alpha_diag * xi(Eigen::all, group_ind);
    new_resid = residual.GetElement(i) - new_pred + prev_pred;
    residual.SetElement(i, new_resid);
    SetPrediction(i, new_pred);
  }
}

void RandomEffectsTracker::RootReset(MultivariateRegressionRandomEffectsModel& rfx_model, 
                                     RandomEffectsDataset& rfx_dataset, ColumnVector& residual) {
  int n = rfx_dataset.NumObservations();
  CHECK_EQ(n, num_observations_);
  double prev_pred;
  double new_pred;
  double new_resid;
  for (int i = 0; i < n; i++) {
    prev_pred = GetPrediction(i);
    new_pred = 0.;
    new_resid = residual.GetElement(i) - new_pred + prev_pred;
    residual.SetElement(i, new_resid);
    SetPrediction(i, new_pred);
  }
}

void MultivariateRegressionRandomEffectsModel::ResetFromSample(RandomEffectsContainer& rfx_container, int sample_num) {
    // Extract parameter vectors
    std::vector<double>& alpha = rfx_container.GetAlpha();
    std::vector<double>& xi = rfx_container.GetXi();
    std::vector<double>& sigma = rfx_container.GetSigma();
    
    // Unpack parameters
    for (int i = 0; i < num_components_; i++) {
      working_parameter_(i) = alpha.at(sample_num*num_components_ + i);
      group_parameter_covariance_(i, i) = sigma.at(sample_num*num_components_ + i);
      for (int j = 0; j < num_groups_; j++) {
        group_parameters_(i,j) = xi.at(sample_num*num_groups_*num_components_ + j*num_components_ + i);
      }
    }
  }

void MultivariateRegressionRandomEffectsModel::SampleRandomEffects(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, 
                                                                   double global_variance, std::mt19937& gen) {
  // Update partial residual to add back in the random effects
  AddCurrentPredictionToResidual(dataset, rfx_tracker, residual);
  
  // Sample random effects
  SampleGroupParameters(dataset, residual, rfx_tracker, global_variance, gen);
  SampleWorkingParameter(dataset, residual, rfx_tracker, global_variance, gen);
  SampleVarianceComponents(dataset, residual, rfx_tracker, global_variance, gen);

  // Update partial residual to remove the random effects
  SubtractNewPredictionFromResidual(dataset, rfx_tracker, residual);
}

void MultivariateRegressionRandomEffectsModel::SampleWorkingParameter(RandomEffectsDataset& dataset, ColumnVector& residual, 
                                                                      RandomEffectsTracker& rfx_tracker, double global_variance, std::mt19937& gen) {
  Eigen::VectorXd posterior_mean = WorkingParameterMean(dataset, residual, rfx_tracker, global_variance);
  Eigen::MatrixXd posterior_covariance = WorkingParameterVariance(dataset, residual, rfx_tracker, global_variance);
  working_parameter_ = normal_sampler_.SampleEigen(posterior_mean, posterior_covariance, gen);
}

void MultivariateRegressionRandomEffectsModel::SampleGroupParameters(RandomEffectsDataset& dataset, ColumnVector& residual, 
                                                                     RandomEffectsTracker& rfx_tracker, double global_variance, std::mt19937& gen) {
  int32_t num_groups = num_groups_;
  Eigen::VectorXd posterior_mean;
  Eigen::MatrixXd posterior_covariance;
  Eigen::VectorXd output;
  for (int i = 0; i < num_groups; i++) {
    posterior_mean = GroupParameterMean(dataset, residual, rfx_tracker, global_variance, i);
    posterior_covariance = GroupParameterVariance(dataset, residual, rfx_tracker, global_variance, i);
    group_parameters_(Eigen::all, i) = normal_sampler_.SampleEigen(posterior_mean, posterior_covariance, gen);
  }  
}

void MultivariateRegressionRandomEffectsModel::SampleVarianceComponents(RandomEffectsDataset& dataset, ColumnVector& residual, 
                                                                        RandomEffectsTracker& rfx_tracker, double global_variance, std::mt19937& gen) {
  int32_t num_components = num_components_;
  double posterior_shape;
  double posterior_scale;
  double output;
  for (int i = 0; i < num_components; i++) {
    posterior_shape = VarianceComponentShape(dataset, residual, rfx_tracker, global_variance, i);
    posterior_scale = VarianceComponentScale(dataset, residual, rfx_tracker, global_variance, i);
    group_parameter_covariance_(i, i) = ig_sampler_.Sample(posterior_shape, posterior_scale, gen);
  }
}

Eigen::VectorXd MultivariateRegressionRandomEffectsModel::WorkingParameterMean(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, 
                                                                               double global_variance){
  int32_t num_components = num_components_;
  int32_t num_groups = num_groups_;
  std::vector<data_size_t> observation_indices;
  Eigen::MatrixXd X_group;
  Eigen::VectorXd y_group;
  Eigen::MatrixXd xi_group;
  Eigen::MatrixXd posterior_denominator = working_parameter_covariance_.inverse();
  Eigen::VectorXd posterior_numerator = Eigen::VectorXd::Zero(num_components);
  Eigen::MatrixXd X = dataset.GetBasis();
  Eigen::VectorXd y = residual.GetData();
  Eigen::MatrixXd xi = group_parameters_;
  for (int i = 0; i < num_groups; i++) {
    observation_indices = rfx_tracker.NodeIndicesInternalIndex(i);
    X_group = X(observation_indices, Eigen::all);
    y_group = y(observation_indices, Eigen::all);
    xi_group = xi(Eigen::all, i);
    posterior_denominator += ((xi_group).asDiagonal() * X_group.transpose() * X_group * (xi_group).asDiagonal()) / global_variance;
    posterior_numerator += (xi_group).asDiagonal() * X_group.transpose() * y_group / global_variance;
  }
  return posterior_denominator.inverse() * posterior_numerator;
}

Eigen::MatrixXd MultivariateRegressionRandomEffectsModel::WorkingParameterVariance(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, double global_variance){
  int32_t num_components = num_components_;
  int32_t num_groups = num_groups_;
  std::vector<data_size_t> observation_indices;
  Eigen::MatrixXd X_group;
  Eigen::VectorXd y_group;
  Eigen::MatrixXd xi_group;
  Eigen::MatrixXd posterior_denominator = working_parameter_covariance_.inverse();
  Eigen::VectorXd posterior_numerator = Eigen::VectorXd::Zero(num_components);
  Eigen::MatrixXd X = dataset.GetBasis();
  Eigen::VectorXd y = residual.GetData();
  Eigen::MatrixXd xi = group_parameters_;
  for (int i = 0; i < num_groups; i++) {
    observation_indices = rfx_tracker.NodeIndicesInternalIndex(i);
    X_group = X(observation_indices, Eigen::all);
    y_group = y(observation_indices, Eigen::all);
    xi_group = xi(Eigen::all, i);
    posterior_denominator += ((xi_group).asDiagonal() * X_group.transpose() * X_group * (xi_group).asDiagonal()) / (global_variance);
  }
  return posterior_denominator.inverse();
}

Eigen::VectorXd MultivariateRegressionRandomEffectsModel::GroupParameterMean(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, double global_variance, int32_t group_id) {
  int32_t num_components = num_components_;
  int32_t num_groups = num_groups_;
  Eigen::MatrixXd X = dataset.GetBasis();
  Eigen::VectorXd y = residual.GetData();
  Eigen::VectorXd alpha = working_parameter_;
  Eigen::MatrixXd posterior_denominator = group_parameter_covariance_.inverse();
  Eigen::VectorXd posterior_numerator = Eigen::VectorXd::Zero(num_components);
  std::vector<data_size_t> observation_indices = rfx_tracker.NodeIndicesInternalIndex(group_id);
  Eigen::MatrixXd X_group = X(observation_indices, Eigen::all);
  Eigen::VectorXd y_group = y(observation_indices, Eigen::all);
  posterior_denominator += ((alpha).asDiagonal() * X_group.transpose() * X_group * (alpha).asDiagonal()) / (global_variance);
  posterior_numerator += (alpha).asDiagonal() * X_group.transpose() * y_group / global_variance;
  return posterior_denominator.inverse() * posterior_numerator;
}

Eigen::MatrixXd MultivariateRegressionRandomEffectsModel::GroupParameterVariance(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, double global_variance, int32_t group_id){
  int32_t num_components = num_components_;
  int32_t num_groups = num_groups_;
  Eigen::MatrixXd X = dataset.GetBasis();
  Eigen::VectorXd y = residual.GetData();
  Eigen::VectorXd alpha = working_parameter_;
  Eigen::MatrixXd posterior_denominator = group_parameter_covariance_.inverse();
//  Eigen::VectorXd posterior_numerator = Eigen::VectorXd::Zero(num_components);
  std::vector<data_size_t> observation_indices = rfx_tracker.NodeIndicesInternalIndex(group_id);
  Eigen::MatrixXd X_group = X(observation_indices, Eigen::all);
//  Eigen::VectorXd y_group = y(observation_indices, Eigen::all);
  posterior_denominator += ((alpha).asDiagonal() * X_group.transpose() * X_group * (alpha).asDiagonal()) / (global_variance);
//  posterior_numerator += (alpha).asDiagonal() * X_group.transpose() * y_group;
  return posterior_denominator.inverse();
}

double MultivariateRegressionRandomEffectsModel::VarianceComponentShape(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, double global_variance, int32_t component_id) {
  return static_cast<double>(variance_prior_shape_ + num_groups_);
}

double MultivariateRegressionRandomEffectsModel::VarianceComponentScale(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, double global_variance, int32_t component_id) {
  int32_t num_groups = num_groups_;
  Eigen::MatrixXd xi = group_parameters_;
  double output = variance_prior_scale_;
  for (int i = 0; i < num_groups; i++) {
    output += xi(component_id, i)*xi(component_id, i);
  }
  return output;
}

void RandomEffectsContainer::AddSample(MultivariateRegressionRandomEffectsModel& model){
  // Increment number of samples
  int sample_ind = num_samples_;
  num_samples_++;

  // Add alpha
  alpha_.resize(num_samples_*num_components_);
  for (int i = 0; i < num_components_; i++) {
    alpha_.at(sample_ind*num_components_ + i) = model.GetWorkingParameter()(i);
  }

  // Add xi and beta
  xi_.resize(num_samples_*num_components_*num_groups_);
  beta_.resize(num_samples_*num_components_*num_groups_);
  for (int i = 0; i < num_components_; i++) {
    for (int j = 0; j < num_groups_; j++) {
      xi_.at(sample_ind*num_groups_*num_components_ + j*num_components_ + i) = model.GetGroupParameters()(i,j);
      beta_.at(sample_ind*num_groups_*num_components_ + j*num_components_ + i) = xi_.at(sample_ind*num_groups_*num_components_ + j*num_components_ + i) * alpha_.at(sample_ind*num_components_ + i);
    }
  }

  // Add sigma
  sigma_xi_.resize(num_samples_*num_components_);
  for (int i = 0; i < num_components_; i++) {
    sigma_xi_.at(sample_ind*num_components_ + i) = model.GetGroupParameterCovariance()(i,i);
  }
}

void RandomEffectsContainer::Predict(RandomEffectsDataset& dataset, LabelMapper& label_mapper, std::vector<double>& output) {
  Eigen::MatrixXd X = dataset.GetBasis();
  std::vector<int32_t> group_labels = dataset.GetGroupLabels();
  CHECK_EQ(X.rows(), group_labels.size());
  int n = X.rows();
  CHECK_EQ(n*num_samples_, output.size());
  std::int32_t group_ind;
  double pred;
  for (int i = 0; i < n; i++) {
    group_ind = label_mapper.CategoryNumber(group_labels[i]);
    for (int j = 0; j < num_samples_; j++) {
      pred = 0;
      for (int k = 0; k < num_components_; k++) {
        pred += X(i,k) * beta_.at(j*num_groups_*num_components_ + group_ind*num_components_ + k);
      }
      output.at(j*n + i) = pred;
    }
  }
}

nlohmann::json RandomEffectsContainer::to_json() {
  json result_obj;
  // Store the non-array fields in json
  result_obj.emplace("num_samples", num_samples_);
  result_obj.emplace("num_components", num_components_);
  result_obj.emplace("num_groups", num_groups_);

  // Store some meta-level information about the containers
  int beta_size = num_groups_*num_components_*num_samples_;
  int alpha_size = num_components_*num_samples_;
  result_obj.emplace("beta_size", beta_size);
  result_obj.emplace("alpha_size", alpha_size);

  // Initialize a map with names of the node vectors and empty json arrays
  std::map<std::string, json> tree_array_map;
  tree_array_map.emplace(std::pair("beta", json::array()));
  tree_array_map.emplace(std::pair("xi", json::array()));
  tree_array_map.emplace(std::pair("alpha", json::array()));
  tree_array_map.emplace(std::pair("sigma_xi", json::array()));

  // Unpack beta and xi into json arrays
  for (int i = 0; i < beta_size; i++) {
    tree_array_map["beta"].emplace_back(beta_.at(i));
    tree_array_map["xi"].emplace_back(xi_.at(i));
  }

  // Unpack alpha and sigma into json arrays
  for (int i = 0; i < alpha_size; i++) {
    tree_array_map["alpha"].emplace_back(alpha_.at(i));
    tree_array_map["sigma_xi"].emplace_back(sigma_xi_.at(i));
  }

  // Unpack the map into the reference JSON object
  for (auto& pair : tree_array_map) {
    result_obj.emplace(pair);
  }

return result_obj;
}

void RandomEffectsContainer::DeleteSample(int sample_num){
  // Decrement number of samples
  num_samples_--;

  // Remove sample_num from alpha
  // ----------------------------
  // This code works because the data are stored in a "column-major" format, 
  // with components comprising rows and and samples comprising columns, so that 
  // element `sample_num*num_components_ + i` will contain the "i"-th component of the 
  // sample indexed by sample_num. Erasing the `sample_num*num_components_ + 0` 
  // element of the vector will move the element that was previously in position 
  // `sample_num*num_components_ + 1` into the position `sample_num*num_components_ + 0`
  // and thus we can repeat `alpha_.erase(alpha_.begin() + sample_num*num_components_);`
  // exactly `num_components_` times to erase each component pertaining to this sample.
  for (int i = 0; i < num_components_; i++) {
    alpha_.erase(alpha_.begin() + sample_num*num_components_);
  }

  // Remove sample_num from xi and beta
  // ----------------------------------
  // This code works as above, with the added nuance of the three-dimensional (Fortran-aligned) array, 
  // in which sample number is the third dimension, group number is the second dimension, and component 
  // number is the third dimension. The nested loop assembles all `num_groups_*num_components_` offsets, 
  // expressed as `j*num_components_ + i`. In order to remove each of the elements stored in these offsets 
  // from `sample_num*num_groups_*num_components_`, we simply need to erase the 
  // `sample_num*num_groups_*num_components_` element, exactly `num_groups_*num_components_` times.
  for (int i = 0; i < num_components_; i++) {
    for (int j = 0; j < num_groups_; j++) {
      xi_.erase(xi_.begin() + sample_num*num_groups_*num_components_);
      beta_.erase(beta_.begin() + sample_num*num_groups_*num_components_);
    }
  }

  // Remove sample_num from sigma
  // ----------------------------
  // This code works as with alpha
  for (int i = 0; i < num_components_; i++) {
    sigma_xi_.erase(sigma_xi_.begin() + sample_num*num_components_);
  }
}

void RandomEffectsContainer::from_json(const nlohmann::json& rfx_container_json) {
  int beta_size = rfx_container_json.at("beta_size");
  int alpha_size = rfx_container_json.at("alpha_size");

  // Clear all internal arrays
  beta_.clear();
  xi_.clear();
  alpha_.clear();
  sigma_xi_.clear();

  // Unpack internal counts
  this->num_samples_ = rfx_container_json.at("num_samples");
  this->num_components_ = rfx_container_json.at("num_components");
  this->num_groups_ = rfx_container_json.at("num_groups");
  
  // Unpack beta and xi
  for (int i = 0; i < beta_size; i++) {
    beta_.push_back(rfx_container_json.at("beta").at(i));
    xi_.push_back(rfx_container_json.at("xi").at(i));
  }
  
  // Unpack alpha and sigma_xi
  for (int i = 0; i < alpha_size; i++) {
    alpha_.push_back(rfx_container_json.at("alpha").at(i));
    sigma_xi_.push_back(rfx_container_json.at("sigma_xi").at(i));
  }
}

void RandomEffectsContainer::append_from_json(const nlohmann::json& rfx_container_json) {
  CHECK_EQ(this->num_components_, rfx_container_json.at("num_components"));
  CHECK_EQ(this->num_groups_, rfx_container_json.at("num_groups"));
  
  // Update internal sample count and extract size of parameter vectors
  int new_num_samples = rfx_container_json.at("num_samples");
  this->num_samples_ += new_num_samples;
  int beta_size = rfx_container_json.at("beta_size");
  int alpha_size = rfx_container_json.at("alpha_size");
  
  // Unpack beta and xi
  for (int i = 0; i < beta_size; i++) {
    beta_.push_back(rfx_container_json.at("beta").at(i));
    xi_.push_back(rfx_container_json.at("xi").at(i));
  }
  
  // Unpack alpha and sigma_xi
  for (int i = 0; i < alpha_size; i++) {
    alpha_.push_back(rfx_container_json.at("alpha").at(i));
    sigma_xi_.push_back(rfx_container_json.at("sigma_xi").at(i));
  }
}

}  // namespace StochTree
