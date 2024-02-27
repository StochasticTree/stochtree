/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include <stochtree/random_effects_sampler.h>

namespace StochTree {

RandomEffectsSampler::RandomEffectsSampler(RegressionRandomEffectsDataset& rfx_dataset, RandomEffectsRegressionGaussianPrior& rfx_prior) {
  int32_t num_components = rfx_prior.GetNumComponents();
  int32_t num_groups = rfx_prior.GetNumGroups();
  W_beta_ = Eigen::MatrixXd(num_components, num_components);
  xi_ = Eigen::MatrixXd(num_components, num_groups);
  alpha_ = Eigen::VectorXd(num_components);
  sigma_xi_ = Eigen::VectorXd(num_components);
  sigma_alpha_ = Eigen::VectorXd(num_components);
  num_components_ = num_components;
  num_groups_ = num_groups;
  SiftGroupIndices(rfx_dataset.group_indices);
}

std::vector<std::int32_t> RandomEffectsSampler::GroupObservationIndices(std::int32_t group_num) const {
  std::size_t const offset_begin = group_index_begin_[group_num];
  std::size_t const offset_end = group_index_end_[group_num];
  if (offset_begin >= sifted_group_observations_.size() || offset_end > sifted_group_observations_.size()) {
    return {};
  }
  return std::vector<std::int32_t>(&sifted_group_observations_[offset_begin], &sifted_group_observations_[offset_end]);
}

void RandomEffectsSampler::InitializeParameters(Eigen::MatrixXd& X, Eigen::MatrixXd& y) {
  for (int i = 0; i < num_components_; i++) {
    alpha_(i) = 1.;
    sigma_alpha_(i) = 1.;
    sigma_xi_(i) = 1.;
    W_beta_(i,i) = 1;
  }
  Sigma_xi_ = sigma_xi_.asDiagonal().toDenseMatrix();
  Sigma_xi_inv_ = Sigma_xi_.inverse();

  std::vector<int32_t> observation_indices;
  Eigen::MatrixXd X_group;
  Eigen::MatrixXd y_group;
  for (int i = 0; i < num_groups_; i++) {
    observation_indices = GroupObservationIndices(i);
    X_group = X(observation_indices,Eigen::all);
    y_group = y(observation_indices,Eigen::all);
    xi_(Eigen::all, i) = (Sigma_xi_inv_ + (W_beta_ * alpha_).asDiagonal() * X_group.transpose() * X_group * (W_beta_ * alpha_).asDiagonal()).inverse() * ((W_beta_ * alpha_).asDiagonal() * X_group.transpose() * y_group);
  }
}

void RandomEffectsSampler::InitializeParameters(RegressionRandomEffectsDataset& rfx_dataset, ColumnVector& residual) {
  Eigen::MatrixXd X = rfx_dataset.basis;
  Eigen::VectorXd y = residual.GetData();
  for (int i = 0; i < num_components_; i++) {
    alpha_(i) = 1.;
    sigma_alpha_(i) = 1.;
    sigma_xi_(i) = 1.;
    W_beta_(i,i) = 1;
  }
  Sigma_xi_ = sigma_xi_.asDiagonal().toDenseMatrix();
  Sigma_xi_inv_ = Sigma_xi_.inverse();

  std::vector<int32_t> observation_indices;
  Eigen::MatrixXd X_group;
  Eigen::MatrixXd y_group;
  for (int i = 0; i < num_groups_; i++) {
    observation_indices = GroupObservationIndices(i);
    X_group = X(observation_indices,Eigen::all);
    y_group = y(observation_indices);
    xi_(Eigen::all, i) = (Sigma_xi_inv_ + (W_beta_ * alpha_).asDiagonal() * X_group.transpose() * X_group * (W_beta_ * alpha_).asDiagonal()).inverse() * ((W_beta_ * alpha_).asDiagonal() * X_group.transpose() * y_group);
  }
}

void RandomEffectsSampler::SampleRandomEffects(RandomEffectsRegressionGaussianPrior& rfx_prior, RegressionRandomEffectsDataset& rfx_dataset, ColumnVector& residual, std::mt19937& gen) {
  Eigen::MatrixXd X = rfx_dataset.basis;
  Eigen::VectorXd y = residual.GetData();
  SampleXi(X, y, gen);
  SampleAlpha(X, y, gen);
  SampleSigma(gen, rfx_prior.GetPriorVarianceShape(), rfx_prior.GetPriorVarianceScale());
}

void RandomEffectsSampler::SampleRandomEffects(Eigen::MatrixXd& X, Eigen::VectorXd& y, std::mt19937& gen, double a, double b) {
  SampleXi(X, y, gen);
  SampleAlpha(X, y, gen);
  SampleSigma(gen, a, b);
}

void RandomEffectsSampler::SampleAlpha(Eigen::MatrixXd& X, Eigen::VectorXd& y, std::mt19937& gen) {
  std::vector<int32_t> observation_indices;
  Eigen::MatrixXd X_group;
  Eigen::VectorXd y_group;
  Eigen::MatrixXd xi_group;
  Eigen::MatrixXd posterior_denominator = Sigma_xi_inv_;
  Eigen::MatrixXd posterior_numerator = Eigen::MatrixXd::Zero(num_components_, 1);
  for (int i = 0; i < num_groups_; i++) {
    observation_indices = GroupObservationIndices(i);
    X_group = X(observation_indices,Eigen::all);
    y_group = y(observation_indices,Eigen::all);
    xi_group = xi_(Eigen::all,i);
    posterior_denominator += W_beta_.transpose() * (xi_group).asDiagonal() * X_group.transpose() * X_group * (xi_group).asDiagonal() * W_beta_;
    posterior_numerator += W_beta_.transpose() * (xi_group).asDiagonal() * X_group.transpose() * y_group;
  }
  Eigen::MatrixXd alpha_mu_post = posterior_denominator.inverse() * posterior_numerator;
  Eigen::MatrixXd alpha_sig_post = posterior_denominator.inverse();
  
  // Cholesky decomposition of alpha_sig_post
  Eigen::LLT<Eigen::MatrixXd> decomposition(alpha_sig_post);
  Eigen::MatrixXd alpha_sig_post_chol = decomposition.matrixL();

  // Sample a vector of standard normal random variables
  std::normal_distribution<double> alpha_dist(0.,1.);
  int coef_dim = alpha_sig_post.cols();
  Eigen::MatrixXd std_norm_vec(coef_dim, 1);
  for (int i = 0; i < coef_dim; i++) {
    std_norm_vec(i,0) = alpha_dist(gen);
  }

  // Convert to alpha posterior
  alpha_ = alpha_mu_post + alpha_sig_post_chol * std_norm_vec;
}

void RandomEffectsSampler::SampleXi(Eigen::MatrixXd& X, Eigen::VectorXd& y, std::mt19937& gen) {
  std::vector<int32_t> observation_indices;
  Eigen::MatrixXd X_group;
  Eigen::VectorXd y_group;
  Eigen::MatrixXd posterior_variance = Eigen::MatrixXd(num_components_, num_components_);
  Eigen::MatrixXd posterior_mean = Eigen::MatrixXd(num_components_, num_components_);
  Eigen::MatrixXd posterior_denominator = Sigma_xi_inv_;
  Eigen::MatrixXd posterior_numerator = Eigen::MatrixXd::Zero(num_components_, 1);
  std::normal_distribution<double> alpha_dist(0.,1.);
  for (int i = 0; i < num_groups_; i++) {
    observation_indices = GroupObservationIndices(i);
    X_group = X(observation_indices,Eigen::all);
    y_group = y(observation_indices,Eigen::all);
    posterior_variance = ((W_beta_ * alpha_).asDiagonal() * X_group.transpose() * X_group * (W_beta_ * alpha_).asDiagonal()).inverse();
    posterior_mean = posterior_variance * ((W_beta_ * alpha_).asDiagonal() * X_group.transpose() * y_group);

    // Cholesky decomposition of alpha_sig_post
    Eigen::LLT<Eigen::MatrixXd> decomposition(posterior_variance);
    Eigen::MatrixXd posterior_variance_chol = decomposition.matrixL();

    // Sample a vector of standard normal random variables
    int coef_dim = posterior_variance.cols();
    Eigen::MatrixXd std_norm_vec(coef_dim, 1);
    for (int i = 0; i < coef_dim; i++) {
      std_norm_vec(i,0) = alpha_dist(gen);
    }

    // Convert to xi posterior for group i
    xi_(Eigen::all, i) = posterior_mean + posterior_variance_chol * std_norm_vec;
  }
}

void RandomEffectsSampler::SampleSigma(std::mt19937& gen, double a, double b) {
  double posterior_ig_shape;
  double posterior_ig_scale;
  double posterior_gamma_scale;
  for (int i = 0; i < num_components_; i++) {
    posterior_ig_shape = a + num_groups_;
    posterior_ig_scale = b + xi_(i, Eigen::all) * xi_(i, Eigen::all).transpose();

    // C++ standard library provides a gamma distribution with scale
    // parameter, but the correspondence between gamma and IG is that 
    // 1 / gamma(a,b) ~ IG(a,b) when b is a __rate__ parameter.
    // Before sampling, we convert ig_scale to a gamma scale parameter by 
    // taking its multiplicative inverse.
    double posterior_gamma_scale = 1./posterior_ig_scale;
    std::gamma_distribution<double> residual_variance_dist(posterior_ig_shape, posterior_gamma_scale);
    // NOTE: the IG-distributed parameter is sampled as (1/residual_variance_dist(gen)) but we store its inverse in Sigma_xi_inv
    Sigma_xi_inv_(i,i) = residual_variance_dist(gen);
  }
}

void RandomEffectsSampler::SiftGroupIndices(std::vector<int32_t>& group_labels) {
  int32_t n = group_labels.size();
  sifted_group_observations_.resize(n);

  // Determine number of unique groups
  std::vector<int32_t> group_index_labels_(group_labels);
  std::sort(group_index_labels_.begin(), group_index_labels_.end());
  auto unique_iter = std::unique(group_index_labels_.begin(), group_index_labels_.end());
  group_index_labels_.erase(unique_iter, group_index_labels_.end());
  int32_t num_groups = group_index_labels_.size();
  group_index_begin_.resize(num_groups);
  group_index_end_.resize(num_groups);

  // Sift the data indices so that they are sorted according to group index
  std::iota(sifted_group_observations_.begin(), sifted_group_observations_.end(), 0);
  auto comp_op = [&](size_t const &l, size_t const &r) { return std::less<int32_t>{}(group_labels[l], group_labels[r]); };
  std::stable_sort(sifted_group_observations_.begin(), sifted_group_observations_.end(), comp_op);
  std::vector<int32_t> group_labels_copy_(group_labels);
  std::stable_sort(group_labels_copy_.begin(), group_labels_copy_.end());

  // Group index construction
  for (int i = 0; i < num_groups; i++) {
    // Add group label to mapping
    label_map_.insert({group_index_labels_[i], i});
    
    // Track beginning and end of each group's sample indices
    auto first = std::find(group_labels_copy_.begin(), group_labels_copy_.end(), group_index_labels_[i]);
    group_index_begin_[i] = std::distance(group_labels_copy_.begin(), first);
    if (i < num_groups - 1) {
      auto last = std::find(group_labels_copy_.begin(), group_labels_copy_.end(), group_index_labels_[i+1]);
      group_index_end_[i] = std::distance(group_labels_copy_.begin(), last);
    } else {
      group_index_end_[i] = std::distance(group_labels_copy_.begin(), group_labels_copy_.end());
    }
  }
}

Eigen::VectorXd RandomEffectsSampler::PredictRandomEffects(Eigen::MatrixXd& X, std::vector<int32_t>& group_labels) {
  CHECK_EQ(X.rows(), group_labels.size());
  int n = X.rows();
  Eigen::VectorXd result(n);
  Eigen::MatrixXd alpha_diag = alpha_.asDiagonal().toDenseMatrix();
  std::uint64_t group_ind;
  for (int i = 0; i < n; i++) {
    group_ind = label_map_[group_labels[i]];
    result(i) = X(i, Eigen::all) * alpha_diag * xi_(Eigen::all, group_ind);
  }
  return result;
}

}  // namespace StochTree