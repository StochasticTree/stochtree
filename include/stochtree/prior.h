/*!
 * Copyright (c) 2024 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_PRIOR_H_
#define STOCHTREE_PRIOR_H_

#include <Eigen/Dense>
#include <stochtree/log.h>

namespace StochTree {

class RandomEffectsGaussianPrior {
 public:
  RandomEffectsGaussianPrior() {}
  virtual ~RandomEffectsGaussianPrior() = default;
};

class RandomEffectsRegressionGaussianPrior : public RandomEffectsGaussianPrior {
 public:
  RandomEffectsRegressionGaussianPrior(double a, double b, int32_t num_components, int32_t num_groups) {
    a_ = a;
    b_ = b;
    num_components_ = num_components;
    num_groups_ = num_groups;
  }
  ~RandomEffectsRegressionGaussianPrior() {}
  double GetPriorVarianceShape() {return a_;}
  double GetPriorVarianceScale() {return b_;}
  int32_t GetNumComponents() {return num_components_;}
  int32_t GetNumGroups() {return num_groups_;}
  void SetPriorVarianceShape(double a) {a_ = a;}
  void SetPriorVarianceScale(double b) {b_ = b;}
  void SetNumComponents(int32_t num_components) {num_components_ = num_components;}
  void SetNumGroups(int32_t num_groups) {num_groups_ = num_groups;}
 private:
  double a_;
  double b_;
  int32_t num_components_; 
  int32_t num_groups_;
};

class TreePrior {
 public:
  TreePrior(double alpha, double beta, int32_t min_samples_in_leaf, int32_t max_depth = -1) {
    alpha_ = alpha;
    beta_ = beta;
    min_samples_in_leaf_ = min_samples_in_leaf;
    max_depth_ = max_depth;
  }
  ~TreePrior() {}
  double GetAlpha() {return alpha_;}
  double GetBeta() {return beta_;}
  int32_t GetMinSamplesLeaf() {return min_samples_in_leaf_;}
  int32_t GetMaxDepth() {return max_depth_;}
  void SetAlpha(double alpha) {alpha_ = alpha;}
  void SetBeta(double beta) {beta_ = beta;}
  void SetMinSamplesLeaf(int32_t min_samples_in_leaf) {min_samples_in_leaf_ = min_samples_in_leaf;}
  void SetMaxDepth(int32_t max_depth) {max_depth_ = max_depth;}
 private:
  double alpha_;
  double beta_;
  int32_t min_samples_in_leaf_;
  int32_t max_depth_;
};

class IGVariancePrior {
 public:
  IGVariancePrior(double shape, double scale) {
    shape_ = shape;
    scale_ = scale;
  }
  ~IGVariancePrior() {}
  double GetShape() {return shape_;}
  double GetScale() {return scale_;}
  void SetShape(double shape) {shape_ = shape;}
  void SetScale(double scale) {scale_ = scale;}
 private:
  double shape_;
  double scale_;
};

} // namespace StochTree

#endif // STOCHTREE_PRIOR_H_