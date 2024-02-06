/*!
 * Copyright (c) 2023 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_PARAMETER_H_
#define STOCHTREE_PARAMETER_H_

#include <Eigen/Dense>

namespace StochTree {

class ModelParameter {
 public:
  ModelParameter() {}
  ModelParameter& operator=(const ModelParameter& param) {}
  virtual ~ModelParameter() = default;
};

class ScalarParameter : public ModelParameter {
 public:
  ScalarParameter(double value) {value_ = value;}
  ScalarParameter& operator=(const ScalarParameter& param) {
    value_ = param.value_;
    return *this;
  }
  ~ScalarParameter() {}
 private:
  double value_;
};

class VectorParameter : public ModelParameter {
 public:
  VectorParameter(Eigen::VectorXd& vector) {vector_ = vector;}
  VectorParameter& operator=(const VectorParameter& param) {
    vector_ = param.vector_;
    return *this;
  }
  ~VectorParameter() {}
 private:
  Eigen::VectorXd vector_;
};

class MatrixParameter : public ModelParameter {
 public:
  MatrixParameter(Eigen::MatrixXd& matrix) {matrix_ = matrix;}
  MatrixParameter& operator=(const MatrixParameter& param) {
    matrix_ = param.matrix_;
    return *this;
  }
  ~MatrixParameter() {}
 private:
  Eigen::MatrixXd matrix_;
};

} // namespace StochTree

#endif // STOCHTREE_PARAMETER_H_
