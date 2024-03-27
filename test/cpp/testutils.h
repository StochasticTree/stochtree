/*!
 * Copyright (c) 2022 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_TESTUTILS_H_
#define STOCHTREE_TESTUTILS_H_

#include <Eigen/Dense>
#include <stochtree/random.h>
#include <vector>

namespace StochTree {

namespace TestUtils {

struct TestDataset {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> covariates;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> omega;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> rfx_basis;
  Eigen::VectorXd outcome;
  std::vector<int32_t> rfx_groups;
  int n;
  int x_cols;
  int omega_cols;
  int rfx_basis_cols;
  bool row_major{true};
};

/*! Creates a small dataset (10 observations) */
TestDataset LoadSmallDatasetUnivariateBasis();

/*! Creates a small dataset (10 observations) with a multivariate basis for leaf regression applications */
TestDataset LoadSmallDatasetMultivariateBasis();

/*! Creates a modest dataset (100 observations) */
TestDataset LoadMediumDatasetUnivariateBasis();

} // namespace TestUtils

} // namespace StochTree

#endif  // STOCHTREE_TESTUTILS_H_
