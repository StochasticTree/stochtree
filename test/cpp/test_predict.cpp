/*!
 * Test of the ensemble prediction method
 */
#include <gtest/gtest.h>
#include <testutils.h>
#include <stochtree/data.h>
#include <stochtree/ensemble.h>
#include <stochtree/interface.h>
#include <stochtree/tree.h>
#include <Eigen/Dense>
#include <iostream>
#include <memory>

/*! \brief Test forest prediction procedures for trees with constants in leaf nodes */
TEST(Ensemble, PredictConstant) {
  // Covariates (uniformly generated in R)
  int n = 10;
  int x_rows = n;
  int x_cols = 5;
  Eigen::MatrixXd covariates(x_rows, x_cols);
  covariates << 0.80195179, 0.56627934, 0.85066760, 0.55333717, 0.67683323,
                0.62574001, 0.71743631, 0.92349582, 0.87779366, 0.68785497,
                0.97678451, 0.96493876, 0.78560157, 0.05775895, 0.43964365,
                0.68112895, 0.69137664, 0.09408876, 0.50819643, 0.15438995,
                0.31852576, 0.17459775, 0.33655242, 0.09688789, 0.22522345,
                0.97831176, 0.34125955, 0.70473897, 0.09524901, 0.57536714,
                0.06442722, 0.50771215, 0.72807792, 0.51536037, 0.15120395,
                0.88865773, 0.25730240, 0.95612361, 0.05282520, 0.39155233,
                0.50874687, 0.39883808, 0.98027747, 0.02904932, 0.37047444,
                0.65685255, 0.62172827, 0.19717188, 0.90977738, 0.92560747;
  
  // Outcome (generated in R as X * (5, 10, 0, 0, 0)' + mean-zero gaussian noise)
  int y_rows = n;
  int y_cols = 1;
  Eigen::MatrixXd outcome(y_rows, y_cols);
  outcome << 9.665316, 10.316500, 14.541611, 10.329346, 3.340881, 8.305549, 5.392728, 7.017450, 6.527128, 9.501689;
  
  // Create a small ensemble
  int output_dim = 1;
  StochTree::TreeEnsemble ensemble(config, output_dim);

  // Simple split rules on both trees
  auto* tree = ensemble.GetTree(0);
  tree->ExpandNode(0, 0, 0.5, true, -5, 5);
  tree = ensemble.GetTree(1);
  tree->ExpandNode(0, 1, 0.5, true, -2.5, 2.5);

  // Run predict on the supplied covariates and check the result
  std::vector<double> result(n*output_dim);
  std::vector<double> expected_pred = {7.5, 7.5, 7.5, 7.5, -7.5, 2.5, -2.5, 2.5, 2.5, 7.5};
  ensemble.PredictInplace(covariates, result, 0);
  for (int i = 0; i < n; i++) {
    ASSERT_NEAR(expected_pred[i], result[i], 0.01);
  }
}

/*! \brief Test forest prediction procedures for trees with a univariate regression in leaf nodes */
TEST(Ensemble, PredictUnivariateRegression) {
  // Covariates (uniformly generated in R)
  int n = 10;
  int x_rows = n;
  int x_cols = 5;
  Eigen::MatrixXd covariates(x_rows, x_cols);
  covariates << 0.80195179, 0.56627934, 0.85066760, 0.55333717, 0.67683323,
                0.62574001, 0.71743631, 0.92349582, 0.87779366, 0.68785497,
                0.97678451, 0.96493876, 0.78560157, 0.05775895, 0.43964365,
                0.68112895, 0.69137664, 0.09408876, 0.50819643, 0.15438995,
                0.31852576, 0.17459775, 0.33655242, 0.09688789, 0.22522345,
                0.97831176, 0.34125955, 0.70473897, 0.09524901, 0.57536714,
                0.06442722, 0.50771215, 0.72807792, 0.51536037, 0.15120395,
                0.88865773, 0.25730240, 0.95612361, 0.05282520, 0.39155233,
                0.50874687, 0.39883808, 0.98027747, 0.02904932, 0.37047444,
                0.65685255, 0.62172827, 0.19717188, 0.90977738, 0.92560747;
  
  // Basis (uniformly generated in R)
  int omega_rows = n;
  int omega_cols = 1;
  Eigen::MatrixXd basis(omega_rows, omega_cols);
  basis << 0.01278643, 0.88747990, 0.14775865, 0.89994737, 0.52761700, 0.94708560, 0.71609799, 0.43333131, 0.42086126, 0.27527916;
  
  // Outcome (generated in R as X * (5, 10, 0, 0, 0)' + mean-zero gaussian noise)
  int y_rows = n;
  int y_cols = 1;
  Eigen::MatrixXd outcome(y_rows, y_cols);
  outcome << 9.665316, 10.316500, 14.541611, 10.329346, 3.340881, 8.305549, 5.392728, 7.017450, 6.527128, 9.501689;
  
  // Create a small ensemble
  int output_dim = 1;
  StochTree::TreeEnsemble ensemble(config, output_dim, false);

  // Simple split rules on both trees
  auto* tree = ensemble.GetTree(0);
  tree->ExpandNode(0, 0, 0.5, true, -5, 5);
  tree = ensemble.GetTree(1);
  tree->ExpandNode(0, 1, 0.5, true, -2.5, 2.5);

  // Run predict on the supplied covariates and basis function and check the result
  std::vector<double> result(n);
  std::vector<double> expected_pred = {0.09589821, 6.65609928, 1.10818989, 6.74960525, -3.95712748, 2.36771399, -1.79024499, 1.08332827, 1.05215315, 2.06459369};
  ensemble.PredictInplace(covariates, basis, result, 0);
  for (int i = 0; i < n; i++) {
    ASSERT_NEAR(expected_pred[i], result[i], 0.01);
  }
}

TEST(Ensemble, PredictMultivariateRegression) {
  // Covariates (uniformly generated in R)
  int n = 10;
  int x_rows = n;
  int x_cols = 5;
  Eigen::MatrixXd covariates(x_rows, x_cols);
  covariates << 0.80195179, 0.56627934, 0.85066760, 0.55333717, 0.67683323,
                0.62574001, 0.71743631, 0.92349582, 0.87779366, 0.68785497,
                0.97678451, 0.96493876, 0.78560157, 0.05775895, 0.43964365,
                0.68112895, 0.69137664, 0.09408876, 0.50819643, 0.15438995,
                0.31852576, 0.17459775, 0.33655242, 0.09688789, 0.22522345,
                0.97831176, 0.34125955, 0.70473897, 0.09524901, 0.57536714,
                0.06442722, 0.50771215, 0.72807792, 0.51536037, 0.15120395,
                0.88865773, 0.25730240, 0.95612361, 0.05282520, 0.39155233,
                0.50874687, 0.39883808, 0.98027747, 0.02904932, 0.37047444,
                0.65685255, 0.62172827, 0.19717188, 0.90977738, 0.92560747;
  
  // Basis (uniformly generated in R)
  int omega_rows = n;
  int omega_cols = 1;
  Eigen::MatrixXd basis(omega_rows, omega_cols);
  basis << 0.01278643, 0.88747990, 0.14775865, 0.89994737, 0.52761700, 0.94708560, 0.71609799, 0.43333131, 0.42086126, 0.27527916;
  
  // Outcome (generated in R as X * (5, 10, 0, 0, 0)' + mean-zero gaussian noise)
  int y_rows = n;
  int y_cols = 1;
  Eigen::MatrixXd outcome(y_rows, y_cols);
  outcome << 9.665316, 10.316500, 14.541611, 10.329346, 3.340881, 8.305549, 5.392728, 7.017450, 6.527128, 9.501689;
  
  // Create a small ensemble
  int output_dim = 1;
  StochTree::TreeEnsemble ensemble(config, output_dim, false);

  // Simple split rules on both trees
  auto* tree = ensemble.GetTree(0);
  tree->ExpandNode(0, 0, 0.5, true, std::vector<double>{-5, -2.5}, std::vector<double>{5, 2.5});
  tree = ensemble.GetTree(1);
  tree->ExpandNode(0, 1, 0.5, true, std::vector<double>{-2.5, -1.25}, std::vector<double>{2.5, 1.25});

  // Run predict on the supplied covariates and basis function and check the result
  std::vector<double> result(n);
  std::vector<double> expected_pred = {4.215666, 3.142668, 6.038436, 2.579774, -3.643255, 1.621931, -3.440814, 3.073815, 2.188391, 9.402637};
  ensemble.PredictInplace(covariates, basis, result, 0);
  for (int i = 0; i < n; i++) {
    ASSERT_NEAR(expected_pred[i], result[i], 0.01);
  }
}

