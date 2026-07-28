#ifndef STOCHTREE_MAINPAGE_H_
#define STOCHTREE_MAINPAGE_H_

/*!
 * \mainpage stochtree C++ Documentation
 * 
 * \section getting-started Getting Started
 * 
 * `stochtree` can be built and run as a standalone C++ program directly from source using `cmake`:
 * 
 * \subsection cloning-repo Cloning the Repository
 * 
 * To clone the repository, you must have git installed, which you can do following <a href="https://learn.microsoft.com/en-us/devops/develop/git/install-and-set-up-git">these instructions</a>. 
 * 
 * Once git is available at the command line, navigate to the folder that will store this project (in bash / zsh, this is done by running `cd` followed by the path to the directory). 
 * Then, clone the `stochtree` repo as a subfolder by running
 * \code{.sh}
 * git clone --recursive https://github.com/StochasticTree/stochtree.git
 * \endcode
 * 
 * <b>NOTE</b>: this project incorporates several dependencies as <a href="https://git-scm.com/book/en/v2/Git-Tools-Submodules">git submodules</a>, 
 * which is why the `--recursive` flag is necessary (some systems may perform a recursive clone without this flag, but 
 * `--recursive` ensures this behavior on all platforms). If you have already cloned the repo without the `--recursive` flag, 
 * you can retrieve the submodules recursively by running `git submodule update --init --recursive` in the main repo directory.
 * 
 * \section key-components Key Components
 * 
 * The stochtree C++ core consists of thousands of lines of C++ code, but it can organized and understood through several components (see [topics](topics.html) for more detail):
 * 
 * - <b>Trees</b>: the most important "primitive" of decision tree algorithms is the \ref tree_group "decision tree itself", which in stochtree is defined by a \ref StochTree::Tree "Tree" class as well as a series of static helper functions for prediction.
 * - <b>Forest</b>: individual trees are combined into a \ref forest_group "forest", or ensemble, which in stochtree is defined by the \ref StochTree::TreeEnsemble "TreeEnsemble" class and a container of forests is defined by the \ref StochTree::ForestContainer "ForestContainer" class.
 * - <b>Dataset</b>: data can be loaded from a variety of sources into a `stochtree` \ref data_group "data layer".
 * - <b>Leaf Model</b>: `stochtree`'s data structures are generalized to support a wide range of models, which are defined via specialized classes in the \ref leaf_model_group "leaf model layer".
 * - <b>Sampler</b>: helper functions that sample forests from training data comprise the \ref sampling_group "sampling layer" of `stochtree`.
 * 
 * \section extending-stochtree Extending stochtree
 * 
 * \subsection custom-leaf-models Custom Leaf Models
 * 
 * The \ref leaf_model_group "leaf model documentation" details the key components of new decision tree models: 
 * custom `LeafModel` and `SuffStat` classes that implement a model's log marginal likelihood and posterior computations. 
 * 
 * Adding a new leaf model will consist largely of implementing new versions of each of these classes which track the 
 * API of the existing classes. Once these classes exist, they need to be reflected in several places. 
 * 
 * Suppose, for the sake of illustration, that the newest custom leaf model is a multinomial logit model.
 * 
 * First, add an entry to the \ref StochTree::ModelType "ModelType" enumeration for this new model type
 * 
 * \code{.cpp}
 * enum ModelType {
 *    kConstantLeafGaussian, 
 *    kUnivariateRegressionLeafGaussian, 
 *    kMultivariateRegressionLeafGaussian, 
 *    kLogLinearVariance, 
 *    kMultinomialLogit, 
 * };
 * \endcode 
 * 
 * Next, add entries to the `std::variants` that bundle related `SuffStat` and `LeafModel` classes
 * 
 * \code{.cpp}
 * using SuffStatVariant = std::variant<GaussianConstantSuffStat, 
 *                                      GaussianUnivariateRegressionSuffStat, 
 *                                      GaussianMultivariateRegressionSuffStat, 
 *                                      LogLinearVarianceSuffStat, 
 *                                      MultinomialLogitSuffStat>;
 * \endcode 
 * 
 * \code{.cpp}
 * using LeafModelVariant = std::variant<GaussianConstantLeafModel, 
 *                                       GaussianUnivariateRegressionLeafModel, 
 *                                       GaussianMultivariateRegressionLeafModel, 
 *                                       LogLinearVarianceLeafModel, 
 *                                       MultinomialLogitLeafModel>;
 * \endcode 
 * 
 * Finally, update the \ref StochTree::suffStatFactory "suffStatFactory" and \ref StochTree::leafModelFactory "leafModelFactory" functions to add a logic branch registering these new objects
 * 
 * \code{.cpp}
 * static inline SuffStatVariant suffStatFactory(ModelType model_type, int basis_dim = 0) {
 *   if (model_type == kConstantLeafGaussian) {
 *     return createSuffStat<GaussianConstantSuffStat>();
 *   } else if (model_type == kUnivariateRegressionLeafGaussian) {
 *     return createSuffStat<GaussianUnivariateRegressionSuffStat>();
 *   } else if (model_type == kMultivariateRegressionLeafGaussian) {
 *     return createSuffStat<GaussianMultivariateRegressionSuffStat, int>(basis_dim);
 *   } else if (model_type == kLogLinearVariance) {
 *     return createSuffStat<LogLinearVarianceSuffStat>();
 *   } else if (model_type == kMultinomialLogit) {
 *     return createSuffStat<MultinomialLogitSuffStat>();
 *   } else {
 *     Log::Fatal("Incompatible model type provided to suff stat factory");
 *   }
 * }
 * \endcode 
 * 
 * \code{.cpp}
 * static inline LeafModelVariant leafModelFactory(ModelType model_type, double tau, Eigen::MatrixXd& Sigma0, double a, double b) {
 *   if (model_type == kConstantLeafGaussian) {
 *     return createLeafModel<GaussianConstantLeafModel, double>(tau);
 *   } else if (model_type == kUnivariateRegressionLeafGaussian) {
 *     return createLeafModel<GaussianUnivariateRegressionLeafModel, double>(tau);
 *   } else if (model_type == kMultivariateRegressionLeafGaussian) {
 *     return createLeafModel<GaussianMultivariateRegressionLeafModel, Eigen::MatrixXd>(Sigma0);
 *   } else if (model_type == kLogLinearVariance) {
 *     return createLeafModel<LogLinearVarianceLeafModel, double, double>(a, b);
 *   } else if (model_type == kMultinomialLogit) {
 *     return createLeafModel<MultinomialLogitLeafModel>();
 *   } else {
 *     Log::Fatal("Incompatible model type provided to leaf model factory");
 *   }
 * }
 * \endcode 
 * 
 */

#endif  // STOCHTREE_MAINPAGE_H_
