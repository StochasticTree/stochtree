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
 * - <b>Sampler</b>: helper functions that sample forests from training data comprise the \ref sampling_group "sampling layer" of `stochtree`.
 * 
 * \section extending-stochtree Extending `stochtree`
 * 
 * 
 * 
 */

#endif  // STOCHTREE_MAINPAGE_H_
