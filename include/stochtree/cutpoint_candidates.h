/*!
 * Copyright (c) 2024 stochtree authors.
 * 
 * Data structures for enumerating potential cutpoint candidates.
 * 
 * This is used in the XBART family of algorithms, which samples split rules
 * based on the log marginal likelihood of every potential cutpoint. For numeric 
 * variables with large sample sizes, it is often unnecessary to consider every 
 * unique value, so we allow for an adaptive "grid" of potential cutpoint values.
 * 
 * Algorithms for enumerating cutpoints take Dataset and SortedNodeSampleTracker objects 
 * as inputs, so that each feature is "pre-sorted" according to its value within a 
 * given node. The size of the adaptive cutpoint grid is set by the 
 * cutpoint_grid_size configuration parameter.
 * 
 * Numeric Features
 * ----------------
 * 
 * When a node has fewer available observations than cutpoint_grid_size, 
 * full enumeration of unique available cutpoints is done via the 
 * `EnumerateNumericCutpointsDeduplication` function
 * 
 * When a node has more available observations than cutpoint_grid_size, 
 * potential cutpoints are "thinned out" by considering every k-th observation, 
 * where k is implied by the number of observations and the target cutpoint_grid_size.
 * 
 * Ordered Categorical Features
 * ----------------------------
 * 
 * In this case, the grid is every unique value of the ordered categorical 
 * feature in ascending order.
 * 
 * Unordered Categorical Features
 * ------------------------------
 * 
 * In this case, the grid is every unique value of the unordered categorical feature, 
 * arranged in an outcome-dependent order, as described in Fisher (1958)
 */
#ifndef STOCHTREE_CUTPOINT_CANDIDATES_H_
#define STOCHTREE_CUTPOINT_CANDIDATES_H_

#include <stochtree/meta.h>
#include <stochtree/partition_tracker.h>

namespace StochTree {

/*! \brief Computing and tracking cutpoints available for a given feature at a given node
 *  Store cutpoint bins in 0-indexed fashion, so that if a given node has 
 */
class FeatureCutpointGrid {
 public:
  FeatureCutpointGrid(int cutpoint_grid_size) : node_stride_begin_{}, node_stride_length_{}, cutpoint_grid_size_{cutpoint_grid_size} {}

  ~FeatureCutpointGrid() {}

  /*! \brief Calculate strides */
  void CalculateStrides(Eigen::MatrixXd& covariates, Eigen::VectorXd& residuals, SortedNodeSampleTracker* feature_node_sort_tracker, int32_t node_id, data_size_t node_begin, data_size_t node_end, int32_t feature_index, std::vector<FeatureType>& feature_types);

  /*! \brief Split numeric / ordered categorical feature and update sort indices */
  void CalculateStridesNumeric(Eigen::MatrixXd& covariates, Eigen::VectorXd& residuals, SortedNodeSampleTracker* feature_node_sort_tracker, int32_t node_id, data_size_t node_begin, data_size_t node_end, int32_t feature_index);

  /*! \brief Split numeric / ordered categorical feature and update sort indices */
  void CalculateStridesOrderedCategorical(Eigen::MatrixXd& covariates, Eigen::VectorXd& residuals, SortedNodeSampleTracker* feature_node_sort_tracker, int32_t node_id, data_size_t node_begin, data_size_t node_end, int32_t feature_index);

  /*! \brief Split unordered categorical feature and update sort indices */
  void CalculateStridesUnorderedCategorical(Eigen::MatrixXd& covariates, Eigen::VectorXd& residuals, SortedNodeSampleTracker* feature_node_sort_tracker, int32_t node_id, data_size_t node_begin, data_size_t node_end, int32_t feature_index);

  /*! \brief Number of potential cutpoints enumerated */
  int32_t NumCutpoints() {return node_stride_begin_.size();}

  /*! \brief Beginning index of bin i */
  int32_t BinStartIndex(int i) {return node_stride_begin_.at(i);}

  /*! \brief Size of bin i */
  int32_t BinLength(int i) {return node_stride_length_.at(i);}

  /*! \brief Beginning index of bin i */
  int32_t BinEndIndex(int i) {return node_stride_begin_.at(i) + node_stride_length_.at(i);}

  /*! \brief Value of the upper-bound (cutpoint) implied by bin i */
  double CutpointValue(int i) {return cutpoint_values_.at(i);}

  /*! \brief Vector of cutpoint values up to and including bin i
   *  Helper function for converting categorical split "value" (as outlined in Fisher 1958) to a set of categories
   */
  std::vector<std::uint32_t> CutpointVector(int i) {
    std::vector<std::uint32_t> out;
    int bin_stop = i + 1;
    for (int j = 0; j < bin_stop; j++) {
      out.push_back(static_cast<std::uint32_t>(cutpoint_values_.at(j)));
    }
    return out;
  }

 private:
  /*! \brief Vectors of node stride starting points and stride lengths */
  std::vector<data_size_t> node_stride_begin_;
  std::vector<data_size_t> node_stride_length_;
  std::vector<double> cutpoint_values_;
  int32_t cutpoint_grid_size_;

  /*! \brief Full enumeration of numeric cutpoints, checking for duplicate value */
  void EnumerateNumericCutpointsDeduplication(Eigen::MatrixXd& covariates, Eigen::VectorXd& residuals, SortedNodeSampleTracker* feature_node_sort_tracker, int32_t node_id, data_size_t node_begin, data_size_t node_end, data_size_t node_size, int32_t feature_index);

  /*! \brief Calculation of numeric cutpoints, thinning out to ensure that, at most, cutpoint_grid_size_ cutpoints are considered */
  void ScanNumericCutpoints(Eigen::MatrixXd& covariates, Eigen::VectorXd& residuals, SortedNodeSampleTracker* feature_node_sort_tracker, int32_t node_id, data_size_t node_begin, data_size_t node_end, data_size_t node_size, int32_t feature_index);
};

/*! \brief Container class for FeatureCutpointGrid objects stored for every feature in a dataset */
class CutpointGridContainer {
 public:
  CutpointGridContainer(Eigen::MatrixXd& covariates, Eigen::VectorXd& residuals, int cutpoint_grid_size) {
    num_features_ = covariates.cols();
    feature_cutpoint_grid_.resize(num_features_);
    for (int i = 0; i < num_features_; i++) {
      feature_cutpoint_grid_[i].reset(new FeatureCutpointGrid(cutpoint_grid_size));
    }
    cutpoint_grid_size_ = cutpoint_grid_size;
  }

  ~CutpointGridContainer() {}

  void Reset(Eigen::MatrixXd& covariates, Eigen::VectorXd& residuals, int cutpoint_grid_size) {
    num_features_ = covariates.cols();
    feature_cutpoint_grid_.resize(num_features_);
    for (int i = 0; i < num_features_; i++) {
      feature_cutpoint_grid_[i].reset(new FeatureCutpointGrid(cutpoint_grid_size));
    }
    cutpoint_grid_size_ = cutpoint_grid_size;
  }

  /*! \brief Calculate strides */
  void CalculateStrides(Eigen::MatrixXd& covariates, Eigen::VectorXd& residuals, SortedNodeSampleTracker* feature_node_sort_tracker, int32_t node_id, data_size_t node_begin, data_size_t node_end, int32_t feature_index, std::vector<FeatureType>& feature_types) {
    feature_cutpoint_grid_[feature_index]->CalculateStrides(covariates, residuals, feature_node_sort_tracker, node_id, node_begin, node_end, feature_index, feature_types);
  }

  /*! \brief Max size of cutpoint grid */
  int32_t CutpointGridSize() {return cutpoint_grid_size_;}

  /*! \brief Number of potential cutpoints enumerated */
  int32_t NumCutpoints(int feature_index) {return feature_cutpoint_grid_[feature_index]->NumCutpoints();}

  /*! \brief Beginning index of bin i */
  int32_t BinStartIndex(int i, int feature_index) {return feature_cutpoint_grid_[feature_index]->BinStartIndex(i);}

  /*! \brief Size of bin i */
  int32_t BinLength(int i, int feature_index) {return feature_cutpoint_grid_[feature_index]->BinLength(i);}

  /*! \brief Beginning index of bin i */
  int32_t BinEndIndex(int i, int feature_index) {return feature_cutpoint_grid_[feature_index]->BinEndIndex(i);}

  /*! \brief Value of the upper-bound (cutpoint) implied by bin i */
  double CutpointValue(int i, int feature_index) {return feature_cutpoint_grid_[feature_index]->CutpointValue(i);}

  /*! \brief Vector of cutpoint values up to and including bin i
   *  Helper function for converting categorical split "value" (as outlined in Fisher 1958) to a set of categories
   */
  std::vector<std::uint32_t> CutpointVector(int i, int feature_index) {
    return feature_cutpoint_grid_[feature_index]->CutpointVector(i);
  }

  FeatureCutpointGrid* GetFeatureCutpointGrid(int feature_num) {return feature_cutpoint_grid_[feature_num].get(); }

 private:
  std::vector<std::unique_ptr<FeatureCutpointGrid>> feature_cutpoint_grid_;
  int num_features_;
  int cutpoint_grid_size_;
};

/*! \brief Computing and tracking cutpoints available for a given feature at a given node */
class NodeCutpointTracker {
 public:
  NodeCutpointTracker(int cutpoint_grid_size) : node_stride_begin_{}, node_stride_length_{}, cutpoint_grid_size_{cutpoint_grid_size}, nodes_enumerated_{} {}

  ~NodeCutpointTracker() {}

  /*! \brief Calculate strides */
  void CalculateStrides(Eigen::MatrixXd& covariates, Eigen::VectorXd& residuals, SortedNodeSampleTracker* feature_node_sort_tracker, int32_t node_id, data_size_t node_begin, data_size_t node_end, int32_t feature_index);

  /*! \brief Split numeric / ordered categorical feature and update sort indices */
  void CalculateStridesNumeric(Eigen::MatrixXd& covariates, Eigen::VectorXd& residuals, SortedNodeSampleTracker* feature_node_sort_tracker, data_size_t node_begin, data_size_t node_end, int32_t feature_index);

  /*! \brief Split unordered categorical feature and update sort indices */
  void CalculateStridesCategorical(Eigen::MatrixXd& covariates, Eigen::VectorXd& residuals, SortedNodeSampleTracker* feature_node_sort_tracker, data_size_t node_begin, data_size_t node_end, int32_t feature_index);

  /*! \brief Number of potential cutpoints enumerated */
  int32_t NumCutpoints() {return node_stride_begin_.size();}

  /*! \brief Whether a cutpoint grid has been enumerated for a given node */
  bool NodeCutpointEvaluated(int32_t node_id) {
    return (std::find(nodes_enumerated_.begin(), nodes_enumerated_.end(), node_id) != nodes_enumerated_.end());
  }

  /*! \brief Node id of the node that has been most recently evaluated */
  int32_t CurrentNodeEvaluated() {return current_node_;}

  /*! \brief Vectors of node stride starting points and stride lengths */
  std::vector<data_size_t> node_stride_begin_;
  std::vector<data_size_t> node_stride_length_;
 
 private:
  int32_t cutpoint_grid_size_;
  std::vector<int32_t> nodes_enumerated_;
  int32_t current_node_;
};

} // namespace StochTree

#endif // STOCHTREE_CUTPOINT_CANDIDATES_H_
