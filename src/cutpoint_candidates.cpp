#include <stochtree/cutpoint_candidates.h>
#include <stochtree/meta.h>

#include <numeric>

namespace StochTree {

void FeatureCutpointGrid::CalculateStrides(Eigen::MatrixXd& covariates, Eigen::VectorXd& residuals, SortedNodeSampleTracker* feature_node_sort_tracker, int32_t node_id, data_size_t node_begin, data_size_t node_end, int32_t feature_index, std::vector<FeatureType>& feature_types) {
  // Reset the stride vectors
  node_stride_begin_.clear();
  node_stride_length_.clear();
  cutpoint_values_.clear();

  // Compute feature strides
  FeatureType feature_type = feature_types[feature_index];
  if (feature_type == FeatureType::kNumeric) {
    CalculateStridesNumeric(covariates, residuals, feature_node_sort_tracker, node_id, node_begin, node_end, feature_index);
  } else if (feature_type == FeatureType::kOrderedCategorical) {
    CalculateStridesOrderedCategorical(covariates, residuals, feature_node_sort_tracker, node_id, node_begin, node_end, feature_index);
  } else if (feature_type == FeatureType::kUnorderedCategorical) {
    CalculateStridesUnorderedCategorical(covariates, residuals, feature_node_sort_tracker, node_id, node_begin, node_end, feature_index);
  }
}

void FeatureCutpointGrid::CalculateStridesNumeric(Eigen::MatrixXd& covariates, Eigen::VectorXd& residuals, SortedNodeSampleTracker* feature_node_sort_tracker, int32_t node_id, data_size_t node_begin, data_size_t node_end, int32_t feature_index) {
  data_size_t node_size = node_end - node_begin;
  // Check if node has fewer observations than cutpoint_grid_size
  if (node_size <= cutpoint_grid_size_) {
    // In this case it is still possible to have "duplicates" if the values of 
    // a numeric feature are very close together which in practice will only 
    // occur when a categorical was imported incorrectly as numeric.
    // For this case, we run through the sorted data, determining the stride length 
    // of all unique values.
    EnumerateNumericCutpointsDeduplication(covariates, residuals, feature_node_sort_tracker, node_id, node_begin, node_end, node_size, feature_index);
  } else {
    // Here we must essentially "thin out" the possible cutpoints
    // First, we determine a step size that ensures there will be as 
    // many potential cutpoints as articulated in cutpoint_grid_size
    ScanNumericCutpoints(covariates, residuals, feature_node_sort_tracker, node_id, node_begin, node_end, node_size, feature_index);
  }
}

void FeatureCutpointGrid::CalculateStridesOrderedCategorical(Eigen::MatrixXd& covariates, Eigen::VectorXd& residuals, SortedNodeSampleTracker* feature_node_sort_tracker, int32_t node_id, data_size_t node_begin, data_size_t node_end, int32_t feature_index) {
  data_size_t node_size = node_end - node_begin;
  
  // Edge case 1: single observation
  double single_value;
  if (node_end - node_begin == 1) {
    node_stride_begin_.push_back(node_begin);
    node_stride_length_.push_back(1);
    single_value = covariates(feature_node_sort_tracker->SortIndex(node_begin, feature_index), feature_index);
    cutpoint_values_.push_back(static_cast<std::uint32_t>(single_value));
    return;
  }

  // Edge case 2: single category
  std::uint32_t first_val = static_cast<std::uint32_t>(covariates(feature_node_sort_tracker->SortIndex(node_begin, feature_index), feature_index));
  std::uint32_t last_val = static_cast<std::uint32_t>(covariates(feature_node_sort_tracker->SortIndex(node_end - 1, feature_index), feature_index));
  if (last_val == first_val) {
    node_stride_begin_.push_back(node_begin);
    node_stride_length_.push_back(node_size);
    single_value = covariates(feature_node_sort_tracker->SortIndex(node_begin, feature_index), feature_index);
    cutpoint_values_.push_back(static_cast<std::uint32_t>(single_value));
    return;
  }
  
  // Run the "regular" algorithm for computing categorical strides
  data_size_t stride_begin = node_begin;
  data_size_t stride_length = 0;
  data_size_t current_sort_ind, next_sort_ind;
  bool last_element;
  bool stride_complete;
  double current_val, next_val;
  for (data_size_t i = node_begin; i < node_end; i++){
    current_sort_ind = feature_node_sort_tracker->SortIndex(i, feature_index);
    current_val = covariates(current_sort_ind, feature_index);
    last_element = ((i == node_end - 1));

    // Increment stride length and bin_sum
    stride_length += 1;
    
    if (last_element) {
      // Update bin vectors
      node_stride_begin_.push_back(stride_begin);
      node_stride_length_.push_back(stride_length);
      cutpoint_values_.push_back(current_val);
    } else {
      next_sort_ind = feature_node_sort_tracker->SortIndex(i + 1, feature_index);
      next_val = covariates(next_sort_ind, feature_index);
      stride_complete = (std::fabs(next_val - current_val) > StochTree::kEpsilon);
      if (stride_complete) {
        // Update bin vectors
        node_stride_begin_.push_back(stride_begin);
        node_stride_length_.push_back(stride_length);
        cutpoint_values_.push_back(current_val);

        // Reset stride and bin tracker
        stride_begin += stride_length;
        stride_length = 0;
      }
    }
  }
}

void FeatureCutpointGrid::CalculateStridesUnorderedCategorical(Eigen::MatrixXd& covariates, Eigen::VectorXd& residuals, SortedNodeSampleTracker* feature_node_sort_tracker, int32_t node_id, data_size_t node_begin, data_size_t node_end, int32_t feature_index) {
  // TODO: refactor so that this initial code is shared between ordered and unordered categorical cutpoint calculation
  data_size_t node_size = node_end - node_begin;
  std::vector<double> bin_sums;
  
  // Edge case 1: single observation
  double single_value;
  if (node_end - node_begin == 1) {
    node_stride_begin_.push_back(node_begin);
    node_stride_length_.push_back(1);
    single_value = covariates(feature_node_sort_tracker->SortIndex(node_begin, feature_index), feature_index);
    cutpoint_values_.push_back(static_cast<std::uint32_t>(single_value));
    return;
  }

  // Edge case 2: single category
  std::uint32_t first_val = static_cast<std::uint32_t>(covariates(feature_node_sort_tracker->SortIndex(node_begin, feature_index), feature_index));
  std::uint32_t last_val = static_cast<std::uint32_t>(covariates(feature_node_sort_tracker->SortIndex(node_end - 1, feature_index), feature_index));
  if (last_val == first_val) {
    node_stride_begin_.push_back(node_begin);
    node_stride_length_.push_back(node_size);
    single_value = covariates(feature_node_sort_tracker->SortIndex(node_begin, feature_index), feature_index);
    cutpoint_values_.push_back(static_cast<std::uint32_t>(single_value));
    return;
  }
  
  // Run the "regular" algorithm for computing categorical strides
  data_size_t stride_begin = node_begin;
  data_size_t stride_length = 0;
  data_size_t current_sort_ind, next_sort_ind;
  bool last_element;
  bool stride_complete;
  double current_val, next_val;
  double current_outcome, next_outcome;
  double bin_sum = 0;
  for (data_size_t i = node_begin; i < node_end; i++){
    current_sort_ind = feature_node_sort_tracker->SortIndex(i, feature_index);
    current_val = covariates(current_sort_ind, feature_index);
    last_element = ((i == node_end - 1));
    
    // Increment stride length and bin_sum
    stride_length += 1;
    bin_sum += residuals(current_sort_ind);

    if (last_element) {
      // Update bin vectors
      node_stride_begin_.push_back(stride_begin);
      node_stride_length_.push_back(stride_length);
      cutpoint_values_.push_back(static_cast<std::uint32_t>(current_val));
      bin_sums.push_back(bin_sum);
    } else {
      next_sort_ind = feature_node_sort_tracker->SortIndex(i + 1, feature_index);
      next_val = covariates(next_sort_ind, feature_index);
      stride_complete = (static_cast<std::uint32_t>(next_val) != static_cast<std::uint32_t>(current_val));
      
      if (stride_complete) {
        // Update bin vectors
        node_stride_begin_.push_back(stride_begin);
        node_stride_length_.push_back(stride_length);
        cutpoint_values_.push_back(current_val);
        bin_sums.push_back(bin_sum);

        // Reset stride and bin tracker
        stride_begin += stride_length;
        stride_length = 0;
        bin_sum = 0;
      }
    }
  }

  // Now re-arrange the categories according to the average outcome as in Fisher (1958)
//  CHECK_EQ(residuals.cols(), 1);
  std::vector<double> bin_avgs(bin_sums.size());
  for (int i = 0; i < bin_sums.size(); i++) {
    bin_avgs[i] = bin_sums[i] / node_stride_length_[i];
  }
  std::vector<int> bin_sort_inds(bin_avgs.size());
  std::iota(bin_sort_inds.begin(), bin_sort_inds.end(), 0);
  auto comp_op = [&](size_t const &l, size_t const &r) { return std::less<double>{}(bin_avgs[l], bin_avgs[r]); };
  std::stable_sort(bin_sort_inds.begin(), bin_sort_inds.end(), comp_op);
  
  std::vector<data_size_t> temp_stride_begin_;
  std::vector<data_size_t> temp_stride_length_;
  std::vector<double> temp_cutpoint_value_;

  std::copy(node_stride_begin_.begin(), node_stride_begin_.end(), std::back_inserter(temp_stride_begin_));
  std::copy(node_stride_length_.begin(), node_stride_length_.end(), std::back_inserter(temp_stride_length_));
  std::copy(cutpoint_values_.begin(), cutpoint_values_.end(), std::back_inserter(temp_cutpoint_value_));

  for (int i = 0; i < node_stride_begin_.size(); i++) {
      node_stride_begin_[i] = temp_stride_begin_[bin_sort_inds[i]];
      node_stride_length_[i] = temp_stride_length_[bin_sort_inds[i]];
      cutpoint_values_[i] = temp_cutpoint_value_[bin_sort_inds[i]];
  }
}

void FeatureCutpointGrid::EnumerateNumericCutpointsDeduplication(Eigen::MatrixXd& covariates, Eigen::VectorXd& residuals, SortedNodeSampleTracker* feature_node_sort_tracker, int32_t node_id, data_size_t node_begin, data_size_t node_end, data_size_t node_size, int32_t feature_index) {
  // Edge case 1: single observation
  double single_value;
  if (node_end - node_begin == 1) {
    node_stride_begin_.push_back(node_begin);
    node_stride_length_.push_back(1);
    single_value = covariates(feature_node_sort_tracker->SortIndex(node_begin, feature_index), feature_index);
    cutpoint_values_.push_back(single_value);
    return;
  }

  // Edge case 2: single unique value
  double first_val = covariates(feature_node_sort_tracker->SortIndex(node_begin, feature_index), feature_index);
  double last_val = covariates(feature_node_sort_tracker->SortIndex(node_end - 1, feature_index), feature_index);
  if (std::fabs(last_val - first_val) < StochTree::kEpsilon) {
    node_stride_begin_.push_back(node_begin);
    node_stride_length_.push_back(node_size);
    cutpoint_values_.push_back(first_val);
    return;
  }
  
  // Run the "regular" algorithm for computing categorical strides
  data_size_t stride_begin = node_begin;
  data_size_t stride_length = 0;
  data_size_t current_sort_ind, next_sort_ind;
  bool last_element;
  bool stride_complete;
  double current_val, next_val;
  for (data_size_t i = node_begin; i < node_end; i++){
    current_sort_ind = feature_node_sort_tracker->SortIndex(i, feature_index);
    current_val = covariates(current_sort_ind, feature_index);
    last_element = ((i == node_end - 1));

    // Increment stride length
    stride_length += 1;
    
    if (last_element) {
      // Update bin vectors
      node_stride_begin_.push_back(stride_begin);
      node_stride_length_.push_back(stride_length);
      cutpoint_values_.push_back(current_val);
    } else {
      next_sort_ind = feature_node_sort_tracker->SortIndex(i + 1, feature_index);
      next_val = covariates(next_sort_ind, feature_index);
      stride_complete = (std::fabs(next_val - current_val) > StochTree::kEpsilon);
      if (stride_complete) {
        // Update bin vectors
        node_stride_begin_.push_back(stride_begin);
        node_stride_length_.push_back(stride_length);
        cutpoint_values_.push_back(current_val);

        // Reset stride and bin tracker
        stride_begin += stride_length;
        stride_length = 0;
      }
    }
  }
}

void FeatureCutpointGrid::ScanNumericCutpoints(Eigen::MatrixXd& covariates, Eigen::VectorXd& residuals, SortedNodeSampleTracker* feature_node_sort_tracker, int32_t node_id, data_size_t node_begin, data_size_t node_end, data_size_t node_size, int32_t feature_index) {
  // Edge case 1: single observation
  double single_value;
  if (node_end - node_begin == 1) {
    node_stride_begin_.push_back(node_begin);
    node_stride_length_.push_back(1);
    single_value = covariates(feature_node_sort_tracker->SortIndex(node_begin, feature_index), feature_index);
    cutpoint_values_.push_back(single_value);
    return;
  }

  // Edge case 2: single unique value
  double first_val = covariates(feature_node_sort_tracker->SortIndex(node_begin, feature_index), feature_index);
  double last_val = covariates(feature_node_sort_tracker->SortIndex(node_end - 1, feature_index), feature_index);
  if (std::fabs(last_val - first_val) < StochTree::kEpsilon) {
    node_stride_begin_.push_back(node_begin);
    node_stride_length_.push_back(node_size);
    cutpoint_values_.push_back(first_val);
    return;
  }
  
  // Run the "regular" algorithm for computing categorical strides
  data_size_t stride_begin = node_begin;
  data_size_t stride_length = 0;
  data_size_t current_sort_ind, next_sort_ind;
  bool last_element;
  bool stride_complete;
  bool bin_complete;
  double step_size = node_size / cutpoint_grid_size_;
  double current_val, next_val;
  for (data_size_t i = node_begin; i < node_end; i++){
    current_sort_ind = feature_node_sort_tracker->SortIndex(i, feature_index);
    current_val = covariates(current_sort_ind, feature_index);
    last_element = ((i == node_end - 1));

    // Increment stride length
    stride_length += 1;
    
    if (last_element) {
      // Update bin vectors
      node_stride_begin_.push_back(stride_begin);
      node_stride_length_.push_back(stride_length);
      cutpoint_values_.push_back(current_val);
    } else {
      next_sort_ind = feature_node_sort_tracker->SortIndex(i + 1, feature_index);
      next_val = covariates(next_sort_ind, feature_index);
      bin_complete = ((stride_length <= step_size) && ((stride_length + 1) > step_size));
      stride_complete = ((bin_complete) && (std::fabs(next_val - current_val) > StochTree::kEpsilon));
      if (stride_complete) {
        // Update bin vectors
        node_stride_begin_.push_back(stride_begin);
        node_stride_length_.push_back(stride_length);
        cutpoint_values_.push_back(current_val);

        // Reset stride and bin tracker
        stride_begin += stride_length;
        stride_length = 0;
      }
    }
  }
}

} // namespace StochTree
