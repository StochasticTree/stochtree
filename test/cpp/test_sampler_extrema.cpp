#include <gtest/gtest.h>
#include <stochtree/tree_sampler.h>

#include <algorithm>
#include <vector>

namespace {

// One numeric feature and one root node; no sampling or random seed is needed.
struct RootData {
  StochTree::ForestDataset dataset;
  std::vector<StochTree::FeatureType> types{StochTree::FeatureType::kNumeric};

  explicit RootData(std::vector<double> values) {
    dataset.AddCovariates(values.data(), values.size(), 1, true);
  }
};

}  // namespace

TEST(SamplerExtrema, RangeIsIndependentOfOrderAndSign) {
  for (auto values : std::vector<std::vector<double>>{
           {1, 2, 3}, {-3, -2, -1}, {-1, 0, 1}, {0, 0}, {-2, -2}, {7}}) {
    const double expected_min = values.front();
    const double expected_max = values.back();
    do {
      SCOPED_TRACE(::testing::PrintToString(values));
      RootData data(values);
      StochTree::ForestTracker tracker(data.dataset.GetCovariates(), data.types, 1, values.size());
      double lower, upper;
      StochTree::VarSplitRange(tracker, data.dataset, 0, 0, 0, lower, upper);
      EXPECT_DOUBLE_EQ(lower, expected_min);
      EXPECT_DOUBLE_EQ(upper, expected_max);
    } while (std::next_permutation(values.begin(), values.end()));
  }
}

TEST(SamplerExtrema, RecognizesConstantAndVaryingNodes) {
  for (auto values : std::vector<std::vector<double>>{
           {1, 2, 3}, {-3, -2, -1}, {0, 0}, {-2, -2}, {2, 2}, {7}}) {
    const bool expected = values.front() != values.back();
    do {
      SCOPED_TRACE(::testing::PrintToString(values));
      RootData data(values);
      StochTree::ForestTracker tracker(data.dataset.GetCovariates(), data.types, 1, values.size());
      EXPECT_EQ(StochTree::NodeNonConstant(data.dataset, tracker, 0, 0), expected);
    } while (std::next_permutation(values.begin(), values.end()));
  }
}

TEST(SamplerExtrema, RecognizesConstantAndVaryingChildren) {
  for (auto values : std::vector<std::vector<double>>{
           {1, 2, 3, 4}, {-4, -3, -2, -1}, {-2, -2, -1, -1}}) {
    const double threshold = (values[1] + values[2]) / 2;
    const bool expected = values[0] != values[1] && values[2] != values[3];
    do {
      SCOPED_TRACE(::testing::PrintToString(values));
      RootData data(values);
      StochTree::ForestTracker tracker(data.dataset.GetCovariates(), data.types, 1, values.size());
      StochTree::TreeSplit split(threshold);
      EXPECT_EQ(StochTree::NodesNonConstantAfterSplit(data.dataset, tracker, split, 0, 0, 0), expected);
    } while (std::next_permutation(values.begin(), values.end()));
  }
}
