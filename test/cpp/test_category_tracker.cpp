#include <gtest/gtest.h>
#include <testutils.h>
#include <stochtree/category_tracker.h>
#include <stochtree/data.h>
#include <stochtree/log.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/tree.h>
#include <iostream>
#include <memory>
#include <vector>

TEST(CategorySampleTracker, BasicOperations) {
  // Create a vector of categorical data
  std::vector<int32_t> category_data {
    3, 4, 3, 2, 2, 4, 3, 3, 3, 4, 3, 4
  };

  // Create a CategorySamplerTracker
  StochTree::CategorySampleTracker category_tracker = StochTree::CategorySampleTracker(category_data);

  // Extract the label map
  std::map<int32_t, int32_t> label_map = category_tracker.GetLabelMap();
  std::map<int32_t, int32_t> expected_label_map {{2, 0}, {3, 1}, {4, 2}};

  // Check that the map was constructed as expected
  ASSERT_EQ(label_map[2], 0);
  ASSERT_EQ(label_map[3], 1);
  ASSERT_EQ(label_map[4], 2);
  ASSERT_EQ(label_map, expected_label_map);
}
