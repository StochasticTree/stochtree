/*!
 * Copyright (c) 2022 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_TESTUTILS_H_
#define STOCHTREE_TESTUTILS_H_

#include <stochtree/random.h>
#include <stochtree/train_data.h>
#include <vector>

namespace StochTree {

namespace TestUtils {

/*!
  * Creates a Dataset from the internal repository examples.
  */
void LoadDatasetFromDemos(const char* filename, const char* config_str, std::unique_ptr<TrainData>& out);

} // namespace TestUtils

} // namespace StochTree

#endif  // STOCHTREE_TESTUTILS_H_
