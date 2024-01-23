/*!
 * Copyright (c) 2023 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_TREE_PRIOR_H_
#define STOCHTREE_TREE_PRIOR_H_

#include <stochtree/meta.h>

namespace StochTree {

/*! \brief Global parameters for gaussian homoskedastic constant leaf outcome model */
struct ClassicTreePrior {
  double alpha;
  double beta;
  data_size_t min_samples_in_leaf;
};

} // namespace StochTree

#endif // STOCHTREE_TREE_PRIOR_H_