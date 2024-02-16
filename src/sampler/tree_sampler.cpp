/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include <stochtree/data.h>
#include <stochtree/ensemble.h>
#include <stochtree/prior.h>
#include <stochtree/sampler.h>

namespace StochTree {

void MCMCTreeSampler::AssignAllSamplesToRoot(int tree_num) {
  sample_node_mapper_->AssignAllSamplesToRoot(tree_num);
}

void GFRTreeSampler::AssignAllSamplesToRoot(int tree_num) {
  sample_node_mapper_->AssignAllSamplesToRoot(tree_num);
}

data_size_t MCMCTreeSampler::GetNodeId(int observation_num, int tree_num) {
  return sample_node_mapper_->GetNodeId(observation_num, tree_num);
}

data_size_t GFRTreeSampler::GetNodeId(int observation_num, int tree_num) {
  return sample_node_mapper_->GetNodeId(observation_num, tree_num);
}

}  // namespace StochTree
