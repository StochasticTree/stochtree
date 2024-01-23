/*!
 * Serialization methods draw primarily from a combination of serialization code in lightGBM and xgboost
 * 
 * LightGBM
 * ========
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 * 
 * xgboost
 * =======
 * Copyright 2015~2023 by XGBoost Contributors
 */
#include <stochtree/ensemble.h>
#include <stochtree/model_draw.h>
#include <stochtree/tree.h>

#include <algorithm>
#include <deque>
#include <random>
#include <unordered_map>

namespace StochTree {

// void XBARTGaussianRegressionModelDraw::SaveModelDrawToFile(const char* filename) {
//     /*! \brief File to write models */
//     auto writer = VirtualFileWriter::Make(filename);
//     if (!writer->Init()) {
//       Log::Fatal("Model file %s is not available for writes", filename);
//     }
//     std::string str_to_write = SaveModelDrawToString();
//     auto size = writer->Write(str_to_write.c_str(), str_to_write.size());
//     // return size > 0;
//   }

// std::string XBARTGaussianRegressionModelDraw::SaveModelDrawToString() const {
//   std::stringstream ss;
//   Common::C_stringstream(ss);

//   // output model type
//   ss << SubModelName() << '\n';

//   // Store the global parameters
//   ss << "sigma_squared=" << sigma_sq_ << '\n';
//   ss << "tau=" << tau_ << '\n';

//   // store the trees
//   int num_trees = tree_ensemble_->NumTrees();
//   std::vector<std::string> tree_strs(num_trees);
//   std::vector<size_t> tree_sizes(num_trees);
//   // output tree models
//   #pragma omp parallel for schedule(static)
//   for (int i = 0; i < num_trees; ++i) {
//     Tree* tree = tree_ensemble_->GetTree(i);
//     const int idx = i;
//     tree_strs[idx] = "Tree=" + std::to_string(idx) + '\n';
//     tree_strs[idx] += tree->ToJSON() + '\n';
//     tree_sizes[idx] = tree->NumNodes();
//   }

//   ss << "tree_sizes=" << CommonC::Join(tree_sizes, " ") << '\n';
//   ss << '\n';

//   for (int i = 0; i < num_trees; ++i) {
//     ss << tree_strs[i];
//     tree_strs[i].clear();
//   }
//   ss << "end of trees" << "\n";
//   return ss.str();
// }

// void BARTGaussianRegressionModelDraw::SaveModelDrawToFile(const char* filename) {
//     /*! \brief File to write models */
//     auto writer = VirtualFileWriter::Make(filename);
//     if (!writer->Init()) {
//       Log::Fatal("Model file %s is not available for writes", filename);
//     }
//     std::string str_to_write = SaveModelDrawToString();
//     auto size = writer->Write(str_to_write.c_str(), str_to_write.size());
//     // return size > 0;
//   }

// std::string BARTGaussianRegressionModelDraw::SaveModelDrawToString() const {
//   std::stringstream ss;
//   Common::C_stringstream(ss);

//   // output model type
//   ss << SubModelName() << '\n';

//   // Store the global parameters
//   ss << "sigma_squared=" << sigma_sq_ << '\n';

//   // store the trees
//   int num_trees = tree_ensemble_->NumTrees();
//   std::vector<std::string> tree_strs(num_trees);
//   std::vector<size_t> tree_sizes(num_trees);
//   // output tree models
//   #pragma omp parallel for schedule(static)
//   for (int i = 0; i < num_trees; ++i) {
//     Tree* tree = tree_ensemble_->GetTree(i);
//     const int idx = i;
//     tree_strs[idx] = "Tree=" + std::to_string(idx) + '\n';
//     tree_strs[idx] += tree->ToJSON() + '\n';
//     tree_sizes[idx] = tree->NumNodes();
//   }

//   ss << "tree_sizes=" << CommonC::Join(tree_sizes, " ") << '\n';
//   ss << '\n';

//   for (int i = 0; i < num_trees; ++i) {
//     ss << tree_strs[i];
//     tree_strs[i].clear();
//   }
//   ss << "end of trees" << "\n";
//   return ss.str();
// }

} // namespace StochTree
