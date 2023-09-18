/*!
 * Copyright (c) 2022 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>
#include <testutils.h>
#include <stochtree/config.h>
#include <stochtree/log.h>
#include <stochtree/random.h>
#include <stochtree/train_data.h>
#include <memory>
#include <string>
#include <thread>
#include <utility>

namespace StochTree {

namespace TestUtils{

  void LoadDatasetFromDemos(const char* filename, const char* config_str, std::unique_ptr<TrainData>& out) {
    std::string fullPath("demo/");
    fullPath += filename;
    Log::Info("Debug sample data path: %s", fullPath.c_str());

    auto param = Config::Str2Map(config_str);
    Config config;
    config.Set(param);
    // OMP_SET_NUM_THREADS(config.num_threads);
    TrainDataLoader loader(config, 1, fullPath.c_str());
    out.reset(loader.LoadFromFile(fullPath.c_str()));
  }

}

}  // namespace StochTree
