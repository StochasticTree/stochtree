/*!
 * Copyright (c) 2023 stochtree authors
 */
#ifndef STOCHTREE_DEBUG_DATA_LOAD_FROM_FILE_H_
#define STOCHTREE_DEBUG_DATA_LOAD_FROM_FILE_H_

#include <stochtree/config.h>
#include <stochtree/data.h>

#include <vector>

namespace StochTree {

  void DebuggingDataLoadFromFile(){    
    // Define any config parameters that aren't defaults
    const char* params = "data=demo/bart_train/test.csv label_columns=0 num_trees=2 min_data_in_leaf=1 alpha=0.95 beta=1.25";
    auto param = Config::Str2Map(params);
    Config config;
    config.Set(param);

    // Extract name of training data file as a C-style const char
    const char* train_filename;
    if (config.data.size() > 0){
      train_filename = config.data.c_str();
      Log::Info("Loading train file: %s", config.data.c_str());
    } else {
      train_filename = nullptr;
      Log::Fatal("No training data filename provided to config");
    }

    // Define data loader
    DataLoader dataset_loader(config, 1, train_filename);

    // Load the data
    std::unique_ptr<Dataset> dataset;
    dataset.reset(dataset_loader.LoadFromFile(train_filename));
  }

} // namespace StochTree

#endif  // STOCHTREE_DEBUG_DATA_LOAD_FROM_FILE_H_
