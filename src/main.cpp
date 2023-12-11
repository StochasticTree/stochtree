/*!
 * Based on the design of the LightGBM command line program, released under the following terms
 * 
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include "../debug/cpp/simulate_data_supervised_learning.h"
#include <stochtree/config.h>
#include <stochtree/io.h>
#include <stochtree/interface.h>

#include <iostream>
#include <random>

namespace StochTree {

Config LoadConfig(int argc, char** argv) {
  Config config_;
  std::unordered_map<std::string, std::vector<std::string>> all_params;
  std::unordered_map<std::string, std::string> params;
  for (int i = 1; i < argc; ++i) {
    Config::KV2Map(&all_params, argv[i]);
  }
  // read parameters from config file
  bool config_file_ok = true;
  if (all_params.count("config") > 0) {
    TextReader<size_t> config_reader(all_params["config"][0].c_str(), false);
    config_reader.ReadAllLines();
    if (!config_reader.Lines().empty()) {
      for (auto& line : config_reader.Lines()) {
        // remove str after "#"
        if (line.size() > 0 && std::string::npos != line.find_first_of("#")) {
          line.erase(line.find_first_of("#"));
        }
        line = Common::Trim(line);
        if (line.size() == 0) {
          continue;
        }
        Config::KV2Map(&all_params, line.c_str());
      }
    } else {
      config_file_ok = false;
    }
  }
  Config::SetVerbosity(all_params);
  // de-duplicate params
  Config::KeepFirstValues(all_params, &params);
  if (!config_file_ok) {
    Log::Warning("Config file %s doesn't exist, will ignore", params["config"].c_str());
  }
  ParameterAlias::KeyAliasTransform(&params);
  config_.Set(params);
  Log::Info("Finished loading parameters");
  return config_;
}

} // namespace StochTree

int main(int argc, char** argv) {
  bool success = false;
  try {
    // Load the config and interface
    StochTree::Config config_ = StochTree::LoadConfig(argc, argv);
    StochTree::StochTreeInterface interface(config_);
    
    // Load training data
    if (config_.data.size() > 0){
      StochTree::Log::Info("Creating dataset from file");
      interface.LoadTrainDataFromFile();
    } else {
      // Override config params to specify 0 as outcome column
      const char* params = "outcome_columns=0";
      auto param = StochTree::Config::Str2Map(params);
      config_.Set(param);

      // Generate simulated dataset and load it
      StochTree::Log::Info("Generating a simulated dataset");
      StochTree::data_size_t n = 5000;
      int p = 10;
      std::vector<double> output_vector = StochTree::SimulateTabularDataset(n, p, 1234);
      interface.LoadTrainDataFromMemory(output_vector.data(), p+1, n, true);
    }

    // Sample from the model
    interface.SampleModel();

    // Set the exit condition to indicate the program ran without exceptions
    success = true;
  }
  catch (const std::exception& ex) {
    std::cerr << "Met Exceptions:" << std::endl;
    std::cerr << ex.what() << std::endl;
  }
  catch (const std::string& ex) {
    std::cerr << "Met Exceptions:" << std::endl;
    std::cerr << ex << std::endl;
  }
  catch (...) {
    std::cerr << "Unknown Exceptions" << std::endl;
  }

  if (!success) {
    exit(-1);
  }
}
