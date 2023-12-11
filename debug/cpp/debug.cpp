/*! Copyright (c) 2023 stochtree authors
 * 
 *  The ParseTask function is a modified version of LoadConfig in src/main.cpp.
 */
#include "data_load_in_memory.h"
#include "data_load_from_file.h"

#include <stochtree/common.h>

#include <string>
#include <unordered_map>
#include <vector>

enum DebugProgram {
  kDataLoadInMemory,
  kDataLoadFromFile
};

void ParseTask(DebugProgram& debug_program, const char* kv) {
  std::vector<std::string> tmp_strs = StochTree::Common::Split(kv, '=');
  if (tmp_strs.size() == 2) {
    std::string key = StochTree::Common::RemoveQuotationSymbol(StochTree::Common::Trim(tmp_strs[0]));
    std::string value = StochTree::Common::RemoveQuotationSymbol(StochTree::Common::Trim(tmp_strs[1]));
    if (value == "DataLoadInMemory") {
      debug_program = DebugProgram::kDataLoadInMemory;
    } else if (value == "DataLoadFromFile") {
      debug_program = DebugProgram::kDataLoadFromFile;
    } else {
      StochTree::Log::Fatal("Unrecognized debug parameter %s", kv);
    }
  } else {
    StochTree::Log::Fatal("Invalid debug parameter %s", kv);
  }
}

int main(int argc, char** argv) {
  // Only parse the first character argument
  DebugProgram debug_program;
  ParseTask(debug_program, argv[1]);

  // Dispatch the correct debugging program
  if (debug_program == DebugProgram::kDataLoadInMemory) {
    StochTree::DebuggingDataLoadInMemory();
  } else if (debug_program == DebugProgram::kDataLoadFromFile) {
    StochTree::DebuggingDataLoadFromFile();
  }
}
