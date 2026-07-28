/*!
 * This file combines simplified / modified versions of the CSV/TXT import/export code in LightGBM, specifically:
 *   array_args.h
 *   binary_writer.h
 *   common.h
 *   file_io.h
 *   parser.h
 *   pipeline_reader.h
 *   text_reader.h
 * 
 * LightGBM is MIT licensed and released with the following copyright header 
 * (with different copyright years in different files):
 *
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_IO_H_
#define STOCHTREE_IO_H_

#include <stochtree/common.h>
#include <stochtree/export.h>
#include <stochtree/meta.h>
#include <stochtree/log.h>
#include <stochtree/random.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// using json = nlohmann::json;

namespace StochTree {

const size_t kGbs = size_t(1024) * 1024 * 1024;

/*!
* \brief Contains some operation for an array, e.g. ArgMax, TopK.
*/
template<typename VAL_T>
class ArrayArgs {
 public:
  inline static size_t ArgMax(const std::vector<VAL_T>& array) {
    if (array.empty()) {
      return 0;
    }
    // if (array.size() > 1024) {
    //   return ArgMaxMT(array);
    // } else {
    size_t arg_max = 0;
    for (size_t i = 1; i < array.size(); ++i) {
      if (array[i] > array[arg_max]) {
        arg_max = i;
      }
    }
    return arg_max;
    // }
  }

  inline static size_t ArgMin(const std::vector<VAL_T>& array) {
    if (array.empty()) {
      return 0;
    }
    size_t arg_min = 0;
    for (size_t i = 1; i < array.size(); ++i) {
      if (array[i] < array[arg_min]) {
        arg_min = i;
      }
    }
    return arg_min;
  }

  inline static size_t ArgMax(const VAL_T* array, size_t n) {
    if (n <= 0) {
      return 0;
    }
    size_t arg_max = 0;
    for (size_t i = 1; i < n; ++i) {
      if (array[i] > array[arg_max]) {
        arg_max = i;
      }
    }
    return arg_max;
  }

  inline static size_t ArgMin(const VAL_T* array, size_t n) {
    if (n <= 0) {
      return 0;
    }
    size_t arg_min = 0;
    for (size_t i = 1; i < n; ++i) {
      if (array[i] < array[arg_min]) {
        arg_min = i;
      }
    }
    return arg_min;
  }

  inline static void Partition(std::vector<VAL_T>* arr, int start, int end, int* l, int* r) {
    int i = start - 1;
    int j = end - 1;
    int p = i;
    int q = j;
    if (start >= end - 1) {
      *l = start - 1;
      *r = end;
      return;
    }
    std::vector<VAL_T>& ref = *arr;
    VAL_T v = ref[end - 1];
    for (;;) {
      while (ref[++i] > v) {}
      while (v > ref[--j]) { if (j == start) { break; } }
      if (i >= j) { break; }
      std::swap(ref[i], ref[j]);
      if (ref[i] == v) { p++; std::swap(ref[p], ref[i]); }
      if (v == ref[j]) { q--; std::swap(ref[j], ref[q]); }
    }
    std::swap(ref[i], ref[end - 1]);
    j = i - 1;
    i = i + 1;
    for (int k = start; k <= p; k++, j--) { std::swap(ref[k], ref[j]); }
    for (int k = end - 2; k >= q; k--, i++) { std::swap(ref[i], ref[k]); }
    *l = j;
    *r = i;
  }

  // Note: k refer to index here. e.g. k=0 means get the max number.
  inline static int ArgMaxAtK(std::vector<VAL_T>* arr, int start, int end, int k) {
    if (start >= end - 1) {
      return start;
    }
    int l = start;
    int r = end - 1;
    Partition(arr, start, end, &l, &r);
    // if find or all elements are the same.
    if ((k > l && k < r) || (l == start - 1 && r == end - 1)) {
      return k;
    } else if (k <= l) {
      return ArgMaxAtK(arr, start, l + 1, k);
    } else {
      return ArgMaxAtK(arr, r, end, k);
    }
  }

  // Note: k is 1-based here. e.g. k=3 means get the top-3 numbers.
  inline static void MaxK(const std::vector<VAL_T>& array, int k, std::vector<VAL_T>* out) {
    out->clear();
    if (k <= 0) {
      return;
    }
    for (auto val : array) {
      out->push_back(val);
    }
    if (static_cast<size_t>(k) >= array.size()) {
      return;
    }
    ArgMaxAtK(out, 0, static_cast<int>(out->size()), k - 1);
    out->erase(out->begin() + k, out->end());
  }

  inline static void Assign(std::vector<VAL_T>* array, VAL_T t, size_t n) {
    array->resize(n);
    for (size_t i = 0; i < array->size(); ++i) {
      (*array)[i] = t;
    }
  }

  inline static bool CheckAllZero(const std::vector<VAL_T>& array) {
    for (size_t i = 0; i < array.size(); ++i) {
      if (array[i] != VAL_T(0)) {
        return false;
      }
    }
    return true;
  }

  inline static bool CheckAll(const std::vector<VAL_T>& array, VAL_T t) {
    for (size_t i = 0; i < array.size(); ++i) {
      if (array[i] != t) {
        return false;
      }
    }
    return true;
  }
};

/*!
  * \brief An interface for serializing binary data to a buffer
  */
struct BinaryWriter {
  /*!
    * \brief Append data to this binary target
    * \param data Buffer to write from
    * \param bytes Number of bytes to write from buffer
    * \return Number of bytes written
    */
  virtual size_t Write(const void* data, size_t bytes) = 0;

  /*!
    * \brief Append data to this binary target aligned on a given byte size boundary
    * \param data Buffer to write from
    * \param bytes Number of bytes to write from buffer
    * \param alignment The size of bytes to align to in whole increments
    * \return Number of bytes written
    */
  size_t AlignedWrite(const void* data, size_t bytes, size_t alignment = 8) {
    auto ret = Write(data, bytes);
    if (bytes % alignment != 0) {
      size_t padding = AlignedSize(bytes, alignment) - bytes;
      std::vector<char> tmp(padding, 0);
      ret += Write(tmp.data(), padding);
    }
    return ret;
  }

  /*!
    * \brief The aligned size of a buffer length.
    * \param bytes The number of bytes in a buffer
    * \param alignment The size of bytes to align to in whole increments
    * \return Number of aligned bytes
    */
  static size_t AlignedSize(size_t bytes, size_t alignment = 8) {
    if (bytes % alignment == 0) {
      return bytes;
    } else {
      return bytes / alignment * alignment + alignment;
    }
  }
};

/*!
 * \brief An interface for writing files from buffers
 */
struct VirtualFileWriter : BinaryWriter {
  virtual ~VirtualFileWriter() {}

  /*!
   * \brief Initialize the writer
   * \return True when the file is available for writes
   */
  virtual bool Init() = 0;

  /*!
   * \brief Create appropriate writer for filename
   * \param filename Filename of the data
   * \return File writer instance
   */
  static std::unique_ptr<VirtualFileWriter> Make(const std::string& filename);

  /*!
   * \brief Check filename existence
   * \param filename Filename of the data
   * \return True when the file exists
   */
  static bool Exists(const std::string& filename);
};

/**
 * \brief An interface for reading files into buffers
 */
struct VirtualFileReader {
  /*!
   * \brief Constructor
   * \param filename Filename of the data
   */
  virtual ~VirtualFileReader() {}
  /*!
   * \brief Initialize the reader
   * \return True when the file is available for read
   */
  virtual bool Init() = 0;
  /*!
   * \brief Read data into buffer
   * \param buffer Buffer to read data into
   * \param bytes Number of bytes to read
   * \return Number of bytes read
   */
  virtual size_t Read(void* buffer, size_t bytes) const = 0;
  /*!
   * \brief Create appropriate reader for filename
   * \param filename Filename of the data
   * \return File reader instance
   */
  static std::unique_ptr<VirtualFileReader> Make(const std::string& filename);
};

/*! \brief Interface for Parser */
class Parser {
 public:
  typedef const char* (*AtofFunc)(const char* p, double* out);

  /*! \brief Default constructor */
  Parser() {}

  /*!
  * \brief Constructor for customized parser. The constructor accepts content not path because need to save/load the config along with model string
  */
  explicit Parser(std::string) {}

  /*! \brief virtual destructor */
  virtual ~Parser() {}

  /*!
  * \brief Parse one line with label
  * \param str One line record, string format, should end with '\0'
  * \param out_features Output columns, store in (column_idx, values)
  */
  virtual void ParseOneLine(const char* str,
                            std::vector<std::pair<int, double>>* out_features) const = 0;

  virtual int NumFeatures() const = 0;

  /*!
  * \brief Create an object of parser, will auto choose the format depend on file
  * \param filename One Filename of data
  * \param header whether input file contains header
  * \param num_features Pass num_features of this data file if you know, <=0 means don't know
  * \param precise_float_parser using precise floating point number parsing if true
  * \return Object of parser
  */
  static Parser* CreateParser(const char* filename, bool header, int num_features, bool precise_float_parser);
};

class CSVParser: public Parser {
 public:
  explicit CSVParser(int total_columns, AtofFunc atof)
    :total_columns_(total_columns), atof_(atof) {
  }

  inline void ParseOneLine(const char* str,
    std::vector<std::pair<int, double>>* out_features) const override {
    int idx = 0;
    double val = 0.0f;
    int offset = 0;
    while (*str != '\0') {
      // Log::Info(str);
      str = atof_(str, &val);
      // Log::Info(("Setting feature value: " + std::to_string(val)).c_str());
      out_features->emplace_back(idx + offset, val);
      ++idx;
      if (*str == ',') {
        ++str;
      } else if (*str != '\0') {
        Log::Fatal("Input format error when parsing as CSV");
      }
    }
  }

  inline int NumFeatures() const override {
    // return total_columns_ - (label_idx_ >= 0);
    return total_columns_;
  }

 private:
  int total_columns_ = -1;
  AtofFunc atof_;
};

/*!
* \brief A pipeline file reader, use 2 threads, one read block from file, the other process the block
*/
class PipelineReader {
 public:
  /*!
  * \brief Read data from a file, use pipeline methods
  * \param filename Filename of data
  * \process_fun Process function
  */
  static size_t Read(const char* filename, int skip_bytes, const std::function<size_t(const char*, size_t)>& process_fun) {
    auto reader = VirtualFileReader::Make(filename);
    if (!reader->Init()) {
      return 0;
    }
    size_t cnt = 0;
    const size_t buffer_size =  16 * 1024 * 1024;
    // buffer used for the process_fun
    auto buffer_process = std::vector<char>(buffer_size);
    // buffer used for the file reading
    auto buffer_read = std::vector<char>(buffer_size);
    size_t read_cnt = 0;
    if (skip_bytes > 0) {
      // skip first k bytes
      read_cnt = reader->Read(buffer_process.data(), skip_bytes);
    }
    // read first block
    read_cnt = reader->Read(buffer_process.data(), buffer_size);

    size_t last_read_cnt = 0;
    while (read_cnt > 0) {
      // start read thread
      std::thread read_worker = std::thread(
        [=, &last_read_cnt, &reader, &buffer_read] {
        last_read_cnt = reader->Read(buffer_read.data(), buffer_size);
      });
      // start process
      cnt += process_fun(buffer_process.data(), read_cnt);
      // wait for read thread
      read_worker.join();
      // exchange the buffer
      std::swap(buffer_process, buffer_read);
      read_cnt = last_read_cnt;
    }
    return cnt;
  }
};

/*!
* \brief Read text data from file
*/
template<typename INDEX_T>
class TextReader {
 public:
  /*!
  * \brief Constructor
  * \param filename Filename of data
  * \param is_skip_first_line True if need to skip header
  */
  TextReader(const char* filename, bool is_skip_first_line, size_t progress_interval_bytes = SIZE_MAX):
    filename_(filename), is_skip_first_line_(is_skip_first_line), read_progress_interval_bytes_(progress_interval_bytes) {
    if (is_skip_first_line_) {
      auto reader = VirtualFileReader::Make(filename);
      if (!reader->Init()) {
        Log::Fatal("Could not open %s", filename);
      }
      std::stringstream str_buf;
      char read_c;
      size_t nread = reader->Read(&read_c, 1);
      while (nread == 1) {
        if (read_c == '\n' || read_c == '\r') {
          break;
        }
        str_buf << read_c;
        ++skip_bytes_;
        nread = reader->Read(&read_c, 1);
      }
      if (read_c == '\r') {
        reader->Read(&read_c, 1);
        ++skip_bytes_;
      }
      if (read_c == '\n') {
        reader->Read(&read_c, 1);
        ++skip_bytes_;
      }
      first_line_ = str_buf.str();
      Log::Debug("Skipped header \"%s\" in file %s", first_line_.c_str(), filename_);
    }
  }
  /*!
  * \brief Destructor
  */
  ~TextReader() {
    Clear();
  }
  /*!
  * \brief Clear cached data
  */
  inline void Clear() {
    lines_.clear();
    lines_.shrink_to_fit();
  }
  /*!
  * \brief return first line of data
  */
  inline std::string first_line() {
    return first_line_;
  }
  /*!
  * \brief Get text data that read from file
  * \return Text data, store in std::vector by line
  */
  inline std::vector<std::string>& Lines() { return lines_; }
  /*!
  * \brief Get joined text data that read from file
  * \return Text data, store in std::string, joined all lines by delimiter
  */
  inline std::string JoinedLines(std::string delimiter = "\n") {
    std::stringstream ss;
    for (auto line : lines_) {
      ss << line << delimiter;
    }
    return ss.str();
  }

  INDEX_T ReadAllAndProcess(const std::function<void(INDEX_T, const char*, size_t)>& process_fun) {
    last_line_ = "";
    INDEX_T total_cnt = 0;
    size_t bytes_read = 0;
    PipelineReader::Read(filename_, skip_bytes_,
        [&process_fun, &bytes_read, &total_cnt, this]
    (const char* buffer_process, size_t read_cnt) {
      size_t cnt = 0;
      size_t i = 0;
      size_t last_i = 0;
      // skip the break between \r and \n
      if (last_line_.size() == 0 && buffer_process[0] == '\n') {
        i = 1;
        last_i = i;
      }
      while (i < read_cnt) {
        if (buffer_process[i] == '\n' || buffer_process[i] == '\r') {
          if (last_line_.size() > 0) {
            last_line_.append(buffer_process + last_i, i - last_i);
            process_fun(total_cnt, last_line_.c_str(), last_line_.size());
            last_line_ = "";
          } else {
            process_fun(total_cnt, buffer_process + last_i, i - last_i);
          }
          ++cnt;
          ++i;
          ++total_cnt;
          // skip end of line
          while ((buffer_process[i] == '\n' || buffer_process[i] == '\r') && i < read_cnt) { ++i; }
          last_i = i;
        } else {
          ++i;
        }
      }
      if (last_i != read_cnt) {
        last_line_.append(buffer_process + last_i, read_cnt - last_i);
      }

      size_t prev_bytes_read = bytes_read;
      bytes_read += read_cnt;
      if (prev_bytes_read / read_progress_interval_bytes_ < bytes_read / read_progress_interval_bytes_) {
        Log::Debug("Read %.1f GBs from %s.", 1.0 * bytes_read / kGbs, filename_);
      }

      return cnt;
    });
    // if last line of file doesn't contain end of line
    if (last_line_.size() > 0) {
      Log::Info("Warning: last line of %s has no end of line, still using this line", filename_);
      process_fun(total_cnt, last_line_.c_str(), last_line_.size());
      ++total_cnt;
      last_line_ = "";
    }
    return total_cnt;
  }

  /*!
  * \brief Read all text data from file in memory
  * \return number of lines of text data
  */
  INDEX_T ReadAllLines() {
    return ReadAllAndProcess(
      [=](INDEX_T, const char* buffer, size_t size) {
      lines_.emplace_back(buffer, size);
    });
  }

  std::vector<char> ReadContent(size_t* out_len) {
    std::vector<char> ret;
    *out_len = 0;
    auto reader = VirtualFileReader::Make(filename_);
    if (!reader->Init()) {
      return ret;
    }
    const size_t buffer_size = 16 * 1024 * 1024;
    auto buffer_read = std::vector<char>(buffer_size);
    size_t read_cnt = 0;
    do {
      read_cnt = reader->Read(buffer_read.data(), buffer_size);
      ret.insert(ret.end(), buffer_read.begin(), buffer_read.begin() + read_cnt);
      *out_len += read_cnt;
    } while (read_cnt > 0);
    return ret;
  }

  INDEX_T SampleFromFile(Random* random, INDEX_T sample_cnt, std::vector<std::string>* out_sampled_data) {
    INDEX_T cur_sample_cnt = 0;
    return ReadAllAndProcess([=, &random, &cur_sample_cnt,
                              &out_sampled_data]
    (INDEX_T line_idx, const char* buffer, size_t size) {
      if (cur_sample_cnt < sample_cnt) {
        out_sampled_data->emplace_back(buffer, size);
        ++cur_sample_cnt;
      } else {
        const size_t idx = static_cast<size_t>(random->NextInt(0, static_cast<int>(line_idx + 1)));
        if (idx < static_cast<size_t>(sample_cnt)) {
          out_sampled_data->operator[](idx) = std::string(buffer, size);
        }
      }
    });
  }
  /*!
  * \brief Read part of text data from file in memory, use filter_fun to filter data
  * \param filter_fun Function that perform data filter
  * \param out_used_data_indices Store line indices that read text data
  * \return The number of total data
  */
  INDEX_T ReadAndFilterLines(const std::function<bool(INDEX_T)>& filter_fun, std::vector<INDEX_T>* out_used_data_indices) {
    out_used_data_indices->clear();
    INDEX_T total_cnt = ReadAllAndProcess(
        [&filter_fun, &out_used_data_indices, this]
    (INDEX_T line_idx , const char* buffer, size_t size) {
      bool is_used = filter_fun(line_idx);
      if (is_used) {
        out_used_data_indices->push_back(line_idx);
        lines_.emplace_back(buffer, size);
      }
    });
    return total_cnt;
  }

  INDEX_T SampleAndFilterFromFile(const std::function<bool(INDEX_T)>& filter_fun, std::vector<INDEX_T>* out_used_data_indices,
    Random* random, INDEX_T sample_cnt, std::vector<std::string>* out_sampled_data) {
    INDEX_T cur_sample_cnt = 0;
    out_used_data_indices->clear();
    INDEX_T total_cnt = ReadAllAndProcess(
        [=, &filter_fun, &out_used_data_indices, &random, &cur_sample_cnt,
         &out_sampled_data]
    (INDEX_T line_idx, const char* buffer, size_t size) {
      bool is_used = filter_fun(line_idx);
      if (is_used) {
        out_used_data_indices->push_back(line_idx);
        if (cur_sample_cnt < sample_cnt) {
          out_sampled_data->emplace_back(buffer, size);
          ++cur_sample_cnt;
        } else {
          const size_t idx = static_cast<size_t>(random->NextInt(0, static_cast<int>(out_used_data_indices->size())));
          if (idx < static_cast<size_t>(sample_cnt)) {
            out_sampled_data->operator[](idx) = std::string(buffer, size);
          }
        }
      }
    });
    return total_cnt;
  }

  INDEX_T CountLine() {
    return ReadAllAndProcess(
      [=](INDEX_T, const char*, size_t) {
    });
  }

  INDEX_T ReadAllAndProcessParallelWithFilter(const std::function<void(INDEX_T, const std::vector<std::string>&)>& process_fun, const std::function<bool(INDEX_T, INDEX_T)>& filter_fun) {
    last_line_ = "";
    INDEX_T total_cnt = 0;
    size_t bytes_read = 0;
    INDEX_T used_cnt = 0;
    PipelineReader::Read(filename_, skip_bytes_,
        [&process_fun, &filter_fun, &total_cnt, &bytes_read, &used_cnt, this]
    (const char* buffer_process, size_t read_cnt) {
      size_t cnt = 0;
      size_t i = 0;
      size_t last_i = 0;
      INDEX_T start_idx = used_cnt;
      // skip the break between \r and \n
      if (last_line_.size() == 0 && buffer_process[0] == '\n') {
        i = 1;
        last_i = i;
      }
      while (i < read_cnt) {
        if (buffer_process[i] == '\n' || buffer_process[i] == '\r') {
          if (last_line_.size() > 0) {
            last_line_.append(buffer_process + last_i, i - last_i);
            if (filter_fun(used_cnt, total_cnt)) {
              lines_.push_back(last_line_);
              ++used_cnt;
            }
            last_line_ = "";
          } else {
            if (filter_fun(used_cnt, total_cnt)) {
              lines_.emplace_back(buffer_process + last_i, i - last_i);
              ++used_cnt;
            }
          }
          ++cnt;
          ++i;
          ++total_cnt;
          // skip end of line
          while ((buffer_process[i] == '\n' || buffer_process[i] == '\r') && i < read_cnt) { ++i; }
          last_i = i;
        } else {
          ++i;
        }
      }
      process_fun(start_idx, lines_);
      lines_.clear();
      if (last_i != read_cnt) {
        last_line_.append(buffer_process + last_i, read_cnt - last_i);
      }

      size_t prev_bytes_read = bytes_read;
      bytes_read += read_cnt;
      if (prev_bytes_read / read_progress_interval_bytes_ < bytes_read / read_progress_interval_bytes_) {
        Log::Debug("Read %.1f GBs from %s.", 1.0 * bytes_read / kGbs, filename_);
      }

      return cnt;
    });
    // if last line of file doesn't contain end of line
    if (last_line_.size() > 0) {
      Log::Info("Warning: last line of %s has no end of line, still using this line", filename_);
      if (filter_fun(used_cnt, total_cnt)) {
        lines_.push_back(last_line_);
        process_fun(used_cnt, lines_);
      }
      lines_.clear();
      ++total_cnt;
      ++used_cnt;
      last_line_ = "";
    }
    return total_cnt;
  }

  INDEX_T ReadAllAndProcessParallel(const std::function<void(INDEX_T, const std::vector<std::string>&)>& process_fun) {
    return ReadAllAndProcessParallelWithFilter(process_fun, [](INDEX_T, INDEX_T) { return true; });
  }

  INDEX_T ReadPartAndProcessParallel(const std::vector<INDEX_T>& used_data_indices, const std::function<void(INDEX_T, const std::vector<std::string>&)>& process_fun) {
    return ReadAllAndProcessParallelWithFilter(process_fun,
      [&used_data_indices](INDEX_T used_cnt, INDEX_T total_cnt) {
      if (static_cast<size_t>(used_cnt) < used_data_indices.size() && total_cnt == used_data_indices[used_cnt]) {
        return true;
      } else {
        return false;
      }
    });
  }

 private:
  /*! \brief Filename of text data */
  const char* filename_;
  /*! \brief Cache the read text data */
  std::vector<std::string> lines_;
  /*! \brief Buffer for last line */
  std::string last_line_;
  /*! \brief first line */
  std::string first_line_ = "";
  /*! \brief is skip first line */
  bool is_skip_first_line_ = false;
  size_t read_progress_interval_bytes_;
  /*! \brief is skip first line */
  int skip_bytes_ = 0;
};

}  // namespace StochTree

#endif   // STOCHTREE_IO_H_
