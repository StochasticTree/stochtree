#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stochtree/interface.h>
#include <stochtree/config.h>
#include <memory>
#include <string>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

class BART_CPP {
 public:
  BART_CPP(std::string param_string) {
    // Convert param_string to a char pointer
    const char* parameters = param_string.c_str();

    // Generate a config object from the provided parameter string
    auto param = StochTree::Config::Str2Map(parameters);
    config.Set(param);
    std::unordered_map<std::string, std::string> method_override = {{"method_type", "bart"}};
    config.Set(method_override);
  }

  void reset_params(std::string param_string) {
    // Convert param_string to a char pointer
    const char* parameters = param_string.c_str();

    // Generate a config object from the provided parameter string
    auto param = StochTree::Config::Str2Map(parameters);
    config.Set(param);
  }

  void sample(py::array model_matrix, int num_col, StochTree::data_size_t num_row) {
    // Extract pointer to contiguous block of memory
    double* data_ptr = static_cast<double*>(model_matrix.mutable_data());
    
    // Configure a BART object
    interface_.reset(new StochTree::StochTreeInterface(config));
    interface_->LoadTrainDataFromMemory(data_ptr, num_col+1, num_row, true);

    // Sample the BART model
    interface_->SampleModel();
  }

  py::array predict(py::array model_matrix, int num_col, StochTree::data_size_t num_row) {
    // Extract pointer to contiguous block of memory
    double* data_ptr = static_cast<double*>(model_matrix.mutable_data());
    
    // Configure a BART object
    interface_->LoadPredictionDataFromMemory(data_ptr, num_col+1, num_row, true, config);

    // Predict from the sampled BART model
    std::vector<double> output_raw = interface_->PredictSamples();

    // Convert result to a matrix
    auto result = py::array_t<double>(py::detail::any_container<ssize_t>({num_row, config.num_samples}));
    py::buffer_info buf = result.request();
    double *ptr = static_cast<double *>(buf.ptr);
    for (size_t i = 0; i < num_row; i++) {
      for (int j = 0; j < config.num_samples; j++) {
        ptr[i*config.num_samples + j] = output_raw[num_row*j + i];
      }
    }

    return result;
  }

 private:
  StochTree::Config config;
  std::unique_ptr<StochTree::StochTreeInterface> interface_;
};

class XBART_CPP {
 public:
  XBART_CPP(std::string param_string) {
    // Convert param_string to a char pointer
    const char* parameters = param_string.c_str();

    // Generate a config object from the provided parameter string
    auto param = StochTree::Config::Str2Map(parameters);
    config.Set(param);
    std::unordered_map<std::string, std::string> method_override = {{"method_type", "xbart"}};
    config.Set(method_override);
  }

  void reset_params(std::string param_string) {
    // Convert param_string to a char pointer
    const char* parameters = param_string.c_str();

    // Generate a config object from the provided parameter string
    auto param = StochTree::Config::Str2Map(parameters);
    config.Set(param);
  }

  void sample(py::array model_matrix, int num_col, StochTree::data_size_t num_row) {
    // Extract pointer to contiguous block of memory
    double* data_ptr = static_cast<double*>(model_matrix.mutable_data());
    
    // Configure an XBART object
    interface_.reset(new StochTree::StochTreeInterface(config));
    interface_->LoadTrainDataFromMemory(data_ptr, num_col+1, num_row, true);

    // Sample the XBART model
    interface_->SampleModel();
  }

  py::array predict(py::array model_matrix, int num_col, StochTree::data_size_t num_row) {
    // Extract pointer to contiguous block of memory
    double* data_ptr = static_cast<double*>(model_matrix.mutable_data());
    
    // Configure an XBART object
    interface_->LoadPredictionDataFromMemory(data_ptr, num_col+1, num_row, true, config);

    // Predict from the sampled XBART model
    std::vector<double> output_raw = interface_->PredictSamples();

    // Convert result to a matrix
    auto result = py::array_t<double>(py::detail::any_container<ssize_t>({num_row, config.num_samples}));
    py::buffer_info buf = result.request();
    double *ptr = static_cast<double *>(buf.ptr);
    for (size_t i = 0; i < num_row; i++) {
      for (int j = 0; j < config.num_samples; j++) {
        ptr[i*config.num_samples + j] = output_raw[num_row*j + i];
      }
    }

    return result;
  }

 private:
  StochTree::Config config;
  std::unique_ptr<StochTree::StochTreeInterface> interface_;
};

PYBIND11_MODULE(stochtree_cpp, m) {
    py::class_<BART_CPP>(m, "BART_CPP")
        .def(py::init<std::string>())
        .def("reset_params", &BART_CPP::reset_params)
        .def("sample", &BART_CPP::sample)
        .def("predict", &BART_CPP::predict);

    py::class_<XBART_CPP>(m, "XBART_CPP")
        .def(py::init<std::string>())
        .def("reset_params", &XBART_CPP::reset_params)
        .def("sample", &XBART_CPP::sample)
        .def("predict", &XBART_CPP::predict);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}