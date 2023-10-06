#include <cpp11.hpp>
#include <stochtree/config.h>
#include <stochtree/interface.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
using namespace cpp11;

[[cpp11::register]]
cpp11::external_pointer<StochTree::StochTreeInterface> bart_sample_cpp(cpp11::doubles_matrix<> model_matrix, cpp11::r_string param_string) {
    // Extract dimensions of matrix X and pointer to its contiguous block of memory
    int p = model_matrix.ncol();
    int n = model_matrix.nrow();
    double* data_ptr = REAL(PROTECT(model_matrix));

    // Convert param_string to a char pointer
    const char* parameters = CHAR(PROTECT(Rf_asChar(param_string)));

    // Generate a config object from the provided parameter string
    auto param = StochTree::Config::Str2Map(parameters);
    StochTree::Config config;
    config.Set(param);
    std::unordered_map<std::string, std::string> method_override = {{"method_type", "bart"}};
    config.Set(method_override);

    // Configure a BART object
    std::unique_ptr<StochTree::StochTreeInterface> bart_ptr_;
    bart_ptr_.reset(new StochTree::StochTreeInterface(config));
    bart_ptr_->LoadTrainDataFromMemory(data_ptr, p, n, false);

    // Sample the BART model
    bart_ptr_->SampleModel();

    // Unprotect character pointers
    UNPROTECT(2);

    return cpp11::external_pointer<StochTree::StochTreeInterface>(bart_ptr_.release());
}

[[cpp11::register]]
cpp11::writable::doubles_matrix<> bart_predict_cpp(cpp11::external_pointer<StochTree::StochTreeInterface> bart_ptr, cpp11::doubles_matrix<> model_matrix, cpp11::r_string param_string) {
    // Extract dimensions of matrix X and pointer to its contiguous block of memory
    int p = model_matrix.ncol();
    int n = model_matrix.nrow();
    double* data_ptr = REAL(PROTECT(model_matrix));

    // Convert param_string to a char pointer
    const char* parameters = CHAR(PROTECT(Rf_asChar(param_string)));

    // Generate a config object from the provided parameter string
    auto param = StochTree::Config::Str2Map(parameters);
    StochTree::Config config;
    config.Set(param);

    // Configure a BART object
    bart_ptr->LoadPredictionDataFromMemory(data_ptr, p, n, false, config);

    // Predict from the sampled BART model
    std::vector<double> output_raw = bart_ptr->PredictSamples();

    // Convert result to a matrix
    cpp11::writable::doubles_matrix<> output(n, config.num_samples);
    for (size_t i = 0; i < n; i++) {
        for (int j = 0; j < config.num_samples; j++) {
            output(i, j) = output_raw[n*j + i];
        }
    }

    // Unprotect character pointers
    UNPROTECT(2);

    return output;
}
