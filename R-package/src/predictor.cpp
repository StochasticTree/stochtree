#include <cpp11.hpp>
#include <stochtree/data.h>
#include <stochtree/container.h>
#include <memory>
#include <vector>
using namespace cpp11;

[[cpp11::register]]
cpp11::writable::doubles_matrix<> predict_forest_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, cpp11::external_pointer<StochTree::ForestDataset> dataset) {
    // Predict from the sampled forests
    std::vector<double> output_raw = forest_samples->Predict(*dataset);

    // Convert result to a matrix
    int n = dataset->GetCovariates().rows();
    int num_samples = forest_samples->NumSamples();
    cpp11::writable::doubles_matrix<> output(n, num_samples);
    for (size_t i = 0; i < n; i++) {
        for (int j = 0; j < num_samples; j++) {
            output(i, j) = output_raw[n*j + i];
        }
    }

    return output;
}