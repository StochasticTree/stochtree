#include <cpp11.hpp>
#include <cmath>

[[cpp11::register]]
double sum_cpp(cpp11::doubles x) {
    double output = 0.0;
    for (int i = 0; i < x.size(); i++) {
        output += x[i];
    }
    return output;
}

[[cpp11::register]]
double mean_cpp(cpp11::doubles x) {
    double output = 0.0;
    for (int i = 0; i < x.size(); i++) {
        output += x[i];
    }
    return output / x.size();
}

[[cpp11::register]]
double var_cpp(cpp11::doubles x) {
    double mean = mean_cpp(x);
    double output = 0.0;
    for (int i = 0; i < x.size(); i++) {
        output += (x[i] - mean) * (x[i] - mean);
    }
    return output / (x.size() - 1);
}

[[cpp11::register]]
double sd_cpp(cpp11::doubles x) {
    return std::sqrt(var_cpp(x));
}
