/*!
 * Copyright (c) 2024 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_DISCRETE_SAMPLER_H_
#define STOCHTREE_DISCRETE_SAMPLER_H_
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

namespace StochTree {

/*! \brief Sample without replacement according to a set of probability weights. 
 * This template function is a C++ variant of numpy's implementation:
 * https://github.com/numpy/numpy/blob/031f44252d613f4524ad181e3eb2ae2791e22187/numpy/random/_generator.pyx#L925
 */
template <typename container_type, typename prob_type>
void sample_without_replacement(container_type* output, prob_type* p, container_type* a, int population_size, int sample_size, std::mt19937& gen) {
  std::vector<prob_type> p_copy(population_size);
  std::memcpy(p_copy.data(), p, sizeof(prob_type) * population_size);
  std::vector<int> indices(sample_size);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  std::vector<prob_type> unif_samples(sample_size);
  std::vector<prob_type> cdf(population_size);
  
  int fulfilled_sample_count = 0;
  int remaining_sample_count = sample_size - fulfilled_sample_count;
  while (fulfilled_sample_count < sample_size) {
    if (fulfilled_sample_count > 0) {
      for (int i = 0; i < fulfilled_sample_count; i++) p_copy[indices[i]] = 0.0;
    }
    std::generate(unif_samples.begin(), unif_samples.begin() + remaining_sample_count, [&gen, &unif](){
      return unif(gen);
    });
    std::partial_sum(p_copy.cbegin(), p_copy.cend(), cdf.begin());
    for (int i = 0; i < cdf.size(); i++) {
      cdf[i] = cdf[i] / cdf[cdf.size()-1];
    }
    std::vector<int> matches(remaining_sample_count);
    for (int i = 0; i < remaining_sample_count; i++) {
      auto match = std::upper_bound(cdf.cbegin(), cdf.cend(), unif_samples[i]);
      if (match != cdf.cend()) {
        matches[i] = std::distance(cdf.cbegin(), match);
      } else {
        matches[i] = std::distance(cdf.cbegin(), cdf.cend());
      }
    }
    std::sort(matches.begin(), matches.end());
    auto last_unique = std::unique(matches.begin(), matches.end());
    matches.erase(last_unique, matches.end());
    for (int i = 0; i < matches.size(); i++) {
      indices[fulfilled_sample_count + i] = matches[i];
    }
    fulfilled_sample_count += matches.size();
    remaining_sample_count -= matches.size();
  }
  for (int i = 0; i < sample_size; i++) {
    output[i] = a[indices[i]];
  }
}

}

#endif // STOCHTREE_DISCRETE_SAMPLER_H_
