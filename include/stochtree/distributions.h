#ifndef STOCHTREE_DISTRIBUTIONS_H
#define STOCHTREE_DISTRIBUTIONS_H
#include <numeric>
#include <random>
/*!
 * \brief A collection of random number generation utilities.
 *
 * This file is vendored from a broader C++ / R distribution 
 * library, where the distributions are subject to rigorous testing.
 * https://github.com/andrewherren/cpp11_r_rng
 */

namespace StochTree {

/*!
 * Generate a standard uniform random variate to 53 bits of precision via two mersenne twisters, see:
 * https://github.com/numpy/numpy/blob/0d7986494b39ace565afda3de68be528ddade602/numpy/random/src/mt19937/mt19937.h#L56
 */
inline double standard_uniform_draw_53bit(std::mt19937& gen) {
  int32_t a = gen() >> 5;
  int32_t b = gen() >> 6;
  return (a * 67108864.0 + b) / 9007199254740992.0;
}

/*!
 * Generate a standard uniform random variate to 32 bits of precision via a single mersenne twister.
 */
inline double standard_uniform_draw_32bit(std::mt19937& gen) {
  constexpr double inv_divisor = 1.0 / static_cast<double>(std::mt19937::max());
  return (gen() * inv_divisor);
}

/*!
 * Standard normal sampler implementing Marsaglia's polar method.
 *
 * Reference: https://en.wikipedia.org/wiki/Marsaglia_polar_method
 */
class standard_normal {
 public:
  standard_normal() {
    has_cached_value_ = false;
    cached_value_ = 0.0;
  }

  inline double operator()(std::mt19937& gen) {
    if (has_cached_value_) {
      has_cached_value_ = false;
      return cached_value_;
    } else {
      double u, v, r, s;
      do {
        u = standard_uniform_draw_53bit(gen) * 2.0 - 1.0;
        v = standard_uniform_draw_53bit(gen) * 2.0 - 1.0;
        s = u * u + v * v;
      } while (s >= 1.0 || s == 0.0);
      r = std::sqrt(-2.0 * std::log(s) / s);
      has_cached_value_ = true;
      cached_value_ = v * r;
      return u * r;
    }
  }

 private:
  bool has_cached_value_;
  double cached_value_;
};

/*!
 * Stateless standard normal sampler implementing Marsaglia's polar method.
 * Without caching, this is half as fast as other methods for repeated normal sampling,
 * but this might be acceptable in cases where a relatively small number of 
 * normal draws is desired.
 * 
 * Reference: https://en.wikipedia.org/wiki/Marsaglia_polar_method
 */
inline double sample_standard_normal(double mean, double sd, std::mt19937& gen) {
  double u, v, r, s;
  do {
    u = standard_uniform_draw_53bit(gen) * 2.0 - 1.0;
    v = standard_uniform_draw_53bit(gen) * 2.0 - 1.0;
    s = u * u + v * v;
  } while (s >= 1.0 || s == 0.0);
  r = std::sqrt(-2.0 * std::log(s) / s);
  return u * r * sd + mean;
};

/*!
 * Stateful gamma distribution which uses caching of normals
 */
class gamma_sampler {
 public:
  gamma_sampler() {
    has_cached_normal_value_ = false;
    cached_normal_value_ = 0.0;
  }

  inline double operator()(std::mt19937& gen, double shape, double scale) {
    if (shape == 1.0) {
      return -std::log(standard_uniform_draw_53bit(gen)) * scale;
    } else if (shape < 1.0) {
      // Modified Ahrens-Dieter used by numpy:
      // https://github.com/numpy/numpy/blob/main/numpy/random/src/distributions/distributions.c
      while (true) {
        double u = standard_uniform_draw_53bit(gen);
        double v0 = standard_uniform_draw_53bit(gen);
        double v = -std::log(v0);
        if (u <= 1.0 - shape) {
          double x = std::pow(u, 1.0 / shape);
          if (x <= v) {
            return x * scale;
          }
        } else {
          double y = -std::log((1 - u) / shape);
          double x = std::pow(1.0 - shape + shape * y, 1.0 / shape);
          if (x <= v + y) {
            return x * scale;
          }
        }
      }
    } else if (shape > 1.0) {
      // Marsaglia-Tsang from numpy
      double b = shape - 1.0 / 3.0;
      double c = 1.0 / std::sqrt(9.0 * b);
      while (true) {
        double x, v;
        do {
          x = normal_draw(gen);
          v = 1.0 + c * x;
        } while (v <= 0.0);
        v = v * v * v;
        double u = standard_uniform_draw_53bit(gen);
        if (u < 1.0 - 0.0331 * (x * x) * (x * x)) {
            return b * v * scale;
        }
        if (std::log(u) < 0.5 * x * x + b * (1.0 - v + std::log(v))) {
            return b * v * scale;
        }
      }
    } else {
      return 0.0;
    }
  }

  inline double normal_draw(std::mt19937& gen) {
    if (has_cached_normal_value_) {
      has_cached_normal_value_ = false;
      return cached_normal_value_;
    } else {
      double u, v, r, s;
      do {
        u = standard_uniform_draw_53bit(gen) * 2.0 - 1.0;
        v = standard_uniform_draw_53bit(gen) * 2.0 - 1.0;
        s = u * u + v * v;
      } while (s >= 1.0 || s == 0.0);
      r = std::sqrt(-2.0 * std::log(s) / s);
      has_cached_normal_value_ = true;
      cached_normal_value_ = v * r;
      return u * r;
    }
  }

 private:
  bool has_cached_normal_value_;
  double cached_normal_value_;
};

/*!
 * Generate a single sample from a gamma distribution using a combination of algorithms
 * When shape < 1.0, use the Ahrens-Dieter method
 * When shape > 1.0, use the Marsaglia-Tsang method
 * When shape == 1.0, sample an exponential via inverse transform method
 * When shape == 0.0, return 0.0
 * https://en.wikipedia.org/wiki/Gamma_distribution#Random_variate_generation
 */
inline double sample_gamma(std::mt19937& gen, double shape, double scale) {
  if (shape == 1.0) {
    return -std::log(standard_uniform_draw_53bit(gen)) * scale;
  } else if (shape < 1.0) {
    // Modified Ahrens-Dieter used by numpy:
    // https://github.com/numpy/numpy/blob/main/numpy/random/src/distributions/distributions.c
    while (true) {
      double u = standard_uniform_draw_53bit(gen);
      double v0 = standard_uniform_draw_53bit(gen);
      double v = -std::log(v0);
      if (u <= 1.0 - shape) {
        double x = std::pow(u, 1.0 / shape);
        if (x <= v) {
          return x * scale;
        }
      } else {
        double y = -std::log((1 - u) / shape);
        double x = std::pow(1.0 - shape + shape * y, 1.0 / shape);
        if (x <= v + y) {
          return x * scale;
        }
      }
    }
  } else if (shape > 1.0) {
    // Marsaglia-Tsang from numpy
    double b = shape - 1.0 / 3.0;
    double c = 1.0 / std::sqrt(9.0 * b);
    while (true) {
      double x, v;
      do {
        // Marsaglia's polar method for standard normal 
        double u1, u2, s;
        do {
          u1 = standard_uniform_draw_53bit(gen) * 2.0 - 1.0;
          u2 = standard_uniform_draw_53bit(gen) * 2.0 - 1.0;
          s = u1 * u1 + u2 * u2;
        } while (s >= 1.0 || s == 0.0);
        x = u1 * std::sqrt(-2.0 * std::log(s) / s);            
        v = 1.0 + c * x;
      } while (v <= 0.0);
      v = v * v * v;
      double u = standard_uniform_draw_53bit(gen);
      if (u < 1.0 - 0.0331 * (x * x) * (x * x)) {
          return b * v * scale;
      }
      if (std::log(u) < 0.5 * x * x + b * (1.0 - v + std::log(v))) {
          return b * v * scale;
      }
    }
  } else {
    return 0.0;
  }
}

/*!
 * Walker-Vose alias method for sampling with replacement from a weighted discrete distribution.
 * 
 * Simplified from https://github.com/boostorg/random/blob/develop/include/boost/random/discrete_distribution.hpp
 * Other references: https://en.wikipedia.org/wiki/Alias_method
 */
class walker_vose {
 public:
  template<typename Iterator>
  walker_vose(Iterator first, Iterator last) {
    n_ = std::distance(first, last);
    probability_.resize(n_);
    alias_.resize(n_);

    // Compute probability normalizing factor
    double sum = 0.0;
    for (auto it = first; it != last; ++it) {
      sum += *it;
    }
    
    // Build alias table using Walker's algorithm
    std::vector<double> p(n_);
    std::vector<int> below_average, above_average;

    for (int i = 0; i < n_; ++i) {
      p[i] = (*(first + i)) * n_ / sum;
      if (p[i] < 1.0) {
        below_average.push_back(i);
      } else {
        above_average.push_back(i);
      }
    }
    
    while (!below_average.empty() && !above_average.empty()) {
      int j = below_average.back(); below_average.pop_back();
      int i = above_average.back(); above_average.pop_back();
      
      probability_[j] = p[j];
      alias_[j] = i;
      p[i] = (p[i] + p[j]) - 1.0;
      
      if (p[i] < 1.0) {
        below_average.push_back(i);
      } else {
        above_average.push_back(i);
      }
    }
    
    while (!above_average.empty()) {
      probability_[above_average.back()] = 1.0;
      above_average.pop_back();
    }
    
    while (!below_average.empty()) {
      probability_[below_average.back()] = 1.0;
      below_average.pop_back();
    }
  }
  
  int operator()(std::mt19937& gen) {
    double u = standard_uniform_draw_53bit(gen);
    int i = static_cast<int>(u * n_);
    double y = u * n_ - i;
    return (y < probability_[i]) ? i : alias_[i];
  }

 private:
  std::vector<double> probability_;
  std::vector<int> alias_;
  int n_;
};

inline int sample_discrete_stateless(std::mt19937& gen, std::vector<double>& weights) {
  double sum_weight = std::accumulate(weights.begin(), weights.end(), 0.0);
  double u = standard_uniform_draw_53bit(gen) * sum_weight;
  double running_total_weight = 0.0;
  for (int i = 0; i < weights.size(); ++i) {
    running_total_weight += weights[i];
    if (running_total_weight > u) {
      return i;
    }
  }
  return weights.size() - 1;
}

inline int sample_discrete_stateless(std::mt19937& gen, std::vector<double>& weights, double sum_weights) {
  double u = standard_uniform_draw_53bit(gen) * sum_weights;
  double running_total_weight = 0.0;
  for (int i = 0; i < weights.size(); ++i) {
    running_total_weight += weights[i];
    if (running_total_weight > u) {
      return i;
    }
  }
  return weights.size() - 1;
}

}

#endif // STOCHTREE_DISTRIBUTIONS_H