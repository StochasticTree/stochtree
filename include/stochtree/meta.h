/*!
 * Macros, constants, and type definitions used elsewhere in the codebase
 * 
 * This code is largely included as-is from LightGBM, which carries 
 * the following copyright information:
 * 
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_META_H_
#define STOCHTREE_META_H_

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>
#include <boost/math/constants/constants.hpp>

#if (defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_AMD64))) || defined(__INTEL_COMPILER) || MM_PREFETCH
  #include <xmmintrin.h>
  #define PREFETCH_T0(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
#elif defined(__GNUC__)
  #define PREFETCH_T0(addr) __builtin_prefetch(reinterpret_cast<const char*>(addr), 0, 3)
#else
  #define PREFETCH_T0(addr) do {} while (0)
#endif

namespace StochTree {

/*! \brief Integer encoding of feature types */
enum FeatureType {
  kNumeric, /*!< Numeric feature */
  kOrderedCategorical, /*!< Ordered categorical feature */
  kUnorderedCategorical /*!< Unordered categorical feature */
};

enum ForestLeafVarianceType {
  kStochastic,
  kFixed
};

// enum ForestLeafPriorType {
//   kConstantLeafGaussian,
//   kUnivariateRegressionLeafGaussian,
//   kMultivariateRegressionLeafGaussian
// };

enum ForestSampler {
  kMCMC,
  kGFR
};

enum ForestType {
  kConstantForest,
  kUnivariateRegressionForest,
  kMultivariateRegressionForest
};

enum RandomEffectsType {
  kConstantRandomEffect,
  kRegressionRandomEffect
};

/*! \brief Double precision pi constant */
static constexpr double pi_constant = boost::math::constants::pi<double>();

/*! \brief Type of data size */
typedef int32_t data_size_t;

/*! \brief Type of feature index */
typedef int32_t feature_size_t;

// Enable following macro to use double for score_t
#define SCORE_T_USE_DOUBLE

// Enable following macro to use double for label_t
#define LABEL_T_USE_DOUBLE

// Enable following macro to use double for label_t
#define TREATMENT_T_USE_DOUBLE

/*! \brief Type of score, and gradients */
#ifdef SCORE_T_USE_DOUBLE
typedef double score_t;
#else
typedef float score_t;
#endif

/*! \brief Type of label */
#ifdef LABEL_T_USE_DOUBLE
typedef double label_t;
#else
typedef float label_t;
#endif

/*! \brief Type of treatment */
#ifdef TREATMENT_T_USE_DOUBLE
typedef double treatment_t;
#else
typedef float treatment_t;
#endif

const score_t kMinScore = -std::numeric_limits<score_t>::infinity();

const score_t kMaxScore = std::numeric_limits<score_t>::infinity();

const score_t kEpsilon = 1e-15f;

const double kZeroThreshold = 1e-35f;

const double kNumericMissingValue = std::nan("");
const int32_t kIntegralMissingValue = -1;

typedef int32_t comm_size_t;

/*! \brief Type of node index */
typedef int32_t node_t;

/*! \brief Type of split condition */
typedef double split_cond_t;

using PredictFunction =
std::function<void(const std::vector<std::pair<int, double>>&, double* output)>;

using PredictSparseFunction =
std::function<void(const std::vector<std::pair<int, double>>&, std::vector<std::unordered_map<int, double>>* output)>;

typedef void(*ReduceFunction)(const char* input, char* output, int type_size, comm_size_t array_size);


typedef void(*ReduceScatterFunction)(char* input, comm_size_t input_size, int type_size,
                                     const comm_size_t* block_start, const comm_size_t* block_len, int num_block, char* output, comm_size_t output_size,
                                     const ReduceFunction& reducer);

typedef void(*AllgatherFunction)(char* input, comm_size_t input_size, const comm_size_t* block_start,
                                 const comm_size_t* block_len, int num_block, char* output, comm_size_t output_size);


#define NO_SPECIFIC (-1)

const int kAlignedSize = 32;

#define SIZE_ALIGNED(t) ((t) + kAlignedSize - 1) / kAlignedSize * kAlignedSize

// Refer to https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-4-c4127?view=vs-2019
#ifdef _MSC_VER
  #pragma warning(disable : 4127)
#endif

}  // namespace StochTree

#endif  // STOCHTREE_META_H_
