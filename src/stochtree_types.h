#include <stochtree/bart.h>
#include <stochtree/bart_sampler.h>
#include <stochtree/container.h>
#include <stochtree/data.h>
#include <stochtree/discrete_sampler.h>
#include <stochtree/ensemble.h>
#include <stochtree/leaf_model.h>
#include <stochtree/meta.h>
#include <stochtree/ordinal_sampler.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/random_effects.h>
#include <stochtree/tree_sampler.h>
#include <cpp11.hpp>

static void check_numeric(cpp11::sexp input, const char* input_name) {
  if (TYPEOF(input) != REALSXP && !Rf_isInteger(input)) {
    cpp11::stop("Parameter %s must be a numeric array (integer or floating point)", input_name);
  }
}

static double* extract_numeric_pointer(cpp11::sexp input, const char* input_name, int& protect_count) {
  if (input == R_NilValue) return nullptr;
  check_numeric(input, input_name);
  cpp11::sexp input_converted = PROTECT(Rf_coerceVector(input, REALSXP));
  protect_count++;
  return REAL(input_converted);
}

static void check_integer(cpp11::sexp input, const char* input_name) {
  if (!Rf_isInteger(input)) {
    cpp11::stop("Parameter %s must be an integer array", input_name);
  }
}

static int* extract_integer_pointer(cpp11::sexp input, const char* input_name, int& protect_count) {
  if (input == R_NilValue) return nullptr;
  check_integer(input, input_name);
  return INTEGER(input);
}

template <typename T>
T get_config_scalar_default(cpp11::list& config_list, const char* config_key, T default_value) {
  cpp11::sexp val = config_list[config_key];
  if (Rf_isNull(val)) return default_value;
  return cpp11::as_cpp<T>(val);
}

template <>
inline int get_config_scalar_default<int>(cpp11::list& config_list, const char* config_key, int default_value) {
  cpp11::sexp val = config_list[config_key];
  if (Rf_isNull(val)) return default_value;
  return Rf_asInteger(val);
}
