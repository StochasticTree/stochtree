#include <cpp11.hpp>
#include "stochtree/random_effects.h"
#include "stochtree_types.h"
#include <stochtree/bart.h>
#include <stochtree/bcf.h>
#include <stochtree/container.h>
#include <initializer_list>
#include <memory>

// Deep-copy a forest container sample-by-sample (so the caller's EXTPTR keeps its own copy).
// File-local (static) helper, matching the convention used elsewhere (e.g. src/stochtree_types.h).
static std::unique_ptr<StochTree::ForestContainer> clone_forest_container(StochTree::ForestContainer* src) {
  auto copy_ptr = std::make_unique<StochTree::ForestContainer>(
      src->NumTrees(), src->OutputDimension(), src->IsLeafConstant(), src->IsExponentiated());
  for (int i = 0; i < src->NumSamples(); i++) copy_ptr->AddSample(*src->GetEnsemble(i));
  return copy_ptr;
}

// Deep-copy a random effects container
static std::unique_ptr<StochTree::RandomEffectsContainer> clone_rfx_container(StochTree::RandomEffectsContainer* src) {
  auto copy_ptr = std::make_unique<StochTree::RandomEffectsContainer>();
  copy_ptr->CopyFromOther(*src);
  return copy_ptr;
}

// Deep-copy a random effects container
static std::unique_ptr<StochTree::LabelMapper> clone_label_mapper(StochTree::LabelMapper* src) {
  auto copy_ptr = std::make_unique<StochTree::LabelMapper>();
  copy_ptr->CopyFromOther(*src);
  return copy_ptr;
}

// Convert std::vector<double> to cpp11::writable::doubles (for returning samples as an R-native vector).
static cpp11::writable::doubles vec_to_doubles(const std::vector<double>& v) {
  cpp11::writable::doubles out(static_cast<R_xlen_t>(v.size()));
  std::copy(v.begin(), v.end(), out.begin());
  return out;
}

// Convert std::vector<double> to cpp11::writable::doubles (for returning samples as an R-native vector).
static cpp11::writable::doubles vec_to_doubles_reshape(const std::vector<double>& v, std::initializer_list<int> dims) {
  auto out = vec_to_doubles(v);
  if (!v.empty()) {
    out.attr("dim") = cpp11::writable::integers(dims);
  }
  return out;
}

// -------------------------------- BARTSamples --------------------------------

[[cpp11::register]]
cpp11::external_pointer<StochTree::BARTSamples> bart_samples_cpp() {
  auto samples = std::make_unique<StochTree::BARTSamples>();
  return cpp11::external_pointer<StochTree::BARTSamples>(samples.release());
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::BARTSamples> bart_samples_from_json_cpp(cpp11::external_pointer<nlohmann::json> json) {
  auto samples = std::make_unique<StochTree::BARTSamples>();
  samples->FromJson(*json);
  return cpp11::external_pointer<StochTree::BARTSamples>(samples.release());
}

[[cpp11::register]]
void append_bart_samples_to_json_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples, cpp11::external_pointer<nlohmann::json> json) {
  samples->AppendToJson(*json);
}

[[cpp11::register]]
int bart_samples_num_samples_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return samples->num_samples;
}

[[cpp11::register]]
double bart_samples_y_bar_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return samples->y_bar;
}

[[cpp11::register]]
double bart_samples_y_std_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return samples->y_std;
}

[[cpp11::register]]
bool bart_samples_has_yhat_train_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return samples->rfx_predictions_train.empty() == false || samples->mean_forest_predictions_train.empty() == false;
}

[[cpp11::register]]
bool bart_samples_has_mean_forest_predictions_train_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return samples->mean_forest_predictions_train.empty() == false;
}

[[cpp11::register]]
bool bart_samples_has_variance_forest_predictions_train_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return samples->variance_forest_predictions_train.empty() == false;
}

[[cpp11::register]]
bool bart_samples_has_rfx_predictions_train_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return samples->rfx_predictions_train.empty() == false;
}

[[cpp11::register]]
bool bart_samples_has_yhat_test_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return samples->rfx_predictions_test.empty() == false || samples->mean_forest_predictions_test.empty() == false;
}

[[cpp11::register]]
bool bart_samples_has_mean_forest_predictions_test_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return samples->mean_forest_predictions_test.empty() == false;
}

[[cpp11::register]]
bool bart_samples_has_variance_forest_predictions_test_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return samples->variance_forest_predictions_test.empty() == false;
}

[[cpp11::register]]
bool bart_samples_has_rfx_predictions_test_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return samples->rfx_predictions_test.empty() == false;
}

[[cpp11::register]]
bool bart_samples_has_cloglog_cutpoint_samples_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return samples->cloglog_cutpoint_samples.empty() == false;
}

[[cpp11::register]]
bool bart_samples_has_global_var_samples_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return samples->global_error_variance_samples.empty() == false;
}

[[cpp11::register]]
bool bart_samples_has_leaf_scale_samples_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return samples->leaf_scale_samples.empty() == false;
}

[[cpp11::register]]
bool bart_samples_has_mean_forest_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return samples->mean_forests != nullptr;
}

[[cpp11::register]]
bool bart_samples_has_rfx_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return samples->rfx_container != nullptr;
}

[[cpp11::register]]
bool bart_samples_has_variance_forest_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return samples->variance_forests != nullptr;
}

[[cpp11::register]]
cpp11::writable::doubles bart_samples_mean_forest_predictions_train_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return vec_to_doubles_reshape(samples->mean_forest_predictions_train, {samples->num_train, samples->num_samples});
}

[[cpp11::register]]
cpp11::writable::doubles bart_samples_variance_forest_predictions_train_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return vec_to_doubles_reshape(samples->variance_forest_predictions_train, {samples->num_train, samples->num_samples});
}

[[cpp11::register]]
cpp11::writable::doubles bart_samples_rfx_predictions_train_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return vec_to_doubles_reshape(samples->rfx_predictions_train, {samples->num_train, samples->num_samples});
}

[[cpp11::register]]
cpp11::writable::doubles bart_samples_yhat_train_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return vec_to_doubles_reshape(samples->OutcomePredictionsTrain(), {samples->num_train, samples->num_samples});
}

[[cpp11::register]]
cpp11::writable::doubles bart_samples_mean_forest_predictions_test_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return vec_to_doubles_reshape(samples->mean_forest_predictions_test, {samples->num_test, samples->num_samples});
}

[[cpp11::register]]
cpp11::writable::doubles bart_samples_variance_forest_predictions_test_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return vec_to_doubles_reshape(samples->variance_forest_predictions_test, {samples->num_test, samples->num_samples});
}

[[cpp11::register]]
cpp11::writable::doubles bart_samples_rfx_predictions_test_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return vec_to_doubles_reshape(samples->rfx_predictions_test, {samples->num_test, samples->num_samples});
}

[[cpp11::register]]
cpp11::writable::doubles bart_samples_yhat_test_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return vec_to_doubles_reshape(samples->OutcomePredictionsTest(), {samples->num_test, samples->num_samples});
}

[[cpp11::register]]
cpp11::writable::doubles bart_samples_global_var_samples_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return vec_to_doubles(samples->global_error_variance_samples);
}

[[cpp11::register]]
cpp11::writable::doubles bart_samples_leaf_scale_samples_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return vec_to_doubles(samples->leaf_scale_samples);
}

[[cpp11::register]]
cpp11::writable::doubles bart_samples_cloglog_cutpoint_samples_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  int ns = samples->num_samples;
  int len = static_cast<int>(samples->cloglog_cutpoint_samples.size());
  int num_cutpoints = (ns > 0) ? len / ns : 0;
  return vec_to_doubles_reshape(samples->cloglog_cutpoint_samples, {num_cutpoints, ns});
}

// Materialize a standalone deep copy of the mean forest container.
[[cpp11::register]]
cpp11::external_pointer<StochTree::ForestContainer> bart_samples_materialize_mean_forest_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  auto copy = clone_forest_container(samples->mean_forests.get());
  return cpp11::external_pointer<StochTree::ForestContainer>(copy.release());
}

// Materialize a standalone deep copy of the variance forest container.
[[cpp11::register]]
cpp11::external_pointer<StochTree::ForestContainer> bart_samples_materialize_variance_forest_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  auto copy = clone_forest_container(samples->variance_forests.get());
  return cpp11::external_pointer<StochTree::ForestContainer>(copy.release());
}

// Materialize a standalone deep copy of the random effects container.
[[cpp11::register]]
cpp11::external_pointer<StochTree::RandomEffectsContainer> bart_samples_materialize_rfx_container_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  auto copy = clone_rfx_container(samples->rfx_container.get());
  return cpp11::external_pointer<StochTree::RandomEffectsContainer>(copy.release());
}

// Materialize a standalone deep copy of the random effects label mapper.
[[cpp11::register]]
cpp11::external_pointer<StochTree::LabelMapper> bart_samples_materialize_rfx_label_mapper_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  auto copy = clone_label_mapper(samples->rfx_label_mapper.get());
  return cpp11::external_pointer<StochTree::LabelMapper>(copy.release());
}

// Borrowed (non-owning) pointers to the samples-owned forest containers, for read-through predict.
// The returned external_pointer does NOT own or finalize the container -- it aliases the one owned
// by `samples`, so it must not outlive it (predict uses it transiently within a single call).
[[cpp11::register]]
cpp11::external_pointer<StochTree::ForestContainer> bart_samples_mean_forest_ptr_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return cpp11::external_pointer<StochTree::ForestContainer>(
      samples->mean_forests.get(), /*use_deleter=*/false, /*finalize_on_exit=*/false);
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::ForestContainer> bart_samples_variance_forest_ptr_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return cpp11::external_pointer<StochTree::ForestContainer>(
      samples->variance_forests.get(), /*use_deleter=*/false, /*finalize_on_exit=*/false);
}

[[cpp11::register]]
void bart_samples_merge_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples,
                            cpp11::external_pointer<StochTree::BARTSamples> other) {
  samples->Merge(*other);
}

// -------------------------------- BCFSamples --------------------------------

[[cpp11::register]]
cpp11::external_pointer<StochTree::BCFSamples> bcf_samples_cpp() {
  auto samples = std::make_unique<StochTree::BCFSamples>();
  return cpp11::external_pointer<StochTree::BCFSamples>(samples.release());
}

[[cpp11::register]]
int bcf_samples_num_samples_cpp(cpp11::external_pointer<StochTree::BCFSamples> samples) {
  return samples->num_samples;
}

[[cpp11::register]]
int bcf_samples_treatment_dim_cpp(cpp11::external_pointer<StochTree::BCFSamples> samples) {
  return samples->treatment_dim;
}

[[cpp11::register]]
double bcf_samples_y_bar_cpp(cpp11::external_pointer<StochTree::BCFSamples> samples) {
  return samples->y_bar;
}

[[cpp11::register]]
double bcf_samples_y_std_cpp(cpp11::external_pointer<StochTree::BCFSamples> samples) {
  return samples->y_std;
}

[[cpp11::register]]
bool bcf_samples_has_mu_forest_cpp(cpp11::external_pointer<StochTree::BCFSamples> samples) {
  return samples->mu_forests != nullptr;
}

[[cpp11::register]]
bool bcf_samples_has_tau_forest_cpp(cpp11::external_pointer<StochTree::BCFSamples> samples) {
  return samples->tau_forests != nullptr;
}

[[cpp11::register]]
bool bcf_samples_has_variance_forest_cpp(cpp11::external_pointer<StochTree::BCFSamples> samples) {
  return samples->variance_forests != nullptr;
}

[[cpp11::register]]
bool bcf_samples_has_rfx_cpp(cpp11::external_pointer<StochTree::BCFSamples> samples) {
  return samples->rfx_container != nullptr;
}

[[cpp11::register]]
cpp11::writable::doubles bcf_samples_global_var_samples_cpp(cpp11::external_pointer<StochTree::BCFSamples> samples) {
  return vec_to_doubles(samples->global_error_variance_samples);
}

[[cpp11::register]]
cpp11::writable::doubles bcf_samples_leaf_scale_mu_samples_cpp(cpp11::external_pointer<StochTree::BCFSamples> samples) {
  return vec_to_doubles(samples->leaf_scale_mu_samples);
}

[[cpp11::register]]
cpp11::writable::doubles bcf_samples_leaf_scale_tau_samples_cpp(cpp11::external_pointer<StochTree::BCFSamples> samples) {
  return vec_to_doubles(samples->leaf_scale_tau_samples);
}

[[cpp11::register]]
cpp11::writable::doubles bcf_samples_tau_0_samples_cpp(cpp11::external_pointer<StochTree::BCFSamples> samples) {
  return vec_to_doubles(samples->tau_0_samples);
}

[[cpp11::register]]
cpp11::writable::doubles bcf_samples_b0_samples_cpp(cpp11::external_pointer<StochTree::BCFSamples> samples) {
  return vec_to_doubles(samples->b0_samples);
}

[[cpp11::register]]
cpp11::writable::doubles bcf_samples_b1_samples_cpp(cpp11::external_pointer<StochTree::BCFSamples> samples) {
  return vec_to_doubles(samples->b1_samples);
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::ForestContainer> bcf_samples_materialize_mu_forest_cpp(cpp11::external_pointer<StochTree::BCFSamples> samples) {
  auto copy = clone_forest_container(samples->mu_forests.get());
  return cpp11::external_pointer<StochTree::ForestContainer>(copy.release());
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::ForestContainer> bcf_samples_materialize_tau_forest_cpp(cpp11::external_pointer<StochTree::BCFSamples> samples) {
  auto copy = clone_forest_container(samples->tau_forests.get());
  return cpp11::external_pointer<StochTree::ForestContainer>(copy.release());
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::ForestContainer> bcf_samples_materialize_variance_forest_cpp(cpp11::external_pointer<StochTree::BCFSamples> samples) {
  auto copy = clone_forest_container(samples->variance_forests.get());
  return cpp11::external_pointer<StochTree::ForestContainer>(copy.release());
}

// Borrowed (non-owning) pointers to the samples-owned forest containers, for read-through predict.
[[cpp11::register]]
cpp11::external_pointer<StochTree::ForestContainer> bcf_samples_mu_forest_ptr_cpp(cpp11::external_pointer<StochTree::BCFSamples> samples) {
  return cpp11::external_pointer<StochTree::ForestContainer>(
      samples->mu_forests.get(), /*use_deleter=*/false, /*finalize_on_exit=*/false);
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::ForestContainer> bcf_samples_tau_forest_ptr_cpp(cpp11::external_pointer<StochTree::BCFSamples> samples) {
  return cpp11::external_pointer<StochTree::ForestContainer>(
      samples->tau_forests.get(), /*use_deleter=*/false, /*finalize_on_exit=*/false);
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::ForestContainer> bcf_samples_variance_forest_ptr_cpp(cpp11::external_pointer<StochTree::BCFSamples> samples) {
  return cpp11::external_pointer<StochTree::ForestContainer>(
      samples->variance_forests.get(), /*use_deleter=*/false, /*finalize_on_exit=*/false);
}

[[cpp11::register]]
void bcf_samples_merge_cpp(cpp11::external_pointer<StochTree::BCFSamples> samples,
                           cpp11::external_pointer<StochTree::BCFSamples> other) {
  samples->Merge(*other);
}
