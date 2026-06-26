#include <cpp11.hpp>
#include "stochtree_types.h"
#include <stochtree/bart.h>
#include <stochtree/bcf.h>
#include <stochtree/container.h>
#include <memory>

// cpp11 bindings for the single-owner BARTSamples / BCFSamples objects. These mirror the
// forest-container bindings (src/forest.cpp): free functions tagged [[cpp11::register]] that take
// and return cpp11::external_pointer handles, wrapped on the R side by an R6 class that holds the
// pointer in a field. The heavy logic lives in core C++ (BARTSamples/BCFSamples methods); these are
// thin marshalling shims, with cpp11::sexp + (x == R_NilValue) for the nullable arguments.

// Deep-copy a forest container sample-by-sample (so the caller's EXTPTR keeps its own copy).
// File-local (static) helper, matching the convention used elsewhere (e.g. src/stochtree_types.h).
static std::unique_ptr<StochTree::ForestContainer> clone_forest_container(StochTree::ForestContainer* src) {
  auto copy = std::make_unique<StochTree::ForestContainer>(
      src->NumTrees(), src->OutputDimension(), src->IsLeafConstant(), src->IsExponentiated());
  for (int i = 0; i < src->NumSamples(); i++) copy->AddSample(*src->GetEnsemble(i));
  return copy;
}

static cpp11::writable::doubles vec_to_doubles(const std::vector<double>& v) {
  cpp11::writable::doubles out(static_cast<R_xlen_t>(v.size()));
  std::copy(v.begin(), v.end(), out.begin());
  return out;
}

// -------------------------------- BARTSamples --------------------------------

[[cpp11::register]]
cpp11::external_pointer<StochTree::BARTSamples> bart_samples_from_components_cpp(
    cpp11::sexp mean_forest, cpp11::sexp variance_forest,
    cpp11::sexp global_var_samples, cpp11::sexp leaf_scale_samples,
    double y_bar, double y_std, int num_samples) {
  auto samples = std::make_unique<StochTree::BARTSamples>();
  // BART supports mean-only, variance-only, or both -- both forests are optional.
  if (mean_forest != R_NilValue) {
    cpp11::external_pointer<StochTree::ForestContainer> fc(mean_forest);
    samples->mean_forests = clone_forest_container(fc.get());
  }
  if (variance_forest != R_NilValue) {
    cpp11::external_pointer<StochTree::ForestContainer> fc(variance_forest);
    samples->variance_forests = clone_forest_container(fc.get());
  }
  if (global_var_samples != R_NilValue) {
    cpp11::doubles gv(global_var_samples);
    samples->global_error_variance_samples.assign(gv.begin(), gv.end());
  }
  if (leaf_scale_samples != R_NilValue) {
    cpp11::doubles ls(leaf_scale_samples);
    samples->leaf_scale_samples.assign(ls.begin(), ls.end());
  }
  samples->y_bar = y_bar;
  samples->y_std = y_std;
  samples->num_samples = num_samples;
  return cpp11::external_pointer<StochTree::BARTSamples>(samples.release());
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
bool bart_samples_has_mean_forest_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return samples->mean_forests != nullptr;
}

[[cpp11::register]]
bool bart_samples_has_variance_forest_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return samples->variance_forests != nullptr;
}

[[cpp11::register]]
cpp11::writable::doubles bart_samples_global_var_samples_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return vec_to_doubles(samples->global_error_variance_samples);
}

[[cpp11::register]]
cpp11::writable::doubles bart_samples_leaf_scale_samples_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  return vec_to_doubles(samples->leaf_scale_samples);
}

// Materialize a standalone deep copy of a forest container (for the deprecated direct forest
// accessor on the R side). The R6 wrapper guards these behind has_*_forest().
[[cpp11::register]]
cpp11::external_pointer<StochTree::ForestContainer> bart_samples_materialize_mean_forest_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  auto copy = clone_forest_container(samples->mean_forests.get());
  return cpp11::external_pointer<StochTree::ForestContainer>(copy.release());
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::ForestContainer> bart_samples_materialize_variance_forest_cpp(cpp11::external_pointer<StochTree::BARTSamples> samples) {
  auto copy = clone_forest_container(samples->variance_forests.get());
  return cpp11::external_pointer<StochTree::ForestContainer>(copy.release());
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
cpp11::external_pointer<StochTree::BCFSamples> bcf_samples_from_components_cpp(
    cpp11::external_pointer<StochTree::ForestContainer> mu_forest,
    cpp11::external_pointer<StochTree::ForestContainer> tau_forest,
    cpp11::sexp variance_forest,
    cpp11::sexp global_var_samples, cpp11::sexp leaf_scale_mu_samples,
    cpp11::sexp leaf_scale_tau_samples, cpp11::sexp tau_0_samples,
    cpp11::sexp b0_samples, cpp11::sexp b1_samples,
    double y_bar, double y_std, int num_samples, int treatment_dim) {
  auto samples = std::make_unique<StochTree::BCFSamples>();
  // mu/tau forests are always present in BCF; variance is optional.
  samples->mu_forests = clone_forest_container(mu_forest.get());
  samples->tau_forests = clone_forest_container(tau_forest.get());
  if (variance_forest != R_NilValue) {
    cpp11::external_pointer<StochTree::ForestContainer> fc(variance_forest);
    samples->variance_forests = clone_forest_container(fc.get());
  }
  auto assign_if = [](cpp11::sexp src, std::vector<double>& dst) {
    if (src != R_NilValue) {
      cpp11::doubles v(src);
      dst.assign(v.begin(), v.end());
    }
  };
  assign_if(global_var_samples, samples->global_error_variance_samples);
  assign_if(leaf_scale_mu_samples, samples->leaf_scale_mu_samples);
  assign_if(leaf_scale_tau_samples, samples->leaf_scale_tau_samples);
  assign_if(tau_0_samples, samples->tau_0_samples);
  assign_if(b0_samples, samples->b0_samples);
  assign_if(b1_samples, samples->b1_samples);
  samples->y_bar = y_bar;
  samples->y_std = y_std;
  samples->num_samples = num_samples;
  samples->treatment_dim = treatment_dim;
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
