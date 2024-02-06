/*! Copyright (c) 2023 by randtree authors. */
#include <stochtree/ensemble.h>
#include <stochtree/model_graph.h>

namespace StochTree {

ModelGraph::ModelGraph(ModelGraph& model_graph) {
  // Copy int vectors and maps
  basis_dim_ = model_graph.basis_dim_;
  forest_map_ = model_graph.forest_map_;
  parameter_map_ = model_graph.parameter_map_;

  // Copy forests
  int num_forests = model_graph.ensembles_.size();
  ensembles_.resize(num_forests);
  for (int i = 0; i < num_forests; i++) {
    ensembles_[i].reset(new TreeEnsemble(*model_graph.ensembles_[i].get()));
  }
  
  // Copy parameters
  int num_params = model_graph.parameters_.size();
  parameters_.resize(num_params);
  for (int i = 0; i < num_params; i++) {
    parameters_[i] = model_graph.parameters_[i];
  }
}

ModelGraph& ModelGraph::operator=(ModelGraph& model_graph) {
  // Copy int vectors and maps
  basis_dim_ = model_graph.basis_dim_;
  forest_map_ = model_graph.forest_map_;
  parameter_map_ = model_graph.parameter_map_;

  // Copy forests
  int num_forests = model_graph.ensembles_.size();
  ensembles_.resize(num_forests);
  for (int i = 0; i < num_forests; i++) {
    ensembles_[i].reset(new TreeEnsemble(*model_graph.ensembles_[i].get()));
  }
  
  // Copy parameters
  int num_params = model_graph.parameters_.size();
  parameters_.resize(num_params);
  for (int i = 0; i < num_params; i++) {
    parameters_[i] = model_graph.parameters_[i];
  }
  
  return *this;
}

} // namespace StochTree
