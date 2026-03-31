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

enum ForestLeafModel {
    kConstant, 
    kUnivariateRegression, 
    kMultivariateRegression
};
