#include <stochtree/container.h>
#include <stochtree/data.h>
#include <stochtree/leaf_model.h>
#include <stochtree/meta.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/random_effects.h>
#include <stochtree/tree_sampler.h>

enum ForestLeafModel {
    kConstant, 
    kUnivariateRegression, 
    kMultivariateRegression
};
