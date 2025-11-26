# stochtree (development version)

## New Features

## Computational Improvements

## Bug Fixes

* Predict random effects correctly in R for univariate random effects models ([#248](https://github.com/StochasticTree/stochtree/pull/248))

## Documentation Improvements

## Other Changes

# stochtree 0.2.0

## New Features

* Support for multithreading in various elements of the GFR and MCMC algorithms ([#182](https://github.com/StochasticTree/stochtree/pull/182))
* Support for binary outcomes in BART and BCF with a probit link ([#164](https://github.com/StochasticTree/stochtree/pull/164))
* Enable "restricted sweep" of tree algorithms over a handful of trees ([#173](https://github.com/StochasticTree/stochtree/pull/173))
* Support for multivariate treatment in R ([#183](https://github.com/StochasticTree/stochtree/pull/183))
* Enable modification of dataset variables (weights, etc...) via low-level interface ([#194](https://github.com/StochasticTree/stochtree/pull/194))

## Computational Improvements

* Modified default random effects initialization ([#190](https://github.com/StochasticTree/stochtree/pull/190))
* Avoid double prediction on training set ([#178](https://github.com/StochasticTree/stochtree/pull/178))

## Bug Fixes

* Fixed indexing bug in cleanup of grow-from-root (GFR) samples in BART and BCF models
* Avoid using covariate preprocessor in `computeForestLeafIndices` function when a `ForestSamples` object is provided (rather than a `bartmodel` or `bcfmodel` object)
* Correctly compute feature-specific split counts in R and Python ([#220](https://github.com/StochasticTree/stochtree/issues/220))
* Avoid override of user-specified `num_burnin` parameter in BCF models with an internal propensity score ([#222](https://github.com/StochasticTree/stochtree/issues/222))
* Outcome predictions correctly incorporate adaptive coding of untreated observations in BCF with binary treatment ([#231](https://github.com/StochasticTree/stochtree/issues/231))

## Documentation Improvements

* Clarify structure / layout of samples when users request multiple chains in BART and BCF models ([#220](https://github.com/StochasticTree/stochtree/issues/220))

## Other Changes

* Standardized naming conventions for data elements of BART and BCF models across R and Python interfaces
    * Covariates / features are always referred to as "`X`"
    * Treatment is always referred to as "`Z`"
    * Propensity scores are referred to as "`propensity`" (rather than "`pi`")
    * Outcomes are referred to as "`y`"
    * Basis vectors for leaf-wise regression models in forest terms are referred to as "`leaf_basis`"
    * Group labels for additive random effects models are referred to as "`rfx_group_ids`"
    * Basis vectors for additive random effects models are referred to as "`rfx_basis`"
* Run-time checks for variables that are treated as continuous but have many "ties" (which presents issues with the current GFR algorithm) when only GFR samples are requested ([#243](https://github.com/StochasticTree/stochtree/pull/243))

# stochtree 0.1.1

* Fixed initialization bug in several R package code examples for random effects models

# stochtree 0.1.0

* Initial release on CRAN.
* Support for sampling stochastic tree ensembles using two algorithms: MCMC and Grow-From-Root (GFR)
* High-level model types supported:
    * Supervised learning with constant leaves or user-specified leaf regression models
    * Causal effect estimation with binary or continuous treatments
* Additional high-level modeling features:
    * Forest-based variance function estimation (heteroskedasticity)
    * Additive (univariate or multivariate) group random effects
    * Multi-chain sampling and support for parallelism
    * "Warm-start" initialization of MCMC forest samplers via the Grow-From-Root (GFR) algorithm
    * Automated preprocessing / handling of categorical variables
* Low-level interface:
    * Ability to combine a forest sampler with other (additive) model terms, without using C++
    * Combine and sample an arbitrary number of forests or random effects terms
