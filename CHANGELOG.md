# stochtree 0.1.2

## R

* Fixed indexing bug in cleanup of grow-from-root (GFR) samples in BART and BCF models
* Avoid using covariate preprocessor in `computeForestLeafIndices` function when a `ForestSamples` object is provided

## Python

* Specify float dtype in numpy ones and zeros (i.e. `np.zeros(num_zeros, dtype=float)`) [#161](https://github.com/StochasticTree/stochtree/pull/161)

# stochtree 0.1.1

## R

* Fixed initialization bug in several R package code examples for random effects models

# stochtree 0.1.0

## Python

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

## R

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
