# Regression Testing for BART and BCF

This directory contains scripts that constitute a lowkey "regression testing" procedure for `stochtree`'s BART and BCF models. The goal is to ensure that the models are functioning correctly and that any changes made to the code do not introduce new bugs.

## Overview

`stochtree` is by its nature a stochastic software tool, meaning that it generates different results each time it is run. This complicates regression testing slightly, as we cannot always distinguish between a genuine performance regression and an expected variation in results.

For this reason, the "regression testing" setup is somewhat informal. It is designed to catch any strinkingly obvious issues. Both BART and BCF are run on a set of example datasets, and the results can then be compared against previously saved outputs.

## Usage

The primary scripts for running the regression tests are located in the `tools/regression/bart` and `tools/regression/bcf` directories. Here, there are scripts that run each models on a variety of datasets and combine and summarize their results (RMSE, coverage, runtime).

This section documents how to run the end-to-end regression test suite for both the R and Python packages of `stochtree`. For more information on the individual tests, see the following section for documentation on the data generating processes tested and the test script parameters.

### R Package

To run the regression tests for the R package, first navigate to the `stochtree` repository in your terminal and then run:

```bash
Rscript tools/regression/bart/regression_test_dispatch_bart.R
Rscript tools/regression/bcf/regression_test_dispatch_bcf.R
```

Then, to combine and analyze the results, run:

```bash
Rscript tools/regression/bart/regression_test_analysis_bart.R
Rscript tools/regression/bcf/regression_test_analysis_bcf.R
```

This can be compared against any previously saved results. We are currently figuring out a solution for hosting previous regression results, so for now, you will need to manually save and compare outputs of the tests locally before and after after making your changes.

### Python Package

To run the regression tests for the Python package, first navigate to the `stochtree` repository in your terminal and then run:

```bash
python tools/regression/bart/regression_test_dispatch_bart.py
python tools/regression/bcf/regression_test_dispatch_bcf.py
```

Then, to combine and analyze the results, run:

```bash
python tools/regression/bart/regression_test_analysis_bart.py
python tools/regression/bcf/regression_test_analysis_bcf.py
```

## Individual Regression Tests

### BART

#### Data-generating processes (DGPs):

1. **DGP 1**: Basic BART without basis or random effects
2. **DGP 2**: BART with basis but no random effects
3. **DGP 3**: BART with random effects but no basis
4. **DGP 4**: BART with both basis and random effects

#### Script

The individual regression tests are dispatched by a `tools/regression/bart/regression_test_dispatch_bart` script for R or Python, both of which accept the following (options) command line arguments:

- `n_iter`: Number of iterations (default: 5)
- `n`: Sample size (default: 1000)
- `p`: Number of covariates (default: 5)
- `num_gfr`: Number of GFR iterations (default: 10)
- `num_mcmc`: Number of MCMC iterations (default: 100)
- `dgp_num`: Data generating process number 1-4 (default: 1)
- `snr`: Signal-to-noise ratio (default: 2.0)
- `test_set_pct`: Test set percentage (default: 0.2)
- `num_threads`: Number of threads, -1 for all available (default: -1)

Run this script in python:

```bash
python tools/regression/bart/individual_regression_test_bart.py [n_iter] [n] [p] [num_gfr] [num_mcmc] [dgp_num] [snr] [test_set_pct] [num_threads]
```

or in R:

```bash
Rscript tools/regression/bart/individual_regression_test_bart.R [n_iter] [n] [p] [num_gfr] [num_mcmc] [dgp_num] [snr] [test_set_pct] [num_threads]
```

#### Output

BART results are saved to CSV files in the `tools/regression/bart/stochtree_bart_python_results/` or `tools/regression/bart/stochtree_bart_r_results/` directory with filenames that encode the parameter values. Each file contains:

- Parameter values (n, p, num_gfr, num_mcmc, dgp_num, snr, test_set_pct, num_threads)
- Iteration number
- RMSE on test set
- Coverage of 95% prediction intervals
- Runtime in seconds

### BCF

#### Data-generating processes (DGPs):

1. **DGP 1**: Basic BCF without random effects
2. **DGP 2**: BCF with multivariate treatment but no random effects
3. **DGP 3**: BCF with random effects but univariate treatment
4. **DGP 4**: BCF with both multivariate treatment and random effects

#### Script

The individual regression tests are dispatched by a `tools/regression/bcf/regression_test_dispatch_bcf` script for R or Python, both of which accept the following (options) command line arguments:

- `n_iter`: Number of iterations (default: 5)
- `n`: Sample size (default: 1000)
- `p`: Number of covariates (default: 5)
- `num_gfr`: Number of GFR iterations (default: 10)
- `num_mcmc`: Number of MCMC iterations (default: 100)
- `dgp_num`: Data generating process number 1-4 (default: 1)
- `snr`: Signal-to-noise ratio (default: 2.0)
- `test_set_pct`: Test set percentage (default: 0.2)
- `num_threads`: Number of threads, -1 for all available (default: -1)

Run this script in python:

```bash
python tools/regression/bcf/individual_regression_test_bcf.py [n_iter] [n] [p] [num_gfr] [num_mcmc] [dgp_num] [snr] [test_set_pct] [num_threads]
```

or in R:

```bash
Rscript tools/regression/bcf/individual_regression_test_bcf.R [n_iter] [n] [p] [num_gfr] [num_mcmc] [dgp_num] [snr] [test_set_pct] [num_threads]
```

#### Outputs

BCF results are saved to CSV files in the `tools/regression/bcf/stochtree_bcf_python_results/` or `tools/regression/bcf/stochtree_bcf_r_results/` directory with filenames that encode the parameter values. Each file contains:

- Parameter values (n, p, num_gfr, num_mcmc, dgp_num, snr, test_set_pct, num_threads)
- Iteration number
- Outcome RMSE and coverage
- Treatment effect RMSE and coverage
- Runtime in seconds
