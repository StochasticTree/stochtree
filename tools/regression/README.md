# Regression Testing for BART and BCF

This directory contains scripts that constitute a lowkey "regression testing" procedure for `stochtree`'s BART and BCF models. The goal is to ensure that the models are functioning correctly and that any changes made to the code do not introduce new bugs.

## Overview

`stochtree` is by its nature a stochastic software tool, meaning that it generates different results each time it is run. This complicates regression testing slightly, as we cannot always distinguish between a genuine performance regression and an expected variation in results.

For this reason, the "regression testing" setup is somewhat informal. It is designed to catch any strinkingly obvious issues. Both BART and BCF are run on a set of example datasets, and the results can then be compared against previously saved outputs.

## Usage

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
