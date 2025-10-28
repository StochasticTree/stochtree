# Release Candidate for StochTree Cloglog BART

This branch serves as a staging / testing zone for the planned incorporation of BART / BCF with a complementary log-log link function into `stochtree`.

## Installation

The cloglog release candidate version of `stochtree` can be installed from github via

```
remotes::install_github("StochasticTree/stochtree", ref="cloglog-bart-rc")
```

## Vignettes and Demos

Before incorporating this functionality into `stochtree`, we intend to develop a rich set of vignettes.
We have included demo scripts for the cloglog model on synthetic ordinal data with 2, 3 and 4 categories in the `tools` subfolder of this branch.
