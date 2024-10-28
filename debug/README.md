# Debugging

This subdirectory contains a debug program for the C++ codebase.
The program takes several command line arguments (in order):

1. Which data-generating process (DGP) to run (integer-coded, see below for a detailed description)
1. Which leaf model to sample (integer-coded, see below for a detailed description)
3. Whether or not to include random effects (0 = no, 1 = yes)
4. Number of grow-from-root (GFR) samples
5. Number of MCMC samples
6. Seed for random number generator (-1 means we defer to C++ `std::random_device`)
7. [Optional] name of data file to load for training, instead of simulating data (leave this blank as `""` if simulated data is desired)
8. [Optional] index of outcome column in data file (leave this blank as `0`)
9. [Optional] comma-delimited string of column indices of covariates (leave this blank as `""`)
10. [Optional] comma-delimited string of column indices of leaf regression bases (leave this blank as `""`)

The DGPs are numbered as follows:

0. Simple leaf regression model with a univariate basis for the leaf model
1. Constant leaf model with a large number of deep interactions between features
2. Simple leaf regression model with a multivariate basis for the leaf model
3. Simple "variance-only" model with a mean of zero but covariate-moderated variance function

The models are numbered as follows:

0. Constant leaf tree model (the "classic" BART / XBART model)
1. "Univariate basis" leaf regression model
2. "Multivariate basis" leaf regression model
3. Log linear heteroskedastic variance model

For an example of how to run this progam for DGP 0, leaf model 1, no random effects, 10 GFR samples, 100 MCMC samples and a default seed (`-1`), run

`./build/debugstochtree 0 1 0 10 100 -1 "" 0 "" ""`

from the main `stochtree` project directory after building with `BUILD_DEBUG_TARGETS` set to `ON`.
