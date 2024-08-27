# Debugging

This subdirectory contains a debug program for the C++ codebase.
The program takes several command line arguments (in order):

1. Which data-generating process (DGP) to run (integer-coded, see below for a detailed description)
1. Which leaf model to sample (integer-coded, see below for a detailed description)
3. Whether or not to include random effects (0 = no, 1 = yes)
4. Number of grow-from-root (GFR) samples
5. Number of MCMC samples
6. Seed for random number generator (-1 means we defer to C++ `std::random_device`)

The DGPs are numbered as follows:

0. Simple leaf regression model with a univariate basis for the leaf model
1. Constant leaf model with a large number of deep interactions between features

The models are numbered as follows:

0. Constant leaf tree model (the "classic" BART / XBART model)
1. "Univariate basis" leaf regression model
2. "Multivariate basis" leaf regression model
