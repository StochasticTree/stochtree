# Debugging

This subdirectory contains a debug program for the C++ codebase.
The program takes several command line arguments (in order):

1. Which data-generating process (DGP) to run (integer-coded, see below for a detailed description)
2. Whether or not to include random effects (0 = no, 1 = yes)
3. Number of grow-from-root (GFR) samples
4. Number of MCMC samples
5. Seed for random number generator (-1 means we defer to C++ `std::random_device`)

The DGPs are numbered as follows:

0. Simple leaf regression model with a univariate basis for the leaf model
1. Constant leaf model with a large number of deep interactions between features
