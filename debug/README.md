# Debugging

This subdirectory contains two standalone C++ debug programs.

---

## `debugstochtree` — low-level API smoke tests

Exercises the low-level forest sampling API directly (no high-level dispatch).
Takes several command line arguments (in order):

1. Which data-generating process (DGP) to run (integer-coded, see below)
2. Which leaf model to sample (integer-coded, see below)
3. Whether or not to include random effects (0 = no, 1 = yes)
4. Number of grow-from-root (GFR) samples
5. Number of MCMC samples
6. Seed for random number generator (-1 = defer to `std::random_device`)
7. [Optional] name of data file to load for training instead of simulating data (`""` = simulate)
8. [Optional] index of outcome column in data file (`0` = none)
9. [Optional] comma-delimited string of column indices of covariates (`""` = none)
10. [Optional] comma-delimited string of column indices of leaf regression bases (`""` = none)
11. [Optional] number of threads to use in the GFR sampler (`-1` = default)

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

Example — DGP 0, leaf model 1, no random effects, 10 GFR samples, 100 MCMC samples, default seed:

```
./build/debugstochtree 0 1 0 10 100 -1 "" 0 "" "" -1
```

Build target: `debugstochtree`

---

## `debug_bart_sampler` — high-level dispatch smoke tests

Exercises `BARTSamplerFit()` across all model types supported by the C++ dispatch API.
Prints pass/fail for basic sanity checks (finite predictions, correct sample counts, etc.).

```
./build/debug_bart_sampler                    # run all smoke tests
./build/debug_bart_sampler --model identity   # run a single model
./build/debug_bart_sampler --help             # list available model names
```

Available model names: `identity`, `probit`, `varforest`, `cloglog`, `ordinal`,
`mean+varforest`, `leaf-reg`, `leaf-reg-mv`, `rfx`, `all` (default).

Build target: `debug_bart_sampler`

---

Both programs are built when `BUILD_DEBUG_TARGETS=ON` (the default):

```
cmake -DBUILD_DEBUG_TARGETS=ON -B build && cmake --build build
```
