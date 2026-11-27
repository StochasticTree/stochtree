---
name: cross-language-bugfix
description: >-
  Use this skill when investigating or fixing a bug in stochtree (across C++, R,
  or Python) to ensure root-cause parity across both host languages, proper
  extension rebuilds, and paired regression tests.
---

# Cross-Language Bugfix Workflow

Because `stochtree` implements its outer Gibbs/GFR sampling loops, data standardization, prior calibration, link transformations, posterior predictive draws, and JSON serialization **in parallel** across `R/{bart,bcf}.R` and `stochtree/{bart,bcf}.py` on top of shared C++ primitives, bugs reported in one language very frequently exist (or have a closely related variant) in the other.

## Step 1: Reproduce & Audit All Three Layers

1. **Reproduce the reported failure** with a minimal script or focused unit test before editing code.
2. **Audit the counterpart implementation** across all three layers, even if the issue only mentions Python or R:
   - **C++ Core**: `include/stochtree/*.h` and shared `src/*.cpp`
   - **Python FFI & High-Level**: `src/py_stochtree.cpp` and `stochtree/{bart,bcf,sampler,forest,serialization,utils,preprocessing}.py`
   - **R FFI & High-Level**: `src/{forest,sampler,R_data,R_random_effects,serialization}.cpp` and `R/{bart,bcf,model,forest,posterior_transformation,serialization,utils}.R`
3. Check **both `BART` and `BCF`** (and both MCMC and GFR / warm-start paths if relevant)—many routines are duplicated between `bart` and `bcf` within the same language (e.g., outcome standardization, residual initialization, `OutcomeModel` parsing, `observation_weights`, `__str__`/`print`, JSON roundtrip).

## Step 2: Implement Fix & Rebuild Affected Targets

- **If C++ headers or core `src/*.cpp` changed**:
  1. Build and run C++ unit tests:
     ```bash
     cmake -S . -B build -DBUILD_TEST=ON && cmake --build build -j && ./build/teststochtree
     ```
  2. Rebuild the Python C++ extension in-place:
     ```bash
     ./venv/bin/pip install --no-build-isolation -e .
     ```
  3. If any `[[cpp11::register]]` signature changed in `src/`, run `Rscript -e "cpp11::cpp_register()"`. (`devtools::load_all('.')` recompiles `src/*.o` automatically for R.)
- **If Roxygen docstrings in `R/*.R` changed**:
  - Run `Rscript -e "devtools::document()"` so `man/*.Rd` and `NAMESPACE` stay in sync.

## Step 3: Add Paired Regression Tests & Verify

1. Add focused, fast deterministic regression tests covering the exact failure mode:
   - **Python**: `test/python/test_*.py` -> `./venv/bin/python -m pytest test/python/<file>.py`
   - **R**: `test/R/testthat/test-*.R` -> `NOT_CRAN=true Rscript -e "devtools::load_all('.'); testthat::test_file('test/R/testthat/<file>.R')"`
   - **C++** (when fixing C++ core primitives): `test/cpp/test_*.cpp` -> `./build/teststochtree`
2. If serialization or legacy model fields were touched, run the backward-compatibility suites:
   - `./venv/bin/python -m pytest test/python/test_serialization_compat.py test/python/test_json.py`
   - `NOT_CRAN=true Rscript -e "devtools::load_all('.'); testthat::test_file('test/R/testthat/test-serialization-compat.R'); testthat::test_file('test/R/testthat/test-serialization.R')"`

## Step 4: Check RNG Reproducibility & `NEWS.md` Impact

- Determine whether the fix alters the RNG draw sequence on valid inputs for a fixed `random_seed` (for standard Gaussian BART/BCF).
- Update `NEWS.md` under the current `.9000` dev heading if appropriate (and note explicitly if fixed-seed outputs change).
