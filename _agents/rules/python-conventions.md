---
trigger: glob
globs: "**/stochtree/**, **/src/py_stochtree.cpp, **/test/python/**, **/pyproject.toml, **/setup.py"
---

# Python Package & NumPy/pybind11 Conventions

Apply these rules when editing `stochtree/*.py`, `src/py_stochtree.cpp`, `test/python/`, or Python build files (`pyproject.toml`, `setup.py`).

## 1. NumPy Dtype & Array Layout Hygiene

Subtle NumPy dtype and memory-order mismatches have historically caused severe numerical bugs in `stochtree`. Enforce these invariants at every public entry point (`sample`, `predict`, `compute_contrast`, `sample_posterior_predictive`, low-level `Dataset` / `Residual`):

- **Explicit `np.float64` Casting**: Cast numeric array inputs (`y_train`, `Z_train`, `Z_test`, `Z`, `propensity_*`, `leaf_basis_*`, `rfx_basis_*`, `observation_weights`) to `np.float64` (`np.asarray(..., dtype=np.float64)` or `.astype(np.float64, copy=False)`) after validation.
  - Never allow boolean arrays (`dtype=bool`) to reach matrix products (`@`) or linear algebra updates—in NumPy, `bool_arr.T @ bool_arr` performs boolean OR-AND algebra (`True` == `1`) instead of summing counts.
  - Never assume user arrays are `float64`; handle `float32` and integer arrays without silent precision/type bugs.
- **Tensor Reshape & Repeat Ordering**: Posterior prediction arrays in Python follow `(n_observations, n_samples)` (2D) or `(n_observations, n_samples, n_draws_per_sample)` / `(n_observations, n_categories, n_samples)` (3D).
  - When expanding a `(n, S)` array by `m = num_draws_per_sample` along the trailing axis via `.reshape(n, S, m)`, use `np.repeat(arr.ravel(), m)` (which repeats each `(i, s)` entry `m` times contiguously), **never** `np.tile(arr.ravel(), m)` (which lays out `(m, n, S)` and scrambles observations upon `.reshape(n, S, m)`).

## 2. API & Diagnostic Consistency

- **Warnings**: Do not emit `DeprecationWarning` for user-facing API deprecations in `stochtree/*.py`—Python filters out `DeprecationWarning` outside `__main__` by default. Use `FutureWarning` or `UserWarning` when warning users, or `ValueError` when a removed option is no longer supported.
- **Model Inspection (`__str__`, `__repr__`, `summary`)**: Derive model descriptions from canonical model state (e.g., `self.outcome_model`, forest config attributes) so printed summaries match the fitted model and R's `print`/`summary` methods. Note that `summary()` prints directly to stdout rather than returning a string (`#310`).
- **Scikit-Learn Estimators**: Keep `StochTreeBARTRegressor` and `StochTreeBARTBinaryClassifier` in `stochtree/sklearn.py` compliant with scikit-learn estimator conventions when `BARTModel` parameters change.
- **Formatting**: Match surrounding PEP 8 / 4-space formatting and avoid unrelated reformatting noise inside bugfix or feature PRs until a repo-wide `ruff` configuration is adopted.

## 3. Rebuilding & Testing

- Editing `src/py_stochtree.cpp` or `include/stochtree/*` requires rebuilding the `stochtree_cpp` extension before running `pytest`:
  `./venv/bin/pip install --no-build-isolation -e .`
- Pure-Python edits in `stochtree/*.py` take effect immediately with an editable install:
  `./venv/bin/python -m pytest test/python`
