---
name: cran-preflight
description: >-
  Use this skill when preparing or validating the R package for CRAN compliance,
  modifying cran-bootstrap.R, upgrading C++ submodules (such as Eigen or fmt),
  or debugging R CMD check warnings and notes.
---

# CRAN Preflight & Bootstrap Check

`stochtree`'s repository root is **not** structured as a standalone CRAN tarball because it relies on `include/` and git submodules in `deps/`, mixes Python `pybind11` code in `src/`, and stores R tests in `test/R/`. `cran-bootstrap.R` transforms the repo into a self-contained CRAN staging directory (`stochtree_cran/`).

## Step 1: Sync Generated R Artifacts

Before running the bootstrap script, ensure `cpp11` wrappers and Roxygen docs at the repo root are current:

```bash
git submodule update --init --recursive
Rscript -e "cpp11::cpp_register(); devtools::document()"
```

## Step 2: Stage the CRAN Package (`stochtree_cran/`)

Run `cran-bootstrap.R` with `(include_vignettes = 0, pkgdown_build = 0, include_tests = 1)` (the exact CRAN submission configuration from `RELEASING.md` and `.github/workflows/r-test.yml`):

```bash
rm -rf stochtree_cran
Rscript cran-bootstrap.R 0 0 1
```

*(Use `Rscript cran-bootstrap.R 1 0 1` if you also want to verify vignette building.)*

Verify what `cran-bootstrap.R` produced:
- `src/py_stochtree.cpp` and `src/Makevars` / `src/*.o` are **not** present in `stochtree_cran/src/`.
- `stochtree_cran/src/include/` contains vendored `stochtree/`, `nlohmann/`, `Eigen/`, `fmt/`, and `fast_double_parser.h`.
- `stochtree_cran/tests/` contains `testthat.R` and `testthat/test-*.R` (without `fixtures/*.json` or `test-serialization-compat.R`).

## Step 3: Build Tarball & Run `R CMD check --as-cran`

```bash
R CMD build stochtree_cran
# Replace <version> with the Version field in DESCRIPTION (e.g. 0.4.6.9000)
R CMD check --as-cran stochtree_<version>.tar.gz
```

## Step 4: Audit Common CRAN Failure Modes

Inspect `stochtree.Rcheck/00check.log` and `stochtree.Rcheck/00install.out` for zero `ERROR`s, zero `WARNING`s, and expected-only `NOTE`s:

1. **Compiler Diagnostic Pragmas in Vendored Headers**:
   - CRAN flags `#pragma clang diagnostic`, `#pragma GCC diagnostic`, `#pragma warning`, and `#pragma diag_suppress` inside `src/include/`.
   - Search staged headers if flagged: `grep -rn "#pragma" stochtree_cran/src/include/` and add the offending header to the pragma-stripping loop at the bottom of `cran-bootstrap.R`.
2. **Forbidden Compiled Symbols (`stdout`, `stderr`, `abort`, `exit`, `printf`)**:
   - Ensure `src/Makevars.in` passes `-DSTOCHTREE_R_BUILD` and all C++ code uses `StochTree::Log` rather than `<iostream>` or `<cstdio>` output functions.
3. **CPU Core Usage > 2**:
   - Verify no R example or test overrides `num_threads` above `2` without `skip_on_cran()`.
4. **Leftover Files in Temp / Working Directory**:
   - Verify every `@examples` block and `testthat` test writing files uses `tempfile()` and calls `unlink()`.

## Step 5: Clean Up Ephemeral Artifacts

Remove temporary build/check artifacts from the repo root when finished:

```bash
rm -rf stochtree_cran stochtree.Rcheck stochtree_*.tar.gz
```
