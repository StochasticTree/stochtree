---
name: release-prep
description: >-
  Use this skill when preparing a new stochtree release or release candidate,
  bumping package version numbers, auditing NEWS.md, testing PyPI sdist builds,
  or checking/regenerating cross-platform reproducibility references.
---

# Release Preparation Workflow

Follow this procedure (grounded in `RELEASING.md`) when preparing a `stochtree` release (`x.y.z`) or bumping to the next development cycle (`x.y.(z+1)-dev` / `x.y.(z+1).9000`).

## Step 1: Synchronize Version Numbers Across All 5 Files

Both R and Python share the same semantic base version `x.y.z`:

| File | Release Format (`x.y.z`) | Dev Cycle Format |
| :--- | :--- | :--- |
| `pyproject.toml` | `version = "x.y.z"` | `version = "x.y.z-dev"` |
| `DESCRIPTION` | `Version: x.y.z` | `Version: x.y.z.9000` |
| `configure.ac` | `AC_INIT([stochtree], [x.y.z], ...)` | `AC_INIT([stochtree], [x.y.z.9000], ...)` |
| `configure` | Regenerate with `autoconf` or update matching `0.x.y*` version strings | Same (`x.y.z.9000`) |
| `Doxyfile` | `PROJECT_NUMBER = x.y.z` | `PROJECT_NUMBER = x.y.z.9000` |

## Step 2: Audit `NEWS.md`

- For a release: rename `# stochtree x.y.z.9000` to `# stochtree x.y.z`.
- **Remove any empty subsections** (`## New Features`, `## Bug Fixes`, `## Documentation and Other Maintenance`) that have no bullet items—CRAN's `NEWS.md` parser flags empty headings.
- For post-release dev bump: prepend a fresh `# stochtree x.y.(z+1).9000` heading at the top of `NEWS.md`.

## Step 3: Packaging & Test Sanity Checks

1. **C++ GoogleTest Suite**:
   ```bash
   cmake -S . -B build -DBUILD_TEST=ON && cmake --build build -j && ./build/teststochtree
   ```
2. **Python `sdist` Build & Install Verification** (catches missing headers in `MANIFEST.in`):
   ```bash
   rm -rf dist && ./venv/bin/python -m build --sdist
   ./venv/bin/python -m pytest test/python
   ```
3. **R Test Suite & CRAN Bootstrap** (invoke the `cran-preflight` skill for full `--as-cran` tarball check):
   ```bash
   NOT_CRAN=true Rscript -e "devtools::load_all('.'); testthat::test_dir('test/R/testthat')"
   ```

## Step 4: Cross-Platform Reproducibility References

Run the reproducibility scripts against the stored CSV references in `tools/reproducibility/{R,python}`:

```bash
./venv/bin/python tools/reproducibility/python/simple-bart.py
./venv/bin/python tools/reproducibility/python/simple-bcf.py
Rscript -e "devtools::load_all('.'); source('tools/reproducibility/R/simple-bart.R')"
Rscript -e "devtools::load_all('.'); source('tools/reproducibility/R/simple-bcf.R')"
```

- **If they fail unexpectedly**: stop and investigate—do **not** regenerate references to mask an unintended regression.
- **If sampler output intentionally changed** in this cycle (and statistical correctness has been verified via exact-posterior / simulation tests and documented in `NEWS.md`): regenerate the 8 reference CSVs under `tools/reproducibility/{R,python}/` and re-run the scripts above to confirm zero mismatches (`*.csv` is in `.gitignore`, so tracked CSV updates show in `git diff`, while any *new* CSV paths require `git add -f`).

## Step 5: Three-Stage GitHub Release Flow

Refer to `RELEASING.md` for cutting the release:
1. **Draft -> Pre-release** (`gh release create vx.y.z ... --prerelease --draft`, then publish pre-release) to trigger slow CI gates (`r-python-slow-api-test`, `regression-test`, `reproducibility_check`, `r-valgrind-check`, `r-devel-check`).
2. **Promote to Full Release** (`gh release edit vx.y.z --prerelease=false --latest`), which triggers `pypi-wheels.yml` and `r-cran-branch.yml` (`r-dev` + `r-x.y.z` tag), followed by manual wheel + sdist `twine upload` and `cran-bootstrap.R 0 0 1` CRAN submission.
