# Release Process

This document describes the end-to-end process for releasing a new version of stochtree. Both the R and Python packages share a version number (`x.y.z`), but they publish on different timelines: Python publishes to PyPI immediately upon submission; R requires a separate CRAN submission that goes through CRAN's review queue.

## Quick reference

| Step | Action |
|------|--------|
| 1 | Open a release tracking issue from the [release template](.github/ISSUE_TEMPLATE/release.md) |
| 2 | Bump versions and clean up NEWS.md |
| 3 | Merge all changes to `main` |
| 4 | Create a GitHub **pre-release** to trigger automated test suite |
| 5 | Review results; fix and re-tag if needed |
| 6 | Promote to **full release** → PyPI publishes, `r-dev` updates automatically |
| 7 | Submit R package to CRAN manually |
| 8 | Bump versions to next dev cycle |

---

## Files to update before releasing

### Version numbers

| File | Field | Example |
|------|-------|---------|
| `pyproject.toml` | `version = "x.y.z"` | `version = "0.4.2"` |
| `R/DESCRIPTION` (generated from `DESCRIPTION` at repo root) | `Version: x.y.z` | `Version: 0.4.2` |
| `configure.ac` | `AC_INIT([stochtree], [x.y.z], [], [stochtree], [])` | `AC_INIT([stochtree], [0.4.2], [], [stochtree], [])` |
| `configure` | Either find and replace or re-generate from `autoconf` |  |
| `Doxyfile` | `PROJECT_NUMBER = x.y.z` | `PROJECT_NUMBER = 0.4.2` |

### Changelog / NEWS

- **`NEWS.md`**: Rename the dev heading (`x.y.z.9000`) to the release version (`x.y.z`). Remove any empty subsections (`## New Features`, `## Bug Fixes`, etc.) that have no entries. After release, add a new `x.y.(z+1).9000` heading for the next cycle. This is the "changelog" source; the documentation website includes it directly.

### Sanity checks

- No empty version section in `NEWS.md` (CRAN will flag this)
- `MANIFEST.in` includes any new C++ dependency header trees added since the last release
- All tests pass locally (`source venv/bin/activate && python -m pytest` and `NOT_CRAN=true Rscript -e "devtools::load_all('.'); testthat::test_dir('test/R/testthat')"`)

---

## GitHub release states and what they trigger

stochtree uses GitHub's three-stage release flow:

```
Draft  →  Pre-release  →  Full release
         (prereleased)    (published)
              ↓                ↓
        slow test suite   PyPI wheel build
        regression tests  r-dev update
        reproducibility
```

### Draft

Create a draft to write the release description and tag without triggering anything. The release is not public at this stage.

**How:** Releases → "Draft a new release" → fill in tag (e.g. `v0.4.2`), target (`main`), title, description → click **"Save draft"**.

### Pre-release

Publishing the draft as a pre-release fires the `prereleased` event, which automatically dispatches:

- `r-python-slow-api-test` (unit + integration tests, 3 OS)
- `regression-test` (benchmark suite, linux)
- `reproducibility_check` (cross-platform RNG consistency, 3 OS)
- `r-valgrind-check` (memory error detection, linux)
- `r-devel-check` (CRAN R-devel compatibility, 3 OS)

**How:** Edit the draft → check **"Set as a pre-release"** → click **"Publish release"**.

Review the Actions tab for results. If a check fails, fix the issue on `main`, then edit the release to point the tag at the new HEAD (delete and recreate the tag, or use the "Edit" → retag flow).

### Full release

Promoting the pre-release to a full release fires `published`, which:

- **Automatically updates the `r-dev` branch** (via `r-cran-branch.yml`) and tags the R release as `r-x.y.z` (which from 0.4.2-on is a git tag but not a separate github release)
- **Triggers wheel builds** (via `pypi-wheels.yml`); PyPI publication is currently a **manual step** — download the wheel artifacts from the Actions run and upload via `twine upload dist/*`. Setting up [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/) (OIDC) will automate this and in upcoming release cycles.

**How:** Edit the pre-release → uncheck **"Set as a pre-release"** → click **"Update release"**.

---

## R package: CRAN submission

After the full release is published and `r-dev` has been updated, submit the R package to CRAN:

```r
# From the repo root
Rscript cran-bootstrap.R 0 0 1   # generates stochtree_cran/
cd stochtree_cran
R CMD build .
R CMD check --as-cran stochtree_0.4.2.tar.gz
# If checks pass:
devtools::submit_cran()           # or use https://cran.r-project.org/submit.html
```

CRAN review typically takes a few days. The GitHub release marks when the package was *submitted*, not when it appears on CRAN.

---

## Post-release

1. On `main`, bump both version files to the next dev cycle:
   - `pyproject.toml`: `"x.y.(z+1)-dev"`
   - `DESCRIPTION`: `x.y.(z+1).9000`
   - `Doxyfile`: `x.y.(z+1).9000`
   - `configure.ac` / `configure`: `x.y.(z+1).9000`
2. Add a new `# stochtree x.y.(z+1).9000` heading (with empty subsections) to `NEWS.md`.
3. Close the release tracking issue.
