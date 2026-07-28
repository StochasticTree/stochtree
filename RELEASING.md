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
- If sampler output changed this cycle, regenerate the reproducibility references under `tools/reproducibility/{R,python}` and confirm the `reproducibility_check` workflow **fails on drift** against the new references. (A stale-reference / lenient-check mismatch caused a post-tag follow-up in 0.4.5.)

---

## Release notes template

Copy this into the GitHub release body (or into a `release_notes.md` file to use with `gh release create --notes-file`). Replace `x.y.z` throughout and paste the matching section from `NEWS.md` under Changes.

````markdown
## Installation

**Python:**

```
pip install stochtree==x.y.z
```

**R:**

**CRAN:** `install.packages("stochtree")` *(pending CRAN review)*
**GitHub, immediate:** `remotes::install_github("StochasticTree/stochtree@r-x.y.z")`

## Changes

<!-- paste the stochtree x.y.z section from NEWS.md here -->
````

To create the draft from the command line instead of the GitHub UI:

```bash
gh release create vx.y.z --title "stochtree x.y.z" --target main --notes-file release_notes.md --prerelease --draft
```

## Updating a pre-release

If pre-release workflows trigger integration test failures or serious performance regressions that require updating code, the cleanest way to "update" the release candidate is to delete the published pre-release (and its associated tag), either in the github UI or via CLI

```
gh release delete v0.4.2 --repo StochasticTree/stochtree
git push origin --delete v0.4.2
```

Then you can start a new draft and pre-release it as above

## Do post-tag fixes need a re-tag?

When a fix lands on `main` after the release tag is cut, decide by what it touches:

- **Re-tag** if it changes anything shipped in the R/Python packages — `R/`, `src/`, `stochtree/`, the version files, or `NEWS.md`.
- **Leave the tag** if it only touches things outside the distributed packages — CI workflows, `tools/reproducibility/` references, benchmarks, docs-site config. The tagged artifact is unchanged, so moving the tag buys nothing. (In 0.4.5, a CI-only reproducibility fix landed post-tag and was correctly left out of the release.)

## Updating a full release (minor fixes after promotion)

For small fixes (doc corrections, man page updates, typos) discovered after promoting to a full release, delete-and-recreate is overkill. Instead, merge the fix to `main` and force-move the tag:

```bash
# After merging the fix to main
git fetch origin
git tag -f vx.y.z origin/main
git push --force origin vx.y.z
```

The GitHub release automatically follows the tag — no UI edits needed.

Because the `published`/`released` events already fired, the packaging workflows won't re-run automatically. Re-dispatch them manually:

```bash
gh workflow run pypi-wheels.yml --repo StochasticTree/stochtree
gh workflow run r-cran-branch.yml --repo StochasticTree/stochtree
```

Or use **Actions → [workflow name] → Run workflow** in the GitHub UI.

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
- **Triggers wheel builds** (via `pypi-wheels.yml`); PyPI publication is currently a **manual step** — see [PyPI publication](#pypi-publication-manual) below.

**How (UI):** Edit the pre-release → uncheck **"Set as a pre-release"** → click **"Update release"**.

**How (CLI):**

```bash
gh release edit vx.y.z --repo StochasticTree/stochtree --prerelease=false --latest
```

`--latest` forces the "Latest" badge (and `/releases/latest` redirect) onto this release rather than relying on GitHub's auto-detection. A release cannot be both prerelease and latest, so these two flags work together. This does not affect which workflows fire — the `published`/`released` events trigger regardless.

### PyPI publication (manual)

After promotion, the `pypi-wheels.yml` run on the `published` event builds the per-platform **wheels** as artifacts (it does **not** build an sdist). Download the wheels, build the sdist locally, then upload both:

```bash
# 1. download the wheel artifacts from the release run
# find the run id (event=release, workflow "Build Python Wheels for PyPI")
gh run list --repo StochasticTree/stochtree --event release
gh run download <run-id> --repo StochasticTree/stochtree --dir dist_wheels

# 2. build the sdist locally (the CI workflow does not produce one)
rm -rf dist && pipx run build --sdist

# 3. upload wheels + sdist
twine upload dist_wheels/**/*.whl dist/*.tar.gz
```

Confirm `x.y.z` appears at https://pypi.org/project/stochtree/. Setting up [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/) (OIDC) will automate this in an upcoming release cycle.

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

---

## Major overhaul releases (release-candidate)

For large or breaking changes — a sampler-core rewrite, a serialization schema change, anything that touches most of the fitting or (de)serialization surface — the normal "merge to `main`, then pre-release" flow is too risky: `main` is what source installers pull from, and any significant code changes deserve some time to be "battle-tested" by an array of users. So, we initially publish the change as a **release-candidate** branch and only merge once it has been thoroughly test. `main` stays on the last stable release the entire time, so patch releases from `main` remain possible.

### Branch topology

- Create a `rc-x.y.z` branch from `main`. The overhaul PR merges into **`rc-x.y.z`, not `main`** (maintainers can -- optionally -- squash commit history at this point).
- Bug fixes found during testing land on `rc-x.y.z` (directly or via small PRs).
- `main` is untouched and releasable throughout — an urgent patch can still ship as `x.(y-1).z+1` (or whatever the appropriate increment of the current version number) from `main` without waiting the RC to merge.

### Building installable release candidates

We have a **`custom-release-candidate.yml`** workflow that produces two purpose-built install branches from `rc-x.y.z`:

| Input | Value | Produces |
|-------|-------|----------|
| `source_branch` | `rc-x.y.z` | (the candidate source) |
| `r_target_branch` | `r-rc-x.y.z` | CRAN-rooted R package (via `cran-bootstrap.R`), installable branch |
| `python_target_branch` | `py-rc-x.y.z` | submodule-flattened, `pip`-installable branch |

Testers then install with no extra toolchain wrangling:

```r
# R
remotes::install_github("StochasticTree/stochtree@r-rc-x.y.z")
```
```bash
# Python
pip install "git+https://github.com/StochasticTree/stochtree@py-rc-x.y.z"
```

We deliberately **do not** publish RC builds to PyPI or an r-universe repo. RC testers can compile from source.

**Keep the install branches fresh:** the RC build is manual by default, so re-run `custom-release-candidate.yml` after every meaningful update to `rc-x.y.z`, or add a `push: branches: [rc-x.y.z]` trigger to it for the duration of the test (see Automation, below) so R and Python testers never drift apart or onto stale code.

### Test and freeze policy

The cost of holding a major overhaul off `main` is drift against any sampler-adjacent change that lands meanwhile. Keep it survivable — and short:

- **Named, time-boxed testing window.** Recruit a specific set of source-building users and set an end date up front.
- **Freeze overhaul-adjacent PRs on `main`** for the duration; route them to `rc-x.y.z` instead. The freeze cost scales with test length — another reason to keep it short.
- **Back-merge `main → rc-x.y.z` frequently** so the eventual `rc → main` merge is small.
- **Track feedback in one place** — a pinned issue or Discussion carrying the promote checklist below.

### Promote gates (run against `rc-x.y.z` before merging)

Dispatch heavy workflows against the RC branch (Actions → workflow → Run workflow → select `rc-x.y.z`); each is `workflow_dispatch`-enabled:

- `r-python-slow-api-test` — unit + slow integration, 3 OS
- `regression-test` — benchmark suite; **compare fit time / memory to the last release**, not just "does it run" (a silent perf regression is exactly what a rewrite risks)
- `reproducibility_check` — cross-platform RNG consistency, 3 OS
- `cross-language-parity` — R↔Python prediction agreement
- `r-valgrind-check` / `r-devel-check` — memory + CRAN R-devel

Plus two gates specific to overhaul PRs, not covered by a single workflow:

- **Serialization forward-compat:** load models saved by the last released `x.y.z` into the RC and confirm correct predictions (the highest-risk surface for a schema change). Also RC-saved → RC-loaded round-trips and, within the all-numeric gate, cross-platform (R↔Python) loads.
- **Reproducibility-reference regeneration — equivalence *then* regenerate.** A sampler change legitimately shifts stored references (v1 golden serialization fixtures, any parameter-bearing fixtures — e.g. models that store `tau_0`, whose stored scale changed — continuation snapshots, and benchmark baselines). The failure mode is regenerating a reference to turn a red test green while a real regression rides along. So: **first prove the new outputs are correct** (bit-identical where we still claim it; statistically-equivalent where we intentionally relaxed it, e.g. continuation), **then** regenerate the references, **on `rc-x.y.z` only** — `main`'s references must stay valid until promote so `main` can still patch-release against them.

### Promote

When the test passes its exit criteria:

1. Open PR and merge `rc-x.y.z → main`.
2. Run the **standard release flow** (top of this document): bump versions off the dev suffix, clean `NEWS.md`, draft → **pre-release** (fires the gate suite automatically) → **full release** (PyPI wheels + `r-dev`/`r-x.y.z`) → CRAN submission.
3. Delete the `rc-x.y.z`, `r-rc-x.y.z`, and `py-rc-x.y.z` branches (or keep `rc-x.y.z` briefly as history).

### Automation status

Most of this is already automated; the remaining gaps are small:

- **Already automated:** the R CRAN-transform + Python submodule-flatten RC build (`custom-release-candidate.yml`); every promote gate (dispatchable against any branch); cross-language parity.
- **Worth adding for the test (highest value first):**
  1. **Auto-refresh the install branches** — a `push: branches: [rc-x.y.z]` trigger on `custom-release-candidate.yml` (or a thin wrapper that calls it) so `r-rc-x.y.z` / `py-rc-x.y.z` rebuild on every RC push. Removes the "did I remember to re-run it?" drift risk.
  2. **One-dispatch gate sweep** — a small umbrella workflow that fans out the promote-gate workflows against `rc-x.y.z`, so a reviewer kicks them all with one click.
  3. *(optional)* a **scheduled `main → rc-x.y.z` back-merge** PR, and **RC version stamping** (`x.y.z.9000` / `x.y.zrcN`) during the RC build.
- **Deliberately kept manual (human judgment):** declaring the test done, reviewing gate results, and the equivalence call that must precede any reference regeneration.
