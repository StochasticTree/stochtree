---
name: Release checklist
about: Track the steps for a new stochtree release
title: "Release x.y.z"
labels: ''
assignees: ''

---

See [RELEASING.md](../../RELEASING.md) for detailed instructions on each step.

## Preparation

- [ ] Version bumped in `pyproject.toml`
- [ ] Version bumped in `DESCRIPTION`
- [ ] Version bumped in `configure.ac`; `configure` regenerated or find-replaced
- [ ] Version bumped in `Doxyfile`
- [ ] `NEWS.md` dev heading renamed to `x.y.z`; empty subsections removed
- [ ] `MANIFEST.in` updated for any new C++ dependency headers
- [ ] All changes merged to `main`

## Pre-release and testing

- [ ] Release notes drafted using template in `RELEASING.md` (installation instructions + NEWS.md section)
- [ ] GitHub draft release created (tag: `vx.y.z`, target: `main`)
- [ ] Draft published as pre-release → automated test suite fires
- [ ] `r-python-slow-api-test` passed (all 3 OSes)
- [ ] `regression-test` passed
- [ ] `reproducibility_check` passed (all 3 OSes)
- [ ] `r-valgrind-check` passed
- [ ] `r-devel-check` passed (all 3 OSes)

## Publishing

- [ ] Pre-release promoted to full release
- [ ] `pypi-wheels` build completed; wheel artifacts downloaded and uploaded to PyPI (`twine upload`)
- [ ] PyPI package available: `pip install stochtree==x.y.z`
- [ ] `r-dev` branch updated and `r-x.y.z` tag created (check Actions tab for `r-cran-branch` run)
- [ ] R CMD check passed locally on `stochtree_cran/` before CRAN submission
- [ ] R package submitted to CRAN

## Post-release

- [ ] `pyproject.toml` bumped to `x.y.(z+1)-dev`
- [ ] `DESCRIPTION` bumped to `x.y.(z+1).9000`
- [ ] `configure.ac` bumped; `configure` regenerated
- [ ] `Doxyfile` bumped
- [ ] New `# stochtree x.y.(z+1).9000` dev heading added to `NEWS.md`
- [ ] This issue closed
