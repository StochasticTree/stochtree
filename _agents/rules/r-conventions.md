---
trigger: glob
globs: "**/R/**, **/man/**, **/NAMESPACE, **/DESCRIPTION, **/configure.ac, **/src/R_*.cpp, **/src/forest.cpp, **/src/sampler.cpp, **/src/serialization.cpp, **/src/kernel.cpp, **/test/R/**"
---

# R Package & CRAN Conventions

Apply these rules when editing R source (`R/`), `cpp11` bindings (`src/`), Roxygen documentation (`man/`, `NAMESPACE`), or R tests (`test/R/`).

## 1. Generated Files & FFI (`cpp11` + `R6`)

- **`cpp11` Registration**: Never hand-edit `src/cpp11.cpp` or `R/cpp11.R`. After adding or changing any `[[cpp11::register]]` signature in `src/`, run `Rscript -e "cpp11::cpp_register()"`.
- **Roxygen2 Documentation**: Never hand-edit `man/*.Rd` or `NAMESPACE`. Edit Roxygen blocks in `R/*.R` and run `Rscript -e "devtools::document()"`.
  - Use **Title Case** for all `@title` entries (CRAN convention enforced since `#310`).
  - Wrap file-I/O `@examples` using `tempfile(fileext = ".json")` and always `unlink()` temporary files before finishing (CRAN forbids writing to the user directory or leaving temp artifacts behind).
- **External Pointers & Serialization**: `R6` classes (`ForestDataset`, `Outcome`, `ForestSamples`, `Forest`, `ForestModel`, `RandomEffects*`) hold `cpp11::external_pointer<T>` handles that cannot be saved with `saveRDS()`. Model persistence goes through the JSON serialization helpers in `R/serialization.R`, `R/bart.R`, and `R/bcf.R`.

## 2. Formatting & Style

- R code is formatted with **Air** (`air.toml`: 80-char line width, 2-space indent, `persistent-line-breaks = true`). Run `air format R test/R` if `air` is installed, or match surrounding formatting strictly.
- Exported high-level helper functions follow `camelCase` (`saveBARTModelToJsonString`, `computeBARTPosteriorInterval`, `extractParameter`), while primary estimators (`bart`, `bcf`) and S3 classes (`bartmodel`, `bcfmodel`) are lowercase.

## 3. Testing & CRAN Guardrails

- **Test Location**: R tests live in `test/R/testthat/` (not `tests/testthat/`) at the repo root. Run locally via:
  - Full suite: `NOT_CRAN=true Rscript -e "devtools::load_all('.'); testthat::test_dir('test/R/testthat')"`
  - Single file: `NOT_CRAN=true Rscript -e "devtools::load_all('.'); testthat::test_file('test/R/testthat/test-bart.R')"`
- **CRAN Resource Limits**:
  - Default `num_threads` must remain `1` everywhere (CRAN rejects packages defaulting to >2 cores).
  - Guard slow MCMC/integration tests and tests requiring non-CRAN fixtures with `skip_on_cran()`.
  - `test-serialization-compat.R` and `test/R/testthat/fixtures/*.json` are intentionally stripped from CRAN tarballs by `cran-bootstrap.R`; keep CRAN-included unit tests self-contained without `jsonlite` fixture files unless guarded with `skip_if_not_installed("jsonlite")` and `skip_on_cran()`.
