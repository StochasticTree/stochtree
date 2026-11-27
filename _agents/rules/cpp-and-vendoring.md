---
trigger: glob
globs: "**/include/**, **/src/**, **/CMakeLists.txt, **/MANIFEST.in, **/cran-bootstrap.R, **/.gitmodules"
---

# C++ Core & Multi-Target Vendoring Rules

Apply these rules whenever editing C++ headers (`include/`), C++ translation units (`src/`), git submodules (`deps/`), or build/packaging manifests—though git submodules (`deps/`) should only ever be modified in exceptional circumstances when other options were not available, and any such modification must be heavily documented in a `DEPENDENCY_PATCHES.md` file checked into the top-level repo directory.

## 1. C++17 & CRAN Portability Constraints

- **Standard & Eigen Macros**: All C++ code must compile cleanly as C++17 with `-DEIGEN_MPL2_ONLY -DEIGEN_DONT_PARALLELIZE`. Use `Eigen::placeholders::all` (never `Eigen::all`, which conflicts across Eigen versions / namespaces).
- **Error Handling & Logging**: Never call `std::cout`, `std::cerr`, `printf`, `fprintf`, `std::exit`, or `std::abort` directly in library headers or `src/` files—`R CMD check --as-cran` flags compiled stdio and abort symbols. Route all logging and fatal checks through `StochTree::Log` (`Log::Info`, `Log::Warning`, `Log::Fatal`) and `CHECK*` macros in `include/stochtree/log.h` (which dispatches to `Rprintf` / `REprintf` under `-DSTOCHTREE_R_BUILD` and throws `std::runtime_error` on fatal errors).
- **Compiler Warnings**: Initialize local scalar accumulators explicitly per loop iteration and avoid unused variables—CRAN checks run with `-Wall -pedantic` on GCC and Clang.
- **OpenMP**: Guard any `<omp.h>` usage via `include/stochtree/openmp_utils.h` so builds without OpenMP (e.g., macOS without `libomp`) compile and run single-threaded cleanly.
- **Formatting**: Match the surrounding style of each file (2-space indent in `include/stochtree/` and core `src/`, 4-space indent in `src/{forest,sampler,R_*}.cpp`) and avoid drive-by reformatting in bugfix or feature PRs until a repo-wide `.clang-format` config is adopted.

## 2. Translation Unit Boundaries in `src/`

`src/` contains three distinct categories of `.cpp` files:

1. **Shared C++ Core** (`container.cpp`, `cutpoint_candidates.cpp`, `data.cpp`, `io.cpp`, `leaf_model.cpp`, `ordinal_sampler.cpp`, `partition_tracker.cpp`, `random_effects.cpp`, `tree.cpp`): Compiled by standalone CMake, Python, and R. Must not include `<cpp11.hpp>` or `<pybind11/...>`.
2. **R `cpp11` Bindings** (`cpp11.cpp`, `forest.cpp`, `kernel.cpp`, `R_data.cpp`, `R_random_effects.cpp`, `R_utils.cpp`, `sampler.cpp`, `serialization.cpp`, `stochtree_types.h`): Compiled only by R (`src/Makevars.in` / `src/Makevars.win.in`); explicitly excluded in `MANIFEST.in`.
3. **Python `pybind11` Binding** (`py_stochtree.cpp`): Compiled only when `BUILD_PYTHON=ON` in `CMakeLists.txt`; explicitly excluded by `cran-bootstrap.R` (`^py_`).

## 3. Four-Way Build & Header Vendoring Sync

Whenever adding, renaming, or removing C++ source files, headers under `include/`, or submodule headers under `deps/`, update **all four** locations together:

1. **`CMakeLists.txt`**: `SOURCES` list and include directories for `stochtree_objs`, `stochtree_cpp`, `teststochtree`, and `debugstochtree`.
2. **`src/Makevars.in` & `src/Makevars.win.in`**: `OBJECTS` list and `-I$(PKGROOT)/...` flags (note: `cran-bootstrap.R` rewrites `-I$(PKGROOT)/include` to `-I$(PKGROOT)/src/include` by exact fixed-string match—preserve indentation in `Makevars*.in`).
3. **`MANIFEST.in`**: Explicit header inclusion rules for PyPI `sdist` builds (and `exclude` list if adding a new R-only `.cpp`/`.h` file in `src/`). Verify with `python -m build --sdist`.
4. **`cran-bootstrap.R`**: Header vendoring into `stochtree_cran/src/include/` and diagnostic pragma stripping. CRAN forbids `#pragma clang diagnostic`, `#pragma GCC diagnostic`, `#pragma warning`, and `#pragma diag_suppress` in bundled C++ headers (e.g., Eigen / JSON headers)—strip any newly introduced pragmas in `cran-bootstrap.R`.
   - Note on Boost: Standalone C++ and Python use `deps/boost_math/include`, whereas CRAN R builds resolve Boost headers via `LinkingTo: BH` in `DESCRIPTION` (do not vendor `boost_math` in `cran-bootstrap.R`).
