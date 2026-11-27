---
name: extend-cpp-api
description: >-
  Use this skill when adding a new C++ class, method, configuration option, or
  LeafModel in include/stochtree/ and exposing it across Python (pybind11), R
  (cpp11/R6), and JSON serialization.
---

# Extending the C++ API Across R, Python, and Serialization

Use this checklist whenever plumbing a new C++ feature, parameter, or leaf model from the C++17 core through both host languages.

## 1. C++17 Core (`include/stochtree/`, `src/`)

- Declare public interfaces in `include/stochtree/*.h` (with Doxygen comments) and put non-trivial implementations in core `src/*.cpp` files (`container.cpp`, `cutpoint_candidates.cpp`, `data.cpp`, `io.cpp`, `leaf_model.cpp`, `ordinal_sampler.cpp`, `partition_tracker.cpp`, `random_effects.cpp`, `tree.cpp`).
- **If adding a new `LeafModel`**, update all 4 dispatch points in `include/stochtree/leaf_model.h`:
  1. `ModelType` enum
  2. `SuffStatVariant` and `LeafModelVariant` (`std::variant<...>`)
  3. `suffStatFactory(...)` and `leafModelFactory(...)`
  4. `LeafModel` methods required by `MCMCSampleOneIter` / `GFRSampleOneIter` (`SplitLogMarginalLikelihood`, `NoSplitLogMarginalLikelihood`, `PosteriorParameterMean`, `SampleLeafParameters`, `RequiresBasis`).
- Add GoogleTest unit coverage in `test/cpp/` and verify with:
  ```bash
  cmake -S . -B build -DBUILD_TEST=ON -DUSE_DEBUG=ON -DUSE_SANITIZER=ON
  cmake --build build -j && ./build/teststochtree
  ```

## 2. Python Binding (`pybind11` + `stochtree/`)

1. Bind the C++ class/method on the corresponding `*Cpp` wrapper class and `PYBIND11_MODULE(stochtree_cpp, m)` block in `src/py_stochtree.cpp`.
2. Expose it in the low-level Python wrapper (`stochtree/{data,forest,sampler,config,random_effects,serialization,kernel}.py`) and, if user-facing, export in `stochtree/__init__.py`.
3. Wire into `BARTModel` (`stochtree/bart.py`) / `BCFModel` (`stochtree/bcf.py`) and `stochtree/sklearn.py` if applicable.
4. Rebuild (`./venv/bin/pip install --no-build-isolation -e .`) and add tests in `test/python/`.

## 3. R Binding (`cpp11` + `R6` + High-Level S3)

1. Add a `[[cpp11::register]]` wrapper in the appropriate R binding file (`src/{forest,sampler,kernel,serialization,R_data,R_random_effects,R_utils}.cpp`) using `cpp11::external_pointer<T>` where needed (check `src/stochtree_types.h` if introducing a new external pointer type).
2. Run `Rscript -e "cpp11::cpp_register()"` to regenerate `src/cpp11.cpp` and `R/cpp11.R`.
3. Expose the method on the `R6` class in `R/{data,forest,model,config,random_effects,serialization,kernel}.R` and wire into `R/{bart,bcf}.R` if applicable.
4. Document with Roxygen2 (Title Case `@title`), run `Rscript -e "devtools::document()"`, and add tests in `test/R/testthat/`.

## 4. JSON Serialization & Backward Compatibility

If the change adds persisted model state:
- **Write path**: Serialize the new field in C++ (`to_json`), R (`saveBARTModelToJson` / `saveBCFModelToJson`), and Python (`to_json`). Keep JSON keys identical between R and Python so cross-language deserialization continues to work.
- **Read path (Backward Compatibility)**: Always guard new JSON fields during deserialization (`contains(...)` in C++, `%in%` / `is.null` in R, `.get(...)` in Python) with sensible defaults so pre-existing saved models still load cleanly.
- **Verify** against golden snapshots in `test/python/test_serialization_compat.py` and `test/R/testthat/test-serialization-compat.R` (and add a stripped-field compat test for the new key).

## 5. Build Manifest Check

If any C++ files or headers were added or renamed, verify all 4 build definitions (`CMakeLists.txt`, `src/Makevars*.in`, `MANIFEST.in`, and `cran-bootstrap.R`).
