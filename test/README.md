# Unit Testing

This directory contains unit tests for the R and Python packages as well as the C++ core. Below, we detail how to run each test suite.

## R Package

To run the R unit tests, first build the package (either via `R CMD build` at the command line or via "Shift + Command + B" in RStudio).
Then in an R console, run `testthat::test_dir("test/R")`.

## Python Package

To run the Python unit tests, first build the package at the command line (activating your virtual environment, if desired, beforehand):

```{bash}
rm -rf stochtree.egg-info; rm -rf .pytest_cache; rm -rf build
pip install . 
```

Then run 

```{bash}
pytest test/python
```

## C++ Core

To run the C++ unit tests, you must build the test executable, which is activated via the `BUILD_TEST` CMake option

```{bash}
rm -rf build                                                 
mkdir build
cmake -S . -B build -DBUILD_TEST=ON -DBUILD_DEBUG_TARGETS=OFF
cmake --build build
```

Then run the unit test suite by running the test executable

```{bash}
./build/teststochtree
```
