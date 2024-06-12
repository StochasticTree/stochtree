# StochasticTree C++ Core

[![Build and test](https://github.com/StochasticTree/stochtree-cpp/actions/workflows/test-multi-platform.yml/badge.svg)](https://github.com/StochasticTree/stochtree-cpp/actions/workflows/test-multi-platform.yml)

This repository hosts the "core" C++ code for defining and sampling stochastic tree ensembles (BART, XBART) for various applications. 
The [R](https://github.com/StochasticTree/stochtree-r) and [Python](https://github.com/StochasticTree/stochtree-python) 
packages have been refactored into separate repositories with installation instructions and demo notebooks. 

## Compilation

### Cloning the Repository

To clone the repository, you must have git installed, which you can do following [these instructions](https://learn.microsoft.com/en-us/devops/develop/git/install-and-set-up-git). 

Once git is available at the command line, navigate to the folder that will store this project (in bash / zsh, this is done by running `cd` followed by the path to the directory). 
Then, clone the `StochasticTree` repo as a subfolder by running
```{bash}
git clone --recursive https://github.com/andrewherren/StochasticTree.git
```

*NOTE*: this project incorporates several dependencies as [git submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules), 
which is why the `--recursive` flag is necessary (some systems may perform a recursive clone without this flag, but 
`--recursive` ensures this behavior on all platforms).

### CMake Build

The C++ project can be built independently from the R / Python packages using `cmake`. 
See [here](https://cmake.org/install/) for details on installing cmake (alternatively, 
on MacOS, `cmake` can be installed using [homebrew](https://formulae.brew.sh/formula/cmake)).
Once `cmake` is installed, you can build the CLI by navigating to the main 
project directory at your command line (i.e. `cd /path/to/stochtree-cpp`) and 
running the following code 

```{bash}
rm -rf build
mkdir build
cmake -S . -B build
cmake --build build
```

The CMake build has two primary targets, which are detailed below

#### Debug Program

`debug/api_debug.cpp` defines a standalone target that can be straightforwardly run with a debugger (i.e. `lldb`, `gdb`) 
while making non-trivial changes to the C++ code.
This debugging program is compiled as part of the CMake build if the `BUILD_DEBUG_TARGETS` option in `CMakeLists.txt` is set to `ON`.

Once the program has been built, it can be run from the command line via `./build/debugstochtree` or attached to a debugger 
via `lldb ./build/debugstochtree` (clang) or `gdb ./build/debugstochtree` (gcc).

#### Unit Tests

We test `stochtree-cpp` using the [GoogleTest](https://google.github.io/googletest/) framework.
Unit tests are compiled into a single target as part of the CMake build if the `BUILD_TEST` option is set to `ON` 
and the test suite can be run after compilation via `./build/teststochtree`
