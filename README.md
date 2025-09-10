# StochTree

[![C++ Tests](https://github.com/StochasticTree/stochtree/actions/workflows/cpp-test.yml/badge.svg)](https://github.com/StochasticTree/stochtree/actions/workflows/cpp-test.yml)
[![Python Tests](https://github.com/StochasticTree/stochtree/actions/workflows/python-test.yml/badge.svg)](https://github.com/StochasticTree/stochtree/actions/workflows/python-test.yml)
[![R Tests](https://github.com/StochasticTree/stochtree/actions/workflows/r-test.yml/badge.svg)](https://github.com/StochasticTree/stochtree/actions/workflows/r-test.yml)

Software for building stochastic tree ensembles (i.e. BART, XBART) for supervised learning and causal inference.

# Getting Started

`stochtree` is composed of a C++ "core" and R / Python interfaces to that core.
Details on installation and use are available below:

* [Python](#python-package)
* [R](#r-package)
* [C++ core](#c-core)

# Python Package

## PyPI (`pip`)

`stochtree`'s Python package can be installed from PyPI via:

```
pip install stochtree
```

## Development Version (Local Build)

The development version of stochtree can be installed from source using pip's [git interface](https://pip.pypa.io/en/stable/topics/vcs-support/).

To proceed, you will need a working version of [git](https://git-scm.com) and python 3.8 or greater (available from several sources, one of the most straightforward being the [anaconda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) suite).


### Quick start

Without worrying about virtual environments (detailed further below), `stochtree` can be installed from the command line

```
pip install numpy scipy pytest pandas scikit-learn pybind11
pip install git+https://github.com/StochasticTree/stochtree.git
```

### Virtual environment installation

Often, users prefer to manage different projects (with different package / python version requirements) in virtual environments. 

#### Conda

Conda provides a straightforward experience in managing python dependencies, avoiding version conflicts / ABI issues / etc.

To build stochtree using a `conda` based workflow, first create and activate a conda environment with the requisite dependencies

```{bash}
conda create -n stochtree-dev -c conda-forge python=3.10 numpy scipy pytest pandas pybind11 scikit-learn matplotlib seaborn
conda activate stochtree-dev
```

Then install the package from github via pip

```{bash}
pip install git+https://github.com/StochasticTree/stochtree.git
```

(*Note*: if you'd also like to run `stochtree`'s notebook examples, you will also need jupyterlab, seaborn, and matplotlib)

```{bash}
conda install matplotlib seaborn
pip install jupyterlab
```

With these dependencies installed, you can [clone the repo](###cloning-the-repository) and run the `demo/` examples.

#### Venv

You could also use venv for environment management. First, navigate to the folder in which you usually store virtual environments 
(i.e. `cd /path/to/envs`) and create and activate a virtual environment:

```{bash}
python -m venv venv
source venv/bin/activate
```

Install all of the package (and demo notebook) dependencies

```{bash}
pip install numpy scipy pytest pandas scikit-learn pybind11
```

Then install stochtree via

```{bash}
pip install git+https://github.com/StochasticTree/stochtree.git
```

As above, if you'd like to run the notebook examples in the `demo/` subfolder, you will also need jupyterlab, seaborn, and matplotlib and you will have to [clone the repo](###cloning-the-repository)

```{bash}
pip install matplotlib seaborn jupyterlab
```

# R Package

The R package can be installed from CRAN via

```
install.packages("stochtree")
```

The development version of `stochtree` can be installed from Github via

```
remotes::install_github("StochasticTree/stochtree", ref="r-dev")
```

# C++ Core

While the C++ core links to both R and Python for a performant, high-level interface, 
the C++ code can be compiled and unit-tested and compiled into a standalone 
[debug program](https://github.com/StochasticTree/stochtree/tree/main/debug).

## Compilation

### Cloning the Repository

To clone the repository, you must have git installed, which you can do following [these instructions](https://learn.microsoft.com/en-us/devops/develop/git/install-and-set-up-git). 

Once git is available at the command line, navigate to the folder that will store this project (in bash / zsh, this is done by running `cd` followed by the path to the directory). 
Then, clone the `stochtree` repo as a subfolder by running
```{bash}
git clone --recursive https://github.com/StochasticTree/stochtree.git
```

*NOTE*: this project incorporates several dependencies as [git submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules), 
which is why the `--recursive` flag is necessary (some systems may perform a recursive clone without this flag, but 
`--recursive` ensures this behavior on all platforms). If you have already cloned the repo without the `--recursive` flag, 
you can retrieve the submodules recursively by running `git submodule update --init --recursive` in the main repo directory.


### CMake Build

The C++ project can be built independently from the R / Python packages using `cmake`. 
See [here](https://cmake.org/install/) for details on installing cmake (alternatively, 
on MacOS, `cmake` can be installed using [homebrew](https://formulae.brew.sh/formula/cmake)).
Once `cmake` is installed, you can build the CLI by navigating to the main 
project directory at your command line (i.e. `cd /path/to/stochtree`) and 
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

We test `stochtree` using the [GoogleTest](https://google.github.io/googletest/) framework.
Unit tests are compiled into a single target as part of the CMake build if the `BUILD_TEST` option is set to `ON` 
and the test suite can be run after compilation via `./build/teststochtree`

## Xcode

While using `gdb` or `lldb` on `debugstochtree` at the command line is very helpful, users may prefer debugging in a full-fledged IDE like xcode. This project's C++ core can be converted to an xcode project from `CMakeLists.txt`, but first you must turn off sanitizers (xcode seems to have its own way of setting this at build time for different configurations, and having injected 
`-fsanitize=address` statically into compiler arguments will cause xcode errors). To do this, modify the `USE_SANITIZER` line in `CMakeLists.txt`:

```
option(USE_SANITIZER "Use santizer flags" OFF)
```

To generate an XCode project based on the build targets and specifications defined in a `CMakeLists.txt`, navigate to the main project folder (i.e. `cd /path/to/project`) and run the following commands:

```{bash}
rm -rf xcode/
mkdir xcode
cd xcode
cmake -G Xcode .. -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=c++ -DUSE_SANITIZER=OFF -DUSE_DEBUG=OFF
cd ..
```

Now, if you navigate to the xcode subfolder (in Finder), you should be able to click on a `.xcodeproj` file and the project will open in XCode.
