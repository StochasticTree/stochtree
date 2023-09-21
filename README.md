# StochasticTree

Stochastic tree ensembles (BART / XBART) for supervised learning and causal inference.

## Installation

To install any of the options below, you must first clone the repository to your local machine. 
To do that, you must have git installed, which you can do following [these instructions](https://learn.microsoft.com/en-us/devops/develop/git/install-and-set-up-git). 

Once git is available at the command line, navigate to the folder that will store this project 
(in bash / zsh, this is done by running `cd` followed by the path to the directory). 
Then, clone the `StochasticTree` repo as a subfolder by running
```{bash}
git clone --recursive https://github.com/andrewherren/StochasticTree.git
```

*NOTE*: this project incorporates several dependencies as [git submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules), 
which is why the `--recursive` flag is necessary (some systems may perform a recursive clone without this flag, but 
`--recursive` ensures this behavior on all platforms).

### R package

The R package is defined in the `R-package` subfolder of `StochasticTree` but it depends on 
C++ code in the main project folder (as specified in `R-package/src/Makevars`). 

There are several ways to install the R package after cloning the repo locally. 
From the R console, you can nagivate to the main project directory 
(i.e. `setwd("/path/to/StochasticTree")`) and then run
```{r}
install.packages(pkgs = "R-package", repos = NULL, type = "source")
```

From the command line, navigate to the main project directory (i.e. `cd /path/to/StochasticTree`) 
and then run 
```{bash}
R CMD INSTALL --preclean R-package
```

### Command line interface

The command line interface can be built from source using `cmake`. 
See [here](https://cmake.org/install/) for details on installing cmake (alternatively, 
on MacOS, `cmake` can be installed using [homebrew](https://formulae.brew.sh/formula/cmake)).
Once `cmake` is installed, you can build the CLI by navigating to the main 
project directory at your command line (i.e. `cd /path/to/StochasticTree`) and 
running the following code 

```{bash}
rm -rf build
mkdir build
cmake -S . -B build
cmake --build build
cmake --install build
```

### Python package

*Coming soon!*

### C++ library

One goal of this project is to provide a unit-tested, low-level core 
of C++ data structures needed to build stochastic tree models, 
so that researchers / implementers don't have to "reinvent the wheel." 
As we develop and stabilize a C++ interface, we will document 
our recommendations for using it in your own C++ project.

## Running the program

### R package

The `R-package` subfolder includes a demo script (`demo/demo.R`) which can be run at the R console on in RStudio.

### CLI

Once built, the CLI can be run directly from the command line, 
with configuration options specified in a `.conf` file. 
See `demo/xbart_train/` for an example.
