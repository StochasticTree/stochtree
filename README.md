# StochasticTree

Stochastic tree ensembles (BART / XBART) for supervised learning and causal inference.

## Installation

### Building the CLI

The command line interface can be built from source using `cmake`. 
See [here](https://cmake.org/install/) for details on installing cmake (alternatively, 
on MacOS, `cmake` can be installed using [homebrew](https://formulae.brew.sh/formula/cmake)).
Once `cmake` is installed, you can build the CLI by navigating to the main 
project directory (i.e. `cd path/to/StochasticTree`) and running the following code 

```{bash}
rm -rf build
mkdir build
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$PWD/build
cmake --build build
cmake --install build
```

## Running the program

### CLI

Once built, the CLI can be run directly from the command line, with configuration specified in a `.conf` file. 
See the demos folder for examples and documentation.
