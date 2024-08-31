# C++ API Documentation

## Building Documentation Locally

We are still working out the best way to deploy the C++ documentation online alongside the R and Python documentation. 
In the meantime, to build the C++ documentation locally, first ensure that you have [doxygen](https://www.doxygen.nl/index.html) installed. 
On MacOS, this can be [done via homebrew](https://formulae.brew.sh/formula/doxygen) (i.e. `brew install doxygen`). 
Next, you will need both the [Sphinx](https://www.sphinx-doc.org/en/master/) and [breathe](https://breathe.readthedocs.io/en/latest/dot_graphs.html) python packages

Now, navigate to the python package's main directory (i.e. `cd [path/to/stochtree]`), build the C++ documentation via `doxygen` and then run `sphinx-build` as below

```
pip install --upgrade pip
pip install -r cpp_docs/requirements.txt
doxygen
sphinx-build -M html cpp_docs/ cpp_docs/build/
```

## Documentation Style

Module (class, function, etc...) documentation follows the format prescribed by [doxygen](https://www.doxygen.nl/manual/docblocks.html) for C++ code.
