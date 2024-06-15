# Python Package Documentation

## Building Documentation Locally

The online documentation is built automatically upon successful PR merge (see [here](https://github.com/StochasticTree/stochtree-python/blob/main/.github/workflows/docs.yml) for the Github workflow).
To build the documentation locally, first ensure that you have [Sphinx](https://www.sphinx-doc.org/en/master/) installed, then navigate to the python package's main directory (i.e. `cd [path/to/stochtree-python]`), 
install the package, and run `sphinx-build` as below

```
pip install .
sphinx-build -M html docs/source/ docs/build/
```

## Documentation Style

Module (class, function, etc...) documentation follows [the numpy standard](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard), 
applied in Sphinx using the [napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) extension.

