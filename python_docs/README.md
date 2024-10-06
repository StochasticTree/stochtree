# Python Package Documentation

## Building Documentation Locally

The online documentation is built in the doc-specific `StochasticTree/stochtree-python` repo (see [here](https://github.com/StochasticTree/stochtree-python/blob/main/.github/workflows/docs.yml) for the Github workflow).
To build the documentation locally, first ensure that you have [Sphinx](https://www.sphinx-doc.org/en/master/) installed, then navigate to the python package's main directory (i.e. `cd [path/to/stochtree]`), 
install the package, and run `sphinx-build` as below

```
pip install --upgrade pip
pip install -r python_docs/requirements.txt
pip install .
sphinx-build -M html python_docs/source/ python_docs/build/
```

## Documentation Style

Module (class, function, etc...) documentation follows [the numpy standard](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard), 
applied in Sphinx using the [napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) extension.

