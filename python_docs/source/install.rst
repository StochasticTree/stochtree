Installation
============

The python package can be installed from source. Clone the repo recursively (including git submodules) by running 

.. code-block:: console

    $ git clone --recursive https://github.com/StochasticTree/stochtree-python.git

Conda
-----

Conda provides a straightforward experience in managing python dependencies, avoiding version conflicts / ABI issues / etc.
Before you begin, make sure you have `conda <https://www.anaconda.com/download>`_ installed. 
Next, create and activate a conda environment with the requisite dependencies

.. code-block:: console

    $ conda create -n stochtree-dev -c conda-forge python=3.10 numpy scipy pytest pandas pybind11 scikit-learn matplotlib seaborn
    $ conda activate stochtree-dev
    $ pip install jupyterlab

Then, navigate to the main ``stochtree-python`` project folder (i.e. ``cd /path/to/stochtree-python``) and install the package locally via pip

.. code-block:: console

    $ pip install .

pip
---

If you would rather avoid installing and setting up conda, you can alternatively setup the dependencies and install ``stochtree`` using only ``pip`` (caveat: this has not been extensively tested 
across platforms and python versions).

First, navigate to the main ``stochtree-python`` project folder (i.e. ``cd /path/to/stochtree-python``) and create and activate a virtual environment as a subfolder of the repo

.. code-block:: console

    $ python -m venv venv
    $ source venv/bin/activate

Install all of the package (and demo notebook) dependencies

.. code-block:: console

    $ pip install numpy scipy pytest pandas scikit-learn pybind11 matplotlib seaborn jupyterlab

Then install stochtree via

.. code-block:: console

    $ pip install .
