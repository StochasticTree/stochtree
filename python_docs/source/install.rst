Installation
============

The python package is not yet on PyPI but can be installed from source using pip's `git interface <https://pip.pypa.io/en/stable/topics/vcs-support/>`_. 
To proceed, you will need a working version of `git <https://git-scm.com>`_ and python 3.8 or greater (available from several sources, one of the most 
straightforward being the `anaconda <https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html>`_ suite).

The python package can be installed from github via pip (and will soon be available on PyPI). 
Clone the repo recursively (including git submodules) by running 

.. code-block:: console

    $ git clone --recursive https://github.com/StochasticTree/stochtree.git

Quick start
-----------

Without worrying about virtual environments (detailed further below), ``stochtree`` can be installed from the command line

.. code-block:: console

    $ pip install numpy scipy pytest pandas scikit-learn pybind11
    $ pip install git+https://github.com/StochasticTree/stochtree.git

Virtual environment installation
--------------------------------

Often, users prefer to manage different projects (with different package / python version requirements) in virtual environments.

Conda
^^^^^

Conda provides a straightforward experience in managing python dependencies, avoiding version conflicts / ABI issues / etc.

To build stochtree using a ``conda`` based workflow, first create and activate a conda environment with the requisite dependencies

.. code-block:: console

    $ conda create -n stochtree-dev -c conda-forge python=3.10 numpy scipy pytest pandas pybind11 scikit-learn matplotlib seaborn
    $ conda activate stochtree-dev

Then install the package from github via pip

.. code-block:: console

    $ pip install git+https://github.com/StochasticTree/stochtree.git

(*Note*: if you'd also like to run ``stochtree``'s notebook examples, you will also need jupyterlab, seaborn, and matplotlib)

.. code-block:: console

    $ conda install matplotlib seaborn
    $ pip install jupyterlab

With these dependencies installed, you can :ref:`clone the repo <cloning-the-repository>` and run the ``demo/`` examples.

Venv
^^^^

You could also use venv for environment management. First, navigate to the folder in which you usually store virtual environments 
(i.e. ``cd /path/to/envs``) and create and activate a virtual environment:

.. code-block:: console

    $ python -m venv venv
    $ source venv/bin/activate

Install all of the package (and demo notebook) dependencies

.. code-block:: console

    $ pip install numpy scipy pytest pandas scikit-learn pybind11

Then install stochtree via

.. code-block:: console

    $ pip install git+https://github.com/StochasticTree/stochtree.git

As above, if you'd like to run the notebook examples in the ``demo/`` subfolder, you will also need jupyterlab, seaborn, and matplotlib and you will have to :ref:`clone the repo <cloning-the-repository>`

.. code-block:: console

    $ pip install matplotlib seaborn jupyterlab

.. _cloning-the-repository:

Cloning the Repository
----------------------

To clone the repository, you must have git installed, which you can do following `these instructions <https://learn.microsoft.com/en-us/devops/develop/git/install-and-set-up-git>`_. 

Once git is available at the command line, navigate to the folder that will store this project (in bash / zsh, this is done by running ``cd`` followed by the path to the directory). 
Then, clone the ``stochtree`` repo as a subfolder by running

.. code-block:: console

    $ git clone --recursive https://github.com/StochasticTree/stochtree.git

*NOTE*: this project incorporates several dependencies as `git submodules <https://git-scm.com/book/en/v2/Git-Tools-Submodules>`_, 
which is why the ``--recursive`` flag is necessary (some systems may perform a recursive clone without this flag, but 
``--recursive`` ensures this behavior on all platforms). If you have already cloned the repo without the ``--recursive`` flag, 
you can retrieve the submodules recursively by running ``git submodule update --init --recursive`` in the main repo directory.