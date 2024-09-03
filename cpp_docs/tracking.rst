Forest Sampling Tracker API
===========================

A truly minimalist tree ensemble library only needs 

* A representation of a decision tree
* A container for grouping / storing ensembles of trees
* In-memory access to / representation of training data
* Routines / functions to construct the trees

Most algorithms for optimizing or sampling tree ensembles frequently perform the following operations

* Determine which leaf a training observation falls into for a decision tree (to compute its prediction and update the residual / outcome)
* Evaluate potential split candidates for a leaf of a decision

With only the "minimalist" tools above, these two tasks proceed largely as follows

* For every observation in the dataset, traverse the tree (runtime depends on the tree topology but in a fully balanced tree with :math:`k` nodes, this has time complexity :math:`O(\log (k))`).
* For every observation in the dataset, determine whether an observation falls into a given node and whether or not a proposed decision rule would be true

These operations both perform unnecessary computation which can be avoided with some additional real-time tracking. Essentially, we want 

1. A mapping from dataset row index to leaf node id for every tree in an ensemble (so that we can skip the tree traversal during prediction)
2. A mapping from leaf node id to dataset row indices every tree in an ensemble (so that we can skip the full pass through the training data at split evaluation)

.. 1. For every observation in a dataset, which leaf node of each tree does the sample fall into?
.. 2. For every leaf in a tree, which training set observations fall into that node?

Forest Tracker
--------------

The ``ForestTracker`` class is a wrapper around several implementations of the mappings discussed above. 

.. doxygenclass:: StochTree::ForestTracker
   :project: StochTree
   :members:
