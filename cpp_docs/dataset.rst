Dataset API
===========

Forest Dataset
--------------

The ``ForestDataset`` class is a wrapper around data needed to sample one or more tree ensembles. 
Its core elements are

* **Covariates**: Features / variables used to partition the forests. Stored internally as a (column-major) ``Eigen::MatrixXd``.
* **Basis**: *[Optional]* basis vector used to define a "leaf regression" --- a partitioned linear model where covariates define the partitions and basis defines the regression variables. 
  Also stored internally as a (column-major) ``Eigen::MatrixXd``.
* **Sample Weights**: *[Optional]* case weights for every observation in a training dataset. These may be heteroskedastic variance parameters or simply survey / case weights. 
  Stored internally as an ``Eigen::VectorXd``.

.. doxygenclass:: StochTree::ForestDataset
   :project: StochTree
   :members:

Random Effects Dataset
----------------------

The ``RandomEffectsDataset`` class is a wrapper around data needed to sample one or more tree ensembles. 
Its core elements are

* **Basis**: Vector of variables that have group-specific random coefficients. In the simplest additive group random effects model, this is a constant intercept of all ones. 
  Stored internally as a (column-major) ``Eigen::MatrixXd``.
* **Group Indices**: Integer-valued indices of group membership. In a model with three groups, these indices would typically be 0, 1, and 2 (remapped from perhaps more descriptive labels in R or Python). 
  Stored internally as an ``std::vector`` of integers.
* **Sample Weights**: *[Optional]* case weights for every observation in a training dataset. These may be heteroskedastic variance parameters or simply survey / case weights. 
  Stored internally as an ``Eigen::VectorXd``.

.. doxygenclass:: StochTree::RandomEffectsDataset
   :project: StochTree
   :members:

Other Classes and Types
-----------------------

.. doxygenenum:: StochTree::FeatureType
   :project: StochTree
   