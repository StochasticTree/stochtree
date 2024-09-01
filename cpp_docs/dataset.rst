Dataset API
===========

ForestDataset
-------------

The ``ForestDataset`` class is a wrapper around data needed to sample one or more tree ensembles. 
Its core elements are

* **Covariates**: Features / variables used to partition the forests. Stored internally as a (column-major) ``Eigen::MatrixXd``.
* **Basis**: *[Optional]* basis vector used to define a "leaf regression" --- a partitioned linear model where covariates define the partitions and basis defines the regression variables. 
  Also stored internally as a (column-major) ``Eigen::MatrixXd``
* **Sample Weights**: *[Optional]* case weights for every observation in a training dataset. These may be heteroskedastic variance parameters or simply survey / case weights. 
  Stored internally as an ``Eigen::VectorXd``

.. doxygenclass:: StochTree::ForestDataset
   :project: StochTree
   :members:
