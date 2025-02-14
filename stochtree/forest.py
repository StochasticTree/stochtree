"""
Python classes wrapping C++ forest container object
"""

import numpy as np
from .data import Dataset
from stochtree_cpp import ForestContainerCpp, ForestCpp
from typing import Union


class ForestContainer:
    """
    Container that stores sampled (and retained) tree ensembles from BART, BCF or a custom sampler.

    Parameters
    ----------
    num_trees : int
        Number of trees that each forest should contain
    output_dimension : int, optional
        Dimension of the leaf node parameters in each tree
    leaf_constant : bool, optional
        Whether the leaf node model is "constant" (i.e. prediction is simply a
        sum of leaf node parameters for every observation in a dataset) or not (i.e.
        each leaf node parameter is multiplied by a "basis vector" before being returned
        as a prediction).
    is_exponentiated : bool, optional
        Whether or not the leaf node parameters are stored in log scale (in which case, they
        must be exponentiated before being returned as predictions).
    """

    def __init__(
        self,
        num_trees: int,
        output_dimension: int = 1,
        leaf_constant: bool = True,
        is_exponentiated: bool = False,
    ) -> None:
        self.forest_container_cpp = ForestContainerCpp(
            num_trees, output_dimension, leaf_constant, is_exponentiated
        )
        self.num_trees = num_trees
        self.output_dimension = output_dimension
        self.leaf_constant = leaf_constant
        self.is_exponentiated = is_exponentiated

    def predict(self, dataset: Dataset) -> np.array:
        """
        Predict from each forest in the container, using the provided `Dataset` object.

        Parameters
        ----------
        dataset : Dataset
            Python object wrapping the "dataset" class used by C++ sampling and prediction data structures.

        Returns
        -------
        np.array
            Numpy array with (`n`, `m`) dimensions, where `n` is the number of observations in `dataset` and `m`
            is the number of samples in the forest container.
        """
        return self.forest_container_cpp.Predict(dataset.dataset_cpp)

    def predict_raw(self, dataset: Dataset) -> np.array:
        """
        Predict raw leaf values for a every forest in the container, using the provided `Dataset` object

        Parameters
        ----------
        dataset : Dataset
            Python object wrapping the "dataset" class used by C++ sampling and prediction data structures.

        Returns
        -------
        np.array
            Numpy array with (`n`, `k`, `m`) dimensions, where `n` is the number of observations in `dataset`,
            `k` is the dimension of the leaf parameter, and `m` is the number of samples in the forest container.
            If `k = 1`, then the returned array is simply (`n`, `m`) dimensions.
        """
        result = self.forest_container_cpp.PredictRaw(dataset.dataset_cpp)
        if result.ndim == 3:
            if result.shape[1] == 1:
                result = result.reshape(result.shape[0], result.shape[2])
        return result

    def predict_raw_single_forest(self, dataset: Dataset, forest_num: int) -> np.array:
        """
        Predict raw leaf values for a specific forest (indexed by `forest_num`), using the provided `Dataset` object

        Parameters
        ----------
        dataset : Dataset
            Python object wrapping the "dataset" class used by C++ sampling and prediction data structures.
        forest_num : int
            Index of the forest from which to predict. Forest indices are 0-based.

        Returns
        -------
        np.array
            Numpy array with (`n`, `k`) dimensions, where `n` is the number of observations in `dataset` and
            `k` is the dimension of the leaf parameter.
        """
        return self.forest_container_cpp.PredictRawSingleForest(
            dataset.dataset_cpp, forest_num
        )

    def predict_raw_single_tree(
        self, dataset: Dataset, forest_num: int, tree_num: int
    ) -> np.array:
        """
        Predict raw leaf values for a specific tree of a specific forest (indexed by `tree_num` and `forest_num`
        respectively), using the provided `Dataset` object.

        Parameters
        ----------
        dataset : Dataset
            Python object wrapping the "dataset" class used by C++ sampling and prediction data structures.
        forest_num : int
            Index of the forest from which to predict. Forest indices are 0-based.
        tree_num : int
            Index of the tree which to predict (within forest indexed by `forest_num`). Tree indices are 0-based.

        Returns
        -------
        np.array
            Numpy array with (`n`, `k`) dimensions, where `n` is the number of observations in `dataset` and
            `k` is the dimension of the leaf parameter.
        """
        return self.forest_container_cpp.PredictRawSingleTree(
            dataset.dataset_cpp, forest_num, tree_num
        )

    def set_root_leaves(
        self, forest_num: int, leaf_value: Union[float, np.array]
    ) -> None:
        """
        Set constant (root) leaf node values for every tree in the forest indexed by `forest_num`.
        Assumes the forest consists of all root (single-node) trees.

        Parameters
        ----------
        forest_num : int
            Index of the forest for which we will set root node parameters.
        leaf_value : float or np.array
            Constant values to which root nodes are to be set. If the trees in forest `forest_num`
            are univariate, then `leaf_value` must be a `float`, while if the trees in forest `forest_num`
            are multivariate, then `leaf_value` must be a `np.array`.
        """
        if not isinstance(leaf_value, np.ndarray) and not isinstance(leaf_value, float):
            raise ValueError("leaf_value must be either a float or np.array")
        if isinstance(leaf_value, np.ndarray):
            leaf_value = np.squeeze(leaf_value)
            if len(leaf_value.shape) != 1:
                raise ValueError("leaf_value must be either a one-dimensional array")
            self.forest_container_cpp.SetRootVector(
                forest_num, leaf_value, leaf_value.shape[0]
            )
        else:
            self.forest_container_cpp.SetRootValue(forest_num, leaf_value)

    def save_to_json_file(self, json_filename: str) -> None:
        """
        Save the forests in the container to a JSON file.

        Parameters
        ----------
        json_filename : str
            Name of JSON file to which forest container state will be saved.
            May contain absolute or relative paths.
        """
        self.forest_container_cpp.SaveToJsonFile(json_filename)

    def load_from_json_file(self, json_filename: str) -> None:
        """
        Load a forest container from output stored in a JSON file.

        Parameters
        ----------
        json_filename : str
            Name of JSON file from which forest container state will be restored.
            May contain absolute or relative paths.
        """
        self.forest_container_cpp.LoadFromJsonFile(json_filename)

    def dump_json_string(self) -> str:
        """
        Dump a forest container into an in-memory JSON string (which can be directly serialized or
        combined with other JSON strings before serialization).

        Returns
        -------
        str
            In-memory string containing state of a forest container.
        """
        return self.forest_container_cpp.DumpJsonString()

    def load_from_json_string(self, json_string: str) -> None:
        """
        Reload a forest container from an in-memory JSON string.

        Parameters
        ----------
        json_string : str
            In-memory string containing state of a forest container.
        """
        self.forest_container_cpp.LoadFromJsonString(json_string)

    def add_sample(self, leaf_value: Union[float, np.array]) -> None:
        """
        Add a new all-root ensemble to the container, with all of the leaves set to the value / vector provided

        Parameters
        ----------
        leaf_value : float or np.array
            Value (or vector of values) to initialize root nodes of every tree in a forest
        """
        if isinstance(leaf_value, np.ndarray):
            leaf_value = np.squeeze(leaf_value)
            self.forest_container_cpp.AddSampleVector(leaf_value)
        else:
            self.forest_container_cpp.AddSampleValue(leaf_value)

    def add_numeric_split(
        self,
        forest_num: int,
        tree_num: int,
        leaf_num: int,
        feature_num: int,
        split_threshold: float,
        left_leaf_value: Union[float, np.array],
        right_leaf_value: Union[float, np.array],
    ) -> None:
        """
        Add a numeric (i.e. X[,i] <= c) split to a given tree in the ensemble

        Parameters
        ----------
        forest_num : int
            Index of the forest which contains the tree to be split
        tree_num : int
            Index of the tree to be split
        leaf_num : int
            Leaf to be split
        feature_num : int
            Feature that defines the new split
        split_threshold : float
            Value that defines the cutoff of the new split
        left_leaf_value : float or np.array
            Value (or array of values) to assign to the newly created left node
        right_leaf_value : float or np.array
            Value (or array of values) to assign to the newly created right node
        """
        if isinstance(left_leaf_value, np.ndarray):
            left_leaf_value = np.squeeze(left_leaf_value)
            right_leaf_value = np.squeeze(right_leaf_value)
            self.forest_container_cpp.AddNumericSplitVector(
                forest_num,
                tree_num,
                leaf_num,
                feature_num,
                split_threshold,
                left_leaf_value,
                right_leaf_value,
            )
        else:
            self.forest_container_cpp.AddNumericSplitValue(
                forest_num,
                tree_num,
                leaf_num,
                feature_num,
                split_threshold,
                left_leaf_value,
                right_leaf_value,
            )

    def get_tree_leaves(self, forest_num: int, tree_num: int) -> np.array:
        """
        Retrieve a vector of indices of leaf nodes for a given tree in a given forest

        Parameters
        ----------
        forest_num : int
            Index of the forest which contains tree `tree_num`
        tree_num : float or np.array
            Index of the tree for which leaf indices will be retrieved

        Returns
        -------
        np.array
            One-dimensional numpy array, containing the indices of leaf nodes in a given tree.
        """
        return self.forest_container_cpp.GetTreeLeaves(forest_num, tree_num)

    def get_tree_split_counts(
        self, forest_num: int, tree_num: int, num_features: int
    ) -> np.array:
        """
        Retrieve a vector of split counts for every training set feature in a given tree in a given forest

        Parameters
        ----------
        forest_num : int
            Index of the forest which contains tree `tree_num`
        tree_num : int
            Index of the tree for which split counts will be retrieved
        num_features : int
            Total number of features in the training set

        Returns
        -------
        np.array
            One-dimensional numpy array with as many elements as in the forest model's training set,
            containing the split count for each feature for a given forest and tree.
        """
        return self.forest_container_cpp.GetTreeSplitCounts(
            forest_num, tree_num, num_features
        )

    def get_forest_split_counts(self, forest_num: int, num_features: int) -> np.array:
        """
        Retrieve a vector of split counts for every training set feature in a given forest

        Parameters
        ----------
        forest_num : int
            Index of the forest which contains tree `tree_num`
        num_features : int
            Total number of features in the training set

        Returns
        -------
        np.array
            One-dimensional numpy array with as many elements as in the forest model's training set,
            containing the split count for each feature for a given forest (summed across every tree in the forest).
        """
        return self.forest_container_cpp.GetForestSplitCounts(forest_num, num_features)

    def get_overall_split_counts(self, num_features: int) -> np.array:
        """
        Retrieve a vector of split counts for every training set feature, aggregated across ensembles and trees.

        Parameters
        ----------
        num_features : int
            Total number of features in the training set

        Returns
        -------
        np.array
            One-dimensional numpy array with as many elements as in the forest model's training set,
            containing the split count for each feature summed across every forest of every tree in the container.
        """
        return self.forest_container_cpp.GetOverallSplitCounts(num_features)

    def get_granular_split_counts(self, num_features: int) -> np.array:
        """
        Retrieve a vector of split counts for every training set variable in a given forest, reported separately for each ensemble and tree

        Parameters
        ----------
        num_features : int
            Total number of features in the training set

        Returns
        -------
        np.array
            Three-dimensional numpy array, containing the number of splits a variable receives in each tree of each forest in a ``ForestContainer``.
            Array will have dimensions (`m`,`b`,`p`) where `m` is the number of forests in the container, `b` is the number of trees in each
            forest, and `p` is the number of features in the forest model's training dataset.
        """
        return self.forest_container_cpp.GetGranularSplitCounts(num_features)

    def num_forest_leaves(self, forest_num: int) -> int:
        """
        Return the total number of leaves for a given forest in the ``ForestContainer``

        Parameters
        ----------
        forest_num : int
            Index of the forest to be queried

        Returns
        -------
        int
            Number of leaves in a given forest in a ``ForestContainer``
        """
        return self.forest_container_cpp.NumLeavesForest(forest_num)

    def sum_leaves_squared(self, forest_num: int) -> float:
        """
        Return the total sum of squared leaf values for a given forest in the ``ForestContainer``

        Parameters
        ----------
        forest_num : int
            Index of the forest to be queried

        Returns
        -------
        float
            Sum of squared leaf values in a given forest in a ``ForestContainer``
        """
        return self.forest_container_cpp.SumLeafSquared(forest_num)

    def is_leaf_node(self, forest_num: int, tree_num: int, node_id: int) -> bool:
        """
        Whether or not a given node of a given tree in a given forest in the ``ForestContainer`` is a leaf

        Parameters
        ----------
        forest_num : int
            Index of the forest to be queried
        tree_num : int
            Index of the tree to be queried
        node_id : int
            Index of the node to be queried

        Returns
        -------
        bool
            `True` if node `node_id` in tree `tree_num` of forest `forest_num` is a leaf, `False` otherwise
        """
        return self.forest_container_cpp.IsLeafNode(forest_num, tree_num, node_id)

    def is_numeric_split_node(
        self, forest_num: int, tree_num: int, node_id: int
    ) -> bool:
        """
        Whether or not a given node of a given tree in a given forest in the ``ForestContainer`` is a numeric split node

        Parameters
        ----------
        forest_num : int
            Index of the forest to be queried
        tree_num : int
            Index of the tree to be queried
        node_id : int
            Index of the node to be queried

        Returns
        -------
        bool
            `True` if node `node_id` in tree `tree_num` of forest `forest_num` is a numeric split node, `False` otherwise
        """
        return self.forest_container_cpp.IsNumericSplitNode(
            forest_num, tree_num, node_id
        )

    def is_categorical_split_node(
        self, forest_num: int, tree_num: int, node_id: int
    ) -> bool:
        """
        Whether or not a given node of a given tree in a given forest in the ``ForestContainer`` is a categorical split node

        Parameters
        ----------
        forest_num : int
            Index of the forest to be queried
        tree_num : int
            Index of the tree to be queried
        node_id : int
            Index of the node to be queried

        Returns
        -------
        bool
            `True` if node `node_id` in tree `tree_num` of forest `forest_num` is a categorical split node, `False` otherwise
        """
        return self.forest_container_cpp.IsCategoricalSplitNode(
            forest_num, tree_num, node_id
        )

    def parent_node(self, forest_num: int, tree_num: int, node_id: int) -> int:
        """
        Parent node of given node of a given tree in a given forest in the ``ForestContainer``

        Parameters
        ----------
        forest_num : int
            Index of the forest to be queried
        tree_num : int
            Index of the tree to be queried
        node_id : int
            Index of the node to be queried

        Returns
        -------
        int
            Index of the parent of node `node_id` in tree `tree_num` of forest `forest_num`.
            If `node_id` is a root node, returns `-1`.
        """
        return self.forest_container_cpp.ParentNode(forest_num, tree_num, node_id)

    def left_child_node(self, forest_num: int, tree_num: int, node_id: int) -> int:
        """
        Left child node of given node of a given tree in a given forest in the ``ForestContainer``

        Parameters
        ----------
        forest_num : int
            Index of the forest to be queried
        tree_num : int
            Index of the tree to be queried
        node_id : int
            Index of the node to be queried

        Returns
        -------
        int
            Index of the left child of node `node_id` in tree `tree_num` of forest `forest_num`.
            If `node_id` is a leaf, returns `-1`.
        """
        return self.forest_container_cpp.LeftChildNode(forest_num, tree_num, node_id)

    def right_child_node(self, forest_num: int, tree_num: int, node_id: int) -> int:
        """
        Right child node of given node of a given tree in a given forest in the ``ForestContainer``

        Parameters
        ----------
        forest_num : int
            Index of the forest to be queried
        tree_num : int
            Index of the tree to be queried
        node_id : int
            Index of the node to be queried

        Returns
        -------
        int
            Index of the right child of node `node_id` in tree `tree_num` of forest `forest_num`.
            If `node_id` is a leaf, returns `-1`.
        """
        return self.forest_container_cpp.RightChildNode(forest_num, tree_num, node_id)

    def node_depth(self, forest_num: int, tree_num: int, node_id: int) -> int:
        """
        Depth of given node of a given tree in a given forest in the ``ForestContainer``.

        Parameters
        ----------
        forest_num : int
            Index of the forest to be queried
        tree_num : int
            Index of the tree to be queried
        node_id : int
            Index of the node to be queried

        Returns
        -------
        int
            Depth of node `node_id` in tree `tree_num` of forest `forest_num`. The root node is defined
            as "depth zero."
        """
        return self.forest_container_cpp.NodeDepth(forest_num, tree_num, node_id)

    def node_split_index(self, forest_num: int, tree_num: int, node_id: int) -> int:
        """
        Split index of given node of a given tree in a given forest in the ``ForestContainer``.
        Returns ``-1`` if the node is a leaf.

        Parameters
        ----------
        forest_num : int
            Index of the forest to be queried
        tree_num : int
            Index of the tree to be queried
        node_id : int
            Index of the node to be queried

        Returns
        -------
        int
            Split index of `node_id` in tree `tree_num` of forest `forest_num`.
        """
        if self.is_leaf_node(forest_num, tree_num, node_id):
            return -1
        else:
            return self.forest_container_cpp.SplitIndex(forest_num, tree_num, node_id)

    def node_split_threshold(
        self, forest_num: int, tree_num: int, node_id: int
    ) -> float:
        """
        Threshold that defines a numeric split for a given node of a given tree in a given forest in the ``ForestContainer``.
        Returns ``np.Inf`` if the node is a leaf or a categorical split node.

        Parameters
        ----------
        forest_num : int
            Index of the forest to be queried
        tree_num : int
            Index of the tree to be queried
        node_id : int
            Index of the node to be queried

        Returns
        -------
        float
            Threshold that defines a numeric split for node `node_id` in tree `tree_num` of forest `forest_num`.
        """
        if self.is_leaf_node(
            forest_num, tree_num, node_id
        ) or self.is_categorical_split_node(forest_num, tree_num, node_id):
            return np.Inf
        else:
            return self.forest_container_cpp.SplitThreshold(
                forest_num, tree_num, node_id
            )

    def node_split_categories(
        self, forest_num: int, tree_num: int, node_id: int
    ) -> np.array:
        """
        Array of category indices that define a categorical split for a given node of a given tree in a given forest in the ``ForestContainer``.
        Returns ``np.array([np.Inf])`` if the node is a leaf or a numeric split node.

        Parameters
        ----------
        forest_num : int
            Index of the forest to be queried
        tree_num : int
            Index of the tree to be queried
        node_id : int
            Index of the node to be queried

        Returns
        -------
        np.array
            Array of category indices that define a categorical split for node `node_id` in tree `tree_num` of forest `forest_num`.
        """
        if self.is_leaf_node(
            forest_num, tree_num, node_id
        ) or self.is_numeric_split_node(forest_num, tree_num, node_id):
            return np.array([np.Inf])
        else:
            return self.forest_container_cpp.SplitCategories(
                forest_num, tree_num, node_id
            )

    def node_leaf_values(
        self, forest_num: int, tree_num: int, node_id: int
    ) -> np.array:
        """
        Node parameter value(s) for a given node of a given tree in a given forest in the ``ForestContainer``.
        Values are stale if the node is a split node.

        Parameters
        ----------
        forest_num : int
            Index of the forest to be queried
        tree_num : int
            Index of the tree to be queried
        node_id : int
            Index of the node to be queried

        Returns
        -------
        np.array
            Array of parameter values for node `node_id` in tree `tree_num` of forest `forest_num`.
        """
        return self.forest_container_cpp.NodeLeafValues(forest_num, tree_num, node_id)

    def num_nodes(self, forest_num: int, tree_num: int) -> int:
        """
        Number of nodes in a given tree in a given forest in the ``ForestContainer``.

        Parameters
        ----------
        forest_num : int
            Index of the forest to be queried
        tree_num : int
            Index of the tree to be queried

        Returns
        -------
        int
            Total number of nodes in tree `tree_num` of forest `forest_num`.
        """
        return self.forest_container_cpp.NumNodes(forest_num, tree_num)

    def num_leaves(self, forest_num: int, tree_num: int) -> int:
        """
        Number of leaves in a given tree in a given forest in the ``ForestContainer``.

        Parameters
        ----------
        forest_num : int
            Index of the forest to be queried
        tree_num : int
            Index of the tree to be queried

        Returns
        -------
        int
            Total number of leaves in tree `tree_num` of forest `forest_num`.
        """
        return self.forest_container_cpp.NumLeaves(forest_num, tree_num)

    def num_leaf_parents(self, forest_num: int, tree_num: int) -> int:
        """
        Number of leaf parents (split nodes with two leaves as children) in a given tree in a given forest in the ``ForestContainer``.

        Parameters
        ----------
        forest_num : int
            Index of the forest to be queried
        tree_num : int
            Index of the tree to be queried

        Returns
        -------
        int
            Total number of leaf parents in tree `tree_num` of forest `forest_num`.
        """
        return self.forest_container_cpp.NumLeafParents(forest_num, tree_num)

    def num_split_nodes(self, forest_num: int, tree_num: int) -> int:
        """
        Number of split_nodes in a given tree in a given forest in the ``ForestContainer``.

        Parameters
        ----------
        forest_num : int
            Index of the forest to be queried
        tree_num : int
            Index of the tree to be queried

        Returns
        -------
        int
            Total number of split nodes in tree `tree_num` of forest `forest_num`.
        """
        return self.forest_container_cpp.NumSplitNodes(forest_num, tree_num)

    def nodes(self, forest_num: int, tree_num: int) -> np.array:
        """
        Array of node indices in a given tree in a given forest in the ``ForestContainer``.

        Parameters
        ----------
        forest_num : int
            Index of the forest to be queried
        tree_num : int
            Index of the tree to be queried

        Returns
        -------
        np.array
            Array of indices of nodes in tree `tree_num` of forest `forest_num`.
        """
        return self.forest_container_cpp.Nodes(forest_num, tree_num)

    def leaves(self, forest_num: int, tree_num: int) -> np.array:
        """
        Array of leaf indices in a given tree in a given forest in the ``ForestContainer``.

        Parameters
        ----------
        forest_num : int
            Index of the forest to be queried
        tree_num : int
            Index of the tree to be queried

        Returns
        -------
        np.array
            Array of indices of leaf nodes in tree `tree_num` of forest `forest_num`.
        """
        return self.forest_container_cpp.Leaves(forest_num, tree_num)

    def delete_sample(self, forest_num: int) -> None:
        """
        Modify the ``ForestContainer`` by removing the forest sample indexed by ``forest_num``.

        Parameters
        ----------
        forest_num : int
            Index of the forest to be removed from the ``ForestContainer``
        """
        return self.forest_container_cpp.DeleteSample(forest_num)


class Forest:
    """
    In-memory python wrapper around a C++ tree ensemble object

    Parameters
    ----------
    num_trees : int
        Number of trees that each forest should contain
    output_dimension : int, optional
        Dimension of the leaf node parameters in each tree
    leaf_constant : bool, optional
        Whether the leaf node model is "constant" (i.e. prediction is simply a
        sum of leaf node parameters for every observation in a dataset) or not (i.e.
        each leaf node parameter is multiplied by a "basis vector" before being returned
        as a prediction).
    is_exponentiated : bool, optional
        Whether or not the leaf node parameters are stored in log scale (in which case, they
        must be exponentiated before being returned as predictions).
    """

    def __init__(
        self,
        num_trees: int,
        output_dimension: int = 1,
        leaf_constant: bool = True,
        is_exponentiated: bool = False,
    ) -> None:
        self.forest_cpp = ForestCpp(
            num_trees, output_dimension, leaf_constant, is_exponentiated
        )
        self.num_trees = num_trees
        self.output_dimension = output_dimension
        self.leaf_constant = leaf_constant
        self.is_exponentiated = is_exponentiated

    def reset_root(self) -> None:
        """
        Reset forest to a forest with all single node (i.e. "root") trees
        """
        self.forest_cpp.ResetRoot()

    def reset(self, forest_container: ForestContainer, forest_num: int) -> None:
        """
        Reset forest to the forest indexed by ``forest_num`` in ``forest_container``

        Parameters
        ----------
        forest_container : `ForestContainer
            Stochtree object storing tree ensembles
        forest_num : int
            Index of the ensemble used to reset the ``Forest``
        """
        self.forest_cpp.Reset(forest_container.forest_container_cpp, forest_num)

    def predict(self, dataset: Dataset) -> np.array:
        """
        Predict from each forest in the container, using the provided `Dataset` object.

        Parameters
        ----------
        dataset : Dataset
            Python object wrapping the "dataset" class used by C++ sampling and prediction data structures.

        Returns
        -------
        np.array
            One-dimensional numpy array with length equal to the number of observations in `dataset`.
        """
        return self.forest_cpp.Predict(dataset.dataset_cpp)

    def predict_raw(self, dataset: Dataset) -> np.array:
        """
        Predict raw leaf values for a every forest in the container, using the provided `Dataset` object

        Parameters
        ----------
        dataset : Dataset
            Python object wrapping the "dataset" class used by C++ sampling and prediction data structures.

        Returns
        -------
        np.array
            Numpy array with (`n`, `k`) dimensions, where `n` is the number of observations in `dataset` and
            `k` is the dimension of the leaf parameter. If `k = 1`, then the returned array is simply one-dimensional
            with `n` observations.
        """
        result = self.forest_cpp.PredictRaw(dataset.dataset_cpp)
        if result.ndim == 3:
            if result.shape[1] == 1:
                result = result.reshape(result.shape[0], result.shape[2])
        return result

    def set_root_leaves(self, leaf_value: Union[float, np.array]) -> None:
        """
        Set constant (root) leaf node values for every tree in the forest.
        Assumes the forest consists of all root (single-node) trees.

        Parameters
        ----------
        leaf_value : float or np.array
            Constant values to which root nodes are to be set. If the trees in forest `forest_num`
            are univariate, then `leaf_value` must be a `float`, while if the trees in forest `forest_num`
            are multivariate, then `leaf_value` must be a `np.array`.
        """
        if not isinstance(leaf_value, np.ndarray) and not isinstance(leaf_value, float):
            raise ValueError("leaf_value must be either a float or np.array")
        if isinstance(leaf_value, np.ndarray):
            leaf_value = np.squeeze(leaf_value)
            if len(leaf_value.shape) != 1:
                raise ValueError("leaf_value must be either a one-dimensional array")
            self.forest_cpp.SetRootVector(leaf_value, leaf_value.shape[0])
        else:
            self.forest_cpp.SetRootValue(leaf_value)

    def add_numeric_split(
        self,
        tree_num: int,
        leaf_num: int,
        feature_num: int,
        split_threshold: float,
        left_leaf_value: Union[float, np.array],
        right_leaf_value: Union[float, np.array],
    ) -> None:
        """
        Add a numeric (i.e. X[,i] <= c) split to a given tree in the forest

        Parameters
        ----------
        tree_num : int
            Index of the tree to be split
        leaf_num : int
            Leaf to be split
        feature_num : int
            Feature that defines the new split
        split_threshold : float
            Value that defines the cutoff of the new split
        left_leaf_value : float or np.array
            Value (or array of values) to assign to the newly created left node
        right_leaf_value : float or np.array
            Value (or array of values) to assign to the newly created right node
        """
        if isinstance(left_leaf_value, np.ndarray):
            left_leaf_value = np.squeeze(left_leaf_value)
            right_leaf_value = np.squeeze(right_leaf_value)
            self.forest_cpp.AddNumericSplitVector(
                tree_num,
                leaf_num,
                feature_num,
                split_threshold,
                left_leaf_value,
                right_leaf_value,
            )
        else:
            self.forest_cpp.AddNumericSplitValue(
                tree_num,
                leaf_num,
                feature_num,
                split_threshold,
                left_leaf_value,
                right_leaf_value,
            )

    def get_tree_leaves(self, tree_num: int) -> np.array:
        """
        Retrieve a vector of indices of leaf nodes for a given tree in the forest

        Parameters
        ----------
        tree_num : float or np.array
            Index of the tree for which leaf indices will be retrieved

        Returns
        -------
        np.array
            One-dimensional numpy array, containing the indices of leaf nodes in a given tree.
        """
        return self.forest_cpp.GetTreeLeaves(tree_num)

    def get_tree_split_counts(self, tree_num: int, num_features: int) -> np.array:
        """
        Retrieve a vector of split counts for every training set variable in a given tree in the forest

        Parameters
        ----------
        tree_num : int
            Index of the tree for which split counts will be retrieved
        num_features : int
            Total number of features in the training set

        Returns
        -------
        np.array
            One-dimensional numpy array with as many elements as in the forest model's training set,
            containing the split count for each feature for a given tree of the forest.
        """
        return self.forest_cpp.GetTreeSplitCounts(tree_num, num_features)

    def get_overall_split_counts(self, num_features: int) -> np.array:
        """
        Retrieve a vector of split counts for every training set variable in the forest

        Parameters
        ----------
        num_features : int
            Total number of features in the training set

        Returns
        -------
        np.array
            One-dimensional numpy array with as many elements as in the forest model's training set,
            containing the overall split count in the forest for each feature.
        """
        return self.forest_cpp.GetOverallSplitCounts(num_features)

    def get_granular_split_counts(self, num_features: int) -> np.array:
        """
        Retrieve a vector of split counts for every training set variable in the forest, reported separately for each tree

        Parameters
        ----------
        num_features : int
            Total number of features in the training set

        Returns
        -------
        np.array
            One-dimensional numpy array with as many elements as in the forest model's training set,
            containing the split count for each feature for a every tree in the forest.
        """
        return self.forest_cpp.GetGranularSplitCounts(num_features)

    def num_forest_leaves(self) -> int:
        """
        Return the total number of leaves in a forest

        Returns
        -------
        int
            Number of leaves in a forest
        """
        return self.forest_cpp.NumLeavesForest()

    def sum_leaves_squared(self) -> float:
        """
        Return the total sum of squared leaf values in a forest

        Returns
        -------
        float
            Sum of squared leaf values in a forest
        """
        return self.forest_cpp.SumLeafSquared()

    def is_leaf_node(self, tree_num: int, node_id: int) -> bool:
        """
        Whether or not a given node of a given tree of a forest is a leaf

        tree_num : int
            Index of the tree to be queried
        node_id : int
            Index of the node to be queried

        Returns
        -------
        bool
            `True` if node `node_id` in tree `tree_num` is a leaf, `False` otherwise
        """
        return self.forest_cpp.IsLeafNode(tree_num, node_id)

    def is_numeric_split_node(self, tree_num: int, node_id: int) -> bool:
        """
        Whether or not a given node of a given tree of a forest is a numeric split node

        Parameters
        ----------
        tree_num : int
            Index of the tree to be queried
        node_id : int
            Index of the node to be queried

        Returns
        -------
        bool
            `True` if node `node_id` in tree `tree_num` is a numeric split node, `False` otherwise
        """
        return self.forest_cpp.IsNumericSplitNode(tree_num, node_id)

    def is_categorical_split_node(self, tree_num: int, node_id: int) -> bool:
        """
        Whether or not a given node of a given tree of a forest is a categorical split node

        Parameters
        ----------
        tree_num : int
            Index of the tree to be queried
        node_id : int
            Index of the node to be queried

        Returns
        -------
        bool
            `True` if node `node_id` in tree `tree_num` is a categorical split node, `False` otherwise
        """
        return self.forest_cpp.IsCategoricalSplitNode(tree_num, node_id)

    def parent_node(self, tree_num: int, node_id: int) -> int:
        """
        Parent node of given node of a given tree of a forest

        Parameters
        ----------
        tree_num : int
            Index of the tree to be queried
        node_id : int
            Index of the node to be queried

        Returns
        -------
        int
            Index of the parent of node `node_id` in tree `tree_num`.
            If `node_id` is a root node, returns `-1`.
        """
        return self.forest_cpp.ParentNode(tree_num, node_id)

    def left_child_node(self, tree_num: int, node_id: int) -> int:
        """
        Left child node of given node of a given tree of a forest

        Parameters
        ----------
        tree_num : int
            Index of the tree to be queried
        node_id : int
            Index of the node to be queried

        Returns
        -------
        int
            Index of the left child of node `node_id` in tree `tree_num`.
            If `node_id` is a leaf, returns `-1`.
        """
        return self.forest_cpp.LeftChildNode(tree_num, node_id)

    def right_child_node(self, tree_num: int, node_id: int) -> int:
        """
        Right child node of given node of a given tree of a forest

        Parameters
        ----------
        tree_num : int
            Index of the tree to be queried
        node_id : int
            Index of the node to be queried

        Returns
        -------
        int
            Index of the right child of node `node_id` in tree `tree_num`.
            If `node_id` is a leaf, returns `-1`.
        """
        return self.forest_cpp.RightChildNode(tree_num, node_id)

    def node_depth(self, tree_num: int, node_id: int) -> int:
        """
        Depth of given node of a given tree of a forest
        Returns ``-1`` if the node is a leaf.

        Parameters
        ----------
        tree_num : int
            Index of the tree to be queried
        node_id : int
            Index of the node to be queried

        Returns
        -------
        int
            Depth of node `node_id` in tree `tree_num`. The root node is defined as "depth zero."
        """
        return self.forest_cpp.NodeDepth(tree_num, node_id)

    def node_split_index(self, tree_num: int, node_id: int) -> int:
        """
        Split index of given node of a given tree of a forest.
        Returns ``-1`` if the node is a leaf.

        Parameters
        ----------
        tree_num : int
            Index of the tree to be queried
        node_id : int
            Index of the node to be queried

        Returns
        -------
        int
            Split index of `node_id` in tree `tree_num`.
        """
        if self.is_leaf_node(tree_num, node_id):
            return -1
        else:
            return self.forest_cpp.SplitIndex(tree_num, node_id)

    def node_split_threshold(self, tree_num: int, node_id: int) -> float:
        """
        Threshold that defines a numeric split for a given node of a given tree of a forest.
        Returns ``np.Inf`` if the node is a leaf or a categorical split node.

        Parameters
        ----------
        tree_num : int
            Index of the tree to be queried
        node_id : int
            Index of the node to be queried

        Returns
        -------
        float
            Threshold that defines a numeric split for node `node_id` in tree `tree_num`.
        """
        if self.is_leaf_node(tree_num, node_id) or self.is_categorical_split_node(
            tree_num, node_id
        ):
            return np.Inf
        else:
            return self.forest_cpp.SplitThreshold(tree_num, node_id)

    def node_split_categories(self, tree_num: int, node_id: int) -> np.array:
        """
        Array of category indices that define a categorical split for a given node of a given tree of a forest.
        Returns ``np.array([np.Inf])`` if the node is a leaf or a numeric split node.

        Parameters
        ----------
        tree_num : int
            Index of the tree to be queried
        node_id : int
            Index of the node to be queried

        Returns
        -------
        np.array
            Array of category indices that define a categorical split for node `node_id` in tree `tree_num`.
        """
        if self.is_leaf_node(tree_num, node_id) or self.is_numeric_split_node(
            tree_num, node_id
        ):
            return np.array([np.Inf])
        else:
            return self.forest_cpp.SplitCategories(tree_num, node_id)

    def node_leaf_values(self, tree_num: int, node_id: int) -> np.array:
        """
        Leaf node value(s) for a given node of a given tree of a forest.
        Values are stale if the node is a split node.

        Parameters
        ----------
        tree_num : int
            Index of the tree to be queried
        node_id : int
            Index of the node to be queried

        Returns
        -------
        np.array
            Array of parameter values for node `node_id` in tree `tree_num`.
        """
        return self.forest_cpp.NodeLeafValues(tree_num, node_id)

    def num_nodes(self, tree_num: int) -> int:
        """
        Number of nodes in a given tree of a forest

        Parameters
        ----------
        tree_num : int
            Index of the tree to be queried

        Returns
        -------
        int
            Total number of nodes in tree `tree_num`.
        """
        return self.forest_cpp.NumNodes(tree_num)

    def num_leaves(self, tree_num: int) -> int:
        """
        Number of leaves in a given tree of a forest

        Parameters
        ----------
        tree_num : int
            Index of the tree to be queried

        Returns
        -------
        int
            Total number of leaves in tree `tree_num`.
        """
        return self.forest_cpp.NumLeaves(tree_num)

    def num_leaf_parents(self, tree_num: int) -> int:
        """
        Number of leaf parents in a given tree of a forest

        Parameters
        ----------
        tree_num : int
            Index of the tree to be queried

        Returns
        -------
        int
            Total number of leaf parents in tree `tree_num`.
        """
        return self.forest_cpp.NumLeafParents(tree_num)

    def num_split_nodes(self, tree_num: int) -> int:
        """
        Number of split_nodes in a given tree of a forest

        Parameters
        ----------
        tree_num : int
            Index of the tree to be queried

        Returns
        -------
        int
            Total number of split nodes in tree `tree_num`.
        """
        return self.forest_cpp.NumSplitNodes(tree_num)

    def nodes(self, tree_num: int) -> np.array:
        """
        Array of node indices in a given tree of a forest

        Parameters
        ----------
        tree_num : int
            Index of the tree to be queried

        Returns
        -------
        np.array
            Array of indices of nodes in tree `tree_num`.
        """
        return self.forest_cpp.Nodes(tree_num)

    def leaves(self, tree_num: int) -> np.array:
        """
        Array of leaf indices in a given tree of a forest

        Parameters
        ----------
        tree_num : int
            Index of the tree to be queried

        Returns
        -------
        np.array
            Array of indices of leaf nodes in tree `tree_num`.
        """
        return self.forest_cpp.Leaves(tree_num)
