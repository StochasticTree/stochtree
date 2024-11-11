"""
Python classes wrapping C++ forest container object
"""
import numpy as np
from .data import Dataset, Residual
# from .serialization import JSONSerializer
from stochtree_cpp import ForestContainerCpp, ForestCpp
from typing import Union

class ForestContainer:
    def __init__(self, num_trees: int, output_dimension: int, leaf_constant: bool, is_exponentiated: bool) -> None:
        # Initialize a ForestContainerCpp object
        self.forest_container_cpp = ForestContainerCpp(num_trees, output_dimension, leaf_constant, is_exponentiated)
    
    def predict(self, dataset: Dataset) -> np.array:
        # Predict samples from Dataset
        return self.forest_container_cpp.Predict(dataset.dataset_cpp)
    
    def predict_raw(self, dataset: Dataset) -> np.array:
        # Predict raw leaf values for a specific forest (indexed by forest_num) from Dataset
        result = self.forest_container_cpp.PredictRaw(dataset.dataset_cpp)
        if result.ndim == 3:
            if result.shape[1] == 1:
                result = result.reshape(result.shape[0], result.shape[2])
        return result
    
    def predict_raw_single_forest(self, dataset: Dataset, forest_num: int) -> np.array:
        # Predict raw leaf values for a specific forest (indexed by forest_num) from Dataset
        return self.forest_container_cpp.PredictRawSingleForest(dataset.dataset_cpp, forest_num)
    
    def predict_raw_single_tree(self, dataset: Dataset, forest_num: int, tree_num: int) -> np.array:
        # Predict raw leaf values for a specific tree from specific forest from Dataset
        return self.forest_container_cpp.PredictRawSingleTree(dataset.dataset_cpp, forest_num, tree_num)
    
    def set_root_leaves(self, forest_num: int, leaf_value: Union[float, np.array]) -> None:
        # Predict raw leaf values for a specific forest (indexed by forest_num) from Dataset
        if not isinstance(leaf_value, np.ndarray) and not isinstance(leaf_value, float):
            raise ValueError("leaf_value must be either a float or np.array")
        if isinstance(leaf_value, np.ndarray):
            leaf_value = np.squeeze(leaf_value)
            if len(leaf_value.shape) != 1:
                raise ValueError("leaf_value must be either a one-dimensional array")
            self.forest_container_cpp.SetRootVector(forest_num, leaf_value, leaf_value.shape[0])
        else:
            self.forest_container_cpp.SetRootValue(forest_num, leaf_value)

    def save_to_json_file(self, json_filename: str) -> None:
        self.forest_container_cpp.SaveToJsonFile(json_filename)

    def load_from_json_file(self, json_filename: str) -> None:
        self.forest_container_cpp.LoadFromJsonFile(json_filename)

    def dump_json_string(self) -> str:
        return self.forest_container_cpp.DumpJsonString()

    def load_from_json_string(self, json_string: str) -> None:
        self.forest_container_cpp.LoadFromJsonString(json_string)
    
    def add_sample(self, leaf_value: Union[float, np.array]) -> None:
        """
        Add a new all-root ensemble to the container, with all of the leaves set to the value / vector provided

        leaf_value : :obj:`float` or :obj:`np.array`
            Value (or vector of values) to initialize root nodes in tree
        """
        if isinstance(leaf_value, np.ndarray):
            leaf_value = np.squeeze(leaf_value)
            self.forest_container_cpp.AddSampleVector(leaf_value)
        else:
            self.forest_container_cpp.AddSampleValue(leaf_value)
    
    def add_numeric_split(self, forest_num: int, tree_num: int, leaf_num: int, feature_num: int, split_threshold: float, 
                          left_leaf_value: Union[float, np.array], right_leaf_value: Union[float, np.array]) -> None:
        """
        Add a numeric (i.e. X[,i] <= c) split to a given tree in the ensemble

        forest_num : :obj:`int`
            Index of the forest which contains the tree to be split
        tree_num : :obj:`int`
            Index of the tree to be split
        leaf_num : :obj:`int`
            Leaf to be split
        feature_num : :obj:`int`
            Feature that defines the new split
        split_threshold : :obj:`float`
            Value that defines the cutoff of the new split
        left_leaf_value : :obj:`float` or :obj:`np.array`
            Value (or array of values) to assign to the newly created left node
        right_leaf_value : :obj:`float` or :obj:`np.array`
            Value (or array of values) to assign to the newly created right node
        """
        if isinstance(left_leaf_value, np.ndarray):
            left_leaf_value = np.squeeze(left_leaf_value)
            right_leaf_value = np.squeeze(right_leaf_value)
            self.forest_container_cpp.AddNumericSplitVector(forest_num, tree_num, leaf_num, feature_num, split_threshold, left_leaf_value, right_leaf_value)
        else:
            self.forest_container_cpp.AddNumericSplitValue(forest_num, tree_num, leaf_num, feature_num, split_threshold, left_leaf_value, right_leaf_value)
    
    def get_tree_leaves(self, forest_num: int, tree_num: int) -> np.array:
        """
        Retrieve a vector of indices of leaf nodes for a given tree in a given forest

        forest_num : :obj:`int`
            Index of the forest which contains tree `tree_num`
        tree_num : :obj:`float` or :obj:`np.array`
            Index of the tree for which leaf indices will be retrieved
        """
        return self.forest_container_cpp.GetTreeLeaves(forest_num, tree_num)

    def get_tree_split_counts(self, forest_num: int, tree_num: int, num_features: int) -> np.array:
        """
        Retrieve a vector of split counts for every training set variable in a given tree in a given forest

        forest_num : :obj:`int`
            Index of the forest which contains tree `tree_num`
        tree_num : :obj:`int`
            Index of the tree for which split counts will be retrieved
        num_features : :obj:`int`
            Total number of features in the training set
        """
        return self.forest_container_cpp.GetTreeSplitCounts(forest_num, tree_num, num_features)

    def get_forest_split_counts(self, forest_num: int, num_features: int) -> np.array:
        """
        Retrieve a vector of split counts for every training set variable in a given forest

        forest_num : :obj:`int`
            Index of the forest which contains tree `tree_num`
        num_features : :obj:`int`
            Total number of features in the training set
        """
        return self.forest_container_cpp.GetForestSplitCounts(forest_num, num_features)

    def get_overall_split_counts(self, num_features: int) -> np.array:
        """
        Retrieve a vector of split counts for every training set variable in a given forest, aggregated across ensembles and trees

        num_features : :obj:`int`
            Total number of features in the training set
        """
        return self.forest_container_cpp.GetOverallSplitCounts(num_features)

    def get_granular_split_counts(self, num_features: int) -> np.array:
        """
        Retrieve a vector of split counts for every training set variable in a given forest, reported separately for each ensemble and tree

        num_features : :obj:`int`
            Total number of features in the training set
        """
        return self.forest_container_cpp.GetGranularSplitCounts(num_features)

    def num_leaves(self, forest_num: int) -> int:
        """
        Return the total number of leaves for a given forest in the ``ForestContainer``

        forest_num : :obj:`int`
            Index of the forest to be queried
        """
        return self.forest_container_cpp.NumLeaves(forest_num)

    def sum_leaves_squared(self, forest_num: int) -> float:
        """
        Return the total sum of squared leaf values for a given forest in the ``ForestContainer``

        forest_num : :obj:`int`
            Index of the forest to be queried
        """
        return self.forest_container_cpp.SumLeafSquared(forest_num)

class Forest:
    def __init__(self, num_trees: int, output_dimension: int, leaf_constant: bool, is_exponentiated: bool) -> None:
        # Initialize a ForestCpp object
        self.forest_cpp = ForestCpp(num_trees, output_dimension, leaf_constant, is_exponentiated)
    
    def predict(self, dataset: Dataset) -> np.array:
        # Predict samples from Dataset
        return self.forest_cpp.Predict(dataset.dataset_cpp)
    
    def predict_raw(self, dataset: Dataset) -> np.array:
        # Predict raw leaf values for a specific forest (indexed by forest_num) from Dataset
        result = self.forest_cpp.PredictRaw(dataset.dataset_cpp)
        if result.ndim == 3:
            if result.shape[1] == 1:
                result = result.reshape(result.shape[0], result.shape[2])
        return result
    
    def set_root_leaves(self, leaf_value: Union[float, np.array]) -> None:
        # Predict raw leaf values for a specific forest (indexed by forest_num) from Dataset
        if not isinstance(leaf_value, np.ndarray) and not isinstance(leaf_value, float):
            raise ValueError("leaf_value must be either a float or np.array")
        if isinstance(leaf_value, np.ndarray):
            leaf_value = np.squeeze(leaf_value)
            if len(leaf_value.shape) != 1:
                raise ValueError("leaf_value must be either a one-dimensional array")
            self.forest_cpp.SetRootVector(leaf_value, leaf_value.shape[0])
        else:
            self.forest_cpp.SetRootValue(leaf_value)

    def add_numeric_split(self, tree_num: int, leaf_num: int, feature_num: int, split_threshold: float, 
                          left_leaf_value: Union[float, np.array], right_leaf_value: Union[float, np.array]) -> None:
        """
        Add a numeric (i.e. X[,i] <= c) split to a given tree in the forest

        tree_num : :obj:`int`
            Index of the tree to be split
        leaf_num : :obj:`int`
            Leaf to be split
        feature_num : :obj:`int`
            Feature that defines the new split
        split_threshold : :obj:`float`
            Value that defines the cutoff of the new split
        left_leaf_value : :obj:`float` or :obj:`np.array`
            Value (or array of values) to assign to the newly created left node
        right_leaf_value : :obj:`float` or :obj:`np.array`
            Value (or array of values) to assign to the newly created right node
        """
        if isinstance(left_leaf_value, np.ndarray):
            left_leaf_value = np.squeeze(left_leaf_value)
            right_leaf_value = np.squeeze(right_leaf_value)
            self.forest_cpp.AddNumericSplitVector(tree_num, leaf_num, feature_num, split_threshold, left_leaf_value, right_leaf_value)
        else:
            self.forest_cpp.AddNumericSplitValue(tree_num, leaf_num, feature_num, split_threshold, left_leaf_value, right_leaf_value)
    
    def get_tree_leaves(self, tree_num: int) -> np.array:
        """
        Retrieve a vector of indices of leaf nodes for a given tree in the forest

        tree_num : :obj:`float` or :obj:`np.array`
            Index of the tree for which leaf indices will be retrieved
        """
        return self.forest_cpp.GetTreeLeaves(tree_num)

    def get_tree_split_counts(self, tree_num: int, num_features: int) -> np.array:
        """
        Retrieve a vector of split counts for every training set variable in a given tree in the forest

        tree_num : :obj:`int`
            Index of the tree for which split counts will be retrieved
        num_features : :obj:`int`
            Total number of features in the training set
        """
        return self.forest_cpp.GetTreeSplitCounts(tree_num, num_features)

    def get_overall_split_counts(self, num_features: int) -> np.array:
        """
        Retrieve a vector of split counts for every training set variable in the forest

        num_features : :obj:`int`
            Total number of features in the training set
        """
        return self.forest_cpp.GetOverallSplitCounts(num_features)

    def get_granular_split_counts(self, num_features: int) -> np.array:
        """
        Retrieve a vector of split counts for every training set variable in the forest, reported separately for each tree

        num_features : :obj:`int`
            Total number of features in the training set
        """
        return self.forest_cpp.GetGranularSplitCounts(num_features)
