"""
Python classes wrapping C++ forest container object
"""
import numpy as np
from .data import Dataset, Residual
# from .serialization import JSONSerializer
from stochtree_cpp import ForestContainerCpp
from typing import Union

class ForestContainer:
    def __init__(self, num_trees: int, output_dimension: int, leaf_constant: bool) -> None:
        # Initialize a ForestContainerCpp object
        self.forest_container_cpp = ForestContainerCpp(num_trees, output_dimension, leaf_constant)
    
    def predict(self, dataset: Dataset) -> np.array:
        # Predict samples from Dataset
        return self.forest_container_cpp.Predict(dataset.dataset_cpp)
    
    def predict_raw(self, dataset: Dataset) -> np.array:
        # Predict raw leaf values for a specific forest (indexed by forest_num) from Dataset
        return self.forest_container_cpp.PredictRaw(dataset.dataset_cpp)
    
    def predict_raw_single_forest(self, dataset: Dataset, forest_num: int) -> np.array:
        # Predict raw leaf values for a specific forest (indexed by forest_num) from Dataset
        return self.forest_container_cpp.PredictRawSingleForest(dataset.dataset_cpp, forest_num)
    
    def set_root_leaves(self, forest_num: int, leaf_value: Union[float, np.array]) -> None:
        # Predict raw leaf values for a specific forest (indexed by forest_num) from Dataset
        if not isinstance(leaf_value, np.ndarray) and not isinstance(leaf_value, float):
            raise ValueError("leaf_value must be either a floating point number or a numpy array")
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
    