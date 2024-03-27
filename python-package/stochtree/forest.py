"""
Python classes wrapping C++ forest container object
"""
import numpy as np
from .data import Dataset
from stochtree_cpp import ForestContainerCpp

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
