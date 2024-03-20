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
    
    def predict(self, dataset: Dataset) -> None:
        # Predict samples from Dataset
        return self.forest_container_cpp.Predict(dataset.dataset_cpp)
