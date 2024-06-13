import warnings
import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.stats import gamma
from .forest import ForestContainer
from stochtree_cpp import JsonCpp

class JSONSerializer:
    """Class that handles serialization and deserialization of stochastic forest models
    """

    def __init__(self) -> None:
        self.json_cpp = JsonCpp()
        self.num_forests = 0
        self.forest_labels = []
    
    def add_forest(self, forest_samples: ForestContainer) -> None:
        """Adds a container of forest samples to a json object

        :param forest_samples: Samples of a tree ensemble
        :type forest_samples: ForestContainer
        """
        forest_label = self.json_cpp.AddForest(forest_samples.forest_container_cpp)
        self.num_forests += 1
        self.forest_labels.append(forest_label)
    
    def add_scalar(self, field_name: str, field_value: float, subfolder_name: str = None) -> None:
        """Adds a scalar (numeric) value to a json object

        :param field_name: Name of the json field / label under which the numeric value will be stored
        :type field_name: str
        :param field_value: Numeric value to be stored
        :type field_value: float
        :param subfolder_name: Name of "subfolder" under which ``field_name`` to be stored in the json hierarchy
        :type subfolder_name: str, optional
        """
        if subfolder_name is None:
            self.json_cpp.AddDouble(field_name, field_value)
        else:
            self.json_cpp.AddDoubleSubfolder(subfolder_name, field_name, field_value)
    
    def add_boolean(self, field_name: str, field_value: bool, subfolder_name: str = None) -> None:
        """Adds a scalar (boolean) value to a json object

        :param field_name: Name of the json field / label under which the boolean value will be stored
        :type field_name: str
        :param field_value: Boolean value to be stored
        :type field_value: bool
        :param subfolder_name: Name of "subfolder" under which ``field_name`` to be stored in the json hierarchy
        :type subfolder_name: str, optional
        """
        if subfolder_name is None:
            self.json_cpp.AddBool(field_name, field_value)
        else:
            self.json_cpp.AddBoolSubfolder(subfolder_name, field_name, field_value)
    
    def add_string(self, field_name: str, field_value: str, subfolder_name: str = None) -> None:
        """Adds a string to a json object

        :param field_name: Name of the json field / label under which the numeric value will be stored
        :type field_name: str
        :param field_value: String field to be stored
        :type field_value: str
        :param subfolder_name: Name of "subfolder" under which ``field_name`` to be stored in the json hierarchy
        :type subfolder_name: str, optional
        """
        if subfolder_name is None:
            self.json_cpp.AddString(field_name, field_value)
        else:
            self.json_cpp.AddStringSubfolder(subfolder_name, field_name, field_value)
    
    def add_numeric_vector(self, field_name: str, field_vector: np.array, subfolder_name: str = None) -> None:
        """Adds a numeric vector (stored as a numpy array) to a json object

        :param field_name: Name of the json field / label under which the numeric vector will be stored
        :type field_name: str
        :param field_vector: Numpy array containing the vector to be stored in json. Should be one-dimensional.
        :type field_vector: np.array
        :param subfolder_name: Name of "subfolder" under which ``field_name`` to be stored in the json hierarchy
        :type subfolder_name: str, optional
        """
        # Runtime checks
        if not isinstance(field_vector, np.ndarray):
            raise ValueError("field_vector must be a numpy array")
        field_vector = np.squeeze(field_vector)
        if field_vector.ndim > 1:
            warnings.warn("field_vector has more than 1 dimension. It will be flattened in row-major order using np.ravel()")
            field_vector = np.ravel(field_vector, order = "C")
        
        if subfolder_name is None:
            self.json_cpp.AddDoubleVector(field_name, field_vector)
        else:
            self.json_cpp.AddDoubleVectorSubfolder(subfolder_name, field_name, field_vector)
    
    def add_string_vector(self, field_name: str, field_vector: list, subfolder_name: str = None) -> None:
        """Adds a list of strings to a json object as an array

        :param field_name: Name of the json field / label under which the string list will be stored
        :type field_name: str
        :param field_vector: Python list of strings containing the array to be stored in json
        :type field_vector: list
        :param subfolder_name: Name of "subfolder" under which ``field_name`` to be stored in the json hierarchy
        :type subfolder_name: str, optional
        """
        # Runtime checks
        if not isinstance(field_vector, list):
            raise ValueError("field_vector must be a list")
        
        if subfolder_name is None:
            self.json_cpp.AddStringVector(field_name, field_vector)
        else:
            self.json_cpp.AddStringVectorSubfolder(subfolder_name, field_name, field_vector)
    
    def get_scalar(self, field_name: str, subfolder_name: str = None) -> float:
        """Retrieves a scalar (numeric) value from a json object

        :param field_name: Name of the json field / label under which the numeric value is stored
        :type field_name: str
        :param subfolder_name: Name of "subfolder" under which ``field_name`` is stored in the json hierarchy
        :type subfolder_name: str, optional
        """
        if subfolder_name is None:
            return self.json_cpp.ExtractDouble(field_name)
        else:
            return self.json_cpp.ExtractDoubleSubfolder(subfolder_name, field_name)
    
    def get_boolean(self, field_name: str, subfolder_name: str = None) -> bool:
        """Retrieves a scalar (boolean) value from a json object

        :param field_name: Name of the json field / label under which the boolean value is stored
        :type field_name: str
        :param subfolder_name: Name of "subfolder" under which ``field_name`` is stored in the json hierarchy
        :type subfolder_name: str, optional
        """
        if subfolder_name is None:
            return self.json_cpp.ExtractBool(field_name)
        else:
            return self.json_cpp.ExtractBoolSubfolder(subfolder_name, field_name)
    
    def get_string(self, field_name: str, subfolder_name: str = None) -> str:
        """Retrieve a string to a json object

        :param field_name: Name of the json field / label under which the numeric value is stored
        :type field_name: str
        :param subfolder_name: Name of "subfolder" under which ``field_name`` is stored in the json hierarchy
        :type subfolder_name: str, optional
        """
        if subfolder_name is None:
            return self.json_cpp.ExtractString(field_name)
        else:
            return self.json_cpp.ExtractStringSubfolder(subfolder_name, field_name)
    
    def get_numeric_vector(self, field_name: str, subfolder_name: str = None) -> np.array:
        """Adds a string to a json object

        :param field_name: Name of the json field / label under which the numeric vector is stored
        :type field_name: str
        :param subfolder_name: Name of "subfolder" under which ``field_name`` to be stored in the json hierarchy
        :type subfolder_name: str, optional
        """
        if subfolder_name is None:
            return self.json_cpp.ExtractDoubleVector(field_name)
        else:
            return self.json_cpp.ExtractDoubleVectorSubfolder(subfolder_name, field_name)
    
    def get_string_vector(self, field_name: str, subfolder_name: str = None) -> list:
        """Adds a string to a json object

        :param field_name: Name of the json field / label under which the string list is stored
        :type field_name: str
        :param subfolder_name: Name of "subfolder" under which ``field_name`` to be stored in the json hierarchy
        :type subfolder_name: str, optional
        """
        if subfolder_name is None:
            return self.json_cpp.ExtractStringVector(field_name)
        else:
            return self.json_cpp.ExtractStringVectorSubfolder(subfolder_name, field_name)
    
    def get_forest_container(self, forest_label: str) -> ForestContainer:
        result = ForestContainer(0, 1, True)
        result.forest_container_cpp.LoadFromJson(self.json_cpp, forest_label)
        return result
