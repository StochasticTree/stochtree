import warnings

import numpy as np
from stochtree_cpp import JsonCpp

from .forest import ForestContainer
from .random_effects import RandomEffectsContainer


class JSONSerializer:
    """
    Class that handles serialization and deserialization of stochastic forest models
    """

    def __init__(self) -> None:
        self.json_cpp = JsonCpp()
        self.num_forests = 0
        self.num_rfx = 0
        self.forest_labels = []
        self.rfx_labels = []

    def return_json_string(self) -> str:
        """
        Convert JSON object to in-memory string

        Returns
        -------
        str
            JSON string representing model metadata (hyperparameters), sampled parameters, and sampled forests
        """
        return self.json_cpp.DumpJson()

    def load_from_json_string(self, json_string: str) -> None:
        """
        Parse in-memory JSON string to `JsonCpp` object

        Parameters
        -------
        json_string : str
            JSON string representing model metadata (hyperparameters), sampled parameters, and sampled forests
        """
        self.json_cpp.LoadFromString(json_string)

    def add_forest(self, forest_samples: ForestContainer) -> None:
        """Adds a container of forest samples to a json object

        Parameters
        ----------
        forest_samples : ForestContainer
            Samples of a tree ensemble
        """
        forest_label = self.json_cpp.AddForest(forest_samples.forest_container_cpp)
        self.num_forests += 1
        self.forest_labels.append(forest_label)

    def add_random_effects(self, rfx_container: RandomEffectsContainer) -> None:
        """Adds a container of random effect samples to a json object

        Parameters
        ----------
        rfx_container : RandomEffectsContainer
            Samples of a random effects model
        """
        _ = self.json_cpp.AddRandomEffectsContainer(rfx_container.rfx_container_cpp)
        _ = self.json_cpp.AddRandomEffectsLabelMapper(
            rfx_container.rfx_label_mapper_cpp
        )
        _ = self.json_cpp.AddRandomEffectsGroupIDs(rfx_container.rfx_group_ids)
        self.json_cpp.IncrementRandomEffectsCount()
        self.num_rfx += 1

    def add_scalar(
        self, field_name: str, field_value: float, subfolder_name: str = None
    ) -> None:
        """Adds a scalar (numeric) value to a json object

        Parameters
        ----------
        field_name : str
            Name of the json field / label under which the numeric value will be stored
        field_value : float
            Numeric value to be stored
        subfolder_name : str, optional
            Name of "subfolder" under which `field_name` to be stored in the json hierarchy
        """
        if subfolder_name is None:
            self.json_cpp.AddDouble(field_name, field_value)
        else:
            self.json_cpp.AddDoubleSubfolder(subfolder_name, field_name, field_value)

    def add_integer(
        self, field_name: str, field_value: int, subfolder_name: str = None
    ) -> None:
        """Adds an integer value to a json object

        Parameters
        ----------
        field_name : str
            Name of the json field / label under which the numeric value will be stored
        field_value : int
            Integer value to be stored
        subfolder_name : str, optional
            Name of "subfolder" under which `field_name` to be stored in the json hierarchy
        """
        if subfolder_name is None:
            self.json_cpp.AddInteger(field_name, field_value)
        else:
            self.json_cpp.AddIntegerSubfolder(subfolder_name, field_name, field_value)

    def add_boolean(
        self, field_name: str, field_value: bool, subfolder_name: str = None
    ) -> None:
        """Adds a scalar (boolean) value to a json object

        Parameters
        ----------
        field_name : str
            Name of the json field / label under which the boolean value will be stored
        field_value : bool
            Boolean value to be stored
        subfolder_name : str, optional
            Name of "subfolder" under which `field_name` to be stored in the json hierarchy
        """
        if subfolder_name is None:
            self.json_cpp.AddBool(field_name, field_value)
        else:
            self.json_cpp.AddBoolSubfolder(subfolder_name, field_name, field_value)

    def add_string(
        self, field_name: str, field_value: str, subfolder_name: str = None
    ) -> None:
        """Adds a string to a json object

        Parameters
        ----------
        field_name : str
            Name of the json field / label under which the numeric value will be stored
        field_value : str
            String field to be stored
        subfolder_name : str, optional
            Name of "subfolder" under which `field_name` to be stored in the json hierarchy
        """
        if subfolder_name is None:
            self.json_cpp.AddString(field_name, field_value)
        else:
            self.json_cpp.AddStringSubfolder(subfolder_name, field_name, field_value)

    def add_numeric_vector(
        self, field_name: str, field_vector: np.array, subfolder_name: str = None
    ) -> None:
        """Adds a numeric vector (stored as a numpy array) to a json object

        Parameters
        ----------
        field_name : str
            Name of the json field / label under which the numeric vector will be stored
        field_vector : np.array
            Numpy array containing the vector to be stored in json. Should be one-dimensional.
        subfolder_name : str, optional
            Name of "subfolder" under which `field_name` to be stored in the json hierarchy
        """
        # Runtime checks
        if not isinstance(field_vector, np.ndarray):
            raise ValueError("field_vector must be a numpy array")
        field_vector = np.squeeze(field_vector)
        if field_vector.ndim > 1:
            warnings.warn(
                "field_vector has more than 1 dimension. It will be flattened in row-major order using np.ravel()"
            )
            field_vector = np.ravel(field_vector, order="C")

        if subfolder_name is None:
            self.json_cpp.AddDoubleVector(field_name, field_vector)
        else:
            self.json_cpp.AddDoubleVectorSubfolder(
                subfolder_name, field_name, field_vector
            )

    def add_integer_vector(
        self, field_name: str, field_vector: np.array, subfolder_name: str = None
    ) -> None:
        """Adds a integer vector (stored as a numpy array) to a json object

        Parameters
        ----------
        field_name : str
            Name of the json field / label under which the integer vector will be stored
        field_vector : np.array
            Numpy array containing the vector to be stored in json. Should be one-dimensional.
        subfolder_name : str, optional
            Name of "subfolder" under which `field_name` to be stored in the json hierarchy
        """
        # Runtime checks
        if not isinstance(field_vector, np.ndarray):
            raise ValueError("field_vector must be a numpy array")
        if not np.issubdtype(field_vector.dtype, np.integer):
            raise ValueError(
                "field_vector must be a numpy array with integer data types"
            )
        field_vector = np.squeeze(field_vector)
        if field_vector.ndim > 1:
            warnings.warn(
                "field_vector has more than 1 dimension. It will be flattened in row-major order using np.ravel()"
            )
            field_vector = np.ravel(field_vector, order="C")

        if subfolder_name is None:
            self.json_cpp.AddIntegerVector(field_name, field_vector)
        else:
            self.json_cpp.AddIntegerVectorSubfolder(
                subfolder_name, field_name, field_vector
            )

    def add_string_vector(
        self, field_name: str, field_vector: list, subfolder_name: str = None
    ) -> None:
        """Adds a list of strings to a json object as an array

        Parameters
        ----------
        field_name : str
            Name of the json field / label under which the string list will be stored
        field_vector : list
            Python list of strings containing the array to be stored in json
        subfolder_name : str, optional
            Name of "subfolder" under which `field_name` to be stored in the json hierarchy
        """
        # Runtime checks
        if not isinstance(field_vector, list) and not isinstance(
            field_vector, np.ndarray
        ):
            raise ValueError("field_vector must be a list or numpy object array")

        if isinstance(field_vector, np.ndarray):
            field_vector = field_vector.tolist()
        if subfolder_name is None:
            self.json_cpp.AddStringVector(field_name, field_vector)
        else:
            self.json_cpp.AddStringVectorSubfolder(
                subfolder_name, field_name, field_vector
            )

    def get_scalar(self, field_name: str, subfolder_name: str = None) -> float:
        """Retrieves a scalar (numeric) value from a json object

        Parameters
        ----------
        field_name : str
            Name of the json field / label under which the numeric value is stored
        subfolder_name : str, optional
            Name of "subfolder" under which `field_name` is stored in the json hierarchy
        """
        if subfolder_name is None:
            return self.json_cpp.ExtractDouble(field_name)
        else:
            return self.json_cpp.ExtractDoubleSubfolder(subfolder_name, field_name)

    def get_integer(self, field_name: str, subfolder_name: str = None) -> int:
        """Retrieves an integer value from a json object

        Parameters
        ----------
        field_name : str
            Name of the json field / label under which the numeric value is stored
        subfolder_name : str, optional
            Name of "subfolder" under which `field_name` is stored in the json hierarchy
        """
        if subfolder_name is None:
            return self.json_cpp.ExtractInteger(field_name)
        else:
            return self.json_cpp.ExtractIntegerSubfolder(subfolder_name, field_name)

    def get_boolean(self, field_name: str, subfolder_name: str = None) -> bool:
        """Retrieves a scalar (boolean) value from a json object

        Parameters
        ----------
        field_name : str
            Name of the json field / label under which the boolean value is stored
        subfolder_name : str, optional
            Name of "subfolder" under which `field_name` is stored in the json hierarchy
        """
        if subfolder_name is None:
            return self.json_cpp.ExtractBool(field_name)
        else:
            return self.json_cpp.ExtractBoolSubfolder(subfolder_name, field_name)

    def get_string(self, field_name: str, subfolder_name: str = None) -> str:
        """Retrieve a string from a json object

        Parameters
        ----------
        field_name : str
            Name of the json field / label under which the string is stored
        subfolder_name : str, optional
            Name of "subfolder" under which `field_name` is stored in the json hierarchy
        """
        if subfolder_name is None:
            return self.json_cpp.ExtractString(field_name)
        else:
            return self.json_cpp.ExtractStringSubfolder(subfolder_name, field_name)

    def get_numeric_vector(
        self, field_name: str, subfolder_name: str = None
    ) -> np.array:
        """Retrieve numeric vector from a json object

        Parameters
        ----------
        field_name : str
            Name of the json field / label under which the numeric vector is stored
        subfolder_name : str, optional
            Name of "subfolder" under which `field_name` to be stored in the json hierarchy
        """
        if subfolder_name is None:
            return self.json_cpp.ExtractDoubleVector(field_name)
        else:
            return self.json_cpp.ExtractDoubleVectorSubfolder(
                subfolder_name, field_name
            )

    def get_integer_vector(
        self, field_name: str, subfolder_name: str = None
    ) -> np.array:
        """Retrieve integer vector from a json object

        Parameters
        ----------
        field_name : str
            Name of the json field / label under which the integer vector is stored
        subfolder_name : str, optional
            Name of "subfolder" under which `field_name` to be stored in the json hierarchy
        """
        if subfolder_name is None:
            return self.json_cpp.ExtractIntegerVector(field_name)
        else:
            return self.json_cpp.ExtractIntegerVectorSubfolder(
                subfolder_name, field_name
            )

    def get_string_vector(self, field_name: str, subfolder_name: str = None) -> list:
        """Adds a string to a json object

        Parameters
        ----------
        field_name : str
            Name of the json field / label under which the string list is stored
        subfolder_name : str, optional
            Name of "subfolder" under which `field_name` to be stored in the json hierarchy
        """
        if subfolder_name is None:
            return self.json_cpp.ExtractStringVector(field_name)
        else:
            return self.json_cpp.ExtractStringVectorSubfolder(
                subfolder_name, field_name
            )

    def get_forest_container(self, forest_str: str) -> ForestContainer:
        """Converts a JSON string for a container of forests to a `ForestContainer` object.

        Parameters
        ----------
        forest_str : str
            String containing the label for a given forest in a JSON object

        Returns
        -------
        ForestContainer
            In-memory `ForestContainer` python object, created from JSON
        """
        # TODO: read this from JSON
        result = ForestContainer(0, 1, True, False)
        result.forest_container_cpp.LoadFromJson(self.json_cpp, forest_str)
        return result

    def get_random_effects_container(
        self, random_effects_str: str
    ) -> RandomEffectsContainer:
        """Converts a JSON string for a random effects container to a `RandomEffectsContainer` object.

        Parameters
        ----------
        random_effects_str : str
            String containing the label for a given random effects term in a JSON object

        Returns
        -------
        RandomEffectsContainer
            In-memory `RandomEffectsContainer` python object, created from JSON
        """
        # TODO: read this from JSON
        result = RandomEffectsContainer()
        result.random_effects_container_cpp.LoadFromJson(
            self.json_cpp, random_effects_str
        )
        return result
