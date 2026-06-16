import warnings

import numpy as np
from stochtree_cpp import JsonCpp

from .forest import ForestContainer
from .random_effects import RandomEffectsContainer

# -----------------------------------------------------------------------------
# Serialized model envelope schema version (RFC 0005).
#
# Integer identifying the *structure* of the serialized BART/BCF JSON envelope.
# Bumped ONLY on a breaking change (rename / remove / re-type a field, change a
# field's meaning, or change a structural convention). Additive, safely-defaulted
# fields do NOT bump it -- they are handled by "augmentation" on read (see the
# defaults registry below and the ``get_*_or_default`` helpers on JSONSerializer).
#
# Kept in sync with the R ``STOCHTREE_SCHEMA_VERSION`` (R/serialization.R). The two
# are independent constants by design (each language owns its serde); their
# agreement is enforced by the cross-platform golden fixtures, not by sharing a value.
SCHEMA_VERSION = 1

# -----------------------------------------------------------------------------
# Augmentation defaults registry (schema_version = 1)
#
# Every OPTIONAL envelope field that may be ABSENT from a model written by an
# earlier release at the SAME schema_version must be read with a default that
# reproduces that model's pre-field behavior. This list is the single source of
# truth for "which fields need defaulting"; read those fields via
# ``get_scalar_or_default`` / ``get_boolean_or_default`` / ``get_string_or_default``
# (never a bare required getter).
#
#   field                        default        (behavior when absent)
#   "outcome"                    "continuous"
#   "link"                       "identity"
#   "multivariate_treatment"     False
#   "internal_propensity_model"  False
#   "has_rfx_basis"              False
#   ... (add a row whenever you add an additive field)
#
# If a new field has NO behavior-preserving default, it is NOT additive: bump
# SCHEMA_VERSION and add a migration step instead.
# -----------------------------------------------------------------------------


def resolve_schema_version(serializer: "JSONSerializer") -> int:
    """Read the envelope ``schema_version`` and enforce the RFC 0005 reader rules.

    Returns the loaded version (``0`` for a legacy / absent stamp). Behavior vs the
    current ``SCHEMA_VERSION``:

    - ``loaded > current`` -> hard error (model written by a newer stochtree).
    - ``loaded == current`` -> parse directly (caller proceeds).
    - ``0 < loaded < current`` -> caller runs the migration ladder (no rungs exist yet
      at ``SCHEMA_VERSION == 1``; this is the hook for future ``migrate_vN_to_vN+1``).
    - ``loaded == 0`` -> legacy model; handled by field-presence default-filling on parse.
    """
    loaded = serializer.get_integer_or_default("schema_version", 0)
    if loaded > SCHEMA_VERSION:
        raise ValueError(
            f"This model was serialized with schema_version={loaded}, but this "
            f"installation of stochtree supports up to schema_version={SCHEMA_VERSION}. "
            "Please upgrade stochtree to load it."
        )
    return loaded


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

    def contains(self, field_name: str, subfolder_name: str = None) -> bool:
        """Whether the json object contains ``field_name`` (optionally under ``subfolder_name``)."""
        if subfolder_name is None:
            return self.json_cpp.ContainsField(field_name)
        return self.json_cpp.ContainsFieldSubfolder(subfolder_name, field_name)

    def get_scalar_or_default(self, field_name, default, subfolder_name=None):
        """Read a numeric field, returning ``default`` if absent (augmentation; see RFC 0005)."""
        if self.contains(field_name, subfolder_name):
            return self.get_scalar(field_name, subfolder_name)
        return default

    def get_integer_or_default(self, field_name, default, subfolder_name=None):
        """Read an integer field, returning ``default`` if absent."""
        if self.contains(field_name, subfolder_name):
            return self.get_integer(field_name, subfolder_name)
        return default

    def get_boolean_or_default(self, field_name, default, subfolder_name=None):
        """Read a boolean field, returning ``default`` if absent."""
        if self.contains(field_name, subfolder_name):
            return self.get_boolean(field_name, subfolder_name)
        return default

    def get_string_or_default(self, field_name, default, subfolder_name=None):
        """Read a string field, returning ``default`` if absent."""
        if self.contains(field_name, subfolder_name):
            return self.get_string(field_name, subfolder_name)
        return default

    def rename_field(self, old_name, new_name, subfolder_name=None):
        """Rename ``old_name`` to ``new_name`` (optionally under ``subfolder_name``).

        No-op if ``old_name`` is absent. Used by JSON schema migrations (RFC 0005)."""
        if subfolder_name is None:
            self.json_cpp.RenameField(old_name, new_name)
        else:
            self.json_cpp.RenameFieldSubfolder(subfolder_name, old_name, new_name)

    def erase_field(self, field_name, subfolder_name=None):
        """Erase ``field_name`` (optionally under ``subfolder_name``). No-op if absent."""
        if subfolder_name is None:
            self.json_cpp.EraseField(field_name)
        else:
            self.json_cpp.EraseFieldSubfolder(subfolder_name, field_name)

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
