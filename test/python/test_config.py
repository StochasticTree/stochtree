import numpy as np
import pytest

from stochtree.config import ForestModelConfig, GlobalModelConfig


class TestConfig:
    def test_forest_config(self):
        with pytest.warns():
            _ = ForestModelConfig(num_trees=10, num_features=5, num_observations=100)
            _ = ForestModelConfig(num_trees=1, num_features=1, num_observations=1)
            _ = ForestModelConfig(
                num_trees=10,
                num_features=5,
                num_observations=100,
                feature_types=[0, 0, 0, 0, 1],
            )
            _ = ForestModelConfig(
                num_trees=1, num_features=1, num_observations=1, feature_types=[2]
            )
            _ = ForestModelConfig(
                num_trees=10,
                num_features=5,
                num_observations=100,
                variable_weights=[0.2, 0.2, 0.2, 0.2, 0.2],
            )
            _ = ForestModelConfig(
                num_trees=1, num_features=1, num_observations=1, variable_weights=[1.0]
            )

        with pytest.raises(ValueError):
            _ = ForestModelConfig()
            _ = ForestModelConfig(
                num_trees=10,
                num_features=6,
                num_observations=100,
                feature_types=[0, 0, 0, 0, 1],
            )
            _ = ForestModelConfig(
                num_trees=10,
                num_features=1,
                num_observations=100,
                feature_types=[0, 0, 0, 0, 1],
            )
            _ = ForestModelConfig(
                num_trees=10,
                num_features=6,
                num_observations=100,
                variable_weights=[0.2, 0.2, 0.2, 0.2, 0.2],
            )
            _ = ForestModelConfig(
                num_trees=10,
                num_features=1,
                num_observations=100,
                variable_weight=[0.2, 0.2, 0.2, 0.2, 0.2],
            )
            _ = ForestModelConfig(
                num_trees=10,
                num_features=1,
                num_observations=100,
                leaf_dimension=2,
                leaf_model_scale=np.array([2, 3], [3, 4], [5, 6]),
            )
            _ = ForestModelConfig(
                num_trees=10, num_features=1, num_observations=100, leaf_model_type=4
            )
            _ = ForestModelConfig(
                num_trees=10, num_features=1, num_observations=100, leaf_model_type=-1
            )

    def test_global_config(self):
        with pytest.raises(ValueError):
            _ = GlobalModelConfig(global_error_variance=0.0)
            _ = GlobalModelConfig(global_error_variance=-1.0)
