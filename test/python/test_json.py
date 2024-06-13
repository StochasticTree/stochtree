import numpy as np
from stochtree import BARTModel, JSONSerializer, ForestContainer, Dataset

class TestJson:
    def test_value(self):
        json_test = JSONSerializer()
        a = 1.5
        b = True
        c = "Example"
        json_test.add_scalar("a", a)
        json_test.add_boolean("b", b)
        json_test.add_string("c", c)
        assert a == json_test.get_scalar("a")
        assert b == json_test.get_boolean("b")
        assert c == json_test.get_string("c")

    def test_array(self):
        json_test = JSONSerializer()
        a = np.array([1.5,2.4,3.3])
        b = ["a","b","c"]
        json_test.add_numeric_vector("a", a)
        json_test.add_string_vector("b", b)
        np.testing.assert_array_equal(a, json_test.get_numeric_vector("a"))
        assert b == json_test.get_string_vector("b")

    def test_forest(self):
        # Generate sample data
        random_seed = 1234
        rng = np.random.default_rng(random_seed)
        n = 1000
        p_X = 10
        p_W = 1
        X = rng.uniform(0, 1, (n, p_X))
        def outcome_mean(X):
            return np.where(
                (X[:,0] >= 0.0) & (X[:,0] < 0.25), -7.5, 
                np.where(
                    (X[:,0] >= 0.25) & (X[:,0] < 0.5), -2.5, 
                    np.where(
                        (X[:,0] >= 0.5) & (X[:,0] < 0.75), 2.5, 
                        7.5
                    )
                )
            )
        epsilon = rng.normal(0, 1, n)
        y = outcome_mean(X) + epsilon

        # Train a BART model
        bart_model = BARTModel()
        bart_model.sample(X_train=X, y_train=y, num_gfr=10, num_mcmc=10)

        # Extract original predictions
        forest_preds_y_mcmc_cached = bart_model.y_hat_train

        # Extract original predictions
        forest_preds_y_mcmc_retrieved = bart_model.predict(X)

        # Roundtrip to / from JSON
        json_test = JSONSerializer()
        json_test.add_forest(bart_model.forest_container)
        forest_container = json_test.get_forest_container("forest_0")

        # Predict from the deserialized forest container
        forest_dataset = Dataset()
        forest_dataset.add_covariates(X)
        forest_preds_json_reload = forest_container.predict(forest_dataset)[:,bart_model.keep_indices]
        forest_preds_json_reload = forest_preds_json_reload*bart_model.y_std + bart_model.y_bar
        # Check the predictions
        np.testing.assert_almost_equal(forest_preds_y_mcmc_cached, forest_preds_json_reload)
        np.testing.assert_almost_equal(forest_preds_y_mcmc_retrieved, forest_preds_json_reload)
        