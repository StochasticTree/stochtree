import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from stochtree import BARTModel, BCFModel


# ---------------------------------------------------------------------------
# Test data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bart_data():
    """Shared simulated dataset for BART extract_parameter tests."""
    rng = np.random.default_rng(42)
    n, p = 100, 5
    X = rng.uniform(0, 1, (n, p))
    y = 5 * X[:, 0] + rng.standard_normal(n)
    sample_inds = np.arange(n)
    train_inds, test_inds = train_test_split(sample_inds, test_size=0.2, random_state=42)
    return {
        "X_train": X[train_inds],
        "X_test": X[test_inds],
        "y_train": y[train_inds],
        "n_train": len(train_inds),
        "n_test": len(test_inds),
        "num_mcmc": 10,
    }


@pytest.fixture(scope="module")
def bcf_data():
    """Shared simulated dataset for BCF extract_parameter tests."""
    rng = np.random.default_rng(42)
    n, p = 100, 5
    X = rng.uniform(0, 1, (n, p))
    pi_X = 0.25 + 0.5 * X[:, 0]
    Z = rng.binomial(1, pi_X, n).astype(float)
    mu_X = 5 * X[:, 0]
    tau_X = 2 * X[:, 1]
    y = mu_X + tau_X * Z + rng.standard_normal(n)
    sample_inds = np.arange(n)
    train_inds, test_inds = train_test_split(sample_inds, test_size=0.2, random_state=42)
    return {
        "X_train": X[train_inds],
        "X_test": X[test_inds],
        "Z_train": Z[train_inds],
        "Z_test": Z[test_inds],
        "y_train": y[train_inds],
        "pi_train": pi_X[train_inds],
        "pi_test": pi_X[test_inds],
        "n_train": len(train_inds),
        "n_test": len(test_inds),
        "num_mcmc": 10,
    }

# ---------------------------------------------------------------------------
# BARTModel.extract_parameter tests
# ---------------------------------------------------------------------------

class TestBARTExtractParameter:
    def test_sigma2_aliases(self, bart_data):
        """sigma2 / global_error_scale / sigma2_global all return the same samples."""
        model = BARTModel()
        model.sample(
            X_train=bart_data["X_train"],
            y_train=bart_data["y_train"],
            X_test=bart_data["X_test"],
            num_gfr=0,
            num_burnin=0,
            num_mcmc=bart_data["num_mcmc"],
            general_params={"sample_sigma2_global": True},
            mean_forest_params={"sample_sigma2_leaf": False},
        )
        s2 = model.extract_parameter("sigma2")
        assert s2.shape == (bart_data["num_mcmc"],)
        assert np.array_equal(s2, model.extract_parameter("global_error_scale"))
        assert np.array_equal(s2, model.extract_parameter("sigma2_global"))

    def test_sigma2_leaf_aliases(self, bart_data):
        """sigma2_leaf / leaf_scale both return the leaf-scale samples."""
        model = BARTModel()
        model.sample(
            X_train=bart_data["X_train"],
            y_train=bart_data["y_train"],
            X_test=bart_data["X_test"],
            num_gfr=0,
            num_burnin=0,
            num_mcmc=bart_data["num_mcmc"],
            general_params={"sample_sigma2_global": False},
            mean_forest_params={"sample_sigma2_leaf": True},
        )
        sl = model.extract_parameter("sigma2_leaf")
        assert sl.shape == (bart_data["num_mcmc"],)
        assert np.array_equal(sl, model.extract_parameter("leaf_scale"))

    def test_y_hat_train(self, bart_data):
        """y_hat_train returns an (n_train, num_mcmc) array."""
        model = BARTModel()
        model.sample(
            X_train=bart_data["X_train"],
            y_train=bart_data["y_train"],
            X_test=bart_data["X_test"],
            num_gfr=0,
            num_burnin=0,
            num_mcmc=bart_data["num_mcmc"],
            general_params={"sample_sigma2_global": False},
            mean_forest_params={"sample_sigma2_leaf": False},
        )
        yht = model.extract_parameter("y_hat_train")
        assert yht.shape == (bart_data["n_train"], bart_data["num_mcmc"])

    def test_y_hat_test(self, bart_data):
        """y_hat_test returns an (n_test, num_mcmc) array."""
        model = BARTModel()
        model.sample(
            X_train=bart_data["X_train"],
            y_train=bart_data["y_train"],
            X_test=bart_data["X_test"],
            num_gfr=0,
            num_burnin=0,
            num_mcmc=bart_data["num_mcmc"],
            general_params={"sample_sigma2_global": False},
            mean_forest_params={"sample_sigma2_leaf": False},
        )
        yht = model.extract_parameter("y_hat_test")
        assert yht.shape == (bart_data["n_test"], bart_data["num_mcmc"])

    def test_sigma2_x_train_aliases(self, bart_data):
        """sigma2_x_train / var_x_train return variance forest in-sample predictions."""
        model = BARTModel()
        model.sample(
            X_train=bart_data["X_train"],
            y_train=bart_data["y_train"],
            X_test=bart_data["X_test"],
            num_gfr=0,
            num_burnin=0,
            num_mcmc=bart_data["num_mcmc"],
            general_params={"sample_sigma2_global": False},
            mean_forest_params={"sample_sigma2_leaf": False},
            variance_forest_params={"num_trees": 10},
        )
        s2x = model.extract_parameter("sigma2_x_train")
        assert s2x.shape == (bart_data["n_train"], bart_data["num_mcmc"])
        assert np.array_equal(s2x, model.extract_parameter("var_x_train"))

    def test_sigma2_x_test_aliases(self, bart_data):
        """sigma2_x_test / var_x_test return variance forest test-set predictions."""
        model = BARTModel()
        model.sample(
            X_train=bart_data["X_train"],
            y_train=bart_data["y_train"],
            X_test=bart_data["X_test"],
            num_gfr=0,
            num_burnin=0,
            num_mcmc=bart_data["num_mcmc"],
            general_params={"sample_sigma2_global": False},
            mean_forest_params={"sample_sigma2_leaf": False},
            variance_forest_params={"num_trees": 10},
        )
        s2x = model.extract_parameter("sigma2_x_test")
        assert s2x.shape == (bart_data["n_test"], bart_data["num_mcmc"])
        assert np.array_equal(s2x, model.extract_parameter("var_x_test"))

    def test_error_sigma2_not_sampled(self, bart_data):
        """Requesting sigma2 when not sampled raises ValueError."""
        model = BARTModel()
        model.sample(
            X_train=bart_data["X_train"],
            y_train=bart_data["y_train"],
            num_gfr=0,
            num_burnin=0,
            num_mcmc=bart_data["num_mcmc"],
            general_params={"sample_sigma2_global": False},
            mean_forest_params={"sample_sigma2_leaf": False},
        )
        with pytest.raises(ValueError, match="global variance"):
            model.extract_parameter("sigma2")

    def test_error_sigma2_leaf_not_sampled(self, bart_data):
        """Requesting sigma2_leaf when not sampled raises ValueError."""
        model = BARTModel()
        model.sample(
            X_train=bart_data["X_train"],
            y_train=bart_data["y_train"],
            num_gfr=0,
            num_burnin=0,
            num_mcmc=bart_data["num_mcmc"],
            general_params={"sample_sigma2_global": False},
            mean_forest_params={"sample_sigma2_leaf": False},
        )
        with pytest.raises(ValueError, match="leaf variance"):
            model.extract_parameter("sigma2_leaf")

    def test_error_y_hat_test_without_test_set(self, bart_data):
        """Requesting y_hat_test when no test set was provided raises ValueError."""
        model = BARTModel()
        model.sample(
            X_train=bart_data["X_train"],
            y_train=bart_data["y_train"],
            num_gfr=0,
            num_burnin=0,
            num_mcmc=bart_data["num_mcmc"],
            general_params={"sample_sigma2_global": False},
            mean_forest_params={"sample_sigma2_leaf": False},
        )
        with pytest.raises(ValueError, match="test set"):
            model.extract_parameter("y_hat_test")

    def test_error_variance_forest_not_fit(self, bart_data):
        """Requesting sigma2_x_train when no variance forest was fit raises ValueError."""
        model = BARTModel()
        model.sample(
            X_train=bart_data["X_train"],
            y_train=bart_data["y_train"],
            num_gfr=0,
            num_burnin=0,
            num_mcmc=bart_data["num_mcmc"],
            general_params={"sample_sigma2_global": False},
            mean_forest_params={"sample_sigma2_leaf": False},
        )
        with pytest.raises(ValueError, match="variance forest"):
            model.extract_parameter("sigma2_x_train")

    def test_error_invalid_term(self, bart_data):
        """An unrecognized term name raises ValueError."""
        model = BARTModel()
        model.sample(
            X_train=bart_data["X_train"],
            y_train=bart_data["y_train"],
            num_gfr=0,
            num_burnin=0,
            num_mcmc=bart_data["num_mcmc"],
            general_params={"sample_sigma2_global": False},
            mean_forest_params={"sample_sigma2_leaf": False},
        )
        with pytest.raises(ValueError, match="not a valid BART model term"):
            model.extract_parameter("not_a_real_term")


# ---------------------------------------------------------------------------
# BCFModel.extract_parameter tests
# ---------------------------------------------------------------------------

class TestBCFExtractParameter:
    def test_sigma2_aliases(self, bcf_data):
        """sigma2 / global_error_scale / sigma2_global all return the same samples."""
        model = BCFModel()
        model.sample(
            X_train=bcf_data["X_train"],
            Z_train=bcf_data["Z_train"],
            y_train=bcf_data["y_train"],
            propensity_train=bcf_data["pi_train"],
            X_test=bcf_data["X_test"],
            Z_test=bcf_data["Z_test"],
            propensity_test=bcf_data["pi_test"],
            num_gfr=0,
            num_burnin=0,
            num_mcmc=bcf_data["num_mcmc"],
            general_params={"sample_sigma2_global": True},
        )
        s2 = model.extract_parameter("sigma2")
        assert s2.shape == (bcf_data["num_mcmc"],)
        assert np.array_equal(s2, model.extract_parameter("global_error_scale"))
        assert np.array_equal(s2, model.extract_parameter("sigma2_global"))

    def test_sigma2_leaf_mu_aliases(self, bcf_data):
        """sigma2_leaf_mu / leaf_scale_mu / mu_leaf_scale return prognostic leaf-scale samples."""
        model = BCFModel()
        model.sample(
            X_train=bcf_data["X_train"],
            Z_train=bcf_data["Z_train"],
            y_train=bcf_data["y_train"],
            propensity_train=bcf_data["pi_train"],
            X_test=bcf_data["X_test"],
            Z_test=bcf_data["Z_test"],
            propensity_test=bcf_data["pi_test"],
            num_gfr=0,
            num_burnin=0,
            num_mcmc=bcf_data["num_mcmc"],
            general_params={"sample_sigma2_global": False},
            prognostic_forest_params={"sample_sigma2_leaf": True},
            treatment_effect_forest_params={"sample_sigma2_leaf": False},
        )
        sl = model.extract_parameter("sigma2_leaf_mu")
        assert sl.shape == (bcf_data["num_mcmc"],)
        assert np.array_equal(sl, model.extract_parameter("leaf_scale_mu"))
        assert np.array_equal(sl, model.extract_parameter("mu_leaf_scale"))

    def test_sigma2_leaf_tau_aliases(self, bcf_data):
        """sigma2_leaf_tau / leaf_scale_tau / tau_leaf_scale return treatment-effect leaf-scale samples."""
        model = BCFModel()
        model.sample(
            X_train=bcf_data["X_train"],
            Z_train=bcf_data["Z_train"],
            y_train=bcf_data["y_train"],
            propensity_train=bcf_data["pi_train"],
            X_test=bcf_data["X_test"],
            Z_test=bcf_data["Z_test"],
            propensity_test=bcf_data["pi_test"],
            num_gfr=0,
            num_burnin=0,
            num_mcmc=bcf_data["num_mcmc"],
            general_params={"sample_sigma2_global": False},
            prognostic_forest_params={"sample_sigma2_leaf": False},
            treatment_effect_forest_params={"sample_sigma2_leaf": True},
        )
        sl = model.extract_parameter("sigma2_leaf_tau")
        assert sl.shape == (bcf_data["num_mcmc"],)
        assert np.array_equal(sl, model.extract_parameter("leaf_scale_tau"))
        assert np.array_equal(sl, model.extract_parameter("tau_leaf_scale"))

    def test_adaptive_coding(self, bcf_data):
        """adaptive_coding returns a (2, num_mcmc) matrix (control row, treated row)."""
        model = BCFModel()
        model.sample(
            X_train=bcf_data["X_train"],
            Z_train=bcf_data["Z_train"],
            y_train=bcf_data["y_train"],
            propensity_train=bcf_data["pi_train"],
            X_test=bcf_data["X_test"],
            Z_test=bcf_data["Z_test"],
            propensity_test=bcf_data["pi_test"],
            num_gfr=0,
            num_burnin=0,
            num_mcmc=bcf_data["num_mcmc"],
            general_params={"sample_sigma2_global": False, "adaptive_coding": True},
        )
        ac = model.extract_parameter("adaptive_coding")
        assert ac.shape == (2, bcf_data["num_mcmc"])

    def test_y_hat_train(self, bcf_data):
        """y_hat_train returns an (n_train, num_mcmc) array."""
        model = BCFModel()
        model.sample(
            X_train=bcf_data["X_train"],
            Z_train=bcf_data["Z_train"],
            y_train=bcf_data["y_train"],
            propensity_train=bcf_data["pi_train"],
            X_test=bcf_data["X_test"],
            Z_test=bcf_data["Z_test"],
            propensity_test=bcf_data["pi_test"],
            num_gfr=0,
            num_burnin=0,
            num_mcmc=bcf_data["num_mcmc"],
            general_params={"sample_sigma2_global": False},
        )
        yht = model.extract_parameter("y_hat_train")
        assert yht.shape == (bcf_data["n_train"], bcf_data["num_mcmc"])

    def test_y_hat_test(self, bcf_data):
        """y_hat_test returns an (n_test, num_mcmc) array."""
        model = BCFModel()
        model.sample(
            X_train=bcf_data["X_train"],
            Z_train=bcf_data["Z_train"],
            y_train=bcf_data["y_train"],
            propensity_train=bcf_data["pi_train"],
            X_test=bcf_data["X_test"],
            Z_test=bcf_data["Z_test"],
            propensity_test=bcf_data["pi_test"],
            num_gfr=0,
            num_burnin=0,
            num_mcmc=bcf_data["num_mcmc"],
            general_params={"sample_sigma2_global": False},
        )
        yht = model.extract_parameter("y_hat_test")
        assert yht.shape == (bcf_data["n_test"], bcf_data["num_mcmc"])

    def test_tau_hat_train(self, bcf_data):
        """tau_hat_train returns an (n_train, num_mcmc) array."""
        model = BCFModel()
        model.sample(
            X_train=bcf_data["X_train"],
            Z_train=bcf_data["Z_train"],
            y_train=bcf_data["y_train"],
            propensity_train=bcf_data["pi_train"],
            X_test=bcf_data["X_test"],
            Z_test=bcf_data["Z_test"],
            propensity_test=bcf_data["pi_test"],
            num_gfr=0,
            num_burnin=0,
            num_mcmc=bcf_data["num_mcmc"],
            general_params={"sample_sigma2_global": False},
        )
        tht = model.extract_parameter("tau_hat_train")
        assert tht.shape == (bcf_data["n_train"], bcf_data["num_mcmc"])

    def test_tau_hat_test(self, bcf_data):
        """tau_hat_test returns an (n_test, num_mcmc) array."""
        model = BCFModel()
        model.sample(
            X_train=bcf_data["X_train"],
            Z_train=bcf_data["Z_train"],
            y_train=bcf_data["y_train"],
            propensity_train=bcf_data["pi_train"],
            X_test=bcf_data["X_test"],
            Z_test=bcf_data["Z_test"],
            propensity_test=bcf_data["pi_test"],
            num_gfr=0,
            num_burnin=0,
            num_mcmc=bcf_data["num_mcmc"],
            general_params={"sample_sigma2_global": False},
        )
        tht = model.extract_parameter("tau_hat_test")
        assert tht.shape == (bcf_data["n_test"], bcf_data["num_mcmc"])

    def test_sigma2_x_train_aliases(self, bcf_data):
        """sigma2_x_train / var_x_train return variance-forest in-sample predictions."""
        model = BCFModel()
        model.sample(
            X_train=bcf_data["X_train"],
            Z_train=bcf_data["Z_train"],
            y_train=bcf_data["y_train"],
            propensity_train=bcf_data["pi_train"],
            X_test=bcf_data["X_test"],
            Z_test=bcf_data["Z_test"],
            propensity_test=bcf_data["pi_test"],
            num_gfr=0,
            num_burnin=0,
            num_mcmc=bcf_data["num_mcmc"],
            general_params={"sample_sigma2_global": False},
            variance_forest_params={"num_trees": 10},
        )
        s2x = model.extract_parameter("sigma2_x_train")
        assert s2x.shape == (bcf_data["n_train"], bcf_data["num_mcmc"])
        assert np.array_equal(s2x, model.extract_parameter("var_x_train"))

    def test_sigma2_x_test_aliases(self, bcf_data):
        """sigma2_x_test / var_x_test return variance-forest test-set predictions."""
        model = BCFModel()
        model.sample(
            X_train=bcf_data["X_train"],
            Z_train=bcf_data["Z_train"],
            y_train=bcf_data["y_train"],
            propensity_train=bcf_data["pi_train"],
            X_test=bcf_data["X_test"],
            Z_test=bcf_data["Z_test"],
            propensity_test=bcf_data["pi_test"],
            num_gfr=0,
            num_burnin=0,
            num_mcmc=bcf_data["num_mcmc"],
            general_params={"sample_sigma2_global": False},
            variance_forest_params={"num_trees": 10},
        )
        s2x = model.extract_parameter("sigma2_x_test")
        assert s2x.shape == (bcf_data["n_test"], bcf_data["num_mcmc"])
        assert np.array_equal(s2x, model.extract_parameter("var_x_test"))

    def test_error_sigma2_not_sampled(self, bcf_data):
        """Requesting sigma2 when not sampled raises ValueError."""
        model = BCFModel()
        model.sample(
            X_train=bcf_data["X_train"],
            Z_train=bcf_data["Z_train"],
            y_train=bcf_data["y_train"],
            propensity_train=bcf_data["pi_train"],
            num_gfr=0,
            num_burnin=0,
            num_mcmc=bcf_data["num_mcmc"],
            general_params={"sample_sigma2_global": False},
        )
        with pytest.raises(ValueError, match="global variance"):
            model.extract_parameter("sigma2")

    def test_error_sigma2_leaf_mu_not_sampled(self, bcf_data):
        """Requesting sigma2_leaf_mu when not sampled raises ValueError."""
        model = BCFModel()
        model.sample(
            X_train=bcf_data["X_train"],
            Z_train=bcf_data["Z_train"],
            y_train=bcf_data["y_train"],
            propensity_train=bcf_data["pi_train"],
            num_gfr=0,
            num_burnin=0,
            num_mcmc=bcf_data["num_mcmc"],
            general_params={"sample_sigma2_global": False},
            prognostic_forest_params={"sample_sigma2_leaf": False},
        )
        with pytest.raises(ValueError, match="prognostic forest leaf variance"):
            model.extract_parameter("sigma2_leaf_mu")

    def test_error_sigma2_leaf_tau_not_sampled(self, bcf_data):
        """Requesting sigma2_leaf_tau when not sampled raises ValueError."""
        model = BCFModel()
        model.sample(
            X_train=bcf_data["X_train"],
            Z_train=bcf_data["Z_train"],
            y_train=bcf_data["y_train"],
            propensity_train=bcf_data["pi_train"],
            num_gfr=0,
            num_burnin=0,
            num_mcmc=bcf_data["num_mcmc"],
            general_params={"sample_sigma2_global": False},
            treatment_effect_forest_params={"sample_sigma2_leaf": False},
        )
        with pytest.raises(ValueError, match="treatment effect forest leaf variance"):
            model.extract_parameter("sigma2_leaf_tau")

    def test_error_adaptive_coding_disabled(self, bcf_data):
        """Requesting adaptive_coding when disabled raises ValueError."""
        model = BCFModel()
        model.sample(
            X_train=bcf_data["X_train"],
            Z_train=bcf_data["Z_train"],
            y_train=bcf_data["y_train"],
            propensity_train=bcf_data["pi_train"],
            num_gfr=0,
            num_burnin=0,
            num_mcmc=bcf_data["num_mcmc"],
            general_params={"sample_sigma2_global": False, "adaptive_coding": False},
        )
        with pytest.raises(ValueError, match="adaptive coding"):
            model.extract_parameter("adaptive_coding")

    def test_error_y_hat_test_without_test_set(self, bcf_data):
        """Requesting y_hat_test when no test set was provided raises ValueError."""
        model = BCFModel()
        model.sample(
            X_train=bcf_data["X_train"],
            Z_train=bcf_data["Z_train"],
            y_train=bcf_data["y_train"],
            propensity_train=bcf_data["pi_train"],
            num_gfr=0,
            num_burnin=0,
            num_mcmc=bcf_data["num_mcmc"],
            general_params={"sample_sigma2_global": False},
        )
        with pytest.raises(ValueError, match="test set"):
            model.extract_parameter("y_hat_test")

    def test_error_tau_hat_test_without_test_set(self, bcf_data):
        """Requesting tau_hat_test when no test set was provided raises ValueError."""
        model = BCFModel()
        model.sample(
            X_train=bcf_data["X_train"],
            Z_train=bcf_data["Z_train"],
            y_train=bcf_data["y_train"],
            propensity_train=bcf_data["pi_train"],
            num_gfr=0,
            num_burnin=0,
            num_mcmc=bcf_data["num_mcmc"],
            general_params={"sample_sigma2_global": False},
        )
        with pytest.raises(ValueError, match="test set"):
            model.extract_parameter("tau_hat_test")

    def test_error_variance_forest_not_fit(self, bcf_data):
        """Requesting sigma2_x_train when no variance forest was fit raises ValueError."""
        model = BCFModel()
        model.sample(
            X_train=bcf_data["X_train"],
            Z_train=bcf_data["Z_train"],
            y_train=bcf_data["y_train"],
            propensity_train=bcf_data["pi_train"],
            num_gfr=0,
            num_burnin=0,
            num_mcmc=bcf_data["num_mcmc"],
            general_params={"sample_sigma2_global": False},
        )
        with pytest.raises(ValueError, match="variance forest"):
            model.extract_parameter("sigma2_x_train")

    def test_error_invalid_term(self, bcf_data):
        """An unrecognized term name raises ValueError."""
        model = BCFModel()
        model.sample(
            X_train=bcf_data["X_train"],
            Z_train=bcf_data["Z_train"],
            y_train=bcf_data["y_train"],
            propensity_train=bcf_data["pi_train"],
            num_gfr=0,
            num_burnin=0,
            num_mcmc=bcf_data["num_mcmc"],
            general_params={"sample_sigma2_global": False},
        )
        with pytest.raises(ValueError, match="not a valid"):
            model.extract_parameter("not_a_real_term")
