import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from stochtree import BARTModel


@pytest.fixture(scope="module")
def bart_data():
    """Shared simulated dataset for all __str__ tests."""
    rng = np.random.default_rng(42)
    n, p = 100, 5
    X = rng.uniform(0, 1, (n, p))
    y = 5 * X[:, 0] + rng.standard_normal(n)
    rfx_group_ids = rng.integers(1, 5, size=n)
    rfx_basis = np.column_stack([np.ones(n), rng.uniform(0, 1, n)])
    sample_inds = np.arange(n)
    train_inds, test_inds = train_test_split(sample_inds, test_size=0.2, random_state=42)
    return {
        "X_train": X[train_inds],
        "X_test": X[test_inds],
        "y_train": y[train_inds],
        "y_test": y[test_inds],
        "rfx_group_ids_train": rfx_group_ids[train_inds],
        "rfx_group_ids_test": rfx_group_ids[test_inds],
        "rfx_basis_train": rfx_basis[train_inds],
        "rfx_basis_test": rfx_basis[test_inds],
    }


class TestBARTModelStr:
    def test_unsampled_model(self):
        assert "Empty BARTModel()" in str(BARTModel())

    def test_one_model_term(self, bart_data):
        """Mean forest only (no global variance, no leaf scale): 1 model term."""
        model = BARTModel()
        model.sample(
            X_train=bart_data["X_train"],
            y_train=bart_data["y_train"],
            X_test=bart_data["X_test"],
            num_gfr=0,
            num_burnin=10,
            num_mcmc=10,
            general_params={"sample_sigma2_global": False},
            mean_forest_params={"sample_sigma2_leaf": False},
        )
        s = str(model)
        assert "BARTModel run with mean forest" in s
        assert "constant leaf prior" in s
        assert "Outcome was standardized" in s
        assert "1 chain" in s
        assert "retaining every iteration" in s

    def test_two_model_terms(self, bart_data):
        """Mean forest + global error variance: 'X and Y' format."""
        model = BARTModel()
        model.sample(
            X_train=bart_data["X_train"],
            y_train=bart_data["y_train"],
            X_test=bart_data["X_test"],
            num_gfr=0,
            num_burnin=10,
            num_mcmc=10,
            general_params={"sample_sigma2_global": True},
            mean_forest_params={"sample_sigma2_leaf": False},
        )
        assert "mean forest and global error variance model" in str(model)

    def test_more_than_two_model_terms(self, bart_data):
        """Mean forest + global variance + leaf scale: Oxford comma format."""
        model = BARTModel()
        model.sample(
            X_train=bart_data["X_train"],
            y_train=bart_data["y_train"],
            X_test=bart_data["X_test"],
            num_gfr=0,
            num_burnin=10,
            num_mcmc=10,
            general_params={"sample_sigma2_global": True},
            mean_forest_params={"sample_sigma2_leaf": True},
        )
        assert ", and mean forest leaf scale model" in str(model)

    def test_leaf_regression(self, bart_data):
        """Leaf basis supplied: leaf regression branch."""
        basis_train = bart_data["rfx_basis_train"][:, :1]
        basis_test = bart_data["rfx_basis_test"][:, :1]
        model = BARTModel()
        model.sample(
            X_train=bart_data["X_train"],
            y_train=bart_data["y_train"],
            leaf_basis_train=basis_train,
            X_test=bart_data["X_test"],
            leaf_basis_test=basis_test,
            num_gfr=0,
            num_burnin=10,
            num_mcmc=10,
            general_params={"sample_sigma2_global": False},
            mean_forest_params={"sample_sigma2_leaf": False},
        )
        assert "leaf regression prior" in str(model)

    def test_intercept_only_rfx(self, bart_data):
        """Intercept-only random effects."""
        model = BARTModel()
        model.sample(
            X_train=bart_data["X_train"],
            y_train=bart_data["y_train"],
            X_test=bart_data["X_test"],
            rfx_group_ids_train=bart_data["rfx_group_ids_train"],
            rfx_group_ids_test=bart_data["rfx_group_ids_test"],
            num_gfr=0,
            num_burnin=10,
            num_mcmc=10,
            general_params={"sample_sigma2_global": False},
            mean_forest_params={"sample_sigma2_leaf": False},
            random_effects_params={"model_spec": "intercept_only"},
        )
        s = str(model)
        assert "additive random effects" in s
        assert "intercept-only" in s

    def test_custom_rfx(self, bart_data):
        """Custom (multi-component) random effects."""
        model = BARTModel()
        model.sample(
            X_train=bart_data["X_train"],
            y_train=bart_data["y_train"],
            X_test=bart_data["X_test"],
            rfx_group_ids_train=bart_data["rfx_group_ids_train"],
            rfx_group_ids_test=bart_data["rfx_group_ids_test"],
            rfx_basis_train=bart_data["rfx_basis_train"],
            rfx_basis_test=bart_data["rfx_basis_test"],
            num_gfr=0,
            num_burnin=10,
            num_mcmc=10,
            general_params={"sample_sigma2_global": False},
            mean_forest_params={"sample_sigma2_leaf": False},
            random_effects_params={"model_spec": "custom"},
        )
        s = str(model)
        assert "additive random effects" in s
        assert "user-supplied basis" in s

    def test_gfr_count(self, bart_data):
        """GFR iteration count is printed correctly."""
        model = BARTModel()
        model.sample(
            X_train=bart_data["X_train"],
            y_train=bart_data["y_train"],
            X_test=bart_data["X_test"],
            num_gfr=10,
            num_burnin=0,
            num_mcmc=10,
            general_params={"sample_sigma2_global": False},
            mean_forest_params={"sample_sigma2_leaf": False},
        )
        assert "10 GFR iterations" in str(model)

    def test_no_standardize(self, bart_data):
        """standardize=False: no 'standardized' line."""
        model = BARTModel()
        model.sample(
            X_train=bart_data["X_train"],
            y_train=bart_data["y_train"],
            X_test=bart_data["X_test"],
            num_gfr=0,
            num_burnin=10,
            num_mcmc=10,
            general_params={"standardize": False, "sample_sigma2_global": False},
            mean_forest_params={"sample_sigma2_leaf": False},
        )
        assert "standardized" not in str(model)

    def test_multi_chain_and_thinning(self, bart_data):
        """Multiple chains and thinning reflected in output."""
        model = BARTModel()
        model.sample(
            X_train=bart_data["X_train"],
            y_train=bart_data["y_train"],
            X_test=bart_data["X_test"],
            num_gfr=0,
            num_burnin=10,
            num_mcmc=10,
            general_params={
                "num_chains": 2,
                "keep_every": 2,
                "sample_sigma2_global": False,
            },
            mean_forest_params={"sample_sigma2_leaf": False},
        )
        s = str(model)
        assert "2 chains" in s
        assert "thinning" in s
