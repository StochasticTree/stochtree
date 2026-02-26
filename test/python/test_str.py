import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from stochtree import BARTModel, BCFModel


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


@pytest.fixture(scope="module")
def bcf_data():
    """Shared simulated dataset for all BCF __str__ and summary tests."""
    rng = np.random.default_rng(42)
    n, p = 100, 5
    X = rng.uniform(0, 1, (n, p))
    pi_X = 0.2 + 0.6 * X[:, 0]
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


class TestBARTModelSummary:
    def test_sigma2_and_leaf_scale_sampled(self, bart_data):
        """With sigma2_global and sigma2_leaf both sampled, both appear in the summary."""
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
        s = model.summary()
        assert "sigma^2" in s
        assert "leaf scale" in s
        assert "in-sample" in s
        assert "test-set" in s

    def test_no_test_set(self, bart_data):
        """Without a test set, the test-set summary line is absent."""
        model = BARTModel()
        model.sample(
            X_train=bart_data["X_train"],
            y_train=bart_data["y_train"],
            num_gfr=0,
            num_burnin=10,
            num_mcmc=10,
            general_params={"sample_sigma2_global": False},
            mean_forest_params={"sample_sigma2_leaf": False},
        )
        s = model.summary()
        assert "in-sample" in s
        assert "test-set" not in s

    def test_intercept_only_rfx(self, bart_data):
        """Intercept-only random effects produce a single-component RFX summary."""
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
        s = model.summary()
        assert "Random effects" in s
        assert "Random effects overall mean" in s

    def test_custom_rfx(self, bart_data):
        """Custom (multi-component) random effects produce a multi-component RFX summary."""
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
        s = model.summary()
        assert "Random effects" in s
        assert "Variance component means" in s

    def test_returns_string(self, bart_data):
        """summary() returns a string."""
        model = BARTModel()
        model.sample(
            X_train=bart_data["X_train"],
            y_train=bart_data["y_train"],
            num_gfr=0,
            num_burnin=10,
            num_mcmc=10,
            general_params={"sample_sigma2_global": False},
            mean_forest_params={"sample_sigma2_leaf": False},
        )
        assert isinstance(model.summary(), str)


class TestBCFModelStr:
    def test_unsampled_model(self):
        assert "Empty BCFModel()" in str(BCFModel())

    def test_default_model(self, bcf_data):
        """Binary treatment, user propensity, adaptive coding (defaults): 2 base terms."""
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
            num_burnin=10,
            num_mcmc=10,
        )
        s = str(model)
        assert "BCFModel run with prognostic forest" in s
        assert "treatment effect forest" in s
        assert "User-provided propensity scores" in s
        assert "adaptive coding" in s
        assert "1 chain" in s
        assert "retaining every iteration" in s

    def test_more_than_two_model_terms(self, bcf_data):
        """Adding sigma2_global gives >2 terms and triggers Oxford-comma format."""
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
            num_burnin=10,
            num_mcmc=10,
            general_params={"sample_sigma2_global": True},
            prognostic_forest_params={"sample_sigma2_leaf": False},
            treatment_effect_forest_params={"sample_sigma2_leaf": False},
        )
        assert ", and global error variance model" in str(model)

    def test_adaptive_coding_disabled(self, bcf_data):
        """Binary treatment without adaptive coding shows 'default coding'."""
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
            num_burnin=10,
            num_mcmc=10,
            general_params={"adaptive_coding": False},
        )
        assert "default coding" in str(model)

    def test_propensity_excluded(self, bcf_data):
        """propensity_covariate='none' shows 'not used in either forest'."""
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
            num_burnin=10,
            num_mcmc=10,
            general_params={"propensity_covariate": "none"},
        )
        assert "not used in either forest" in str(model)

    def test_continuous_treatment(self, bcf_data):
        """Non-binary treatment shows 'univariate but not binary'."""
        rng = np.random.default_rng(7)
        n_train = bcf_data["X_train"].shape[0]
        n_test = bcf_data["X_test"].shape[0]
        Z_cont_train = rng.standard_normal(n_train)
        Z_cont_test = rng.standard_normal(n_test)
        y_cont_train = bcf_data["y_train"] + 0.5 * Z_cont_train
        model = BCFModel()
        model.sample(
            X_train=bcf_data["X_train"],
            Z_train=Z_cont_train,
            y_train=y_cont_train,
            X_test=bcf_data["X_test"],
            Z_test=Z_cont_test,
            num_gfr=0,
            num_burnin=10,
            num_mcmc=10,
            general_params={"propensity_covariate": "none"},
        )
        assert "univariate but not binary" in str(model)

    def test_multi_chain_and_thinning(self, bcf_data):
        """Multiple chains and thinning reflected in output."""
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
            num_burnin=10,
            num_mcmc=10,
            general_params={"num_chains": 2, "keep_every": 2},
        )
        s = str(model)
        assert "2 chains" in s
        assert "thinning" in s


class TestBCFModelSummary:
    def test_sigma2_and_leaf_scales_and_adaptive_coding(self, bcf_data):
        """sigma^2, both leaf scales, and adaptive coding parameters all appear when sampled."""
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
            num_burnin=10,
            num_mcmc=10,
            general_params={"sample_sigma2_global": True, "adaptive_coding": True},
            prognostic_forest_params={"sample_sigma2_leaf": True},
            treatment_effect_forest_params={"sample_sigma2_leaf": True},
        )
        s = model.summary()
        assert "sigma^2" in s
        assert "prognostic forest leaf scale" in s
        assert "treatment effect forest leaf scale" in s
        assert "adaptive coding parameters" in s
        assert "in-sample" in s
        assert "test-set" in s
        assert "CATEs" in s

    def test_no_test_set(self, bcf_data):
        """Without a test set, test-set summary lines are absent."""
        model = BCFModel()
        model.sample(
            X_train=bcf_data["X_train"],
            Z_train=bcf_data["Z_train"],
            y_train=bcf_data["y_train"],
            propensity_train=bcf_data["pi_train"],
            num_gfr=0,
            num_burnin=10,
            num_mcmc=10,
            general_params={"sample_sigma2_global": False},
        )
        s = model.summary()
        assert "in-sample" in s
        assert "test-set" not in s

    def test_returns_string(self, bcf_data):
        """summary() returns a string."""
        model = BCFModel()
        model.sample(
            X_train=bcf_data["X_train"],
            Z_train=bcf_data["Z_train"],
            y_train=bcf_data["y_train"],
            propensity_train=bcf_data["pi_train"],
            num_gfr=0,
            num_burnin=10,
            num_mcmc=10,
            general_params={"sample_sigma2_global": False},
        )
        assert isinstance(model.summary(), str)
