import numpy as np
from scipy.stats import norm

from stochtree import BARTModel, BCFModel, OutcomeModel


class TestInitParamsHonored:
    """Regression tests that user-supplied initialization / calibration parameters
    are threaded into the C++ sampler rather than silently dropped.

    Each test would fail if the corresponding parameter were ignored on the
    C++ path (as it was before these fixes)."""

    def test_bart_sigma2_init_honored(self):
        rng = np.random.default_rng(1)
        n, p = 200, 3
        X = rng.uniform(size=(n, p))
        y = X[:, 0] + rng.normal(size=n)

        def fit(s):
            m = BARTModel()
            general_params = {"standardize": False, "random_seed": 1}
            if s is not None:
                general_params["sigma2_init"] = s
            m.sample(
                X_train=X,
                y_train=y,
                num_gfr=0,
                num_burnin=0,
                num_mcmc=5,
                general_params=general_params,
            )
            return m

        m_set = fit(9.0)
        m_default = fit(None)
        # The user-supplied global variance init must be honored, not hardcoded to 1.0.
        assert np.isclose(m_set.sigma2_init, 9.0)
        assert not np.isclose(m_default.sigma2_init, 9.0)

    def test_bart_var_forest_leaf_init_honored(self):
        rng = np.random.default_rng(2)
        n, p = 200, 3
        X = rng.uniform(size=(n, p))
        # Heteroskedastic outcome so the variance forest is meaningfully fit.
        y = X[:, 0] + rng.normal(size=n) * np.exp(0.5 * X[:, 1])

        def fit(v):
            m = BARTModel()
            variance_forest_params = {"num_trees": 50}
            if v is not None:
                variance_forest_params["var_forest_leaf_init"] = v
            m.sample(
                X_train=X,
                y_train=y,
                X_test=X,
                num_gfr=0,
                num_burnin=0,
                num_mcmc=3,
                general_params={"standardize": False, "random_seed": 1},
                mean_forest_params={"num_trees": 50},
                variance_forest_params=variance_forest_params,
            )
            return np.asarray(m.sigma2_x_train)

        # Same seed; differing only by the variance-forest leaf init must change output.
        assert not np.allclose(fit(0.05), fit(2.0))

    def test_bcf_delta_max_honored(self):
        rng = np.random.default_rng(3)
        n, p = 300, 5
        X = rng.uniform(size=(n, p))
        Z = rng.binomial(1, 0.5, size=n).astype(float)
        propensity = np.full(n, 0.5)
        prob = norm.cdf(X[:, 0] - 0.5 + 0.5 * Z)
        y = rng.binomial(1, prob).astype(float)
        num_trees_tau = 50

        def fit(delta_max):
            m = BCFModel()
            m.sample(
                X_train=X,
                Z_train=Z,
                y_train=y,
                propensity_train=propensity,
                num_gfr=0,
                num_burnin=0,
                num_mcmc=5,
                general_params={
                    "random_seed": 1,
                    "outcome_model": OutcomeModel(outcome="binary", link="probit"),
                },
                treatment_effect_forest_params={
                    "num_trees": num_trees_tau,
                    "delta_max": delta_max,
                },
            )
            return m.sigma2_leaf_tau_init

        def expected(delta_max):
            p_coverage = 0.6827
            q_quantile = norm.ppf((p_coverage + 1) / 2)
            return ((delta_max / (q_quantile * norm.pdf(0))) ** 2) / num_trees_tau

        v_05 = fit(0.5)
        v_09 = fit(0.9)
        # delta_max must drive the treatment-effect leaf scale calibration.
        assert not np.isclose(v_05, v_09)
        assert np.isclose(v_05, expected(0.5))
        assert np.isclose(v_09, expected(0.9))

    def test_bart_observation_weights_honored(self):
        rng = np.random.default_rng(4)
        n, p = 200, 3
        X = rng.uniform(size=(n, p))
        y = X[:, 0] + rng.normal(size=n)
        w = rng.uniform(0.1, 2.0, size=n)

        def fit(weights):
            m = BARTModel()
            m.sample(
                X_train=X,
                y_train=y,
                X_test=X,
                num_gfr=0,
                num_burnin=0,
                num_mcmc=5,
                general_params={"standardize": False, "random_seed": 1},
                observation_weights_train=weights,
            )
            return np.asarray(m.y_hat_test)

        # Non-uniform observation weights must change the fit (same seed).
        assert not np.allclose(fit(None), fit(w))

    def test_bcf_observation_weights_honored(self):
        rng = np.random.default_rng(5)
        n, p = 200, 5
        X = rng.uniform(size=(n, p))
        Z = rng.binomial(1, 0.5, size=n).astype(float)
        propensity = np.full(n, 0.5)
        y = X[:, 0] + Z * X[:, 1] + rng.normal(size=n)
        w = rng.uniform(0.1, 2.0, size=n)

        def fit(weights):
            m = BCFModel()
            m.sample(
                X_train=X,
                Z_train=Z,
                y_train=y,
                propensity_train=propensity,
                X_test=X,
                Z_test=Z,
                propensity_test=propensity,
                num_gfr=0,
                num_burnin=0,
                num_mcmc=5,
                general_params={"standardize": False, "random_seed": 1},
                observation_weights_train=weights,
            )
            return np.asarray(m.y_hat_test)

        assert not np.allclose(fit(None), fit(w))

    def test_bcf_internal_propensity_reproducible(self):
        rng = np.random.default_rng(7)
        n, p = 200, 5
        X = rng.uniform(size=(n, p))
        Z = rng.binomial(1, 0.5, size=n).astype(float)
        y = X[:, 0] + Z * X[:, 1] + rng.normal(size=n)

        def fit():
            m = BCFModel()
            # No propensity_train -> BCF estimates it with an internal BART model.
            m.sample(
                X_train=X,
                Z_train=Z,
                y_train=y,
                num_gfr=0,
                num_burnin=0,
                num_mcmc=10,
                general_params={"random_seed": 99},
            )
            return np.asarray(m.y_hat_train)

        # The internally-estimated propensity (and hence the full fit) must be
        # reproducible across runs with a fixed random_seed.
        np.testing.assert_allclose(fit(), fit())
