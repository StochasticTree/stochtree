import numpy as np
import pytest

from stochtree import BARTModel, BCFModel


def _make_bcf_data(seed=202, n=200, p=5):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, p))
    propensity = 0.3 + 0.4 * X[:, 1]
    Z = rng.binomial(1, propensity).astype(np.float64).reshape(-1, 1)
    pi = propensity.reshape(-1, 1)
    mu = 1.0 + 2.0 * X[:, 0]
    tau = 1.5 * X[:, 2]
    y = mu + tau * Z[:, 0] + 0.5 * rng.standard_normal(n)
    return X, Z, y, pi


def _make_data(seed=101, n=200, p=5):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, p))
    y = 2.0 * X[:, 0] + 0.5 * rng.standard_normal(n)
    return X, y


class TestBARTContinuation:
    def test_continue_matches_one_shot(self):
        # A continued chain (N then +M) must match a single run of N+M to within
        # floating-point noise. The residual is reconstructed at continue-time, so the
        # two chains differ only by ~1e-15 rounding, not by RNG/leaf-cache state.
        X, y = _make_data()
        N, M, seed = 10, 8, 1234

        cont = BARTModel()
        cont.sample(
            X_train=X, y_train=y, num_gfr=0, num_burnin=0, num_mcmc=N,
            general_params={"random_seed": seed},
        )
        cont.continue_sampling(X_train=X, y_train=y, num_mcmc=M, num_burnin=0)

        one_shot = BARTModel()
        one_shot.sample(
            X_train=X, y_train=y, num_gfr=0, num_burnin=0, num_mcmc=N + M,
            general_params={"random_seed": seed},
        )

        assert cont.num_samples == one_shot.num_samples == N + M
        pred_cont = cont.predict(X, terms="y_hat")
        pred_one_shot = one_shot.predict(X, terms="y_hat")
        np.testing.assert_allclose(pred_cont, pred_one_shot, atol=1e-10, rtol=0)

        # Sampled scalars should match too (global variance is sampled by default)
        np.testing.assert_allclose(
            cont.global_var_samples, one_shot.global_var_samples, atol=1e-10, rtol=0
        )

    def test_continuation_is_deterministic(self):
        # Two continuations from the same fitted model produce identical results.
        X, y = _make_data()

        def run():
            m = BARTModel()
            m.sample(
                X_train=X, y_train=y, num_gfr=0, num_burnin=0, num_mcmc=10,
                general_params={"random_seed": 1234},
            )
            m.continue_sampling(X_train=X, y_train=y, num_mcmc=8, num_burnin=0)
            return m.predict(X, terms="y_hat")

        np.testing.assert_array_equal(run(), run())

    def test_override_seed_changes_draws(self):
        # Supplying a new random_seed re-seeds the continued draws (no resume).
        X, y = _make_data()
        m = BARTModel()
        m.sample(
            X_train=X, y_train=y, num_gfr=0, num_burnin=0, num_mcmc=10,
            general_params={"random_seed": 1234},
        )
        resumed = BARTModel()
        resumed.sample(
            X_train=X, y_train=y, num_gfr=0, num_burnin=0, num_mcmc=10,
            general_params={"random_seed": 1234},
        )
        m.continue_sampling(X_train=X, y_train=y, num_mcmc=8, num_burnin=0)
        resumed.continue_sampling(
            X_train=X, y_train=y, num_mcmc=8, num_burnin=0, random_seed=999
        )
        # The newly drawn samples should differ between resume and override.
        assert not np.allclose(
            m.predict(X, terms="y_hat")[:, 10:],
            resumed.predict(X, terms="y_hat")[:, 10:],
        )

    def test_continuation_guards(self):
        X, y = _make_data()
        # Variance forest is not yet supported for continuation.
        var_model = BARTModel()
        var_model.sample(
            X_train=X, y_train=y, num_gfr=0, num_burnin=0, num_mcmc=5,
            variance_forest_params={"num_trees": 20},
        )
        with pytest.raises(NotImplementedError):
            var_model.continue_sampling(X_train=X, y_train=y, num_mcmc=5)


class TestBCFContinuation:
    def test_continue_matches_one_shot(self):
        # A continued BCF chain (N then +M) must match a single run of N+M to within
        # floating-point noise (bit-identical RNG resume + warm-start residual reconstruction).
        X, Z, y, pi = _make_bcf_data()
        N, M, seed = 10, 8, 1234

        cont = BCFModel()
        cont.sample(
            X_train=X, Z_train=Z, y_train=y, propensity_train=pi,
            num_gfr=0, num_burnin=0, num_mcmc=N,
            general_params={"random_seed": seed},
        )
        cont.continue_sampling(
            X_train=X, Z_train=Z, y_train=y, propensity_train=pi, num_mcmc=M, num_burnin=0
        )

        one_shot = BCFModel()
        one_shot.sample(
            X_train=X, Z_train=Z, y_train=y, propensity_train=pi,
            num_gfr=0, num_burnin=0, num_mcmc=N + M,
            general_params={"random_seed": seed},
        )

        assert cont.num_samples == one_shot.num_samples == N + M
        pred_cont = cont.predict(X, Z, propensity=pi, terms="y_hat")
        pred_one_shot = one_shot.predict(X, Z, propensity=pi, terms="y_hat")
        np.testing.assert_allclose(pred_cont, pred_one_shot, atol=1e-9, rtol=0)

        # Sampled scalars should match too (global variance + prognostic leaf scale + tau_0).
        np.testing.assert_allclose(
            cont.global_var_samples, one_shot.global_var_samples, atol=1e-9, rtol=0
        )
        if getattr(cont, "sample_sigma2_leaf_mu", False):
            np.testing.assert_allclose(
                cont.leaf_scale_mu_samples, one_shot.leaf_scale_mu_samples, atol=1e-9, rtol=0
            )
        if getattr(cont, "sample_tau_0", False):
            np.testing.assert_allclose(
                cont.tau_0_samples, one_shot.tau_0_samples, atol=1e-9, rtol=0
            )

    def test_continuation_is_deterministic(self):
        X, Z, y, pi = _make_bcf_data()

        def run():
            m = BCFModel()
            m.sample(
                X_train=X, Z_train=Z, y_train=y, propensity_train=pi,
                num_gfr=0, num_burnin=0, num_mcmc=10,
                general_params={"random_seed": 1234},
            )
            m.continue_sampling(
                X_train=X, Z_train=Z, y_train=y, propensity_train=pi, num_mcmc=8, num_burnin=0
            )
            return m.predict(X, Z, propensity=pi, terms="y_hat")

        np.testing.assert_array_equal(run(), run())

    def test_override_seed_changes_draws(self):
        X, Z, y, pi = _make_bcf_data()
        m = BCFModel()
        m.sample(
            X_train=X, Z_train=Z, y_train=y, propensity_train=pi,
            num_gfr=0, num_burnin=0, num_mcmc=10,
            general_params={"random_seed": 1234},
        )
        resumed = BCFModel()
        resumed.sample(
            X_train=X, Z_train=Z, y_train=y, propensity_train=pi,
            num_gfr=0, num_burnin=0, num_mcmc=10,
            general_params={"random_seed": 1234},
        )
        m.continue_sampling(
            X_train=X, Z_train=Z, y_train=y, propensity_train=pi, num_mcmc=8, num_burnin=0
        )
        resumed.continue_sampling(
            X_train=X, Z_train=Z, y_train=y, propensity_train=pi, num_mcmc=8, num_burnin=0,
            random_seed=999,
        )
        # The newly drawn samples should differ between resume and override.
        assert not np.allclose(
            m.predict(X, Z, propensity=pi, terms="y_hat")[:, 10:],
            resumed.predict(X, Z, propensity=pi, terms="y_hat")[:, 10:],
        )

    def test_continuation_guards(self):
        X, Z, y, pi = _make_bcf_data()
        # Variance forest is not yet supported for continuation.
        var_model = BCFModel()
        var_model.sample(
            X_train=X, Z_train=Z, y_train=y, propensity_train=pi,
            num_gfr=0, num_burnin=0, num_mcmc=5,
            variance_forest_params={"num_trees": 20},
        )
        with pytest.raises(NotImplementedError):
            var_model.continue_sampling(
                X_train=X, Z_train=Z, y_train=y, propensity_train=pi, num_mcmc=5
            )
