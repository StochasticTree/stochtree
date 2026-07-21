import numpy as np
import pandas as pd
import pytest

from stochtree import BARTModel, BCFModel, OutcomeModel


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
        # The retained history (first N draws) is carried forward verbatim, so it is bit-identical
        # to a one-shot run of N+M. The continued draws are only *statistically* equivalent, not
        # bit-identical: continuation resumes the RNG stream but not the sampler's pre-drawn
        # leaf-normal cache, so the realized draws differ while targeting the same posterior.
        # (See test_continuation_is_deterministic for the reproducibility guarantee.)
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
        # History is retained verbatim (bit-identical).
        np.testing.assert_allclose(
            pred_cont[:, :N], pred_one_shot[:, :N], atol=1e-10, rtol=0
        )
        np.testing.assert_allclose(
            cont.global_var_samples[:N], one_shot.global_var_samples[:N], atol=1e-10, rtol=0
        )
        # The full posterior (history + continued draws) targets the same distribution, so the
        # per-observation posterior-mean prediction agrees to within Monte Carlo noise.
        np.testing.assert_allclose(
            pred_cont.mean(axis=1), pred_one_shot.mean(axis=1), atol=0.25, rtol=0
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
            X_train=X, y_train=y, num_mcmc=8, num_burnin=0,
            general_params={"random_seed": 999},
        )
        # The newly drawn samples should differ between resume and override.
        assert not np.allclose(
            m.predict(X, terms="y_hat")[:, 10:],
            resumed.predict(X, terms="y_hat")[:, 10:],
        )

    def test_continuation_variance_forest_supported(self):
        # Variance forest continuation is supported: the variance forest warm-starts from its
        # last retained sample and new draws append.
        X, y = _make_data()
        var_model = BARTModel()
        var_model.sample(
            X_train=X, y_train=y, num_gfr=0, num_burnin=0, num_mcmc=5,
            variance_forest_params={"num_trees": 20},
        )
        var_model.continue_sampling(X_train=X, y_train=y, num_mcmc=5)
        assert var_model.num_samples == 10
        vf = var_model.predict(X, terms="variance_forest")
        assert vf.shape == (X.shape[0], 10)
        assert np.all(vf > 0) and np.all(np.isfinite(vf))

    def test_continuation_rfx_requires_group_ids(self):
        # An rfx model must be re-supplied its group ids to continue.
        rng = np.random.default_rng(3)
        n, p = 200, 4
        X = rng.uniform(0, 1, (n, p))
        g = rng.integers(0, 3, n).astype(np.int32)
        rb = np.ones((n, 1))
        y = 2.0 * X[:, 0] + g * 0.5 + 0.3 * rng.standard_normal(n)
        m = BARTModel()
        m.sample(
            X_train=X, y_train=y, num_gfr=0, num_burnin=0, num_mcmc=5,
            rfx_group_ids_train=g, rfx_basis_train=rb,
        )
        with pytest.raises(ValueError):
            m.continue_sampling(X_train=X, y_train=y, num_mcmc=5)  # missing rfx_group_ids_train
        # random_effects_params is a changeable dict on continuation (no warning, proceeds).
        m.continue_sampling(
            X_train=X, y_train=y, num_mcmc=5,
            rfx_group_ids_train=g, rfx_basis_train=rb,
            random_effects_params={"variance_prior_shape": 2.0, "variance_prior_scale": 2.0},
        )
        assert m.num_samples == 10

    def test_continuation_supports_test_data(self):
        # A model fit with no test set can be given one on continuation. Test predictions are
        # recomputed in full from all retained forests, so the stored trace covers every sample and
        # is bit-identical to a fresh predict() on the same X_test.
        X, y = _make_data(seed=7)
        rng = np.random.default_rng(8)
        X_test = rng.uniform(0, 1, (50, X.shape[1]))

        m = BARTModel()
        m.sample(X_train=X, y_train=y, num_gfr=0, num_burnin=5, num_mcmc=10,
                 general_params={"random_seed": 7})
        m.continue_sampling(X_train=X, y_train=y, X_test=X_test, num_mcmc=8)

        assert m.num_samples == 18
        assert m.has_test
        stored = m.y_hat_test
        assert stored.shape == (50, 18)
        np.testing.assert_allclose(
            stored, m.predict(X_test, terms="y_hat"), atol=1e-10, rtol=0
        )

    def test_continuation_drops_stale_test_data_with_warning(self):
        # Fit WITH a test set, then continue WITHOUT re-supplying it: the stale test predictions are
        # dropped (they would otherwise cover only the pre-continuation draws) and a warning is raised.
        X, y = _make_data(seed=7)
        rng = np.random.default_rng(8)
        X_test = rng.uniform(0, 1, (50, X.shape[1]))

        m = BARTModel()
        m.sample(X_train=X, y_train=y, X_test=X_test, num_gfr=0, num_burnin=5, num_mcmc=10,
                 general_params={"random_seed": 7})
        assert m.has_test

        with pytest.warns(UserWarning, match="test-set predictions are stale"):
            m.continue_sampling(X_train=X, y_train=y, num_mcmc=8)

        assert not m.has_test
        assert m.y_hat_test is None
        assert m.num_samples == 18

    def test_continuation_probit_supported(self):
        def run():
            rng = np.random.default_rng(7)
            n, p = 300, 4
            X = rng.uniform(0, 1, (n, p))
            z = (2 * X[:, 0] - 1) + rng.normal(0, 1, n)
            y = (z > 0).astype(float)
            m = BARTModel()
            m.sample(X_train=X, y_train=y, num_gfr=0, num_burnin=5, num_mcmc=10,
                     general_params={"random_seed": 1, "sample_sigma2_global": False,
                                     "outcome_model": OutcomeModel(outcome="binary", link="probit")})
            m.continue_sampling(X_train=X, y_train=y, num_mcmc=8)
            return m
        m = run()
        assert m.num_samples == 18
        Xt = np.random.default_rng(99).uniform(0, 1, (20, 4))
        np.testing.assert_allclose(run().predict(Xt, terms="y_hat"), m.predict(Xt, terms="y_hat"))

    def test_continuation_multivariate_leaf_supported(self):
        def run():
            rng = np.random.default_rng(7)
            n, p = 300, 4
            X = rng.uniform(0, 1, (n, p))
            W = rng.uniform(-1, 1, (n, 2))  # 2-col leaf basis -> multivariate leaf regression
            y = W[:, 0] * X[:, 0] + W[:, 1] * X[:, 1] + 0.3 * rng.standard_normal(n)
            m = BARTModel()
            m.sample(X_train=X, y_train=y, leaf_basis_train=W, num_gfr=0, num_burnin=5, num_mcmc=10,
                     general_params={"random_seed": 1})
            m.continue_sampling(X_train=X, y_train=y, leaf_basis_train=W, num_mcmc=8)
            return m
        m = run()
        assert m.num_samples == 18
        Xt = np.random.default_rng(99).uniform(0, 1, (20, 4))
        Wt = np.random.default_rng(5).uniform(-1, 1, (20, 2))
        np.testing.assert_allclose(run().predict(Xt, leaf_basis=Wt, terms="y_hat"),
                                   m.predict(Xt, leaf_basis=Wt, terms="y_hat"))

    def test_continuation_respecifies_split_variable_selection(self):
        X, y = _make_data(seed=303, n=200, p=5)

        def fresh():
            m = BARTModel()
            m.sample(X_train=X, y_train=y, num_gfr=0, num_burnin=5, num_mcmc=6,
                     general_params={"random_seed": 3})
            return m

        # keep_vars / drop_vars + variable_weights are changeable via the param dicts: no warning,
        # continuation proceeds.
        m1 = fresh()
        m1.continue_sampling(
            X_train=X, y_train=y, num_mcmc=6,
            mean_forest_params={"drop_vars": [2, 3, 4]},
        )
        assert m1.num_samples == 12

        m2 = fresh()
        m2.continue_sampling(
            X_train=X, y_train=y, num_mcmc=6,
            general_params={"variable_weights": np.array([0.5, 0.5, 0.0, 0.0, 0.0])},
        )
        assert m2.num_samples == 12

        # Malformed variable_weights are rejected (wrong length).
        with pytest.raises(ValueError):
            fresh().continue_sampling(
                X_train=X, y_train=y, num_mcmc=6,
                general_params={"variable_weights": np.array([0.5, 0.5])},
            )

        # An unchangeable parameter is ignored with a warning.
        with pytest.warns(UserWarning):
            fresh().continue_sampling(
                X_train=X, y_train=y, num_mcmc=6,
                mean_forest_params={"num_trees": 42},
            )

        # keep_vars by name works on a DataFrame-fit model.
        Xdf = pd.DataFrame(X, columns=[f"v{i + 1}" for i in range(X.shape[1])])
        md = BARTModel()
        md.sample(X_train=Xdf, y_train=y, num_gfr=0, num_burnin=5, num_mcmc=6,
                  general_params={"random_seed": 3})
        md.continue_sampling(
            X_train=Xdf, y_train=y, num_mcmc=6,
            mean_forest_params={"keep_vars": ["v1", "v2"]},
        )
        assert md.num_samples == 12


class TestBCFContinuation:
    def test_continue_matches_one_shot(self):
        # The retained history (first N draws) is carried forward verbatim, so it is bit-identical
        # to a one-shot run of N+M. The continued draws are only *statistically* equivalent, not
        # bit-identical: continuation resumes the RNG stream but not the sampler's pre-drawn
        # leaf-normal cache, so the realized draws differ while targeting the same posterior.
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
        # History is retained verbatim (bit-identical).
        np.testing.assert_allclose(
            pred_cont[:, :N], pred_one_shot[:, :N], atol=1e-9, rtol=0
        )
        np.testing.assert_allclose(
            cont.global_var_samples[:N], one_shot.global_var_samples[:N], atol=1e-9, rtol=0
        )
        # The full posterior targets the same distribution: posterior-mean prediction agrees to
        # within Monte Carlo noise.
        np.testing.assert_allclose(
            pred_cont.mean(axis=1), pred_one_shot.mean(axis=1), atol=0.25, rtol=0
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
            general_params={"random_seed": -1},
        )
        # The newly drawn samples should differ between resume and override.
        assert not np.allclose(
            m.predict(X, Z, propensity=pi, terms="y_hat")[:, 10:],
            resumed.predict(X, Z, propensity=pi, terms="y_hat")[:, 10:],
        )

    def test_continuation_multivariate_treatment_supported(self):
        # Multivariate treatment (step 2) is now supported.
        def run():
            rng = np.random.default_rng(303)
            n, p = 300, 5
            X = rng.uniform(0, 1, (n, p))
            Z = rng.uniform(0, 1, (n, 2))  # multivariate (treatment_dim=2)
            y = 1.0 * X[:, 0] + Z[:, 0] * X[:, 1] - Z[:, 1] * X[:, 2] + 0.3 * rng.standard_normal(n)
            m = BCFModel()
            m.sample(X_train=X, Z_train=Z, y_train=y, num_gfr=0, num_burnin=5, num_mcmc=10,
                     general_params={"random_seed": 1})
            m.continue_sampling(X_train=X, Z_train=Z, y_train=y, num_mcmc=8)
            return m
        m = run()
        assert m.num_samples == 18
        assert m.tau_hat_train.shape == (300, 2, 18)
        np.testing.assert_allclose(run().y_hat_train, m.y_hat_train, atol=1e-10, rtol=0)

    def test_continuation_drops_stale_test_data_with_warning(self):
        # BCF continuation takes no test set, so a model fit WITH one has its stale test preds dropped.
        rng = np.random.default_rng(505)
        n, p = 200, 5
        X = rng.uniform(0, 1, (n, p))
        pi = 0.3 + 0.4 * X[:, 1]
        Z = rng.binomial(1, pi).astype(np.float64).reshape(-1, 1)
        y = 1.0 + 2.0 * X[:, 0] + 1.5 * X[:, 2] * Z[:, 0] + 0.5 * rng.standard_normal(n)
        X_test = rng.uniform(0, 1, (40, p))
        Z_test = rng.binomial(1, 0.3 + 0.4 * X_test[:, 1]).astype(np.float64).reshape(-1, 1)
        pi_test = (0.3 + 0.4 * X_test[:, 1]).reshape(-1, 1)
        m = BCFModel()
        m.sample(X_train=X, Z_train=Z, y_train=y, propensity_train=pi.reshape(-1, 1),
                 X_test=X_test, Z_test=Z_test, propensity_test=pi_test,
                 num_gfr=0, num_burnin=5, num_mcmc=10, general_params={"random_seed": 1})
        assert m.has_test
        with pytest.warns(UserWarning, match="test-set predictions are stale"):
            m.continue_sampling(X_train=X, Z_train=Z, y_train=y, propensity_train=pi.reshape(-1, 1), num_mcmc=8)
        assert not m.has_test
        assert m.y_hat_test is None
        assert m.num_samples == 18

    def test_continuation_variance_forest_supported(self):
        X, Z, y, pi = _make_bcf_data()
        m = BCFModel()
        m.sample(X_train=X, Z_train=Z, y_train=y, propensity_train=pi, num_gfr=0, num_burnin=5,
                 num_mcmc=10, variance_forest_params={"num_trees": 20},
                 general_params={"random_seed": 1})
        m.continue_sampling(X_train=X, Z_train=Z, y_train=y, propensity_train=pi, num_mcmc=8)
        assert m.num_samples == 18
        assert m.sigma2_x_train.shape == (X.shape[0], 18)

    def test_continuation_adaptive_coding_supported(self):
        X, Z, y, pi = _make_bcf_data()
        m = BCFModel()
        m.sample(X_train=X, Z_train=Z, y_train=y, propensity_train=pi, num_gfr=0, num_burnin=5,
                 num_mcmc=10, general_params={"adaptive_coding": True, "random_seed": 1})
        m.continue_sampling(X_train=X, Z_train=Z, y_train=y, propensity_train=pi, num_mcmc=8)
        assert m.num_samples == 18
        assert m.b0_samples.shape == (18,) and m.b1_samples.shape == (18,)

    def test_continuation_rfx_supported(self):
        rng = np.random.default_rng(404)
        n, p = 200, 5
        X = rng.uniform(0, 1, (n, p))
        pi = 0.3 + 0.4 * X[:, 1]
        Z = rng.binomial(1, pi).astype(np.float64).reshape(-1, 1)
        g = rng.integers(0, 3, n).astype(np.int32)
        rb = np.ones((n, 1))
        y = 1.0 + 2.0 * X[:, 0] + 1.5 * X[:, 2] * Z[:, 0] + (-2 * (g == 0) + 2 * (g == 1)) + 0.5 * rng.standard_normal(n)

        def run():
            m = BCFModel()
            m.sample(X_train=X, Z_train=Z, y_train=y, propensity_train=pi.reshape(-1, 1),
                     rfx_group_ids_train=g, rfx_basis_train=rb, num_gfr=0, num_burnin=5, num_mcmc=10,
                     general_params={"random_seed": 1})
            m.continue_sampling(X_train=X, Z_train=Z, y_train=y, propensity_train=pi.reshape(-1, 1),
                                rfx_group_ids_train=g, rfx_basis_train=rb, num_mcmc=8)
            return m

        m = run()
        assert m.num_samples == 18
        assert m.extract_random_effect_samples()["beta_samples"].shape[-1] == 18
        # rfx model must be re-supplied its group ids to continue
        m2 = BCFModel()
        m2.sample(X_train=X, Z_train=Z, y_train=y, propensity_train=pi.reshape(-1, 1),
                  rfx_group_ids_train=g, rfx_basis_train=rb, num_gfr=0, num_burnin=0, num_mcmc=5)
        with pytest.raises(ValueError):
            m2.continue_sampling(X_train=X, Z_train=Z, y_train=y, propensity_train=pi.reshape(-1, 1), num_mcmc=5)
        # Deterministic
        np.testing.assert_allclose(run().y_hat_train, m.y_hat_train, atol=1e-10, rtol=0)

    def test_continuation_probit_supported(self):
        def run():
            rng = np.random.default_rng(7)
            n, p = 300, 5
            X = rng.uniform(0, 1, (n, p))
            pi = 0.3 + 0.4 * X[:, 1]
            Z = rng.binomial(1, pi).astype(np.float64).reshape(-1, 1)
            lin = 1.0 * X[:, 0] + 0.8 * X[:, 2] * Z[:, 0]
            y = (lin + rng.normal(0, 1, n) > 0.9).astype(float)
            m = BCFModel()
            m.sample(X_train=X, Z_train=Z, y_train=y, propensity_train=pi.reshape(-1, 1),
                     num_gfr=0, num_burnin=5, num_mcmc=10,
                     general_params={"random_seed": 1, "sample_sigma2_global": False,
                                     "outcome_model": OutcomeModel(outcome="binary", link="probit")})
            m.continue_sampling(X_train=X, Z_train=Z, y_train=y, propensity_train=pi.reshape(-1, 1), num_mcmc=8)
            return m
        m = run()
        assert m.num_samples == 18
        np.testing.assert_allclose(run().y_hat_train, m.y_hat_train, atol=1e-10, rtol=0)

    def test_continuation_respecifies_split_variable_selection(self):
        X, Z, y, pi = _make_bcf_data()

        def fresh():
            m = BCFModel()
            m.sample(X_train=X, Z_train=Z, y_train=y, propensity_train=pi, num_gfr=0,
                     num_burnin=5, num_mcmc=6, general_params={"random_seed": 7})
            return m

        # keep_vars / drop_vars (per forest) + variable_weights are changeable: continuation proceeds.
        m1 = fresh()
        m1.continue_sampling(
            X_train=X, Z_train=Z, y_train=y, propensity_train=pi, num_mcmc=6,
            prognostic_forest_params={"keep_vars": [0, 1]},
            treatment_effect_forest_params={"drop_vars": [2, 3, 4]},
        )
        assert m1.num_samples == 12

        m2 = fresh()
        m2.continue_sampling(
            X_train=X, Z_train=Z, y_train=y, propensity_train=pi, num_mcmc=6,
            general_params={"variable_weights": np.array([0.5, 0.5, 0.0, 0.0, 0.0])},
        )
        assert m2.num_samples == 12

        # Malformed variable_weights are rejected (wrong length).
        with pytest.raises(ValueError):
            fresh().continue_sampling(
                X_train=X, Z_train=Z, y_train=y, propensity_train=pi, num_mcmc=6,
                general_params={"variable_weights": np.array([0.5, 0.5])},
            )

        # An unchangeable parameter is ignored with a warning.
        with pytest.warns(UserWarning):
            fresh().continue_sampling(
                X_train=X, Z_train=Z, y_train=y, propensity_train=pi, num_mcmc=6,
                prognostic_forest_params={"num_trees": 42},
            )

        # keep_vars by name works on a DataFrame-fit model.
        Xdf = pd.DataFrame(X, columns=[f"v{i + 1}" for i in range(X.shape[1])])
        md = BCFModel()
        md.sample(X_train=Xdf, Z_train=Z, y_train=y, propensity_train=pi, num_gfr=0,
                  num_burnin=5, num_mcmc=6, general_params={"random_seed": 7})
        md.continue_sampling(
            X_train=Xdf, Z_train=Z, y_train=y, propensity_train=pi, num_mcmc=6,
            treatment_effect_forest_params={"keep_vars": ["v1", "v2"]},
        )
        assert md.num_samples == 12
