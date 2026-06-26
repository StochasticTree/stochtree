import numpy as np

from stochtree import BARTModel, BCFModel
from stochtree_cpp import BARTSamplesCpp, BCFSamplesCpp


class TestBARTSamplesCpp:
    """Isolated tests for the thin single-owner wrapper around StochTree::BARTSamples.

    These exercise the wrapper plumbing (FromJson -> accessors / param-vector marshalling /
    materialize-on-demand) independently of the model re-point, so the wrapper can be validated
    before bart.py routes through it.
    """

    def _fit(self, seed=42, n=100, p=4):
        rng = np.random.default_rng(seed)
        X = rng.uniform(0, 1, (n, p))
        y = X[:, 0] * 2 + rng.normal(0, 0.5, n)
        model = BARTModel()
        model.sample(
            X_train=X, y_train=y, num_gfr=0, num_burnin=0, num_mcmc=10,
            general_params={"random_seed": seed},
        )
        return model

    def test_from_json_accessors_and_materialize(self):
        model = self._fit()
        samples = BARTSamplesCpp.from_json_string(model.to_json())

        # Scalars / counts match the model
        assert samples.num_samples() == model.num_samples
        assert np.isclose(samples.y_bar(), model.y_bar)
        assert np.isclose(samples.y_std(), model.y_std)

        # Parameter traces round-trip to the model's arrays (one entry per draw)
        if model.sample_sigma2_global:
            gv = samples.global_var_samples()
            assert gv.shape[0] == model.num_samples
            np.testing.assert_allclose(gv, model.global_var_samples)
        if model.sample_sigma2_leaf:
            np.testing.assert_allclose(
                samples.leaf_scale_samples(), model.leaf_scale_samples
            )

        # Materialized mean forest is a faithful deep copy (byte-identical serialization)
        assert samples.has_mean_forest()
        fc = samples.materialize_mean_forest()
        assert fc.NumSamples() == model.num_samples
        assert (
            fc.DumpJsonString()
            == model.forest_container_mean.forest_container_cpp.DumpJsonString()
        )

        # This model has no variance forest
        assert not samples.has_variance_forest()
        assert samples.materialize_variance_forest() is None

    def test_to_json_string_round_trips(self):
        # The wrapper's own ToJson -> FromJson preserves the samples subtree.
        model = self._fit()
        samples = BARTSamplesCpp.from_json_string(model.to_json())
        restored = BARTSamplesCpp.from_json_string(samples.to_json_string())
        assert restored.num_samples() == samples.num_samples()
        np.testing.assert_allclose(
            restored.global_var_samples(), samples.global_var_samples()
        )
        assert (
            restored.materialize_mean_forest().DumpJsonString()
            == samples.materialize_mean_forest().DumpJsonString()
        )

    def test_merge_appends_draws(self):
        # Build two wrappers from the SAME model's JSON so they share standardization (Merge guards
        # against mismatched y_bar/y_std). Merging concatenates their draws.
        js = self._fit(seed=1).to_json()
        a = BARTSamplesCpp.from_json_string(js)
        b = BARTSamplesCpp.from_json_string(js)
        n_a, n_b = a.num_samples(), b.num_samples()
        gv_a = a.global_var_samples().copy()
        gv_b = b.global_var_samples().copy()

        a.merge(b)

        assert a.num_samples() == n_a + n_b
        assert a.materialize_mean_forest().NumSamples() == n_a + n_b
        np.testing.assert_allclose(a.global_var_samples(), np.concatenate([gv_a, gv_b]))


class TestBCFSamplesCpp:
    """Isolated tests for the single-owner wrapper around StochTree::BCFSamples."""

    def _fit(self, seed=42, n=120, p=4):
        rng = np.random.default_rng(seed)
        X = rng.uniform(0, 1, (n, p))
        pi = 0.3 + 0.4 * X[:, 1]
        Z = rng.binomial(1, pi).astype(np.float64).reshape(-1, 1)
        mu = 1.0 + 2.0 * X[:, 0]
        tau = 1.5 * X[:, 2]
        y = mu + tau * Z[:, 0] + 0.5 * rng.standard_normal(n)
        model = BCFModel()
        model.sample(
            X_train=X, Z_train=Z, y_train=y, propensity_train=pi.reshape(-1, 1),
            num_gfr=0, num_burnin=0, num_mcmc=10,
            general_params={"random_seed": seed},
        )
        return model

    def test_from_json_accessors_and_materialize(self):
        model = self._fit()
        samples = BCFSamplesCpp.from_json_string(model.to_json())

        assert samples.num_samples() == model.num_samples
        assert samples.treatment_dim() == model.treatment_dim
        assert np.isclose(samples.y_bar(), model.y_bar)
        assert np.isclose(samples.y_std(), model.y_std)

        if model.sample_sigma2_global:
            np.testing.assert_allclose(samples.global_var_samples(), model.global_var_samples)
        if model.sample_sigma2_leaf_mu:
            np.testing.assert_allclose(
                samples.leaf_scale_mu_samples(), model.leaf_scale_mu_samples
            )

        # Both prognostic and treatment forests materialize as faithful deep copies
        assert samples.has_mu_forest()
        assert samples.has_tau_forest()
        assert (
            samples.materialize_mu_forest().DumpJsonString()
            == model.forest_container_mu.forest_container_cpp.DumpJsonString()
        )
        assert (
            samples.materialize_tau_forest().DumpJsonString()
            == model.forest_container_tau.forest_container_cpp.DumpJsonString()
        )
        assert not samples.has_variance_forest()
        assert samples.materialize_variance_forest() is None

    def test_merge_appends_draws(self):
        js = self._fit(seed=1).to_json()
        a = BCFSamplesCpp.from_json_string(js)
        b = BCFSamplesCpp.from_json_string(js)
        n_a, n_b = a.num_samples(), b.num_samples()

        a.merge(b)

        assert a.num_samples() == n_a + n_b
        assert a.materialize_mu_forest().NumSamples() == n_a + n_b
        assert a.materialize_tau_forest().NumSamples() == n_a + n_b
