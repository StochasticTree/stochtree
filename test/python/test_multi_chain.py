"""Tests for multi-chain BART and BCF sampling.

Covers sample-count correctness, GFR warm-start path, chain independence,
extract_parameter dimensions, and the num_gfr >= num_chains validation.
"""
import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from stochtree import BARTModel, BCFModel


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bart_data():
    rng = np.random.default_rng(42)
    n, p = 200, 5
    X = rng.uniform(0, 1, (n, p))
    y = 5 * X[:, 0] + rng.standard_normal(n)
    idx = np.arange(n)
    train_inds, test_inds = train_test_split(idx, test_size=0.2, random_state=42)
    return {
        "X_train": X[train_inds],
        "X_test": X[test_inds],
        "y_train": y[train_inds],
        "n_train": len(train_inds),
        "n_test": len(test_inds),
    }


@pytest.fixture(scope="module")
def bcf_data():
    rng = np.random.default_rng(42)
    n, p = 200, 5
    X = rng.uniform(0, 1, (n, p))
    pi_X = 0.25 + 0.5 * X[:, 0]
    Z = rng.binomial(1, pi_X, n).astype(float)
    y = 5 * X[:, 0] + 2 * X[:, 1] * Z + rng.standard_normal(n)
    idx = np.arange(n)
    train_inds, test_inds = train_test_split(idx, test_size=0.2, random_state=42)
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
    }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _bart(data, *, num_gfr, num_burnin, num_mcmc, num_chains, **kw):
    m = BARTModel()
    m.sample(
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_test=data["X_test"],
        num_gfr=num_gfr,
        num_burnin=num_burnin,
        num_mcmc=num_mcmc,
        general_params={"num_chains": num_chains, "num_threads": 1, **kw},
    )
    return m


def _bcf(data, *, num_gfr, num_burnin, num_mcmc, num_chains, **kw):
    m = BCFModel()
    m.sample(
        X_train=data["X_train"],
        Z_train=data["Z_train"],
        y_train=data["y_train"],
        propensity_train=data["pi_train"],
        X_test=data["X_test"],
        Z_test=data["Z_test"],
        propensity_test=data["pi_test"],
        num_gfr=num_gfr,
        num_burnin=num_burnin,
        num_mcmc=num_mcmc,
        general_params={"num_chains": num_chains, "num_threads": 1, **kw},
    )
    return m


# ---------------------------------------------------------------------------
# BARTModel multi-chain tests
# ---------------------------------------------------------------------------

class TestBARTMultiChain:
    NUM_MCMC = 10
    NUM_CHAINS = 3

    def test_sample_counts_no_gfr(self, bart_data):
        """Total kept samples = num_chains * num_mcmc when num_gfr=0."""
        n_chains, n_mcmc = self.NUM_CHAINS, self.NUM_MCMC
        m = _bart(bart_data, num_gfr=0, num_burnin=0, num_mcmc=n_mcmc, num_chains=n_chains)
        expected = n_chains * n_mcmc
        assert m.global_var_samples.shape == (expected,)
        assert m.y_hat_train.shape == (bart_data["n_train"], expected)
        assert m.y_hat_test.shape == (bart_data["n_test"], expected)

    def test_sample_counts_with_gfr(self, bart_data):
        """With GFR, total kept samples = num_chains * num_mcmc (GFR dropped by default)."""
        n_chains, n_mcmc, n_gfr = self.NUM_CHAINS, self.NUM_MCMC, 6
        m = _bart(bart_data, num_gfr=n_gfr, num_burnin=5, num_mcmc=n_mcmc, num_chains=n_chains)
        expected = n_chains * n_mcmc
        assert m.global_var_samples.shape == (expected,)
        assert m.y_hat_train.shape == (bart_data["n_train"], expected)

    def test_leaf_scale_sample_count(self, bart_data):
        """Leaf-scale samples also have num_chains * num_mcmc entries."""
        n_chains, n_mcmc = self.NUM_CHAINS, self.NUM_MCMC
        m = _bart(
            bart_data,
            num_gfr=0, num_burnin=0, num_mcmc=n_mcmc, num_chains=n_chains,
            sample_sigma2_global=False,
        )
        m2 = BARTModel()
        m2.sample(
            X_train=bart_data["X_train"],
            y_train=bart_data["y_train"],
            X_test=bart_data["X_test"],
            num_gfr=0,
            num_burnin=0,
            num_mcmc=n_mcmc,
            general_params={"num_chains": n_chains, "num_threads": 1, "sample_sigma2_global": False},
            mean_forest_params={"sample_sigma2_leaf": True},
        )
        assert m2.leaf_scale_samples.shape == (n_chains * n_mcmc,)

    def test_chain_independence_no_gfr(self, bart_data):
        """With 2 chains, sigma2 samples from different chains are not identical."""
        m = _bart(bart_data, num_gfr=0, num_burnin=0, num_mcmc=self.NUM_MCMC, num_chains=2)
        chain1 = m.global_var_samples[: self.NUM_MCMC]
        chain2 = m.global_var_samples[self.NUM_MCMC :]
        assert not np.allclose(chain1, chain2), (
            "Chains produced identical sigma2 samples; they should be independent."
        )

    def test_chain_independence_with_gfr(self, bart_data):
        """With GFR warm-start, different chains still produce distinct sigma2 samples."""
        n_gfr, n_mcmc = 4, self.NUM_MCMC
        m = _bart(bart_data, num_gfr=n_gfr, num_burnin=5, num_mcmc=n_mcmc, num_chains=2)
        chain1 = m.global_var_samples[:n_mcmc]
        chain2 = m.global_var_samples[n_mcmc:]
        assert not np.allclose(chain1, chain2)

    def test_extract_parameter_multi_chain(self, bart_data):
        """extract_parameter returns num_chains * num_mcmc samples."""
        n_chains, n_mcmc = self.NUM_CHAINS, self.NUM_MCMC
        m = _bart(bart_data, num_gfr=0, num_burnin=0, num_mcmc=n_mcmc, num_chains=n_chains)
        s2 = m.extract_parameter("sigma2_global")
        assert s2.shape == (n_chains * n_mcmc,)
        yht = m.extract_parameter("y_hat_train")
        assert yht.shape == (bart_data["n_train"], n_chains * n_mcmc)

    def test_predict_multi_chain(self, bart_data):
        """predict() returns correct shape and finite values for multi-chain BART."""
        n_chains, n_mcmc = self.NUM_CHAINS, self.NUM_MCMC
        m = _bart(bart_data, num_gfr=0, num_burnin=0, num_mcmc=n_mcmc, num_chains=n_chains)
        expected_cols = n_chains * n_mcmc
        n_test = bart_data["n_test"]
        result = m.predict(X=bart_data["X_test"], terms="y_hat")
        assert result.shape == (n_test, expected_cols)
        assert np.all(np.isfinite(result))

    def test_predict_multi_chain_gfr(self, bart_data):
        """predict() stays finite with GFR warm-start + multiple chains."""
        m = _bart(bart_data, num_gfr=6, num_burnin=5, num_mcmc=self.NUM_MCMC, num_chains=self.NUM_CHAINS)
        result = m.predict(X=bart_data["X_test"], terms="y_hat")
        assert result.shape == (bart_data["n_test"], self.NUM_CHAINS * self.NUM_MCMC)
        assert np.all(np.isfinite(result))

    def test_num_gfr_less_than_num_chains_raises(self, bart_data):
        """num_chains > num_gfr must raise a ValueError."""
        with pytest.raises((ValueError, Exception)):
            _bart(bart_data, num_gfr=2, num_burnin=0, num_mcmc=5, num_chains=4)

    def test_samples_finite_multi_chain_gfr(self, bart_data):
        """sigma2 samples are finite and positive with GFR warm-start + multiple chains."""
        m = _bart(bart_data, num_gfr=6, num_burnin=10, num_mcmc=self.NUM_MCMC, num_chains=3)
        assert np.all(np.isfinite(m.global_var_samples))
        assert np.all(m.global_var_samples > 0)


# ---------------------------------------------------------------------------
# BCFModel multi-chain tests
# ---------------------------------------------------------------------------

class TestBCFMultiChain:
    NUM_MCMC = 10
    NUM_CHAINS = 3
    NUM_GFR = 6

    def test_sample_counts_no_gfr(self, bcf_data):
        """Total kept samples = num_chains * num_mcmc when num_gfr=0."""
        n_chains, n_mcmc = self.NUM_CHAINS, self.NUM_MCMC
        m = _bcf(bcf_data, num_gfr=0, num_burnin=10, num_mcmc=n_mcmc, num_chains=n_chains)
        expected = n_chains * n_mcmc
        assert m.global_var_samples.shape == (expected,)
        assert m.tau_hat_train.shape == (bcf_data["n_train"], expected)
        assert m.mu_hat_train.shape == (bcf_data["n_train"], expected)
        assert m.tau_hat_test.shape == (bcf_data["n_test"], expected)

    def test_sample_counts_with_gfr(self, bcf_data):
        """GFR warm-start path: all parameter arrays have num_chains * num_mcmc entries."""
        n_chains = self.NUM_CHAINS
        n_mcmc = self.NUM_MCMC
        n_gfr = self.NUM_GFR
        m = _bcf(
            bcf_data,
            num_gfr=n_gfr,
            num_burnin=5,
            num_mcmc=n_mcmc,
            num_chains=n_chains,
            adaptive_coding=True,
        )
        expected = n_chains * n_mcmc
        assert m.global_var_samples.shape == (expected,)
        # BCF-specific samples
        assert m.tau_0_samples.shape == (1, expected)
        assert m.b0_samples.shape == (expected,)
        assert m.b1_samples.shape == (expected,)
        assert m.leaf_scale_mu_samples.shape == (expected,)
        # Predictions
        assert m.tau_hat_train.shape == (bcf_data["n_train"], expected)
        assert m.mu_hat_train.shape == (bcf_data["n_train"], expected)

    def test_chain_independence_no_gfr(self, bcf_data):
        """With 2 chains (no GFR), sigma2 samples differ across chains."""
        m = _bcf(bcf_data, num_gfr=0, num_burnin=10, num_mcmc=self.NUM_MCMC, num_chains=2)
        chain1 = m.global_var_samples[: self.NUM_MCMC]
        chain2 = m.global_var_samples[self.NUM_MCMC :]
        assert not np.allclose(chain1, chain2)

    def test_chain_independence_with_gfr(self, bcf_data):
        """With GFR warm-start, chains produce distinct sigma2 samples."""
        n_mcmc = self.NUM_MCMC
        m = _bcf(bcf_data, num_gfr=4, num_burnin=5, num_mcmc=n_mcmc, num_chains=2)
        chain1 = m.global_var_samples[:n_mcmc]
        chain2 = m.global_var_samples[n_mcmc:]
        assert not np.allclose(chain1, chain2)

    def test_samples_finite_gfr_multi_chain(self, bcf_data):
        """sigma2 samples remain finite with GFR warm-start + multiple chains.

        This exercises the tau_0 / adaptive-coding reset logic introduced to
        prevent residual blowup when transitioning between chains.
        """
        m = _bcf(
            bcf_data,
            num_gfr=self.NUM_GFR,
            num_burnin=20,
            num_mcmc=self.NUM_MCMC,
            num_chains=self.NUM_CHAINS,
            adaptive_coding=True,
        )
        assert np.all(np.isfinite(m.global_var_samples)), (
            "sigma2 samples contain non-finite values; possible chain-transition blowup."
        )
        assert np.all(m.global_var_samples > 0)
        assert np.all(np.isfinite(m.tau_0_samples))
        assert np.all(np.isfinite(m.b0_samples))
        assert np.all(np.isfinite(m.b1_samples))

    def test_extract_parameter_multi_chain(self, bcf_data):
        """extract_parameter returns num_chains * num_mcmc samples for BCF."""
        n_chains, n_mcmc = self.NUM_CHAINS, self.NUM_MCMC
        m = _bcf(bcf_data, num_gfr=0, num_burnin=10, num_mcmc=n_mcmc, num_chains=n_chains)
        expected = n_chains * n_mcmc
        s2 = m.extract_parameter("sigma2_global")
        assert s2.shape == (expected,)
        cate = m.extract_parameter("cate_test")
        assert cate.shape == (bcf_data["n_test"], expected)
        prog = m.extract_parameter("prognostic_function_test")
        assert prog.shape == (bcf_data["n_test"], expected)

    def test_predict_terms_multi_chain_no_gfr(self, bcf_data):
        """predict() returns correct shape and finite values for each forest term (no GFR)."""
        n_chains, n_mcmc = self.NUM_CHAINS, self.NUM_MCMC
        m = _bcf(bcf_data, num_gfr=0, num_burnin=10, num_mcmc=n_mcmc, num_chains=n_chains)
        expected_cols = n_chains * n_mcmc
        n_test = bcf_data["n_test"]
        kw = dict(X=bcf_data["X_test"], Z=bcf_data["Z_test"], propensity=bcf_data["pi_test"])
        for term in ["y_hat", "cate", "prognostic_function", "mu", "tau"]:
            result = m.predict(**kw, terms=term)
            assert result.shape == (n_test, expected_cols), f"shape mismatch for term={term!r}"
            assert np.all(np.isfinite(result)), f"non-finite values for term={term!r}"

    def test_predict_terms_multi_chain_with_gfr(self, bcf_data):
        """predict() returns correct shape and finite values for each forest term (GFR path)."""
        n_chains, n_mcmc = self.NUM_CHAINS, self.NUM_MCMC
        m = _bcf(bcf_data, num_gfr=self.NUM_GFR, num_burnin=5, num_mcmc=n_mcmc, num_chains=n_chains)
        expected_cols = n_chains * n_mcmc
        n_test = bcf_data["n_test"]
        kw = dict(X=bcf_data["X_test"], Z=bcf_data["Z_test"], propensity=bcf_data["pi_test"])
        for term in ["y_hat", "cate", "prognostic_function", "mu", "tau"]:
            result = m.predict(**kw, terms=term)
            assert result.shape == (n_test, expected_cols), f"shape mismatch for term={term!r}"
            assert np.all(np.isfinite(result)), f"non-finite values for term={term!r}"

    def test_predict_variance_forest_multi_chain(self, bcf_data):
        """predict() returns correct shape and positive values for variance forest term."""
        n_chains, n_mcmc = self.NUM_CHAINS, self.NUM_MCMC
        m = BCFModel()
        m.sample(
            X_train=bcf_data["X_train"],
            Z_train=bcf_data["Z_train"],
            y_train=bcf_data["y_train"],
            propensity_train=bcf_data["pi_train"],
            X_test=bcf_data["X_test"],
            Z_test=bcf_data["Z_test"],
            propensity_test=bcf_data["pi_test"],
            num_gfr=0,
            num_burnin=10,
            num_mcmc=n_mcmc,
            general_params={"num_chains": n_chains, "num_threads": 1},
            variance_forest_params={"num_trees": 10},
        )
        result = m.predict(
            X=bcf_data["X_test"],
            Z=bcf_data["Z_test"],
            propensity=bcf_data["pi_test"],
            terms="variance_forest",
        )
        assert result.shape == (bcf_data["n_test"], n_chains * n_mcmc)
        assert np.all(np.isfinite(result))
        assert np.all(result > 0)

    def test_num_gfr_less_than_num_chains_raises(self, bcf_data):
        """num_chains > num_gfr must raise an error."""
        with pytest.raises((ValueError, Exception)):
            _bcf(bcf_data, num_gfr=2, num_burnin=0, num_mcmc=5, num_chains=4)

    def test_serialization_round_trip_multi_chain(self, bcf_data):
        """Serialize and reload a multi-chain BCF model; predictions must match."""
        import json
        n_chains, n_mcmc, n_gfr = 2, self.NUM_MCMC, 4
        m = _bcf(bcf_data, num_gfr=n_gfr, num_burnin=5, num_mcmc=n_mcmc, num_chains=n_chains)
        json_str = m.to_json()

        m2 = BCFModel()
        m2.from_json(json_str)

        pred_orig = m.predict(
            X=bcf_data["X_test"],
            Z=bcf_data["Z_test"],
            propensity=bcf_data["pi_test"],
            terms="cate",
        )
        pred_rt = m2.predict(
            X=bcf_data["X_test"],
            Z=bcf_data["Z_test"],
            propensity=bcf_data["pi_test"],
            terms="cate",
        )
        assert pred_orig.shape == pred_rt.shape
        np.testing.assert_allclose(pred_orig, pred_rt)
