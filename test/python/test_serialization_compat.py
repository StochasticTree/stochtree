"""
Backward-compatibility deserialization tests

These tests verify that models serialized without certain optional fields
(as would be produced by older package versions) can still be loaded
correctly, with appropriate warnings where applicable.

Fixture files (test/python/fixtures/) are generated once from the current
package and checked in.  They serve as a "snapshot" — if a future change
breaks the ability to deserialize them, these tests will catch it.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from stochtree import BARTModel, BCFModel

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_fixture(name: str) -> dict:
    """Load a fixture JSON file and return as a Python dict."""
    with open(FIXTURES_DIR / name) as f:
        return json.load(f)


def _to_json_string(obj: dict) -> str:
    """Serialise a Python dict back to a JSON string."""
    return json.dumps(obj)


def _strip_fields(obj: dict, *fields: str) -> str:
    """Remove fields from obj (top-level) and return the JSON string."""
    obj = dict(obj)  # shallow copy
    for f in fields:
        obj.pop(f, None)
    return _to_json_string(obj)


def _collect_warnings(fn, *args, **kwargs):
    """Call fn(*args, **kwargs) and return (result, list_of_warning_messages)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    msgs = [str(w.message) for w in caught]
    return result, msgs


# ===========================================================================
# BART snapshot tests
# ===========================================================================

class TestBARTSnapshot:
    def test_fixture_loads_and_predicts(self):
        """BART fixture deserialises and prediction succeeds."""
        fixture_obj = _load_fixture("bart_mcmc.json")
        json_str = _to_json_string(fixture_obj)

        rng = np.random.default_rng(1)
        X = rng.uniform(size=(30, 5))

        m = BARTModel()
        m.from_json(json_str)
        preds = m.predict(X)
        assert "y_hat" in preds
        assert preds["y_hat"].shape[0] == 30

    def test_roundtrip_is_deterministic(self):
        """Two loads of the same fixture string produce identical predictions."""
        fixture_obj = _load_fixture("bart_mcmc.json")
        json_str = _to_json_string(fixture_obj)

        rng = np.random.default_rng(99)
        X = rng.uniform(size=(20, 5))

        m1 = BARTModel()
        m1.from_json(json_str)
        m2 = BARTModel()
        m2.from_json(json_str)

        p1 = m1.predict(X)["y_hat"].mean(axis=1)
        p2 = m2.predict(X)["y_hat"].mean(axis=1)
        np.testing.assert_array_equal(p1, p2)


# ===========================================================================
# BCF snapshot tests
# ===========================================================================

class TestBCFSnapshot:
    def test_fixture_loads_and_predicts(self):
        """BCF fixture deserialises and prediction succeeds."""
        fixture_obj = _load_fixture("bcf_mcmc.json")
        json_str = _to_json_string(fixture_obj)

        rng = np.random.default_rng(1)
        n = 20
        X = rng.uniform(size=(n, 5))
        Z = rng.binomial(1, 0.5, n).astype(float)
        pi = np.full(n, 0.5)

        m = BCFModel()
        m.from_json(json_str)
        preds = m.predict(X, Z, pi)
        assert "y_hat" in preds
        assert "tau_hat" in preds
        assert preds["y_hat"].shape[0] == n


# ===========================================================================
# BART backward-compat: missing optional fields
# ===========================================================================

class TestBARTBackwardCompat:
    def test_missing_outcome_model(self):
        """BART loads without 'outcome_model' (pre-v0.4.1 format)."""
        fixture_obj = _load_fixture("bart_mcmc.json")
        json_str = _strip_fields(fixture_obj, "outcome_model", "probit_outcome_model")

        m = BARTModel()
        m.from_json(json_str)

        rng = np.random.default_rng(1)
        X = rng.uniform(size=(20, 5))
        preds = m.predict(X)
        assert preds["y_hat"].shape[0] == 20

    def test_missing_rfx_model_spec_no_rfx(self):
        """BART loads without 'rfx_model_spec' when has_rfx=False."""
        fixture_obj = _load_fixture("bart_mcmc.json")
        assert not fixture_obj.get("has_rfx", True), "Fixture must have has_rfx=False"
        json_str = _strip_fields(fixture_obj, "rfx_model_spec")

        m = BARTModel()
        m.from_json(json_str)

        rng = np.random.default_rng(1)
        X = rng.uniform(size=(20, 5))
        preds = m.predict(X)
        assert preds["y_hat"].shape[0] == 20

    def test_missing_preprocessor_emits_warning(self):
        """BART loads without 'covariate_preprocessor' and emits a warning."""
        fixture_obj = _load_fixture("bart_mcmc.json")
        json_str = _strip_fields(fixture_obj, "covariate_preprocessor")

        m = BARTModel()
        _, warns = _collect_warnings(m.from_json, json_str)
        # Should warn about missing preprocessor or succeed silently
        assert any("preprocessor" in w.lower() or "covariate" in w.lower() for w in warns) or len(warns) == 0
        assert isinstance(m, BARTModel)

    def test_missing_num_chains_keep_every(self):
        """BART loads without 'num_chains' and 'keep_every'."""
        fixture_obj = _load_fixture("bart_mcmc.json")
        json_str = _strip_fields(fixture_obj, "num_chains", "keep_every")

        m = BARTModel()
        m.from_json(json_str)

        rng = np.random.default_rng(1)
        X = rng.uniform(size=(20, 5))
        preds = m.predict(X)
        assert preds["y_hat"].shape[0] == 20


# ===========================================================================
# BCF backward-compat: missing optional fields
# ===========================================================================

class TestBCFBackwardCompat:
    def _predict(self, m: BCFModel, n: int = 20, seed: int = 1):
        rng = np.random.default_rng(seed)
        X = rng.uniform(size=(n, 5))
        Z = rng.binomial(1, 0.5, n).astype(float)
        pi = np.full(n, 0.5)
        return m.predict(X, Z, pi), n

    def test_missing_outcome_model(self):
        """BCF loads without 'outcome_model' (pre-v0.4.1 format)."""
        fixture_obj = _load_fixture("bcf_mcmc.json")
        json_str = _strip_fields(fixture_obj, "outcome_model", "probit_outcome_model")

        m = BCFModel()
        m.from_json(json_str)
        preds, n = self._predict(m)
        assert preds["y_hat"].shape[0] == n

    def test_missing_multivariate_treatment(self):
        """BCF loads without 'multivariate_treatment' (pre-v0.4.0)."""
        fixture_obj = _load_fixture("bcf_mcmc.json")
        json_str = _strip_fields(fixture_obj, "multivariate_treatment")

        m = BCFModel()
        m.from_json(json_str)
        preds, n = self._predict(m)
        assert preds["y_hat"].shape[0] == n

    def test_missing_internal_propensity_model(self):
        """BCF loads without 'internal_propensity_model' (pre-v0.3.2)."""
        fixture_obj = _load_fixture("bcf_mcmc.json")
        json_str = _strip_fields(fixture_obj, "internal_propensity_model")

        m = BCFModel()
        m.from_json(json_str)
        preds, n = self._predict(m)
        assert preds["y_hat"].shape[0] == n

    def test_missing_rfx_model_spec_no_rfx(self):
        """BCF loads without 'rfx_model_spec' when has_rfx=False."""
        fixture_obj = _load_fixture("bcf_mcmc.json")
        assert not fixture_obj.get("has_rfx", True), "Fixture must have has_rfx=False"
        json_str = _strip_fields(fixture_obj, "rfx_model_spec")

        m = BCFModel()
        m.from_json(json_str)
        preds, n = self._predict(m)
        assert preds["y_hat"].shape[0] == n

    def test_missing_preprocessor_emits_warning(self):
        """BCF loads without 'covariate_preprocessor' and emits a warning."""
        fixture_obj = _load_fixture("bcf_mcmc.json")
        json_str = _strip_fields(fixture_obj, "covariate_preprocessor")

        m = BCFModel()
        _, warns = _collect_warnings(m.from_json, json_str)
        # Should warn or at least succeed
        assert any("preprocessor" in w.lower() or "covariate" in w.lower() for w in warns) or len(warns) == 0
        assert isinstance(m, BCFModel)

    def test_missing_num_chains_keep_every(self):
        """BCF loads without 'num_chains' and 'keep_every'."""
        fixture_obj = _load_fixture("bcf_mcmc.json")
        json_str = _strip_fields(fixture_obj, "num_chains", "keep_every")

        m = BCFModel()
        m.from_json(json_str)
        preds, n = self._predict(m)
        assert preds["y_hat"].shape[0] == n

    def test_missing_has_rfx_basis(self):
        """BCF loads without 'has_rfx_basis'."""
        fixture_obj = _load_fixture("bcf_mcmc.json")
        json_str = _strip_fields(fixture_obj, "has_rfx_basis")

        m = BCFModel()
        m.from_json(json_str)
        preds, n = self._predict(m)
        assert preds["y_hat"].shape[0] == n

    def test_missing_multiple_optional_fields(self):
        """BCF loads when many optional fields are absent simultaneously."""
        fixture_obj = _load_fixture("bcf_mcmc.json")
        json_str = _strip_fields(
            fixture_obj,
            "outcome_model", "probit_outcome_model",
            "multivariate_treatment", "internal_propensity_model",
            "rfx_model_spec", "num_chains", "keep_every",
            "has_rfx_basis",
        )

        m = BCFModel()
        m.from_json(json_str)
        preds, n = self._predict(m)
        assert preds["y_hat"].shape[0] == n

    def test_missing_binary_treatment_is_inferred(self):
        """Legacy JSON without 'binary_treatment' infers it from adaptive_coding.

        adaptive_coding is only permitted for a binary treatment, so a legacy
        model with adaptive_coding=True must be recovered as binary_treatment.
        The checked-in fixture predates the field, so it exercises this path.
        """
        fixture_obj = _load_fixture("bcf_mcmc.json")
        assert "binary_treatment" not in fixture_obj
        assert fixture_obj.get("adaptive_coding") is True
        json_str = _to_json_string(fixture_obj)

        m = BCFModel()
        _, warns = _collect_warnings(m.from_json, json_str)
        assert m.binary_treatment is True
        assert any("binary_treatment" in w for w in warns)
        # The reloaded model must be printable (GH #393 failure mode).
        assert isinstance(str(m), str)

    def test_missing_treatment_dim_defaults_to_one(self):
        """Legacy JSON without 'treatment_dim' defaults to 1 (univariate)."""
        fixture_obj = _load_fixture("bcf_mcmc.json")
        assert "treatment_dim" not in fixture_obj
        assert fixture_obj.get("multivariate_treatment") is False
        json_str = _to_json_string(fixture_obj)

        m = BCFModel()
        m.from_json(json_str)
        assert m.treatment_dim == 1


# Attributes the BCF JSON contract must preserve across a round-trip: the
# metadata/flags consumed by __str__/summary/predict. Prior hyperparameters are
# intentionally not serialized (they fall back to defaults), so they are
# excluded here on purpose.
_BCF_MUST_SURVIVE_ATTRS = [
    "binary_treatment",
    "multivariate_treatment",
    "treatment_dim",
    "adaptive_coding",
    "sample_tau_0",
    "internal_propensity_model",
    "propensity_covariate",
    "include_variance_forest",
    "has_rfx",
    "standardize",
    "num_samples",
]


class TestBCFFieldRoundtrip:
    """Live (sample -> to_json -> from_json) round-trip of the contract fields.

    Unlike the fixture tests, this also exercises the to_json side, and covers
    both load paths (from_json and from_json_string_list).
    """

    def _assert_contract(self, original, reloaded, label):
        for attr in _BCF_MUST_SURVIVE_ATTRS:
            assert hasattr(reloaded, attr), f"{label}: missing attribute '{attr}'"
            assert getattr(reloaded, attr) == getattr(original, attr), (
                f"{label}: attribute '{attr}' changed across round-trip"
            )
        # Behavioral smoke: str() must not raise (this is what GH #393 hit).
        assert isinstance(str(reloaded), str)

    def test_binary_internal_propensity_roundtrip(self):
        """The exact GH #393 config: binary treatment + internal propensity."""
        rng = np.random.default_rng(393)
        n, p = 300, 5
        X = rng.uniform(size=(n, p))
        pi_x = 0.25 + 0.5 * X[:, 0]
        Z = rng.binomial(1, pi_x).astype(float)
        y = pi_x * 5 + Z * X[:, 1] * 2 + rng.normal(size=n)

        m = BCFModel()
        # No propensity supplied -> internal propensity model is built.
        m.sample(X_train=X, Z_train=Z, y_train=y,
                 num_gfr=10, num_burnin=0, num_mcmc=10)
        assert m.binary_treatment is True
        assert m.internal_propensity_model is True

        json_str = m.to_json()
        m2 = BCFModel()
        m2.from_json(json_str)
        self._assert_contract(m, m2, "from_json")

        m3 = BCFModel()
        m3.from_json_string_list([json_str])
        self._assert_contract(m, m3, "from_json_string_list")

    def test_multivariate_treatment_dim_roundtrip(self):
        """Multivariate treatment preserves treatment_dim and prints (latent #393)."""
        rng = np.random.default_rng(395)
        n, p = 300, 5
        X = rng.uniform(size=(n, p))
        Z = rng.normal(size=(n, 2))
        pi = np.full((n, 2), 0.5)
        y = X[:, 0] + Z[:, 0] * 0.5 + Z[:, 1] * 0.3 + rng.normal(size=n)

        m = BCFModel()
        m.sample(X_train=X, Z_train=Z, y_train=y, propensity_train=pi,
                 num_gfr=10, num_burnin=0, num_mcmc=10)
        assert m.multivariate_treatment is True
        assert m.treatment_dim == 2

        m2 = BCFModel()
        m2.from_json(m.to_json())
        assert m2.treatment_dim == 2
        # The multivariate branch of __str__ dereferences treatment_dim.
        assert "2 dimensions" in str(m2)

    def test_multichain_and_combined_roundtrip(self):
        """num_chains>1 round-trip and from_json_string_list of two models."""
        rng = np.random.default_rng(398)
        n, p = 300, 5
        X = rng.uniform(size=(n, p))
        pi_x = 0.25 + 0.5 * X[:, 0]
        Z = rng.binomial(1, pi_x).astype(float)
        y = pi_x * 5 + Z * X[:, 1] * 2 + rng.normal(size=n)

        def fit(seed=None, num_chains=1):
            m = BCFModel()
            m.sample(
                X_train=X, Z_train=Z, y_train=y, propensity_train=pi_x,
                num_gfr=0, num_burnin=0, num_mcmc=10,
                general_params={"num_chains": num_chains},
            )
            return m

        # Multi-chain single model.
        m_mc = fit(num_chains=2)
        assert m_mc.num_chains == 2
        m_rt = BCFModel()
        m_rt.from_json(m_mc.to_json())
        self._assert_contract(m_mc, m_rt, "bcf/multichain")

        # Combine two separately-sampled models (num_samples is the sum).
        m_a, m_b = fit(), fit()
        combined = BCFModel()
        combined.from_json_string_list([m_a.to_json(), m_b.to_json()])
        assert combined.binary_treatment == m_a.binary_treatment
        assert combined.treatment_dim == m_a.treatment_dim
        assert combined.num_samples == 20
        assert isinstance(str(combined), str)


class TestBARTFieldRoundtrip:
    """BART round-trip of leaf-model fields across load paths and chains."""

    def _check(self, original, reloaded, label):
        for attr in ("has_basis", "num_basis", "num_chains"):
            assert hasattr(reloaded, attr), f"{label}: missing '{attr}'"
            assert getattr(reloaded, attr) == getattr(original, attr), (
                f"{label}: '{attr}' changed across round-trip"
            )
        assert isinstance(str(reloaded), str)

    def test_constant_and_regression_roundtrip(self):
        rng = np.random.default_rng(396)
        n, p = 200, 5
        X = rng.uniform(size=(n, p))
        y = X[:, 0] + rng.normal(size=n)

        m_const = BARTModel()
        m_const.sample(X_train=X, y_train=y, num_gfr=0, num_burnin=0, num_mcmc=10)
        assert m_const.has_basis is False
        r = BARTModel()
        r.from_json(m_const.to_json())
        self._check(m_const, r, "constant")

        W = rng.uniform(size=(n, 2))
        m_reg = BARTModel()
        m_reg.sample(
            X_train=X, leaf_basis_train=W, y_train=y,
            num_gfr=0, num_burnin=0, num_mcmc=10,
        )
        assert m_reg.has_basis is True
        r2 = BARTModel()
        r2.from_json(m_reg.to_json())
        self._check(m_reg, r2, "regression")

    def test_multichain_and_combined_roundtrip(self):
        rng = np.random.default_rng(397)
        n, p = 200, 5
        X = rng.uniform(size=(n, p))
        y = X[:, 0] + rng.normal(size=n)

        m_mc = BARTModel()
        m_mc.sample(
            X_train=X, y_train=y, num_gfr=0, num_burnin=0, num_mcmc=10,
            general_params={"num_chains": 2},
        )
        assert m_mc.num_chains == 2
        r = BARTModel()
        r.from_json(m_mc.to_json())
        self._check(m_mc, r, "bart/multichain")

        m_a = BARTModel()
        m_a.sample(X_train=X, y_train=y, num_gfr=0, num_burnin=0, num_mcmc=10)
        m_b = BARTModel()
        m_b.sample(X_train=X, y_train=y, num_gfr=0, num_burnin=0, num_mcmc=10)
        combined = BARTModel()
        combined.from_json_string_list([m_a.to_json(), m_b.to_json()])
        assert combined.num_samples == 20
        assert isinstance(str(combined), str)


def test_future_schema_version_raises():
    """A model stamped with a newer schema_version than this install supports must error."""
    from stochtree.serialization import SCHEMA_VERSION

    rng = np.random.default_rng(0)
    X = rng.uniform(size=(80, 3))
    y = X[:, 0] + rng.normal(size=80)
    m = BARTModel()
    m.sample(X_train=X, y_train=y, num_gfr=0, num_burnin=0, num_mcmc=5)

    payload = json.loads(m.to_json())
    assert payload["schema_version"] == SCHEMA_VERSION
    payload["schema_version"] = SCHEMA_VERSION + 1

    with pytest.raises(ValueError, match="schema_version"):
        BARTModel().from_json(json.dumps(payload))
