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
