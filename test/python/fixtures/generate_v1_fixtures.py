"""Generate the v1 (schema_version=1) golden-fixture matrix for the unified envelope.

Run once to (re)mint the checked-in v1 fixtures:

    python test/python/fixtures/generate_v1_fixtures.py

The matrix is {bart, bcf} x {numeric, categorical} x {no-rfx, rfx} = 8 models,
covering the format-relevant axes: forest naming (BART vs BCF), the covariate
preprocessor (identity vs categorical encoding -> the cross-platform-portable
axis), and random effects (which add the random_effects subfolder).

Fixtures only need to be structurally complete and loadable, not statistically
meaningful, so the ensembles are tiny (5 trees, 2 MCMC) to keep the checked-in
files small (a few KB each). The legacy v0 fixtures (bart_mcmc.json /
bcf_mcmc.json) are kept separately and exercise the v0 -> v1 migration path.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stochtree import BARTModel, BCFModel

OUT = Path(__file__).parent

_NUM_MCMC = 2
_SMALL = {"num_trees": 5}
N = 200
P = 4  # numeric covariate columns (categorical models add one "cat" column)

# Superseded ad-hoc fixtures from the first cut (replaced by the matrix below).
_SUPERSEDED = ("bart_mcmc_v1.json", "bart_rfx_v1.json", "bcf_mcmc_v1.json")


def _make_covariates(rng, n, categorical):
    """Return (X, x0) where x0 is a numeric column used to build the outcome."""
    X_num = rng.uniform(size=(n, P))
    if not categorical:
        return X_num, X_num[:, 0]
    X = pd.DataFrame(X_num, columns=[f"x{i}" for i in range(P)])
    X["cat"] = pd.Categorical(rng.choice(["a", "b", "c"], size=n))
    return X, X_num[:, 0]


def _bart(seed, *, rfx, categorical):
    rng = np.random.default_rng(seed)
    X, x0 = _make_covariates(rng, N, categorical)
    y = x0 + rng.normal(scale=0.5, size=N)
    kw = dict(num_gfr=0, num_burnin=0, num_mcmc=_NUM_MCMC, mean_forest_params=_SMALL)
    if rfx:
        g = rng.integers(0, 3, size=N)
        y = y + g
        kw["rfx_group_ids_train"] = g
        kw["rfx_basis_train"] = np.ones((N, 1))
    m = BARTModel()
    m.sample(X_train=X, y_train=y, **kw)
    return m.to_json()


def _bcf(seed, *, rfx, categorical):
    rng = np.random.default_rng(seed)
    X, x0 = _make_covariates(rng, N, categorical)
    pi = 0.25 + 0.5 * x0
    Z = rng.binomial(1, pi).astype(float)
    y = pi * 5 + Z * x0 * 2 + rng.normal(size=N)
    kw = dict(
        num_gfr=0,
        num_burnin=0,
        num_mcmc=_NUM_MCMC,
        prognostic_forest_params=_SMALL,
        treatment_effect_forest_params=_SMALL,
    )
    if rfx:
        g = rng.integers(0, 3, size=N)
        y = y + g
        kw["rfx_group_ids_train"] = g
        kw["rfx_basis_train"] = np.ones((N, 1))
    m = BCFModel()
    m.sample(X_train=X, Z_train=Z, y_train=y, propensity_train=pi, **kw)
    return m.to_json()


def main():
    seed = 110
    for model, fn in (("bart", _bart), ("bcf", _bcf)):
        for categorical in (False, True):
            for rfx in (False, True):
                kind = "categorical" if categorical else "numeric"
                suffix = "_rfx" if rfx else ""
                name = f"{model}_{kind}{suffix}_v1.json"
                js = fn(seed, rfx=rfx, categorical=categorical)
                seed += 1
                payload = json.loads(js)
                assert payload["schema_version"] == 1, f"{name}: expected schema_version=1"
                (OUT / name).write_text(js)
                print(f"wrote {name} (schema_version={payload['schema_version']})")

    for old in _SUPERSEDED:
        p = OUT / old
        if p.exists():
            p.unlink()
            print(f"removed superseded {old}")


if __name__ == "__main__":
    main()
