#!/usr/bin/env python3
"""
bart_dispatch_benchmark.py

Compare C++ fast-path vs Python slow-path dispatch times for every supported
BART model variant.

Usage:
    python tools/perf/bart_dispatch_benchmark.py
    python tools/perf/bart_dispatch_benchmark.py identity probit rfx
    python tools/perf/bart_dispatch_benchmark.py --list

Available scenarios:
    identity          Continuous outcome, identity link
    identity_rfx      Continuous + intercept-only random effects
    identity_basis    Continuous + univariate leaf basis (leaf regression)
    identity_basis_rfx  Continuous + leaf basis + RFX
    variance_forest   Continuous + heteroskedastic variance forest
    probit            Binary outcome, probit link
    cloglog_binary    Binary outcome, cloglog link
    cloglog_ordinal   Ordinal outcome (K=3), cloglog link
    cloglog_rfx       Ordinal outcome (K=3), cloglog + RFX

Set STOCHTREE_USE_CPP_DISPATCH=0 in the environment to force the slow path
globally (the benchmark controls this automatically per run).
"""

import os
import sys
import time
import numpy as np

# Parameters
N_REPS    = 10
N_TRAIN   = 2000
N_TEST    = 500
P         = 10
NUM_GFR   = 10
NUM_BURNIN = 0
NUM_MCMC  = 100

# ── helpers ───────────────────────────────────────────────────────────────────

def step_f(x: np.ndarray) -> np.ndarray:
    """Piecewise-constant signal used as the mean across scenarios."""
    return np.where(
        x < 0.25, -7.5,
        np.where(x < 0.50, -2.5,
        np.where(x < 0.75,  2.5, 7.5))
    )

def split_xy(X: np.ndarray, y: np.ndarray, n_test: int = N_TEST):
    """Return (X_train, X_test, y_train, y_test, train_idx, test_idx)."""
    n = X.shape[0]
    rng_split = np.random.default_rng(0)
    test_idx  = np.sort(rng_split.choice(n, n_test, replace=False))
    train_idx = np.setdiff1d(np.arange(n), test_idx)
    return (X[train_idx], X[test_idx],
            y[train_idx], y[test_idx],
            train_idx, test_idx)

def time_reps(fn, reps: int = N_REPS) -> list[float]:
    """Run fn() `reps` times; return list of elapsed seconds."""
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times

# ── shared base data ──────────────────────────────────────────────────────────

rng = np.random.default_rng(42)
X_base = rng.uniform(0, 1, (N_TRAIN + N_TEST, P))
f_base = step_f(X_base[:, 0])

X_tr_base, X_te_base, y_tr_base, y_te_base, tr_idx, te_idx = split_xy(
    X_base, f_base + rng.normal(0, 1, N_TRAIN + N_TEST)
)

# ── scenario definitions ──────────────────────────────────────────────────────
# Each scenario is a dict with:
#   setup() -> data_dict     called once to generate data
#   run(data)                calls BARTModel.sample() and returns the model

from stochtree import BARTModel, OutcomeModel

SCENARIOS: dict[str, dict] = {}

def _add(name, setup_fn, run_fn):
    SCENARIOS[name] = {"setup": setup_fn, "run": run_fn}


# ── identity ──────────────────────────────────────────────────────────────────
def _identity_setup():
    return dict(X_train=X_tr_base, X_test=X_te_base, y_train=y_tr_base)

def _identity_run(d):
    m = BARTModel()
    m.sample(d["X_train"], d["y_train"], X_test=d["X_test"],
             num_gfr=NUM_GFR, num_burnin=NUM_BURNIN, num_mcmc=NUM_MCMC)
    return m

_add("identity", _identity_setup, _identity_run)


# ── identity + RFX ────────────────────────────────────────────────────────────
def _rfx_setup():
    rfx_ids  = rng.integers(0, 4, N_TRAIN + N_TEST)
    rfx_coef = np.array([-6.0, -2.0, 2.0, 6.0])
    y_full   = f_base + rfx_coef[rfx_ids] + rng.normal(0, 1, N_TRAIN + N_TEST)
    _, _, y_tr, _, _, _ = split_xy(X_base, y_full)
    return dict(
        X_train=X_tr_base, X_test=X_te_base,
        y_train=y_tr,
        rfx_group_ids_train=rfx_ids[tr_idx].astype(np.int32),
        rfx_group_ids_test =rfx_ids[te_idx].astype(np.int32),
    )

def _rfx_run(d):
    m = BARTModel()
    m.sample(d["X_train"], d["y_train"], X_test=d["X_test"],
             rfx_group_ids_train=d["rfx_group_ids_train"],
             rfx_group_ids_test =d["rfx_group_ids_test"],
             num_gfr=NUM_GFR, num_burnin=NUM_BURNIN, num_mcmc=NUM_MCMC,
             random_effects_params={"model_spec": "intercept_only"})
    return m

_add("identity_rfx", _rfx_setup, _rfx_run)


# ── identity + univariate leaf basis ──────────────────────────────────────────
def _basis_setup():
    W      = rng.uniform(0, 1, (N_TRAIN + N_TEST, 2))
    y_full = f_base * W[:, 0] + rng.normal(0, 1, N_TRAIN + N_TEST)
    _, _, y_tr, _, _, _ = split_xy(X_base, y_full)
    return dict(
        X_train=X_tr_base, X_test=X_te_base, y_train=y_tr,
        W_train=W[tr_idx], W_test=W[te_idx],
    )

def _basis_run(d):
    m = BARTModel()
    m.sample(d["X_train"], d["y_train"], X_test=d["X_test"],
             leaf_basis_train=d["W_train"], leaf_basis_test=d["W_test"],
             num_gfr=NUM_GFR, num_burnin=NUM_BURNIN, num_mcmc=NUM_MCMC,
             mean_forest_params={"sample_sigma2_leaf": False})
    return m

_add("identity_basis", _basis_setup, _basis_run)


# ── identity + leaf basis + RFX ───────────────────────────────────────────────
def _basis_rfx_setup():
    W        = rng.uniform(0, 1, (N_TRAIN + N_TEST, 2))
    rfx_ids  = rng.integers(0, 3, N_TRAIN + N_TEST)
    rfx_coef = np.array([-4.0, 0.0, 4.0])
    y_full   = f_base * W[:, 0] + rfx_coef[rfx_ids] + rng.normal(0, 1, N_TRAIN + N_TEST)
    _, _, y_tr, _, _, _ = split_xy(X_base, y_full)
    return dict(
        X_train=X_tr_base, X_test=X_te_base, y_train=y_tr,
        W_train=W[tr_idx], W_test=W[te_idx],
        rfx_group_ids_train=rfx_ids[tr_idx].astype(np.int32),
        rfx_group_ids_test =rfx_ids[te_idx].astype(np.int32),
    )

def _basis_rfx_run(d):
    m = BARTModel()
    m.sample(d["X_train"], d["y_train"], X_test=d["X_test"],
             leaf_basis_train=d["W_train"], leaf_basis_test=d["W_test"],
             rfx_group_ids_train=d["rfx_group_ids_train"],
             rfx_group_ids_test =d["rfx_group_ids_test"],
             num_gfr=NUM_GFR, num_burnin=NUM_BURNIN, num_mcmc=NUM_MCMC,
             mean_forest_params    ={"sample_sigma2_leaf": False},
             random_effects_params ={"model_spec": "intercept_only"})
    return m

_add("identity_basis_rfx", _basis_rfx_setup, _basis_rfx_run)


# ── variance forest ───────────────────────────────────────────────────────────
def _vf_setup():
    return dict(X_train=X_tr_base, X_test=X_te_base, y_train=y_tr_base)

def _vf_run(d):
    m = BARTModel()
    m.sample(d["X_train"], d["y_train"], X_test=d["X_test"],
             num_gfr=NUM_GFR, num_burnin=NUM_BURNIN, num_mcmc=NUM_MCMC,
             general_params={"include_variance_forest": True})
    return m

_add("variance_forest", _vf_setup, _vf_run)


# ── probit ────────────────────────────────────────────────────────────────────
def _probit_setup():
    z_full = f_base / 5.0 + rng.normal(0, 1, N_TRAIN + N_TEST)
    y_full = (z_full > 0).astype(int)
    _, _, y_tr, _, _, _ = split_xy(X_base, y_full.astype(float))
    return dict(X_train=X_tr_base, X_test=X_te_base, y_train=y_tr.astype(int))

def _probit_run(d):
    m = BARTModel()
    m.sample(d["X_train"], d["y_train"], X_test=d["X_test"],
             num_gfr=NUM_GFR, num_burnin=NUM_BURNIN, num_mcmc=NUM_MCMC,
             general_params={
                 "outcome_model": OutcomeModel(outcome="binary", link="probit"),
                 "sample_sigma2_global": False,
             })
    return m

_add("probit", _probit_setup, _probit_run)


# ── cloglog binary ────────────────────────────────────────────────────────────
def _cloglog_bin_setup():
    p_pos  = 1.0 / (1.0 + np.exp(-f_base / 5.0))
    y_full = rng.binomial(1, p_pos).astype(int)
    _, _, y_tr, _, _, _ = split_xy(X_base, y_full.astype(float))
    return dict(X_train=X_tr_base, X_test=X_te_base, y_train=y_tr.astype(int))

def _cloglog_bin_run(d):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = BARTModel()
        m.sample(d["X_train"], d["y_train"], X_test=d["X_test"],
                 num_gfr=NUM_GFR, num_burnin=NUM_BURNIN, num_mcmc=NUM_MCMC,
                 general_params={
                     "outcome_model": OutcomeModel(outcome="binary", link="cloglog"),
                     "sample_sigma2_global": False,
                 })
    return m

_add("cloglog_binary", _cloglog_bin_setup, _cloglog_bin_run)


# ── cloglog ordinal (K=3) ─────────────────────────────────────────────────────
def _cloglog_ord_setup():
    z_full = f_base / 5.0
    cuts   = [-0.5, 0.5]
    y_full = np.digitize(z_full, cuts).astype(int)   # 0, 1, 2
    _, _, y_tr, _, _, _ = split_xy(X_base, y_full.astype(float))
    return dict(X_train=X_tr_base, X_test=X_te_base, y_train=y_tr.astype(int))

def _cloglog_ord_run(d):
    m = BARTModel()
    m.sample(d["X_train"], d["y_train"], X_test=d["X_test"],
             num_gfr=NUM_GFR, num_burnin=NUM_BURNIN, num_mcmc=NUM_MCMC,
             general_params={
                 "outcome_model": OutcomeModel(outcome="ordinal", link="cloglog"),
                 "sample_sigma2_global": False,
             })
    return m

_add("cloglog_ordinal", _cloglog_ord_setup, _cloglog_ord_run)


# ── cloglog ordinal + RFX ─────────────────────────────────────────────────────
def _cloglog_rfx_setup():
    rfx_ids  = rng.integers(0, 3, N_TRAIN + N_TEST)
    rfx_coef = np.array([-1.0, 0.0, 1.0])
    z_full   = f_base / 5.0 + rfx_coef[rfx_ids]
    cuts     = [-0.5, 0.5]
    y_full   = np.digitize(z_full, cuts).astype(int)
    _, _, y_tr, _, _, _ = split_xy(X_base, y_full.astype(float))
    return dict(
        X_train=X_tr_base, X_test=X_te_base, y_train=y_tr.astype(int),
        rfx_group_ids_train=rfx_ids[tr_idx].astype(np.int32),
        rfx_group_ids_test =rfx_ids[te_idx].astype(np.int32),
    )

def _cloglog_rfx_run(d):
    m = BARTModel()
    m.sample(d["X_train"], d["y_train"], X_test=d["X_test"],
             rfx_group_ids_train=d["rfx_group_ids_train"],
             rfx_group_ids_test =d["rfx_group_ids_test"],
             num_gfr=NUM_GFR, num_burnin=NUM_BURNIN, num_mcmc=NUM_MCMC,
             general_params={
                 "outcome_model": OutcomeModel(outcome="ordinal", link="cloglog"),
                 "sample_sigma2_global": False,
             },
             random_effects_params={"model_spec": "intercept_only"})
    return m

_add("cloglog_rfx", _cloglog_rfx_setup, _cloglog_rfx_run)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if "--list" in args:
        print("Available scenarios:")
        for name in SCENARIOS:
            print(f"  {name}")
        return

    selected = [a for a in args if a in SCENARIOS] if args else list(SCENARIOS)
    if not selected:
        print("No matching scenarios found. Use --list to see available names.")
        return

    print(f"\nBART dispatch benchmark  |  n_train={N_TRAIN}  n_test={N_TEST}"
          f"  p={P}  reps={N_REPS}")
    print(f"  GFR={NUM_GFR}  burnin={NUM_BURNIN}  MCMC={NUM_MCMC}\n")
    print(f"{'scenario':<22}  {'fast (s)':>10}  {'slow (s)':>10}  {'speedup':>8}")
    print("-" * 57)

    for name in selected:
        sc   = SCENARIOS[name]
        data = sc["setup"]()

        os.environ["STOCHTREE_USE_CPP_DISPATCH"] = "1"
        fast_t = time_reps(lambda: sc["run"](data))

        os.environ["STOCHTREE_USE_CPP_DISPATCH"] = "0"
        slow_t = time_reps(lambda: sc["run"](data))

        del os.environ["STOCHTREE_USE_CPP_DISPATCH"]  # restore default

        fast_mean = sum(fast_t) / len(fast_t)
        slow_mean = sum(slow_t) / len(slow_t)
        speedup   = slow_mean / fast_mean
        print(f"{name:<22}  {fast_mean:>10.3f}  {slow_mean:>10.3f}  {speedup:>7.2f}x")

    print(f"\n(times are mean elapsed seconds across {N_REPS} repetitions)")


if __name__ == "__main__":
    main()
