"""
Dispatch benchmark: slow path vs BARTSampler (C++).

Compares two execution paths for BARTModel.sample():
  - slow_path  : existing Python sampling loop (fast_path=False)
  - bartsampler: C++ BARTSamplerFit dispatch (fast_path=True)

Wall-time is measured with timeit over a small number of repetitions.
Results are printed as a table with columns: scenario, n, T, S,
slow_path_ms, bartsampler_ms, speedup.

Usage:
    source venv/bin/activate
    python tools/perf/bart_dispatch_benchmark.py
    python tools/perf/bart_dispatch_benchmark.py --reps 5
    python tools/perf/bart_dispatch_benchmark.py --scenario identity probit
"""

import argparse
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from stochtree import BARTModel


# ── Data generators ───────────────────────────────────────────────────────────

def make_continuous(n, p, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = 2.0 * X[:, 0] - X[:, 1] + 0.5 * rng.standard_normal(n)
    return X, y


def make_binary(n, p, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    eta = X[:, 0] - 0.5 * X[:, 1]
    from scipy.stats import norm
    prob = norm.cdf(eta)
    y = (rng.uniform(size=n) < prob).astype(float)
    return X, y


def make_ordinal(n, p, K, seed=99):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    eta = X[:, 0] - 0.5 * X[:, 1]
    cuts = np.linspace(-2.0, 2.0, K + 1)[1:-1]  # K-1 interior cuts
    u = rng.uniform(size=n)
    y = np.full(n, float(K - 1))
    for k, c in enumerate(cuts):
        cdf_k = 1.0 - np.exp(-np.exp(c + eta))
        y = np.where(u < cdf_k, np.minimum(y, float(k)), y)
    return X, y.astype(float)


def make_heterosked(n, p, seed=789):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, p))
    s = np.where(X[:, 0] < 0.25, 0.5,
        np.where(X[:, 0] < 0.50, 1.0,
        np.where(X[:, 0] < 0.75, 2.0, 3.0)))
    y = s * rng.standard_normal(n)
    return X, y


def make_rfx(n, p, num_groups, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    alpha = rng.normal(0, 0.5, size=num_groups)
    groups = np.arange(n, dtype=np.int32) % num_groups
    y = 2.0 * X[:, 0] - X[:, 1] + alpha[groups] + 0.3 * rng.standard_normal(n)
    return X, y, groups


# ── Timing helper ─────────────────────────────────────────────────────────────

def time_ms(fn, reps):
    """Run fn() reps times, return average wall-time in ms."""
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1000.0


# ── Scenario definition ───────────────────────────────────────────────────────

@dataclass
class Scenario:
    label: str
    n: int
    p: int
    num_trees: int
    num_gfr: int
    num_mcmc: int
    num_chains: int = 1
    link: str = "identity"         # "identity", "probit", "cloglog"
    cloglog_K: int = 2
    variance_forest: bool = False
    num_trees_variance: int = 0
    rfx: bool = False
    rfx_num_groups: int = 0

    @property
    def num_samples(self):
        return self.num_gfr + self.num_mcmc * self.num_chains


def build_call(s: Scenario, fast_path):
    """Return a zero-argument callable that runs the full sampling for scenario s."""
    # Build data once per scenario (not counted in timing)
    if s.link == "cloglog":
        X, y = make_ordinal(s.n, s.p, s.cloglog_K)
    elif s.link == "probit":
        X, y = make_binary(s.n, s.p)
    elif s.variance_forest and s.num_trees == 0:
        X, y = make_heterosked(s.n, s.p)
    elif s.rfx:
        X, y, groups = make_rfx(s.n, s.p, s.rfx_num_groups)
    else:
        X, y = make_continuous(s.n, s.p)

    general_params = {"num_chains": s.num_chains}
    if s.link in ("probit", "cloglog"):
        general_params["link"] = s.link

    mean_forest_params = {"num_trees": s.num_trees}

    variance_forest_params = {}
    if s.variance_forest:
        variance_forest_params = {
            "include_variance_forest": True,
            "num_trees": s.num_trees_variance,
        }

    random_effects_params = {}
    if s.rfx:
        random_effects_params = {"model_spec": "intercept_only"}

    kwargs = dict(
        X_train=X,
        y_train=y,
        num_gfr=s.num_gfr,
        num_mcmc=s.num_mcmc,
        general_params=general_params,
        mean_forest_params=mean_forest_params,
        variance_forest_params=variance_forest_params or None,
        random_effects_params=random_effects_params or None,
        fast_path=fast_path,
    )
    if s.rfx:
        kwargs["rfx_group_ids_train"] = groups

    def fn():
        m = BARTModel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.sample(**kwargs)
        return m

    return fn


ALL_SCENARIOS = [
    Scenario("GFR-only (id)",         500, 10, 200, 10,   0, 1),
    Scenario("MCMC-only (id)",        500, 10, 200,  0, 100, 1),
    Scenario("Warm-start (id)",       500, 10, 200,  5, 100, 1),
    Scenario("Multi-chain (id)",      500, 10, 200,  3,  50, 3),
    Scenario("Large-n (id)",         2000, 10, 200,  5, 100, 1),
    Scenario("Warm-start (probit)",   500, 10, 200,  5, 100, 1, link="probit"),
    Scenario("Warm-start (cloglog2)", 500, 10, 200,  5, 100, 1, link="cloglog", cloglog_K=2),
    Scenario("Warm-start (cloglog3)", 500, 10, 200,  5, 100, 1, link="cloglog", cloglog_K=3),
    Scenario("Warm-start (mean+var)", 500, 10, 200,  5, 100, 1,
             variance_forest=True, num_trees_variance=50),
    Scenario("Warm-start (rfx-10g)", 500, 10, 200,  5, 100, 1, rfx=True, rfx_num_groups=10),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reps", type=int, default=3,
                        help="Number of timed repetitions per cell (default: 3)")
    parser.add_argument("--scenario", nargs="+", metavar="LABEL",
                        help="Run only scenarios whose label contains these strings")
    args = parser.parse_args()

    scenarios = ALL_SCENARIOS
    if args.scenario:
        filters = [f.lower() for f in args.scenario]
        scenarios = [s for s in scenarios
                     if any(f in s.label.lower() for f in filters)]
    if not scenarios:
        print("No matching scenarios.")
        return

    W = (28, 6, 5, 5, 14, 14, 10)
    header = f"{'Scenario':<{W[0]}} {'n':>{W[1]}} {'T':>{W[2]}} {'S':>{W[3]}} " \
             f"{'slow_path(ms)':>{W[4]}} {'bartsampler(ms)':>{W[5]}} {'speedup':>{W[6]}}"
    sep = "-" * sum(W + (len(W) - 1,))

    print(f"\nDispatch benchmark  (reps={args.reps})")
    print(sep)
    print(header)
    print(sep)

    for s in scenarios:
        fn_slow = build_call(s, fast_path=False)
        fn_smp  = build_call(s, fast_path=True)

        ms_slow = time_ms(fn_slow, args.reps)
        ms_smp  = time_ms(fn_smp,  args.reps)
        speedup = ms_slow / ms_smp if ms_smp > 0 else float("nan")

        label_str = f"{s.label:<{W[0]}}"
        print(f"{label_str} {s.n:>{W[1]}} {s.num_trees:>{W[2]}} {s.num_samples:>{W[3]}} "
              f"{ms_slow:>{W[4]}.1f} {ms_smp:>{W[5]}.1f} {speedup:>{W[6]}.2f}x")

    print(sep)
    print("  speedup > 1.0x: BARTSampler (fast_path=True) is faster")
    print("  speedup < 1.0x: slow path is faster (regression — investigate)\n")


if __name__ == "__main__":
    main()
