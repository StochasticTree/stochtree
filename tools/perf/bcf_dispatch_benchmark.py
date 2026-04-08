"""
Dispatch benchmark: slow path vs BCFSampler (C++).

Compares two execution paths for BCFModel.sample():
  - slow_path  : existing Python sampling loop (fast_path=False)
  - bcfsampler : C++ BCFSamplerFit dispatch (fast_path=True)

Wall-time is measured with timeit over a small number of repetitions.
Results are printed as a table with columns: scenario, n, T_mu, T_tau, S,
slow_path_ms, bcfsampler_ms, speedup.

Usage:
    source venv/bin/activate
    python tools/perf/bcf_dispatch_benchmark.py
    python tools/perf/bcf_dispatch_benchmark.py --reps 5
    python tools/perf/bcf_dispatch_benchmark.py --scenario identity propensity
"""

import argparse
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from stochtree import BCFModel


# ── Data generator ────────────────────────────────────────────────────────────

def make_bcf_data(n, p=5, seed=42):
    """
    Friedman-like BCF DGP:
      mu(X)  = sin(pi*X0*X1) + 2*(X2-0.5)^2 + X3 + 0.5*X4
      tau(X) = 1 + X2 + X3
      pi_hat = Phi(0.5*X0 - 0.25*X1)
      Z      ~ Bernoulli(pi_hat)
      y      = mu(X) + tau(X)*Z + N(0,1)
    """
    rng = np.random.default_rng(seed)
    X = np.empty((n, p))
    X[:, :2] = rng.standard_normal((n, 2))
    if p > 2:
        X[:, 2:] = rng.uniform(size=(n, p - 2))

    mu = (np.sin(np.pi * X[:, 0] * X[:, 1])
          + 2.0 * (X[:, 2] - 0.5) ** 2
          + X[:, 3]
          + 0.5 * X[:, 4])
    tau = 1.0 + X[:, 2] + X[:, 3]
    pi_hat = norm.cdf(0.5 * X[:, 0] - 0.25 * X[:, 1])
    Z = (rng.uniform(size=n) < pi_hat).astype(float)
    y = mu + tau * Z + rng.standard_normal(n)

    return X, y, Z, pi_hat


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
    num_trees_mu: int
    num_trees_tau: int
    num_gfr: int
    num_mcmc: int
    propensity: bool = True          # include pi_hat
    adaptive_coding: bool = False
    variance_forest: bool = False
    num_trees_variance: int = 0

    @property
    def num_samples(self):
        return self.num_gfr + self.num_mcmc


def build_call(s: Scenario, fast_path: bool):
    """Return a zero-argument callable that runs the full BCF sampling."""
    X, y, Z, pi_hat = make_bcf_data(s.n)

    general_params = {
        "adaptive_coding": s.adaptive_coding,
        "propensity_covariate": "prognostic" if s.propensity else "none",
    }
    prognostic_forest_params = {"num_trees": s.num_trees_mu}
    treatment_effect_forest_params = {
        "num_trees": s.num_trees_tau,
        "sample_sigma2_leaf": False,
    }

    variance_forest_params = {}
    if s.variance_forest:
        variance_forest_params = {
            "include_variance_forest": True,
            "num_trees": s.num_trees_variance,
        }

    kwargs = dict(
        X_train=X,
        Z_train=Z,
        y_train=y,
        propensity_train=pi_hat if s.propensity else None,
        num_gfr=s.num_gfr,
        num_mcmc=s.num_mcmc,
        general_params=general_params,
        prognostic_forest_params=prognostic_forest_params,
        treatment_effect_forest_params=treatment_effect_forest_params,
        variance_forest_params=variance_forest_params or None,
        fast_path=fast_path,
    )

    def fn():
        m = BCFModel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.sample(**kwargs)
        return m

    return fn


ALL_SCENARIOS = [
    Scenario("MCMC-only (id)",         500,  250, 100,   0, 100),
    Scenario("Warm-start (id)",        500,  250, 100,  10, 100),
    Scenario("Large-n (id)",          2000,  250, 100,  10, 100),
    Scenario("No propensity",          500,  250, 100,  10, 100, propensity=False),
    Scenario("Adaptive coding",        500,  250, 100,  10, 100, adaptive_coding=True),
    Scenario("Mean+var forest",        500,  250, 100,  10, 100,
             variance_forest=True, num_trees_variance=50),
    Scenario("Small trees",            500,   50,  25,  10, 100),
    Scenario("Many trees",             500,  500, 200,  10, 100),
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

    W = (24, 6, 6, 6, 5, 14, 14, 10)
    header = (f"{'Scenario':<{W[0]}} {'n':>{W[1]}} {'T_mu':>{W[2]}} {'T_tau':>{W[3]}} "
              f"{'S':>{W[4]}} {'slow_path(ms)':>{W[5]}} {'bcfsampler(ms)':>{W[6]}} {'speedup':>{W[7]}}")
    sep = "-" * (sum(W) + len(W) - 1)

    print(f"\nBCF dispatch benchmark  (reps={args.reps})")
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
        print(f"{label_str} {s.n:>{W[1]}} {s.num_trees_mu:>{W[2]}} {s.num_trees_tau:>{W[3]}} "
              f"{s.num_samples:>{W[4]}} {ms_slow:>{W[5]}.1f} {ms_smp:>{W[6]}.1f} "
              f"{speedup:>{W[7]}.2f}x")

    print(sep)
    print("  speedup > 1.0x: BCFSampler (fast_path=True) is faster")
    print("  speedup < 1.0x: slow path is faster (regression — investigate)\n")


if __name__ == "__main__":
    main()
