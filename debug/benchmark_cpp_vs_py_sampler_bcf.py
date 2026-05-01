"""Benchmark: C++ sampler loop vs. Python sampler loop -- BCF.

Compares runtime and test-set RMSE across run_cpp=True / False in BCFModel.sample().
Simplest BCF case: univariate binary treatment, no RFX, no adaptive coding,
no treatment intercept.

DGP: mu(X) is a step function on X[:,0].  tau(X) = 2 + 4*X[:,2] (linear CATE).
     pi(X) = 0.2 + 0.6*X[:,3] (mild confounding).  Z ~ Bernoulli(pi(X)).
     y = mu(X) + tau(X)*Z + noise.

Usage:
    source venv/bin/activate
    python debug/benchmark_cpp_vs_py_sampler_bcf.py
"""

import argparse
import time
import numpy as np
from stochtree import BCFModel

parser = argparse.ArgumentParser()
parser.add_argument("--num-chains", type=int, default=1)
args = parser.parse_args()
num_chains = args.num_chains

# ---------------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------------
rng = np.random.default_rng(1234)

n = 2000
p = 10
X = rng.uniform(size=(n, p))

mu_X = (
    np.where((X[:, 0] >= 0.00) & (X[:, 0] < 0.25), -7.5, 0) +
    np.where((X[:, 0] >= 0.25) & (X[:, 0] < 0.50), -2.5, 0) +
    np.where((X[:, 0] >= 0.50) & (X[:, 0] < 0.75),  2.5, 0) +
    np.where((X[:, 0] >= 0.75) & (X[:, 0] < 1.00),  7.5, 0)
)
tau_X = 2 + 4 * X[:, 2]
pi_X  = 0.2 + 0.6 * X[:, 3]
Z     = rng.binomial(1, pi_X).astype(float)

noise_sd = 1.0
y = mu_X + tau_X * Z + rng.normal(scale=noise_sd, size=n)

test_frac  = 0.2
n_test     = round(test_frac * n)
n_train    = n - n_test
test_inds  = rng.choice(n, size=n_test, replace=False)
train_inds = np.setdiff1d(np.arange(n), test_inds)

X_train,  X_test  = X[train_inds],    X[test_inds]
Z_train,  Z_test  = Z[train_inds],    Z[test_inds]
pi_train, pi_test = pi_X[train_inds], pi_X[test_inds]
y_train,  y_test  = y[train_inds],    y[test_inds]
mu_test           = mu_X[test_inds]
tau_test          = tau_X[test_inds]
f_test            = mu_test + tau_test * Z_test

# ---------------------------------------------------------------------------
# Benchmark settings
# ---------------------------------------------------------------------------
num_gfr       = 10
num_burnin    = 0
num_mcmc      = 100
num_trees_mu  = 200
num_trees_tau = 50
n_reps        = 3

print(
    f"n_train={n_train}  n_test={n_test}  p={p}\n"
    f"mu_trees={num_trees_mu}  tau_trees={num_trees_tau}  "
    f"num_gfr={num_gfr}  num_burnin={num_burnin}  num_mcmc={num_mcmc}  "
    f"num_chains={num_chains}  reps={n_reps}\n"
)

# ---------------------------------------------------------------------------
# Helper: run one configuration and return timing + accuracy metrics
# ---------------------------------------------------------------------------
def run_once(run_cpp: bool, seed: int) -> dict:
    m = BCFModel()
    t0 = time.perf_counter()
    m.sample(
        X_train=X_train,
        Z_train=Z_train,
        y_train=y_train,
        propensity_train=pi_train,
        X_test=X_test,
        Z_test=Z_test,
        propensity_test=pi_test,
        num_gfr=num_gfr,
        num_burnin=num_burnin,
        num_mcmc=num_mcmc,
        general_params={
            "adaptive_coding": False,
            "random_seed": seed,
            "num_chains": num_chains,
            "propensity_covariate": "prognostic",
        },
        prognostic_forest_params={"num_trees": num_trees_mu},
        treatment_effect_forest_params={
            "num_trees": num_trees_tau,
            "sample_intercept": False,
        },
        run_cpp=run_cpp,
    )
    elapsed = time.perf_counter() - t0

    y_hat   = m.y_hat_test.mean(axis=1)
    tau_hat = m.tau_hat_test.mean(axis=1)

    return {
        "elapsed":  elapsed,
        "rmse_y":   float(np.sqrt(np.mean((y_hat   - y_test)   ** 2))),
        "rmse_f":   float(np.sqrt(np.mean((y_hat   - f_test)   ** 2))),
        "rmse_tau": float(np.sqrt(np.mean((tau_hat - tau_test) ** 2))),
    }

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------
seeds = [1000 + i for i in range(1, n_reps + 1)]

results_cpp = []
results_py  = []

print("Running C++ sampler (run_cpp=True)...")
for i, seed in enumerate(seeds, 1):
    print(f"  rep {i}/{n_reps}")
    results_cpp.append(run_once(run_cpp=True, seed=seed))

print("\nRunning Python sampler (run_cpp=False)...")
for i, seed in enumerate(seeds, 1):
    print(f"  rep {i}/{n_reps}")
    results_py.append(run_once(run_cpp=False, seed=seed))

# ---------------------------------------------------------------------------
# Summarise
# ---------------------------------------------------------------------------
def summarise(results: list) -> dict:
    keys = ["elapsed", "rmse_y", "rmse_f", "rmse_tau"]
    out = {k: float(np.mean([r[k] for r in results])) for k in keys}
    out["elapsed_sd"] = float(np.std([r["elapsed"] for r in results], ddof=1))
    return out

s_cpp = summarise(results_cpp)
s_py  = summarise(results_py)
rows  = [("cpp (run_cpp=True)", s_cpp), ("py  (run_cpp=False)", s_py)]

print("\n--- Results ---")
print(
    f"{'Sampler':<22}  {'Time (s)':>10}  {'SD':>10}  "
    f"{'RMSE (obs)':>12}  {'RMSE (f)':>12}  {'RMSE (tau)':>12}"
)
print("-" * 84)
for label, s in rows:
    print(
        f"{label:<22}  {s['elapsed']:>10.3f}  {s['elapsed_sd']:>10.3f}  "
        f"{s['rmse_y']:>12.4f}  {s['rmse_f']:>12.4f}  {s['rmse_tau']:>12.4f}"
    )

speedup = s_py["elapsed"] / s_cpp["elapsed"]
print(f"\nSpeedup (py / cpp): {speedup:.2f}x")
print(
    f"RMSE delta (cpp - py):  "
    f"obs={s_cpp['rmse_y'] - s_py['rmse_y']:.4f}  "
    f"f={s_cpp['rmse_f'] - s_py['rmse_f']:.4f}  "
    f"tau={s_cpp['rmse_tau'] - s_py['rmse_tau']:.4f}"
)
