"""Benchmark: C++ sampler loop vs. Python sampler loop – heteroskedastic BART.

Compares runtime, mean-forest RMSE (vs. true f(X)) and RMSE of the estimated
conditional standard deviation (vs. the true s(X)) across run_cpp=True /
False in BARTModel.sample() with both a mean forest and a variance forest
(num_trees_variance > 0).

DGP: f(X) is a step function of X[:,0]; s(X) varies by quadrant of X[:,0]
and linearly with X[:,2], matching the heteroskedastic_bart.R debug script.

Note: A variance-only model (num_trees_mean=0, num_trees_variance>0) is now
supported in the C++ path.  The mean-forest RMSE is reported as NaN in that
case since there is no mean forest to evaluate.

Usage:
    source venv/bin/activate
    python debug/benchmark_cpp_vs_py_sampler_heteroskedastic.py
"""

import argparse
import time
import numpy as np
from stochtree import BARTModel

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

# True conditional mean and conditional std dev
f_X = (
    np.where((X[:, 0] >= 0.00) & (X[:, 0] < 0.25), -3.0, 0) +
    np.where((X[:, 0] >= 0.25) & (X[:, 0] < 0.50), -1.0, 0) +
    np.where((X[:, 0] >= 0.50) & (X[:, 0] < 0.75),  1.0, 0) +
    np.where((X[:, 0] >= 0.75) & (X[:, 0] < 1.00),  3.0, 0)
)
s_X = (
    np.where((X[:, 0] >= 0.00) & (X[:, 0] < 0.25), 0.5 * X[:, 2], 0) +
    np.where((X[:, 0] >= 0.25) & (X[:, 0] < 0.50), 1.0 * X[:, 2], 0) +
    np.where((X[:, 0] >= 0.50) & (X[:, 0] < 0.75), 2.0 * X[:, 2], 0) +
    np.where((X[:, 0] >= 0.75) & (X[:, 0] < 1.00), 3.0 * X[:, 2], 0)
)
y = f_X + rng.standard_normal(n) * s_X

test_frac = 0.2
n_test = round(test_frac * n)
n_train = n - n_test
test_inds  = rng.choice(n, size=n_test, replace=False)
train_inds = np.setdiff1d(np.arange(n), test_inds)

X_train, X_test = X[train_inds], X[test_inds]
y_train, y_test = y[train_inds], y[test_inds]
f_test = f_X[test_inds]
s_test = s_X[test_inds]

# ---------------------------------------------------------------------------
# Benchmark settings
# ---------------------------------------------------------------------------
num_gfr    = 10
num_burnin = 0
num_mcmc   = 100
num_trees_mean     = 200
num_trees_variance = 50
n_reps = 3

print(
    f"n_train={n_train}  n_test={n_test}  p={p}  "
    f"num_trees_mean={num_trees_mean}  num_trees_variance={num_trees_variance}  "
    f"num_gfr={num_gfr}  num_burnin={num_burnin}  num_mcmc={num_mcmc}  "
    f"num_chains={num_chains}  reps={n_reps}\n"
)

# ---------------------------------------------------------------------------
# Helper: run one configuration and return timing + metrics
# ---------------------------------------------------------------------------
def run_once(run_cpp: bool, seed: int) -> dict:
    m = BARTModel()
    t0 = time.perf_counter()
    m.sample(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        num_gfr=num_gfr,
        num_burnin=num_burnin,
        num_mcmc=num_mcmc,
        general_params={
            "random_seed": seed,
            "sample_sigma2_global": False,
            "num_chains": num_chains,
        },
        mean_forest_params={"num_trees": num_trees_mean},
        variance_forest_params={"num_trees": num_trees_variance},
        run_cpp=run_cpp,
    )
    elapsed = time.perf_counter() - t0

    # mean-forest RMSE vs. true f(X) – only defined when a mean forest was fitted
    if num_trees_mean > 0:
        f_hat = m.y_hat_test.mean(axis=1)
        rmse_f = float(np.sqrt(np.mean((f_hat - f_test) ** 2)))
    else:
        rmse_f = float("nan")
    # sigma2_x_test has shape (n_test, num_mcmc); take posterior mean of cond. std dev
    s_hat = np.sqrt(m.sigma2_x_test).mean(axis=1)
    rmse_s = float(np.sqrt(np.mean((s_hat - s_test) ** 2)))
    return {"elapsed": elapsed, "rmse_f": rmse_f, "rmse_s": rmse_s}

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------
seeds = [1000 + i for i in range(1, n_reps + 1)]

results_cpp = []
results_py  = []

print("Running C++ sampler (run_cpp=True)...")
for i, seed in enumerate(seeds, 1):
    print(f"  rep {i}/{n_reps}")
    results_cpp.append(run_once(run_cpp=True,  seed=seed))

print("\nRunning Python sampler (run_cpp=False)...")
for i, seed in enumerate(seeds, 1):
    print(f"  rep {i}/{n_reps}")
    results_py.append(run_once(run_cpp=False, seed=seed))

# ---------------------------------------------------------------------------
# Summarise
# ---------------------------------------------------------------------------
def summarise(results: list) -> dict:
    elapsed = [r["elapsed"] for r in results]
    rmse_f  = [r["rmse_f"]  for r in results]
    rmse_s  = [r["rmse_s"]  for r in results]
    return {
        "elapsed_mean": float(np.mean(elapsed)),
        "elapsed_sd":   float(np.std(elapsed, ddof=1)),
        "rmse_f_mean":  float(np.nanmean(rmse_f)),
        "rmse_s_mean":  float(np.mean(rmse_s)),
    }

s_cpp = summarise(results_cpp)
s_py  = summarise(results_py)
rows  = [("cpp (run_cpp=True)", s_cpp), ("py  (run_cpp=False)", s_py)]

print("\n--- Results ---")
print(
    f"{'Sampler':<22}  {'Time (s)':>10}  {'SD':>10}  {'RMSE f(X)':>12}  {'RMSE s(X)':>12}"
)
print("-" * 74)
for label, s in rows:
    print(
        f"{label:<22}  {s['elapsed_mean']:>10.3f}  {s['elapsed_sd']:>10.3f}"
        f"  {s['rmse_f_mean']:>12.4f}  {s['rmse_s_mean']:>12.4f}"
    )

speedup = s_py["elapsed_mean"] / s_cpp["elapsed_mean"]
print(f"\nSpeedup (py / cpp): {speedup:.2f}x")
print(
    f"RMSE f(X) delta (cpp - py): {s_cpp['rmse_f_mean'] - s_py['rmse_f_mean']:.4f}\n"
    f"RMSE s(X) delta (cpp - py): {s_cpp['rmse_s_mean'] - s_py['rmse_s_mean']:.4f}"
)
