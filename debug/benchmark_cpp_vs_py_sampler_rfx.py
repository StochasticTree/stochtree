"""Benchmark: C++ sampler loop vs. Python sampler loop – BART with random effects.

Compares runtime and RMSE (vs. test outcomes and vs. true mean) across
run_cpp=True / False in BARTModel.sample().

DGP: continuous outcome with an additive intercept-only random effect.
  y_i = f(X_i) + alpha_{g_i} + eps_i,  eps ~ N(0, 0.5^2)
  f(X) is a piecewise-constant step function on X[:,0].
  Group intercepts alpha_g ~ N(0, 1), 10 groups.

Usage:
    source venv/bin/activate
    python debug/benchmark_cpp_vs_py_sampler_rfx.py
"""

import time
import numpy as np
from stochtree import BARTModel

# ---------------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------------
rng = np.random.default_rng(1234)

n = 2000
p = 10
num_groups = 10

X = rng.uniform(size=(n, p))
f_X = np.where(X[:, 0] < 0.25, -7.5,
      np.where(X[:, 0] < 0.5,  -2.5,
      np.where(X[:, 0] < 0.75,  2.5, 7.5)))

group_ids = rng.integers(0, num_groups, size=n).astype(np.int32)
rfx_coefs = rng.normal(0, 1, size=num_groups)
rfx_term  = rfx_coefs[group_ids]

mu_true = f_X + rfx_term
y = mu_true + rng.normal(0, 0.5, size=n)

test_frac  = 0.2
n_test     = round(test_frac * n)
test_inds  = rng.choice(n, size=n_test, replace=False)
train_inds = np.setdiff1d(np.arange(n), test_inds)

X_train, X_test           = X[train_inds],        X[test_inds]
y_train, y_test           = y[train_inds],         y[test_inds]
group_ids_train           = group_ids[train_inds]
group_ids_test            = group_ids[test_inds]
rfx_basis_train           = np.ones((len(train_inds), 1))
rfx_basis_test            = np.ones((n_test, 1))
mu_test                   = mu_true[test_inds]

# ---------------------------------------------------------------------------
# Benchmark settings
# ---------------------------------------------------------------------------
num_gfr    = 10
num_burnin = 100
num_mcmc   = 100
num_trees  = 200
n_reps     = 3

print(
    f"n_train={len(train_inds)}  n_test={n_test}  p={p}  num_groups={num_groups}  "
    f"num_trees={num_trees}  num_gfr={num_gfr}  num_burnin={num_burnin}  "
    f"num_mcmc={num_mcmc}  reps={n_reps}\n"
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
        rfx_group_ids_train=group_ids_train,
        rfx_group_ids_test=group_ids_test,
        rfx_basis_train=rfx_basis_train,
        rfx_basis_test=rfx_basis_test,
        mean_forest_params={"num_trees": num_trees},
        general_params={"random_seed": seed},
        run_cpp=run_cpp,
    )
    elapsed = time.perf_counter() - t0

    y_hat = m.y_hat_test.mean(axis=1)
    rmse_y   = float(np.sqrt(np.mean((y_hat - y_test)  ** 2)))
    rmse_mu  = float(np.sqrt(np.mean((y_hat - mu_test) ** 2)))

    return {"elapsed": elapsed, "rmse_y": rmse_y, "rmse_mu": rmse_mu}

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------
seeds = [100 + i for i in range(1, n_reps + 1)]

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
    rmse_y  = [r["rmse_y"]  for r in results]
    rmse_mu = [r["rmse_mu"] for r in results]
    return {
        "elapsed_mean": np.mean(elapsed),
        "elapsed_sd":   np.std(elapsed, ddof=1),
        "rmse_y_mean":  np.mean(rmse_y),
        "rmse_mu_mean": np.mean(rmse_mu),
    }

s_cpp = summarise(results_cpp)
s_py  = summarise(results_py)
rows  = [("cpp (run_cpp=True)", s_cpp), ("py  (run_cpp=False)", s_py)]

print("\n--- Results ---")
print(
    f"{'Sampler':<22}  {'Time (s)':>10}  {'SD':>10}  "
    f"{'RMSE (y)':>10}  {'RMSE (mu)':>10}"
)
print("-" * 70)
for label, s in rows:
    print(
        f"{label:<22}  {s['elapsed_mean']:>10.3f}  {s['elapsed_sd']:>10.3f}"
        f"  {s['rmse_y_mean']:>10.4f}  {s['rmse_mu_mean']:>10.4f}"
    )

speedup = s_py["elapsed_mean"] / s_cpp["elapsed_mean"]
print(f"\nSpeedup (py / cpp): {speedup:.2f}x")
print(
    f"RMSE-y  delta (cpp - py): {s_cpp['rmse_y_mean']  - s_py['rmse_y_mean']:.4f}\n"
    f"RMSE-mu delta (cpp - py): {s_cpp['rmse_mu_mean'] - s_py['rmse_mu_mean']:.4f}"
)
