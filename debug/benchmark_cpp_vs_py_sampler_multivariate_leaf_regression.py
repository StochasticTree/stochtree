"""Benchmark: C++ sampler loop vs. Python sampler loop -- multivariate leaf regression.

Compares runtime and test-set RMSE across run_cpp=True / False in BARTModel.sample()
with a 2-column leaf basis (multivariate leaf regression).

DGP: f(X, Z) = tau_1(X)*Z_1 + tau_2(X)*Z_2, where tau_1/tau_2 are step functions
of X[:,0] and Z_1, Z_2 are drawn uniform [0, 1].  A constant noise term is added.
The leaf basis passed to the sampler is [Z_1, Z_2] (shape n x 2).

Usage:
    source venv/bin/activate
    python debug/benchmark_cpp_vs_py_sampler_multivariate_leaf_regression.py
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
Z1 = rng.uniform(size=n)
Z2 = rng.uniform(size=n)

# Heterogeneous slopes on Z1 and Z2, partitioned by X[:,0]
tau1_X = (
    np.where((X[:, 0] >= 0.00) & (X[:, 0] < 0.25), -2.0, 0) +
    np.where((X[:, 0] >= 0.25) & (X[:, 0] < 0.50), -1.0, 0) +
    np.where((X[:, 0] >= 0.50) & (X[:, 0] < 0.75),  1.0, 0) +
    np.where((X[:, 0] >= 0.75) & (X[:, 0] < 1.00),  2.0, 0)
)
tau2_X = (
    np.where((X[:, 0] >= 0.00) & (X[:, 0] < 0.25),  1.0, 0) +
    np.where((X[:, 0] >= 0.25) & (X[:, 0] < 0.50),  2.0, 0) +
    np.where((X[:, 0] >= 0.50) & (X[:, 0] < 0.75), -1.0, 0) +
    np.where((X[:, 0] >= 0.75) & (X[:, 0] < 1.00), -2.0, 0)
)
f_XZ = tau1_X * Z1 + tau2_X * Z2
noise_sd = 1.0
y = f_XZ + rng.normal(scale=noise_sd, size=n)

test_frac = 0.2
n_test = round(test_frac * n)
n_train = n - n_test
test_inds  = rng.choice(n, size=n_test, replace=False)
train_inds = np.setdiff1d(np.arange(n), test_inds)

X_train, X_test   = X[train_inds],  X[test_inds]
Z1_train, Z1_test = Z1[train_inds], Z1[test_inds]
Z2_train, Z2_test = Z2[train_inds], Z2[test_inds]
y_train, y_test   = y[train_inds],  y[test_inds]
f_test = f_XZ[test_inds]

# Leaf basis matrices (n x 2)
basis_train = np.column_stack([Z1_train, Z2_train])
basis_test  = np.column_stack([Z1_test,  Z2_test])

# ---------------------------------------------------------------------------
# Benchmark settings
# ---------------------------------------------------------------------------
num_gfr    = 10
num_burnin = 0
num_mcmc   = 100
num_trees  = 200
n_reps     = 3

# Optional: pass a 2x2 prior covariance for the leaf coefficients
sigma2_leaf_init = np.array([[0.5, 0.0], [0.0, 0.5]])

print(
    f"n_train={n_train}  n_test={n_test}  p={p}  basis_dim=2\n"
    f"num_trees={num_trees}  num_gfr={num_gfr}  num_burnin={num_burnin}  "
    f"num_mcmc={num_mcmc}  num_chains={num_chains}  reps={n_reps}\n"
)

# ---------------------------------------------------------------------------
# Helper: run one configuration and return timing + RMSE
# ---------------------------------------------------------------------------
def run_once(run_cpp: bool, seed: int) -> dict:
    m = BARTModel()
    t0 = time.perf_counter()
    m.sample(
        X_train=X_train,
        y_train=y_train,
        leaf_basis_train=basis_train,
        X_test=X_test,
        leaf_basis_test=basis_test,
        num_gfr=num_gfr,
        num_burnin=num_burnin,
        num_mcmc=num_mcmc,
        general_params={"random_seed": seed, "num_chains": num_chains},
        mean_forest_params={
            "num_trees": num_trees,
            "sigma2_leaf_init": sigma2_leaf_init,
            "sample_sigma2_leaf": False,
        },
        run_cpp=run_cpp,
    )
    elapsed = time.perf_counter() - t0

    yhat   = m.y_hat_test.mean(axis=1)
    rmse   = float(np.sqrt(np.mean((yhat - y_test) ** 2)))
    rmse_f = float(np.sqrt(np.mean((yhat - f_test) ** 2)))
    return {"elapsed": elapsed, "rmse": rmse, "rmse_f": rmse_f}

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

# # ---------------------------------------------------------------------------
# # Summarise
# # ---------------------------------------------------------------------------
# def summarise(results: list) -> dict:
#     elapsed = [r["elapsed"] for r in results]
#     rmse    = [r["rmse"]    for r in results]
#     rmse_f  = [r["rmse_f"]  for r in results]
#     return {
#         "elapsed_mean": float(np.mean(elapsed)),
#         "elapsed_sd":   float(np.std(elapsed, ddof=1)),
#         "rmse_mean":    float(np.mean(rmse)),
#         "rmse_f_mean":  float(np.mean(rmse_f)),
#     }

# s_cpp = summarise(results_cpp)
# s_py  = summarise(results_py)
# rows  = [("cpp (run_cpp=True)", s_cpp), ("py  (run_cpp=False)", s_py)]

# print("\n--- Results ---")
# header = f"{'sampler':<22}  {'elapsed (s)':>14}  {'RMSE y':>10}  {'RMSE f(X,Z)':>12}"
# print(header)
# print("-" * len(header))
# for label, s in rows:
#     print(
#         f"{label:<22}  "
#         f"{s['elapsed_mean']:>7.2f} ± {s['elapsed_sd']:>4.2f}  "
#         f"{s['rmse_mean']:>10.4f}  "
#         f"{s['rmse_f_mean']:>12.4f}"
#     )
# ---------------------------------------------------------------------------
# Summarise
# ---------------------------------------------------------------------------
def summarise(results: list) -> dict:
    elapsed = [r["elapsed"] for r in results]
    rmse    = [r["rmse"]    for r in results]
    rmse_f  = [r["rmse_f"]  for r in results]
    return {
        "elapsed_mean": float(np.mean(elapsed)),
        "elapsed_sd":   float(np.std(elapsed, ddof=1)),
        "rmse_mean":    float(np.mean(rmse)),
        "rmse_f_mean":  float(np.mean(rmse_f)),
    }

s_cpp = summarise(results_cpp)
s_py  = summarise(results_py)
rows  = [("cpp (run_cpp=True)", s_cpp), ("py  (run_cpp=False)", s_py)]

print("\n--- Results ---")
print(
    f"{'Sampler':<22}  {'Time (s)':>10}  {'SD':>10}  {'RMSE (obs)':>12}  {'RMSE f(X,Z)':>13}"
)
print("-" * 74)
for label, s in rows:
    print(
        f"{label:<22}  {s['elapsed_mean']:>10.3f}  {s['elapsed_sd']:>10.3f}"
        f"  {s['rmse_mean']:>12.4f}  {s['rmse_f_mean']:>13.4f}"
    )

speedup = s_py["elapsed_mean"] / s_cpp["elapsed_mean"]
print(f"\nSpeedup (py / cpp): {speedup:.2f}x")
print(
    f"RMSE delta (cpp - py):  "
    f"obs={s_cpp['rmse_mean'] - s_py['rmse_mean']:.4f}  "
    f"f={s_cpp['rmse_f_mean'] - s_py['rmse_f_mean']:.4f}"
)
