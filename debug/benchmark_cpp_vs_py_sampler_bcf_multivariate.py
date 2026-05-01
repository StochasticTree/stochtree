"""Benchmark: C++ sampler loop vs. Python sampler loop -- multivariate treatment BCF.

Compares runtime and accuracy across run_cpp=True / False in BCFModel.sample()
with a 2-column binary treatment (multivariate BCF).

DGP: propensity pi(X) = [0.25 + 0.5*X[:,0], 0.75 - 0.5*X[:,1]] (2-column).
     mu(X)  = 5*pi[:,0] + 2*pi[:,1] + 2*X[:,2].
     tau(X) = [X[:,1], X[:,2]] (2-column CATE).
     Z ~ Bernoulli(pi(X)) column-wise (binary, shape n x 2).
     y = mu(X) + sum(Z * tau(X), axis=1) + noise.
Adaptive coding is disabled; sigma2_leaf is not sampled for the tau forest.

Usage:
    source venv/bin/activate
    python debug/benchmark_cpp_vs_py_sampler_bcf_multivariate.py
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
p = 5
snr = 2.0

X     = rng.uniform(size=(n, p))
pi_x  = np.c_[0.25 + 0.5 * X[:, 0], 0.75 - 0.5 * X[:, 1]]
mu_x  = pi_x[:, 0] * 5 + pi_x[:, 1] * 2 + 2 * X[:, 2]
tau_x = np.c_[X[:, 1], X[:, 2]]
Z     = (rng.uniform(size=(n, 2)) < pi_x).astype(float)
E_XZ  = mu_x + (Z * tau_x).sum(axis=1)
y     = E_XZ + rng.normal(size=n) * (E_XZ.std() / snr)

test_frac  = 0.2
n_test     = round(test_frac * n)
n_train    = n - n_test
test_inds  = rng.choice(n, size=n_test, replace=False)
train_inds = np.setdiff1d(np.arange(n), test_inds)

X_train,  X_test  = X[train_inds],    X[test_inds]
Z_train,  Z_test  = Z[train_inds],    Z[test_inds]
pi_train, pi_test = pi_x[train_inds], pi_x[test_inds]
y_train,  y_test  = y[train_inds],    y[test_inds]
mu_test           = mu_x[test_inds]
tau_test          = tau_x[test_inds]
f_test            = mu_test + (Z_test * tau_test).sum(axis=1)

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
    f"n_train={n_train}  n_test={n_test}  p={p}  treatment_dim=2\n"
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
        },
        prognostic_forest_params={"num_trees": num_trees_mu},
        treatment_effect_forest_params={
            "num_trees": num_trees_tau,
            "sample_sigma2_leaf": False,
            "sample_intercept": False,
        },
        run_cpp=run_cpp,
    )
    elapsed = time.perf_counter() - t0

    mu_hat   = m.mu_hat_test.mean(axis=1)
    tau_hat1 = m.tau_hat_test[:, 0, :].mean(axis=1)
    tau_hat2 = m.tau_hat_test[:, 1, :].mean(axis=1)
    y_hat    = m.y_hat_test.mean(axis=1)

    return {
        "elapsed":  elapsed,
        "rmse_y":   float(np.sqrt(np.mean((y_hat    - y_test)         ** 2))),
        "rmse_f":   float(np.sqrt(np.mean((y_hat    - f_test)         ** 2))),
        "rmse_mu":  float(np.sqrt(np.mean((mu_hat   - mu_test)        ** 2))),
        "rmse_tau1": float(np.sqrt(np.mean((tau_hat1 - tau_test[:, 0]) ** 2))),
        "rmse_tau2": float(np.sqrt(np.mean((tau_hat2 - tau_test[:, 1]) ** 2))),
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
    keys = ["elapsed", "rmse_y", "rmse_f", "rmse_mu", "rmse_tau1", "rmse_tau2"]
    out = {k: float(np.mean([r[k] for r in results])) for k in keys}
    out["elapsed_sd"] = float(np.std([r["elapsed"] for r in results], ddof=1))
    return out

s_cpp = summarise(results_cpp)
s_py  = summarise(results_py)
rows  = [("cpp (run_cpp=True)", s_cpp), ("py  (run_cpp=False)", s_py)]

print("\n--- Results ---")
print(
    f"{'Sampler':<22}  {'Time (s)':>8}  {'SD':>8}  "
    f"{'RMSE(y)':>9}  {'RMSE(f)':>9}  {'RMSE(mu)':>9}  "
    f"{'RMSE(tau1)':>10}  {'RMSE(tau2)':>10}"
)
print("-" * 97)
for label, s in rows:
    print(
        f"{label:<22}  {s['elapsed']:>8.3f}  {s['elapsed_sd']:>8.3f}  "
        f"{s['rmse_y']:>9.4f}  {s['rmse_f']:>9.4f}  {s['rmse_mu']:>9.4f}  "
        f"{s['rmse_tau1']:>10.4f}  {s['rmse_tau2']:>10.4f}"
    )

speedup = s_py["elapsed"] / s_cpp["elapsed"]
print(f"\nSpeedup (py / cpp): {speedup:.2f}x")
print(
    f"RMSE delta (cpp - py):  "
    f"y={s_cpp['rmse_y'] - s_py['rmse_y']:.4f}  "
    f"f={s_cpp['rmse_f'] - s_py['rmse_f']:.4f}  "
    f"mu={s_cpp['rmse_mu'] - s_py['rmse_mu']:.4f}  "
    f"tau1={s_cpp['rmse_tau1'] - s_py['rmse_tau1']:.4f}  "
    f"tau2={s_cpp['rmse_tau2'] - s_py['rmse_tau2']:.4f}"
)
