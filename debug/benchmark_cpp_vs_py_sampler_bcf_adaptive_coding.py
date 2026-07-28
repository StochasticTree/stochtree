"""Benchmark: C++ sampler loop vs. Python sampler loop -- BCF with adaptive coding.

Exercises SampleAdaptiveCodingParameters() for binary treatment.  Verifies:
  - The C++ path runs without error.
  - b_0_samples and b_1_samples are populated with the right shape.
  - mu_hat + Z * tau_hat == y_hat (internal decomposition check on train set).
  - CATE RMSE (cpp) is close to CATE RMSE (python) -- large differences indicate
    a residual accounting bug in SampleAdaptiveCodingParameters or the
    mu/tau prediction split.
  - Speedup is reported for reference, though the primary goal is correctness.

DGP:
    mu(X) = step function on X[:,0]
    tau_forest(X) = 2 * X[:,2]      (forest component, scale-free)
    true_b_0 = -0.5
    true_b_1 =  1.5
    CATE(X)  = (true_b_1 - true_b_0) * tau_forest(X)  = 4 * X[:,2]
    pi(X)    = 0.2 + 0.6 * X[:,3]
    Z ~ Bernoulli(pi(X))
    y = mu(X) + (true_b_0*(1-Z) + true_b_1*Z) * tau_forest(X) + noise

Usage:
    source venv/bin/activate
    python debug/benchmark_cpp_vs_py_sampler_bcf_adaptive_coding.py
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
# Shared RNG and dataset sizes
# ---------------------------------------------------------------------------
rng = np.random.default_rng(1234)

n = 2000
p = 10
noise_sd = 1.0
test_frac = 0.2
n_test = round(test_frac * n)
n_train = n - n_test

num_gfr = 10
num_burnin = 0
num_mcmc = 100
num_trees_mu = 200
num_trees_tau = 50
n_reps = 3

# ---------------------------------------------------------------------------
# DGP: binary treatment with adaptive coding
# ---------------------------------------------------------------------------
X_all = rng.uniform(size=(n, p))

mu_X = (
    np.where((X_all[:, 0] >= 0.00) & (X_all[:, 0] < 0.25), -7.5, 0) +
    np.where((X_all[:, 0] >= 0.25) & (X_all[:, 0] < 0.50), -2.5, 0) +
    np.where((X_all[:, 0] >= 0.50) & (X_all[:, 0] < 0.75),  2.5, 0) +
    np.where((X_all[:, 0] >= 0.75) & (X_all[:, 0] < 1.00),  7.5, 0)
)
TRUE_B_0 = -0.5
TRUE_B_1 =  1.5
tau_forest_X = 2.0 * X_all[:, 2]                           # forest component (no coding)
tau_X = (TRUE_B_1 - TRUE_B_0) * tau_forest_X               # CATE = (b_1 - b_0) * tau_forest
pi_X = 0.2 + 0.6 * X_all[:, 3]
Z_all = rng.binomial(1, pi_X).astype(float)
coded_basis = TRUE_B_0 * (1 - Z_all) + TRUE_B_1 * Z_all
y_all = mu_X + coded_basis * tau_forest_X + rng.normal(scale=noise_sd, size=n)

test_inds  = rng.choice(n, size=n_test, replace=False)
train_inds = np.setdiff1d(np.arange(n), test_inds)

X_train  = X_all[train_inds];   X_test  = X_all[test_inds]
Z_train  = Z_all[train_inds];   Z_test  = Z_all[test_inds]
pi_train = pi_X[train_inds];    pi_test = pi_X[test_inds]
y_train  = y_all[train_inds];   y_test  = y_all[test_inds]
mu_test  = mu_X[test_inds]
tau_test = tau_X[test_inds]     # true CATE = (b_1-b_0)*tau_forest(X_test)
# true fitted value: mu(X) + (b_0*(1-Z) + b_1*Z) * tau_forest(X)
coded_test = TRUE_B_0 * (1 - Z_test) + TRUE_B_1 * Z_test
f_test = mu_test + coded_test * tau_forest_X[test_inds]

print(
    f"n_train={n_train}  n_test={n_test}  p={p}\n"
    f"mu_trees={num_trees_mu}  tau_trees={num_trees_tau}  "
    f"num_gfr={num_gfr}  num_burnin={num_burnin}  num_mcmc={num_mcmc}  "
    f"num_chains={num_chains}  reps={n_reps}\n"
    f"true_b_0={TRUE_B_0}  true_b_1={TRUE_B_1}\n"
)

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_once(run_cpp: bool, seed: int) -> dict:
    m = BCFModel()
    t0 = time.perf_counter()
    m.sample(
        X_train=X_train,
        Z_train=Z_train,
        y_train=y_train,
        propensity_train=pi_train,
        num_gfr=num_gfr,
        num_burnin=num_burnin,
        num_mcmc=num_mcmc,
        general_params={
            "adaptive_coding": True,
            "random_seed": seed,
            "num_chains": num_chains,
            "propensity_covariate": "prognostic",
        },
        prognostic_forest_params={"num_trees": num_trees_mu},
        treatment_effect_forest_params={"num_trees": num_trees_tau},
        run_cpp=run_cpp,
    )
    elapsed_sample = time.perf_counter() - t0

    # Internal consistency check on train set (still available on the model object)
    max_decomp_err_train = float(np.max(np.abs(
        m.y_hat_train - (m.mu_hat_train + Z_train[:, None] * m.tau_hat_train)
    )))

    t1 = time.perf_counter()
    preds = m.predict(X=X_test, Z=Z_test, propensity=pi_test, run_cpp=run_cpp)
    elapsed_predict = time.perf_counter() - t1

    yhat   = preds["y_hat"].mean(axis=1)
    tauhat = preds["tau_hat"].mean(axis=1)

    max_decomp_err_test = float(np.max(np.abs(
        preds["y_hat"] - (preds["mu_hat"] + Z_test[:, None] * preds["tau_hat"])
    )))

    b0_mean = float(np.mean(m.b0_samples))
    b1_mean = float(np.mean(m.b1_samples))

    return {
        "elapsed_sample":       elapsed_sample,
        "elapsed_predict":      elapsed_predict,
        "b0_mean":              b0_mean,
        "b1_mean":              b1_mean,
        "b0_shape":             m.b0_samples.shape,
        "b1_shape":             m.b1_samples.shape,
        "max_decomp_err_train": max_decomp_err_train,
        "max_decomp_err_test":  max_decomp_err_test,
        "rmse_y":               float(np.sqrt(np.mean((yhat   - y_test) ** 2))),
        "rmse_f":               float(np.sqrt(np.mean((yhat   - f_test) ** 2))),
        "rmse_tau":             float(np.sqrt(np.mean((tauhat - tau_test) ** 2))),
    }

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
seeds = [1000 + i for i in range(1, n_reps + 1)]

print("=" * 60)
print("BINARY TREATMENT  (adaptive_coding=True)")
print(f"True b_0={TRUE_B_0}  b_1={TRUE_B_1}  CATE=(b1-b0)*tau_forest(X)")
print("=" * 60)

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

def summarise(results):
    keys = ["elapsed_sample", "elapsed_predict", "b0_mean", "b1_mean",
            "max_decomp_err_train", "max_decomp_err_test",
            "rmse_y", "rmse_f", "rmse_tau"]
    out = {k: float(np.mean([r[k] for r in results])) for k in keys}
    out["elapsed"] = out["elapsed_sample"] + out["elapsed_predict"]
    out["elapsed_sd"] = float(np.std(
        [r["elapsed_sample"] + r["elapsed_predict"] for r in results], ddof=1
    ))
    out["b0_shape"]   = results[0]["b0_shape"]
    out["b1_shape"]   = results[0]["b1_shape"]
    return out

s_cpp = summarise(results_cpp)
s_py  = summarise(results_py)

print("\n--- Results ---")
print(f"b0_samples shape  cpp={s_cpp['b0_shape']}  py={s_py['b0_shape']}")
print(f"b1_samples shape  cpp={s_cpp['b1_shape']}  py={s_py['b1_shape']}")
print()
print(f"{'Sampler':<22}  {'Total (s)':>9}  {'Samp (s)':>9}  {'Pred (s)':>9}  {'SD':>6}  "
      f"{'b_0 mean':>8}  {'b_1 mean':>8}  "
      f"{'max_decomp_tr':>13}  {'max_decomp_te':>13}  "
      f"{'RMSE(y)':>8}  {'RMSE(f)':>8}  {'RMSE(tau)':>10}")
print("-" * 140)
for label, s in [("cpp (run_cpp=True)", s_cpp), ("py  (run_cpp=False)", s_py)]:
    print(
        f"{label:<22}  {s['elapsed']:>9.3f}  {s['elapsed_sample']:>9.3f}  "
        f"{s['elapsed_predict']:>9.3f}  {s['elapsed_sd']:>6.3f}  "
        f"{s['b0_mean']:>8.4f}  {s['b1_mean']:>8.4f}  "
        f"{s['max_decomp_err_train']:>13.2e}  {s['max_decomp_err_test']:>13.2e}  "
        f"{s['rmse_y']:>8.4f}  {s['rmse_f']:>8.4f}  {s['rmse_tau']:>10.4f}"
    )
print(f"True b_0={TRUE_B_0:.4f}  b_1={TRUE_B_1:.4f}")
speedup = s_py["elapsed"] / s_cpp["elapsed"]
print(f"Speedup (py / cpp): {speedup:.2f}x")
print(
    f"RMSE delta (cpp - py):  "
    f"y={s_cpp['rmse_y'] - s_py['rmse_y']:.4f}  "
    f"f={s_cpp['rmse_f'] - s_py['rmse_f']:.4f}  "
    f"tau={s_cpp['rmse_tau'] - s_py['rmse_tau']:.4f}"
)
