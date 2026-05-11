"""Benchmark: C++ sampler loop vs. Python sampler loop -- BCF with treatment intercept.

Exercises SampleParametricTreatmentEffect() for both univariate and multivariate
treatment by toggling sample_intercept=True.  Verifies:
  - The C++ path runs without error.
  - tau_0_samples is populated and has the right shape.
  - CATE RMSE (cpp) is close to CATE RMSE (python) -- large differences indicate
    a residual accounting bug in the new intercept step.
  - Speedup is reported for reference, though the primary goal here is correctness.

DGP (univariate):
    mu(X) = step function on X[:,0]
    tau_0  = 1.5  (true global intercept)
    tau(X) = tau_0 + 2*X[:,2]  (full CATE)
    pi(X)  = 0.2 + 0.6*X[:,3]
    Z ~ Bernoulli(pi(X))
    y = mu(X) + (tau_0 + tau(X)) * Z + noise

DGP (multivariate, treatment_dim=2):
    Same mu(X); tau_0 = [0.5, 1.0]; tau_k(X) = tau_0[k] + X[:,k+1]

Usage:
    source venv/bin/activate
    python debug/benchmark_cpp_vs_py_sampler_bcf_tau_intercept.py
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
# DGP: univariate binary treatment with global tau_0
# ---------------------------------------------------------------------------
X_all = rng.uniform(size=(n, p))

mu_X = (
    np.where((X_all[:, 0] >= 0.00) & (X_all[:, 0] < 0.25), -7.5, 0) +
    np.where((X_all[:, 0] >= 0.25) & (X_all[:, 0] < 0.50), -2.5, 0) +
    np.where((X_all[:, 0] >= 0.50) & (X_all[:, 0] < 0.75),  2.5, 0) +
    np.where((X_all[:, 0] >= 0.75) & (X_all[:, 0] < 1.00),  7.5, 0)
)
TRUE_TAU0_UNIVARIATE = 1.5
tau_forest_X = 2.0 * X_all[:, 2]                        # forest component only
tau_X = TRUE_TAU0_UNIVARIATE + tau_forest_X              # full CATE = tau_0 + tau(X)
pi_X = 0.2 + 0.6 * X_all[:, 3]
Z_all = rng.binomial(1, pi_X).astype(float)
y_all = mu_X + tau_X * Z_all + rng.normal(scale=noise_sd, size=n)

test_inds = rng.choice(n, size=n_test, replace=False)
train_inds = np.setdiff1d(np.arange(n), test_inds)

X_train_u  = X_all[train_inds];   X_test_u  = X_all[test_inds]
Z_train_u  = Z_all[train_inds];   Z_test_u  = Z_all[test_inds]
pi_train_u = pi_X[train_inds];    pi_test_u = pi_X[test_inds]
y_train_u  = y_all[train_inds];   y_test_u  = y_all[test_inds]
mu_test_u  = mu_X[test_inds]
tau_test_u = tau_X[test_inds]
f_test_u   = mu_test_u + tau_test_u * Z_test_u

# ---------------------------------------------------------------------------
# DGP: multivariate (2-column) treatment with per-arm global tau_0
# ---------------------------------------------------------------------------
X_all_mv = rng.uniform(size=(n, p))
TRUE_TAU0_MV = np.array([0.5, 1.0])
pi_mv = np.c_[0.25 + 0.5 * X_all_mv[:, 0], 0.75 - 0.5 * X_all_mv[:, 1]]
mu_mv = pi_mv[:, 0] * 5 + pi_mv[:, 1] * 2 + 2 * X_all_mv[:, 2]
tau_forest_mv = np.c_[X_all_mv[:, 1], X_all_mv[:, 2]]           # forest component only
tau_mv = TRUE_TAU0_MV + tau_forest_mv                            # full CATE
Z_mv = (rng.uniform(size=(n, 2)) < pi_mv).astype(float)
y_mv = mu_mv + (Z_mv * tau_mv).sum(axis=1) + rng.normal(size=n) * noise_sd

test_inds_mv  = rng.choice(n, size=n_test, replace=False)
train_inds_mv = np.setdiff1d(np.arange(n), test_inds_mv)

X_train_mv  = X_all_mv[train_inds_mv];  X_test_mv  = X_all_mv[test_inds_mv]
Z_train_mv  = Z_mv[train_inds_mv];      Z_test_mv  = Z_mv[test_inds_mv]
pi_train_mv = pi_mv[train_inds_mv];     pi_test_mv = pi_mv[test_inds_mv]
y_train_mv  = y_mv[train_inds_mv];      y_test_mv  = y_mv[test_inds_mv]
mu_test_mv  = mu_mv[test_inds_mv]
tau_test_mv = tau_mv[test_inds_mv]
f_test_mv   = mu_test_mv + (Z_test_mv * tau_test_mv).sum(axis=1)

print(
    f"n_train={n_train}  n_test={n_test}  p={p}\n"
    f"mu_trees={num_trees_mu}  tau_trees={num_trees_tau}  "
    f"num_gfr={num_gfr}  num_burnin={num_burnin}  num_mcmc={num_mcmc}  "
    f"num_chains={num_chains}  reps={n_reps}\n"
)

# ---------------------------------------------------------------------------
# Runner: univariate treatment with sample_intercept=True
# ---------------------------------------------------------------------------
def run_once_univariate(run_cpp: bool, seed: int) -> dict:
    m = BCFModel()
    t0 = time.perf_counter()
    m.sample(
        X_train=X_train_u,
        Z_train=Z_train_u,
        y_train=y_train_u,
        propensity_train=pi_train_u,
        X_test=X_test_u,
        Z_test=Z_test_u,
        propensity_test=pi_test_u,
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
            "sample_intercept": True,
        },
        run_cpp=run_cpp,
    )
    elapsed = time.perf_counter() - t0

    # tau_0_samples shape: (1, num_mcmc) for univariate
    tau_0_shape = getattr(m, "tau_0_samples", None)
    tau_0_mean = float(np.mean(m.tau_0_samples)) if tau_0_shape is not None else float("nan")

    y_hat   = m.y_hat_test.mean(axis=1)
    tau_hat = m.tau_hat_test.mean(axis=1)

    return {
        "elapsed":   elapsed,
        "tau_0_mean": tau_0_mean,
        "tau_0_shape": tuple(m.tau_0_samples.shape) if tau_0_shape is not None else None,
        "rmse_y":    float(np.sqrt(np.mean((y_hat   - y_test_u)   ** 2))),
        "rmse_f":    float(np.sqrt(np.mean((y_hat   - f_test_u)   ** 2))),
        "rmse_tau":  float(np.sqrt(np.mean((tau_hat - tau_test_u) ** 2))),
    }

# ---------------------------------------------------------------------------
# Runner: multivariate (2-column) treatment with sample_intercept=True
# ---------------------------------------------------------------------------
def run_once_multivariate(run_cpp: bool, seed: int) -> dict:
    m = BCFModel()
    t0 = time.perf_counter()
    m.sample(
        X_train=X_train_mv,
        Z_train=Z_train_mv,
        y_train=y_train_mv,
        propensity_train=pi_train_mv,
        X_test=X_test_mv,
        Z_test=Z_test_mv,
        propensity_test=pi_test_mv,
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
            "sample_intercept": True,
        },
        run_cpp=run_cpp,
    )
    elapsed = time.perf_counter() - t0

    # tau_0_samples shape: (2, num_mcmc) for treatment_dim=2
    tau_0_samples = getattr(m, "tau_0_samples", None)
    tau_0_mean = (
        m.tau_0_samples.mean(axis=1).tolist() if tau_0_samples is not None else [float("nan")] * 2
    )

    tau_hat1 = m.tau_hat_test[:, 0, :].mean(axis=1)
    tau_hat2 = m.tau_hat_test[:, 1, :].mean(axis=1)
    y_hat    = m.y_hat_test.mean(axis=1)

    return {
        "elapsed":    elapsed,
        "tau_0_mean": tau_0_mean,
        "tau_0_shape": tuple(m.tau_0_samples.shape) if tau_0_samples is not None else None,
        "rmse_y":     float(np.sqrt(np.mean((y_hat    - y_test_mv)         ** 2))),
        "rmse_f":     float(np.sqrt(np.mean((y_hat    - f_test_mv)         ** 2))),
        "rmse_tau1":  float(np.sqrt(np.mean((tau_hat1 - tau_test_mv[:, 0]) ** 2))),
        "rmse_tau2":  float(np.sqrt(np.mean((tau_hat2 - tau_test_mv[:, 1]) ** 2))),
    }

# ---------------------------------------------------------------------------
# Run: univariate
# ---------------------------------------------------------------------------
seeds = [1000 + i for i in range(1, n_reps + 1)]

print("=" * 60)
print("UNIVARIATE TREATMENT  (sample_intercept=True)")
print(f"True tau_0 = {TRUE_TAU0_UNIVARIATE}")
print("=" * 60)

results_cpp_u = []
results_py_u  = []

print("Running C++ sampler (run_cpp=True)...")
for i, seed in enumerate(seeds, 1):
    print(f"  rep {i}/{n_reps}")
    results_cpp_u.append(run_once_univariate(run_cpp=True, seed=seed))

print("\nRunning Python sampler (run_cpp=False)...")
for i, seed in enumerate(seeds, 1):
    print(f"  rep {i}/{n_reps}")
    results_py_u.append(run_once_univariate(run_cpp=False, seed=seed))

def summarise_u(results):
    keys = ["elapsed", "rmse_y", "rmse_f", "rmse_tau"]
    out = {k: float(np.mean([r[k] for r in results])) for k in keys}
    out["elapsed_sd"]  = float(np.std([r["elapsed"]    for r in results], ddof=1))
    out["tau_0_mean"]  = float(np.mean([r["tau_0_mean"] for r in results]))
    out["tau_0_shape"] = results[0]["tau_0_shape"]
    return out

s_cpp_u = summarise_u(results_cpp_u)
s_py_u  = summarise_u(results_py_u)

print("\n--- Univariate Results ---")
print(f"tau_0_samples shape  cpp={s_cpp_u['tau_0_shape']}  py={s_py_u['tau_0_shape']}")
print(
    f"{'Sampler':<22}  {'Time (s)':>8}  {'SD':>8}  "
    f"{'tau_0 mean':>10}  {'RMSE(y)':>9}  {'RMSE(f)':>9}  {'RMSE(tau)':>10}"
)
print("-" * 90)
for label, s in [("cpp (run_cpp=True)", s_cpp_u), ("py  (run_cpp=False)", s_py_u)]:
    print(
        f"{label:<22}  {s['elapsed']:>8.3f}  {s['elapsed_sd']:>8.3f}  "
        f"{s['tau_0_mean']:>10.4f}  {s['rmse_y']:>9.4f}  {s['rmse_f']:>9.4f}  {s['rmse_tau']:>10.4f}"
    )
print(f"True tau_0:  {TRUE_TAU0_UNIVARIATE:.4f}")
speedup_u = s_py_u["elapsed"] / s_cpp_u["elapsed"]
print(f"Speedup (py / cpp): {speedup_u:.2f}x")
print(
    f"RMSE delta (cpp - py):  "
    f"y={s_cpp_u['rmse_y'] - s_py_u['rmse_y']:.4f}  "
    f"f={s_cpp_u['rmse_f'] - s_py_u['rmse_f']:.4f}  "
    f"tau={s_cpp_u['rmse_tau'] - s_py_u['rmse_tau']:.4f}"
)

# ---------------------------------------------------------------------------
# Run: multivariate
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("MULTIVARIATE TREATMENT  (treatment_dim=2, sample_intercept=True)")
print(f"True tau_0 = {TRUE_TAU0_MV.tolist()}")
print("=" * 60)

results_cpp_mv = []
results_py_mv  = []

print("Running C++ sampler (run_cpp=True)...")
for i, seed in enumerate(seeds, 1):
    print(f"  rep {i}/{n_reps}")
    results_cpp_mv.append(run_once_multivariate(run_cpp=True, seed=seed))

print("\nRunning Python sampler (run_cpp=False)...")
for i, seed in enumerate(seeds, 1):
    print(f"  rep {i}/{n_reps}")
    results_py_mv.append(run_once_multivariate(run_cpp=False, seed=seed))

def summarise_mv(results):
    keys = ["elapsed", "rmse_y", "rmse_f", "rmse_tau1", "rmse_tau2"]
    out = {k: float(np.mean([r[k] for r in results])) for k in keys}
    out["elapsed_sd"]   = float(np.std([r["elapsed"] for r in results], ddof=1))
    out["tau_0_mean_0"] = float(np.mean([r["tau_0_mean"][0] for r in results]))
    out["tau_0_mean_1"] = float(np.mean([r["tau_0_mean"][1] for r in results]))
    out["tau_0_shape"]  = results[0]["tau_0_shape"]
    return out

s_cpp_mv = summarise_mv(results_cpp_mv)
s_py_mv  = summarise_mv(results_py_mv)

print("\n--- Multivariate Results ---")
print(f"tau_0_samples shape  cpp={s_cpp_mv['tau_0_shape']}  py={s_py_mv['tau_0_shape']}")
print(
    f"{'Sampler':<22}  {'Time (s)':>8}  {'SD':>8}  "
    f"{'tau_0[0]':>9}  {'tau_0[1]':>9}  "
    f"{'RMSE(y)':>8}  {'RMSE(f)':>8}  {'RMSE(tau1)':>10}  {'RMSE(tau2)':>10}"
)
print("-" * 105)
for label, s in [("cpp (run_cpp=True)", s_cpp_mv), ("py  (run_cpp=False)", s_py_mv)]:
    print(
        f"{label:<22}  {s['elapsed']:>8.3f}  {s['elapsed_sd']:>8.3f}  "
        f"{s['tau_0_mean_0']:>9.4f}  {s['tau_0_mean_1']:>9.4f}  "
        f"{s['rmse_y']:>8.4f}  {s['rmse_f']:>8.4f}  {s['rmse_tau1']:>10.4f}  {s['rmse_tau2']:>10.4f}"
    )
print(f"True tau_0:  [{TRUE_TAU0_MV[0]:.4f}, {TRUE_TAU0_MV[1]:.4f}]")
speedup_mv = s_py_mv["elapsed"] / s_cpp_mv["elapsed"]
print(f"Speedup (py / cpp): {speedup_mv:.2f}x")
print(
    f"RMSE delta (cpp - py):  "
    f"y={s_cpp_mv['rmse_y'] - s_py_mv['rmse_y']:.4f}  "
    f"f={s_cpp_mv['rmse_f'] - s_py_mv['rmse_f']:.4f}  "
    f"tau1={s_cpp_mv['rmse_tau1'] - s_py_mv['rmse_tau1']:.4f}  "
    f"tau2={s_cpp_mv['rmse_tau2'] - s_py_mv['rmse_tau2']:.4f}"
)
