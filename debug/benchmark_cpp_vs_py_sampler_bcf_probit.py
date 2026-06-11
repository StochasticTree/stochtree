"""Benchmark: C++ sampler loop vs. Python sampler loop -- probit BCF.

Compares runtime, Brier score on outcome, and latent-scale tau RMSE across
run_cpp=True / False in BCFModel.sample().

DGP (latent-index model):
    w = mu(X) + tau(X)*Z + eps,  eps ~ N(0, 1)
    y = 1(w > 0)
    mu(X) = 1 + 2*X[:,0] + X[:,1]           (confounded with propensity)
    tau(X) = 0.5 + X[:,2]                    (latent-scale CATE)
    pi(X) = 0.4 + 0.2*X[:,0]                (mild confounding)
    Z ~ Bernoulli(pi(X))

Metrics:
    Brier score: mean((mean_s Phi(mu_hat[i,s] + tau_hat[i,s]*Z[i]) - y[i])^2)
    RMSE(tau):   sqrt(mean((mean_s tau_hat_test[i,s] - tau_test[i])^2))

Usage:
    source venv/bin/activate
    python debug/benchmark_cpp_vs_py_sampler_bcf_probit.py
"""

import argparse
import time
import numpy as np
from scipy.stats import norm
from stochtree import BCFModel, OutcomeModel

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
X = rng.uniform(size=(n, p))

mu_X  = 1 + 2 * X[:, 0] + X[:, 1]
tau_X = 0.5 + X[:, 2]
pi_X  = 0.4 + 0.2 * X[:, 0]
Z     = rng.binomial(1, pi_X).astype(float)

w = mu_X + tau_X * Z + rng.standard_normal(n)
y = (w > 0).astype(float)

test_frac  = 0.2
n_test     = round(test_frac * n)
n_train    = n - n_test
test_inds  = rng.choice(n, size=n_test, replace=False)
train_inds = np.setdiff1d(np.arange(n), test_inds)

X_train,  X_test  = X[train_inds],    X[test_inds]
Z_train,  Z_test  = Z[train_inds],    Z[test_inds]
pi_train, pi_test = pi_X[train_inds], pi_X[test_inds]
y_train,  y_test  = y[train_inds],    y[test_inds]
tau_test          = tau_X[test_inds]

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
        num_gfr=num_gfr,
        num_burnin=num_burnin,
        num_mcmc=num_mcmc,
        general_params={
            "random_seed": seed,
            "num_chains": num_chains,
            "outcome_model": OutcomeModel(outcome="binary", link="probit"),
            "sample_sigma2_global": False,
        },
        prognostic_forest_params={
            "num_trees": num_trees_mu,
            "sample_sigma2_leaf": False,
        },
        treatment_effect_forest_params={
            "num_trees": num_trees_tau,
            "sample_sigma2_leaf": False,
            "sample_intercept": False,
        },
        run_cpp=run_cpp,
    )
    elapsed_sample = time.perf_counter() - t0

    # Request latent-scale mu and tau (scale="linear", no probit transform applied)
    t1 = time.perf_counter()
    preds = m.predict(
        X=X_test, Z=Z_test, propensity=pi_test,
        terms=["mu", "tau"], scale="linear", run_cpp=run_cpp
    )
    elapsed_predict = time.perf_counter() - t1

    # mu_hat, tau_hat: (n_test, num_samples) — latent scale
    mu_hat  = preds["mu_hat"]   # (n_test, num_samples)
    tau_hat = preds["tau_hat"]  # (n_test, num_samples)

    # P(Y=1 | X, Z, sample s) = Phi(mu_hat[i,s] + tau_hat[i,s] * Z_test[i])
    linear_pred = mu_hat + tau_hat * Z_test[:, np.newaxis]
    p_hat_samples = norm.cdf(linear_pred)          # (n_test, num_samples)
    p_hat_mean    = p_hat_samples.mean(axis=1)     # (n_test,)

    tau_hat_mean  = tau_hat.mean(axis=1)           # (n_test,)

    brier    = float(np.mean((p_hat_mean - y_test) ** 2))
    rmse_tau = float(np.sqrt(np.mean((tau_hat_mean - tau_test) ** 2)))

    return {
        "elapsed_sample":  elapsed_sample,
        "elapsed_predict": elapsed_predict,
        "brier":    brier,
        "rmse_tau": rmse_tau,
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
    keys = ["elapsed_sample", "elapsed_predict", "brier", "rmse_tau"]
    out = {k: float(np.mean([r[k] for r in results])) for k in keys}
    out["elapsed"] = out["elapsed_sample"] + out["elapsed_predict"]
    out["elapsed_sd"] = float(np.std(
        [r["elapsed_sample"] + r["elapsed_predict"] for r in results], ddof=1
    ))
    return out

s_cpp = summarise(results_cpp)
s_py  = summarise(results_py)
rows  = [("cpp (run_cpp=True)", s_cpp), ("py  (run_cpp=False)", s_py)]

print("\n--- Results ---")
print(
    f"{'Sampler':<22}  {'Total (s)':>10}  {'Samp (s)':>10}  {'Pred (s)':>10}  {'SD':>10}  "
    f"{'Brier':>10}  {'RMSE (tau)':>12}"
)
print("-" * 92)
for label, s in rows:
    print(
        f"{label:<22}  {s['elapsed']:>10.3f}  {s['elapsed_sample']:>10.3f}  "
        f"{s['elapsed_predict']:>10.3f}  {s['elapsed_sd']:>10.3f}  "
        f"{s['brier']:>10.4f}  {s['rmse_tau']:>12.4f}"
    )

speedup = s_py["elapsed"] / s_cpp["elapsed"]
print(f"\nSpeedup (py / cpp): {speedup:.2f}x")
print(
    f"Delta (cpp - py):  "
    f"brier={s_cpp['brier'] - s_py['brier']:.4f}  "
    f"rmse_tau={s_cpp['rmse_tau'] - s_py['rmse_tau']:.4f}"
)
