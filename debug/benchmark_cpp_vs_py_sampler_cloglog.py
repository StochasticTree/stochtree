"""Benchmark: C++ sampler loop vs. Python sampler loop – cloglog BART.

Compares runtime, Brier score, and RMSE-to-truth (vs. true P(Y=1|X)) across
run_cpp=True / False in BARTModel.sample().

DGP uses the cloglog link: P(Y=1|X) = 1 - exp(-exp(f(X))).
The step function for f(X) is kept in the range [-2, 1] so that the implied
probabilities span roughly 0.13 to 0.93 and are well-identified.

Usage:
    conda activate stochtree-book   # or: source venv/bin/activate
    python debug/benchmark_cpp_vs_py_sampler_cloglog.py
"""

import time
import numpy as np
from stochtree import BARTModel, OutcomeModel

# ---------------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------------
rng = np.random.default_rng(1234)

n = 2000
p = 10
X = rng.uniform(size=(n, p))

# Latent mean on the cloglog (log-log) scale.
# P(Y=1|X) = 1 - exp(-exp(f_X)); values chosen so probabilities are moderate.
f_X = (
    np.where((X[:, 0] >= 0.00) & (X[:, 0] < 0.25), -2.0, 0) +
    np.where((X[:, 0] >= 0.25) & (X[:, 0] < 0.50), -0.5, 0) +
    np.where((X[:, 0] >= 0.50) & (X[:, 0] < 0.75),  0.5, 0) +
    np.where((X[:, 0] >= 0.75) & (X[:, 0] < 1.00),  1.0, 0)
)
p_X = 1.0 - np.exp(-np.exp(f_X))          # true P(Y = 1 | X)
y = rng.binomial(1, p_X).astype(float)    # observed binary outcome

test_frac = 0.2
n_test = round(test_frac * n)
n_train = n - n_test
test_inds  = rng.choice(n, size=n_test, replace=False)
train_inds = np.setdiff1d(np.arange(n), test_inds)

X_train, X_test = X[train_inds], X[test_inds]
y_train, y_test = y[train_inds], y[test_inds]
p_test = p_X[test_inds]

# ---------------------------------------------------------------------------
# Benchmark settings
# ---------------------------------------------------------------------------
num_gfr   = 10
num_mcmc  = 100
num_trees = 200
n_reps    = 3

print(
    f"n_train={n_train}  n_test={n_test}  p={p}  "
    f"num_trees={num_trees}  num_gfr={num_gfr}  num_mcmc={num_mcmc}  reps={n_reps}\n"
)

# ---------------------------------------------------------------------------
# Helper: run one configuration and return timing + metrics
# ---------------------------------------------------------------------------
def run_once(run_cpp: bool, num_gfr: int, num_mcmc: int, seed: int) -> dict:
    m = BARTModel()
    t0 = time.perf_counter()
    m.sample(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        num_gfr=num_gfr,
        num_burnin=0,
        num_mcmc=num_mcmc,
        mean_forest_params={"num_trees": num_trees},
        general_params={
            "random_seed": seed,
            "outcome_model": OutcomeModel(outcome="binary", link="cloglog"),
            "sample_sigma2_global": False,
        },
        run_cpp=run_cpp,
    )
    elapsed = time.perf_counter() - t0

    # Posterior-mean predicted probability on the test set
    preds = m.predict(X=X_test, scale="probability")
    p_hat = preds["y_hat"].mean(axis=1)  # (n_test,)

    brier  = float(np.mean((p_hat - y_test) ** 2))
    rmse_p = float(np.sqrt(np.mean((p_hat - p_test) ** 2)))

    return {"elapsed": elapsed, "brier": brier, "rmse_p": rmse_p}

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------
seeds = [1000 + i for i in range(1, n_reps + 1)]

results_cpp = []
results_py  = []

print("Running C++ sampler (run_cpp=True)...")
for i, seed in enumerate(seeds, 1):
    print(f"  rep {i}/{n_reps}")
    results_cpp.append(run_once(run_cpp=True,  num_gfr=num_gfr, num_mcmc=num_mcmc, seed=seed))

print("\nRunning Python sampler (run_cpp=False)...")
for i, seed in enumerate(seeds, 1):
    print(f"  rep {i}/{n_reps}")
    results_py.append(run_once(run_cpp=False, num_gfr=num_gfr, num_mcmc=num_mcmc, seed=seed))

# ---------------------------------------------------------------------------
# Summarise
# ---------------------------------------------------------------------------
def summarise(results: list) -> dict:
    elapsed = [r["elapsed"] for r in results]
    brier   = [r["brier"]   for r in results]
    rmse_p  = [r["rmse_p"]  for r in results]
    return {
        "elapsed_mean": np.mean(elapsed),
        "elapsed_sd":   np.std(elapsed, ddof=1),
        "brier_mean":   np.mean(brier),
        "rmse_p_mean":  np.mean(rmse_p),
    }

s_cpp = summarise(results_cpp)
s_py  = summarise(results_py)
rows  = [("cpp (run_cpp=True)", s_cpp), ("py  (run_cpp=False)", s_py)]

print("\n--- Results ---")
print(
    f"{'Sampler':<22}  {'Time (s)':>10}  {'SD':>10}  "
    f"{'Brier':>12}  {'RMSE (vs truth)':>15}"
)
print("-" * 75)
for label, s in rows:
    print(
        f"{label:<22}  {s['elapsed_mean']:>10.3f}  {s['elapsed_sd']:>10.3f}"
        f"  {s['brier_mean']:>12.4f}  {s['rmse_p_mean']:>15.4f}"
    )

speedup = s_py["elapsed_mean"] / s_cpp["elapsed_mean"]
print(f"\nSpeedup (py / cpp): {speedup:.2f}x")
print(
    f"Brier delta (cpp - py):  {s_cpp['brier_mean']  - s_py['brier_mean']:.4f}\n"
    f"RMSE-p delta (cpp - py): {s_cpp['rmse_p_mean'] - s_py['rmse_p_mean']:.4f}"
)
