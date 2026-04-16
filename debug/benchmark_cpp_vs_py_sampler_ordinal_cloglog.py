"""Benchmark: C++ sampler loop vs. Python sampler loop – ordinal cloglog BART.

Compares runtime, mean Brier score, and mean RMSE-to-truth (vs. true class
probabilities) across run_cpp=True / False in BARTModel.sample().

DGP uses 4 ordinal categories with a cloglog link.
The latent step function f(X) is on the log-log scale, and each category
boundary (gamma_k) is fixed at log(k) for k = 1, 2, 3 so the four
cumulative probabilities are P(Y <= k | X) = 1 - exp(-exp(f(X) - gamma_k)).

Usage:
    conda activate stochtree-book   # or: source venv/bin/activate
    python debug/benchmark_cpp_vs_py_sampler_ordinal_cloglog.py
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

# Latent step function on the cloglog scale
f_X = (
    np.where((X[:, 0] >= 0.00) & (X[:, 0] < 0.25), -2.0, 0.0) +
    np.where((X[:, 0] >= 0.25) & (X[:, 0] < 0.50), -0.5, 0.0) +
    np.where((X[:, 0] >= 0.50) & (X[:, 0] < 0.75),  0.5, 0.0) +
    np.where((X[:, 0] >= 0.75) & (X[:, 0] < 1.00),  1.0, 0.0)
)

# Fixed log-scale cutpoints (gamma_k); K = 4 categories => K-1 = 3 cutpoints
# gamma_0 is fixed at 0 for identifiability; gamma_1 = log(2), gamma_2 = log(3)
K = 4
gamma_true = np.array([0.0, np.log(2), np.log(3)])

# True cumulative probabilities: P(Y <= k | X) = 1 - exp(-exp(f_X - gamma_k))
# Shape: (n, K-1)
cum_prob = 1.0 - np.exp(-np.exp(f_X[:, None] - gamma_true[None, :]))

# True class probabilities: P(Y = k | X), shape (n, K)
p_X = np.column_stack([
    cum_prob[:, 0],
    cum_prob[:, 1] - cum_prob[:, 0],
    cum_prob[:, 2] - cum_prob[:, 1],
    1.0 - cum_prob[:, 2],
])

# Draw ordinal outcomes (0-indexed: 0, 1, 2, 3)
u = rng.uniform(size=n)
y = (
    (u > cum_prob[:, 0]).astype(int) +
    (u > cum_prob[:, 1]).astype(int) +
    (u > cum_prob[:, 2]).astype(int)
).astype(float)

test_frac = 0.2
n_test = round(test_frac * n)
n_train = n - n_test
test_inds  = rng.choice(n, size=n_test, replace=False)
train_inds = np.setdiff1d(np.arange(n), test_inds)

X_train, X_test = X[train_inds], X[test_inds]
y_train, y_test = y[train_inds], y[test_inds]
p_test = p_X[test_inds]  # (n_test, K) true class probabilities

# ---------------------------------------------------------------------------
# Benchmark settings
# ---------------------------------------------------------------------------
num_gfr   = 10
num_mcmc  = 100
num_trees = 200
n_reps    = 3

print(
    f"K={K}  n_train={n_train}  n_test={n_test}  p={p}  "
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
            "outcome_model": OutcomeModel(outcome="ordinal", link="cloglog"),
            "sample_sigma2_global": False,
        },
        run_cpp=run_cpp,
    )
    elapsed = time.perf_counter() - t0

    # predict() returns a dict; for ordinal probability scale the value for
    # "y_hat" has shape (n_test, K, num_mcmc)
    preds = m.predict(X=X_test, scale="probability")
    p_hat = preds["y_hat"].mean(axis=2)  # (n_test, K) posterior mean

    # Mean Brier score across all cells
    brier  = float(np.mean((p_hat - p_test) ** 2))
    # Per-class RMSE vs. true probs, averaged over classes
    rmse_p = float(np.mean(np.sqrt(np.mean((p_hat - p_test) ** 2, axis=0))))

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
        "elapsed_mean": float(np.mean(elapsed)),
        "elapsed_sd":   float(np.std(elapsed, ddof=1)),
        "brier_mean":   float(np.mean(brier)),
        "rmse_p_mean":  float(np.mean(rmse_p)),
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
