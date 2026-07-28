# Benchmark cloglog BART in Python using shared data (multiple replications)
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from stochtree import BARTModel, OutcomeModel

# Load benchmark data
script_dir = "tools/debug"
train = np.genfromtxt(
    os.path.join(script_dir, "cloglog_benchmark_train.csv"),
    delimiter=",", skip_header=1,
)
test = np.genfromtxt(
    os.path.join(script_dir, "cloglog_benchmark_test.csv"),
    delimiter=",", skip_header=1,
)
true_probs_train = np.genfromtxt(
    os.path.join(script_dir, "cloglog_benchmark_true_probs_train.csv"),
    delimiter=",", skip_header=1,
)
true_probs_test = np.genfromtxt(
    os.path.join(script_dir, "cloglog_benchmark_true_probs_test.csv"),
    delimiter=",", skip_header=1,
)

# Extract X and y
p = train.shape[1] - 1
X_train = train[:, :p]
y_train = train[:, p]
X_test = test[:, :p]
y_test = test[:, p]
n_categories = len(np.unique(y_train))

print(f"Train: {X_train.shape[0]} obs, Test: {X_test.shape[0]} obs")
unique, counts = np.unique(y_train, return_counts=True)
print(f"Outcome distribution: {dict(zip(unique.astype(int), counts))}")

# Run multiple replications with explicit C++ seeds
n_reps = 10
seeds = list(range(1, n_reps + 1))
train_cors = np.zeros((n_reps, n_categories))
test_cors = np.zeros((n_reps, n_categories))

for rep, seed in enumerate(seeds):
    start_time = time.time()
    bart_model = BARTModel()
    bart_model.sample(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        num_gfr=0,
        num_burnin=1000,
        num_mcmc=1000,
        general_params=dict(
            cutpoint_grid_size=100,
            sample_sigma2_global=False,
            keep_every=1,
            num_chains=1,
            random_seed=seed,
            outcome_model=OutcomeModel(outcome="ordinal", link="cloglog"),
        ),
        mean_forest_params=dict(num_trees=50),
    )
    runtime = time.time() - start_time

    est_probs_train = bart_model.predict(X=X_train, scale="probability", terms="y_hat")
    est_probs_test = bart_model.predict(X=X_test, scale="probability", terms="y_hat")

    for j in range(n_categories):
        train_cors[rep, j] = np.corrcoef(
            true_probs_train[:, j], np.mean(est_probs_train[:, j, :], axis=1)
        )[0, 1]
        test_cors[rep, j] = np.corrcoef(
            true_probs_test[:, j], np.mean(est_probs_test[:, j, :], axis=1)
        )[0, 1]

    print(f"Rep {rep + 1} (seed={seed}, {runtime:.1f}s): "
          f"train=[{', '.join(f'{c:.4f}' for c in train_cors[rep, :])}] "
          f"test=[{', '.join(f'{c:.4f}' for c in test_cors[rep, :])}]")

# Summary
print("\n--- Summary across replications ---")
print(f"{'':>20s}  {'Cat 1':>12s}  {'Cat 2':>12s}  {'Cat 3':>12s}")
print(f"{'Train mean':>20s}  {train_cors[:, 0].mean():>12.4f}  {train_cors[:, 1].mean():>12.4f}  {train_cors[:, 2].mean():>12.4f}")
print(f"{'Train std':>20s}  {train_cors[:, 0].std():>12.4f}  {train_cors[:, 1].std():>12.4f}  {train_cors[:, 2].std():>12.4f}")
print(f"{'Test mean':>20s}  {test_cors[:, 0].mean():>12.4f}  {test_cors[:, 1].mean():>12.4f}  {test_cors[:, 2].mean():>12.4f}")
print(f"{'Test std':>20s}  {test_cors[:, 0].std():>12.4f}  {test_cors[:, 1].std():>12.4f}  {test_cors[:, 2].std():>12.4f}")
