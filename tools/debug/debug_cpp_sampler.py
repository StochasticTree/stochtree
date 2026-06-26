"""Minimal script for debugging the C++ sampler under lldb.

Usage:
    source venv/bin/activate
    lldb -- python debug/debug_cpp_sampler.py
    # then at the (lldb) prompt:
    #   run
    #   bt          (after the crash, to get a backtrace)
    #   frame info  (to see the crashing frame)
"""

import numpy as np
from stochtree import BARTModel

seed  = 1001
n     = 10000
p     = 10
rng   = np.random.default_rng(1234)

X     = rng.uniform(size=(n, p))
f_X   = (
    np.where((X[:, 0] >= 0.00) & (X[:, 0] < 0.25), -7.5, 0) +
    np.where((X[:, 0] >= 0.25) & (X[:, 0] < 0.50), -2.5, 0) +
    np.where((X[:, 0] >= 0.50) & (X[:, 0] < 0.75),  2.5, 0) +
    np.where((X[:, 0] >= 0.75) & (X[:, 0] < 1.00),  7.5, 0)
)
y     = f_X + rng.normal(scale=1.0, size=n)

n_test    = round(0.2 * n)
test_inds = rng.choice(n, size=n_test, replace=False)
train_inds = np.setdiff1d(np.arange(n), test_inds)

X_train, X_test = X[train_inds], X[test_inds]
y_train, y_test = y[train_inds], y[test_inds]

print(f"n_train={len(train_inds)}  n_test={n_test}  p={p}  seed={seed}")
print("Calling BARTModel.sample() with run_cpp=True ...")

m = BARTModel()
m.sample(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    num_gfr=10,
    num_burnin=0,
    num_mcmc=100,
    general_params={"random_seed": seed},
    mean_forest_params={"num_trees": 200},
    run_cpp=True,
)

print("Completed successfully.")
print(f"y_hat_test shape: {m.y_hat_test.shape}")
