"""
Replication script from:
https://github.com/StochasticTree/stochtree/issues/376
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from threadpoolctl import threadpool_limits
from stochtree import BCFModel


def confounded_dgp(n=445, p=8, n_seed=0):
    """Small confounded DGP roughly mimicking LaLonde-style earnings data."""
    rng = np.random.default_rng(n_seed)
    X = rng.normal(size=(n, p))
    e = 1 / (1 + np.exp(-(1.5 * X[:, 0] - 0.7 * X[:, 1])))   # confounded propensity
    T = rng.binomial(1, e).astype(float)
    tau = 1500 + 800 * X[:, 2] - 500 * X[:, 3]               # heterogeneous, mean ~1500
    Y0 = np.maximum(np.random.default_rng(n_seed + 1).normal(
        loc=3000 + 1500 * X[:, 0], scale=4000, size=n), 0)
    Y = Y0 + T * tau
    return X.astype(float), T, Y.astype(float)

X, T, Y = confounded_dgp(n_seed=0)

for seed in [0, 1, 7, 13, 42, 100, 2026]:
    propensity = cross_val_predict(
        HistGradientBoostingClassifier(max_iter=200, random_state=int(seed)),
        X, T.astype(int), method="predict_proba", cv=5,
    )[:, 1]
    propensity = np.clip(propensity, 0.01, 0.99)

    m = BCFModel()
    with threadpool_limits(limits=1):
        m.sample(
            X_train=X, Z_train=T, y_train=Y,
            propensity_train=propensity,
            num_gfr=5, num_burnin=200, num_mcmc=200,
            general_params={"random_seed": int(seed) % (2**31)},
        )
    tau_hat = m.predict(X=X, Z=T, propensity=propensity,
                        type="posterior", terms="cate")
    print(f"seed={seed:>4}  BCF ATE={float(np.mean(tau_hat)):>+8.0f}")

