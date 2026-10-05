"""Check that the MCMC tree sampler targets the exact posterior on a tiny problem.

One tree, one covariate taking values 1, 2, 3, 4, outcomes all zero, alpha=0.6,
beta=2, min_samples_leaf=1, and global and leaf variances fixed at 1. With so few
observations every tree can be enumerated, so the posterior over the number of
leaves is known exactly, and a long chain must reproduce it.

The sampler's target over trees is

    p(T | y)  ∝  prod_internal [ pg(d) / (width of the node's covariate range) ]
                * prod_leaves  [ (1 - pg(d)) * (1 + n_leaf)^(-1/2) ],   pg(d) = alpha (1+d)^-beta

where 1/width is the uniform cutpoint density integrated over a unit gap between
adjacent integers, and (1 + n)^(-1/2) is a leaf's marginal likelihood for zero
outcomes, up to a constant shared by every tree.

The covariate is fed in ascending and descending order: the posterior is the same,
but the order changes which observation each node scans first when computing its
split range, which is how the GH #425 extrema bug shows up.
"""

import numpy as np
import pytest

from stochtree import BARTModel

X_VALUES = np.array([1.0, 2.0, 3.0, 4.0])
ALPHA, BETA = 0.6, 2.0


def exact_leaf_distribution():
    """Posterior probability of 1..4 leaves, by enumerating every tree."""

    def pg(depth):
        return ALPHA * (1 + depth) ** -BETA

    def node(lo, hi, depth):
        # {num_leaves: unnormalized posterior mass} for the subtree on points lo..hi-1
        out = {1: (1 - pg(depth)) * (1 + hi - lo) ** -0.5}
        width = X_VALUES[hi - 1] - X_VALUES[lo]
        for cut in range(lo + 1, hi):
            weight = pg(depth) * (X_VALUES[cut] - X_VALUES[cut - 1]) / width
            for kl, ml in node(lo, cut, depth + 1).items():
                for kr, mr in node(cut, hi, depth + 1).items():
                    out[kl + kr] = out.get(kl + kr, 0.0) + weight * ml * mr
        return out

    mass = node(0, len(X_VALUES), 0)
    total = sum(mass.values())
    return np.array([mass.get(k, 0.0) / total for k in range(1, len(X_VALUES) + 1)])


def test_exact_leaf_distribution():
    # Hand-checkable anchor for the enumeration itself
    np.testing.assert_allclose(
        exact_leaf_distribution(), [0.50183, 0.42176, 0.07196, 0.00446], atol=1e-5
    )


@pytest.mark.parametrize("descending", [False, True], ids=["ascending", "descending"])
def test_mcmc_matches_exact_posterior(descending):
    X = (X_VALUES[::-1] if descending else X_VALUES).reshape(-1, 1)
    model = BARTModel()
    model.sample(
        X_train=X,
        y_train=np.zeros(len(X_VALUES)),
        num_gfr=0,
        num_burnin=5000,
        num_mcmc=100000,
        general_params={
            "standardize": False,
            "sample_sigma2_global": False,
            "sigma2_init": 1.0,
            "random_seed": 20260907,
        },
        mean_forest_params={
            "num_trees": 1,
            "alpha": ALPHA,
            "beta": BETA,
            "min_samples_leaf": 1,
            "sample_sigma2_leaf": False,
            "sigma2_leaf_init": 1.0,
        },
    )
    forests = model.forest_container_mean
    leaves = np.array([forests.num_forest_leaves(s) for s in range(forests.num_samples())])
    observed = np.array([np.mean(leaves == k) for k in range(1, len(X_VALUES) + 1)])

    # With 100k draws the Monte Carlo standard error of each share is under 0.2
    # percentage points; the pre-fix sampler missed by 8-10 points (or never split).
    np.testing.assert_allclose(observed, exact_leaf_distribution(), atol=0.01)
