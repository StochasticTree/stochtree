#!/usr/bin/env python3
"""
Generate cross-language parity fixtures using Python stochtree.

Fixture layout for each scenario <name>
-----------------------------------------
  <name>.json            metadata: seed, n_train, n_test, p, num_gfr, num_burnin,
                         num_mcmc, and any model-specific params
  <name>_<term>.csv      (n_obs x num_mcmc) float64 matrix, no header, comma-separated
                         rows = observations, columns = posterior draws

Scenarios
---------
  bart_basic      continuous BART, no RFX
                  terms: yhat_train, yhat_test
  bart_rfx        continuous BART with group random effects              [TODO]
                  terms: yhat_train, yhat_test, rfx_train, rfx_test
  bcf_basic       BCF with sample_tau_0=True, no variance forest        [TODO]
                  terms: yhat_train, yhat_test, tau_train, tau_test,
                         mu_train, mu_test
  bcf_varforest   BCF with variance forest                              [TODO]
                  terms: same as bcf_basic + sigma2x_train, sigma2x_test

Usage
-----
  python test/cross_language/generate_predictions.py [--output-dir DIR]
"""
import argparse
import json
import os

import numpy as np

# Fixed seed used for both data generation and model sampling.
# R side must use the same seed value.
GLOBAL_SEED = 42


def write_matrix(path: str, mat: np.ndarray) -> None:
    """Write (n, S) float64 matrix as CSV with no header."""
    np.savetxt(path, mat, delimiter=",", fmt="%.17g")


def write_metadata(path: str, meta: dict) -> None:
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Scenario: bart_basic
# ---------------------------------------------------------------------------

def scenario_bart_basic(output_dir: str) -> None:
    """Continuous BART, no RFX. Writes X_train, y_train, X_test, and prediction matrices."""
    from stochtree import BARTModel

    n = 250
    p = 5
    n_test = 50
    n_train = n - n_test
    num_gfr = 10
    num_burnin = 10
    num_mcmc = 50

    # Generate data — written to fixture so R reads identical inputs
    rng = np.random.default_rng(GLOBAL_SEED)
    X = rng.uniform(size=(n, p))
    f_x = np.sin(np.pi * X[:, 0]) * X[:, 1]
    y = f_x + rng.normal(scale=0.5, size=n)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train = y[:n_train]

    write_matrix(os.path.join(output_dir, "bart_basic_X_train.csv"), X_train)
    write_matrix(os.path.join(output_dir, "bart_basic_y_train.csv"), y_train.reshape(-1, 1))
    write_matrix(os.path.join(output_dir, "bart_basic_X_test.csv"), X_test)

    # Fit model and write predictions
    model = BARTModel()
    model.sample(
        X_train, y_train,
        num_gfr=num_gfr,
        num_burnin=num_burnin,
        num_mcmc=num_mcmc,
        general_params={"random_seed": GLOBAL_SEED},
    )
    write_matrix(
        os.path.join(output_dir, "bart_basic_yhat_train.csv"),
        model.predict(X_train, terms="y_hat"),
    )
    write_matrix(
        os.path.join(output_dir, "bart_basic_yhat_test.csv"),
        model.predict(X_test, terms="y_hat"),
    )

    meta = {
        "scenario": "bart_basic",
        "seed": GLOBAL_SEED,
        "n": n,
        "p": p,
        "n_train": n_train,
        "n_test": n_test,
        "num_gfr": num_gfr,
        "num_burnin": num_burnin,
        "num_mcmc": num_mcmc,
        "params": {},
    }
    write_metadata(os.path.join(output_dir, "bart_basic.json"), meta)

    print("  bart_basic — OK")


# ---------------------------------------------------------------------------
# Scenario: bcf_basic
# ---------------------------------------------------------------------------

def scenario_bcf_basic(output_dir: str) -> None:
    """BCF with sample_tau_0=True, no variance forest, no RFX.

    DGP: mu(X) and tau(X) are step functions of X[:,0] and X[:,1] respectively,
    propensity pi(X) is a step function of X[:,0].  Matches the example in the
    predict.bcfmodel Rd docs so the scenario is easy to reason about.

    Terms written: yhat, tau (= tau_0 + tau(X)), mu (prognostic forest).
    """
    from stochtree import BCFModel

    n = 500
    p = 5
    n_test = 100
    n_train = n - n_test
    num_gfr = 10
    num_burnin = 10
    num_mcmc = 50

    rng = np.random.default_rng(GLOBAL_SEED)
    X = rng.uniform(size=(n, p))

    # Step-function propensity and DGP (canonical BCF example)
    pi_x = (
        ((X[:, 0] >= 0.00) & (X[:, 0] < 0.25)) * 0.2
        + ((X[:, 0] >= 0.25) & (X[:, 0] < 0.50)) * 0.4
        + ((X[:, 0] >= 0.50) & (X[:, 0] < 0.75)) * 0.6
        + ((X[:, 0] >= 0.75) & (X[:, 0] <= 1.00)) * 0.8
    )
    Z = rng.binomial(1, pi_x).astype(float)
    mu_x = (
        ((X[:, 0] >= 0.00) & (X[:, 0] < 0.25)) * (-7.5)
        + ((X[:, 0] >= 0.25) & (X[:, 0] < 0.50)) * (-2.5)
        + ((X[:, 0] >= 0.50) & (X[:, 0] < 0.75)) * 2.5
        + ((X[:, 0] >= 0.75) & (X[:, 0] <= 1.00)) * 7.5
    )
    tau_x = (
        ((X[:, 1] >= 0.00) & (X[:, 1] < 0.25)) * 0.5
        + ((X[:, 1] >= 0.25) & (X[:, 1] < 0.50)) * 1.0
        + ((X[:, 1] >= 0.50) & (X[:, 1] < 0.75)) * 1.5
        + ((X[:, 1] >= 0.75) & (X[:, 1] <= 1.00)) * 2.0
    )
    y = mu_x + tau_x * Z + rng.normal(scale=1.0, size=n)

    X_train, X_test = X[:n_train], X[n_train:]
    Z_train, Z_test = Z[:n_train], Z[n_train:]
    pi_train, pi_test = pi_x[:n_train], pi_x[n_train:]
    y_train = y[:n_train]

    # Write data fixtures (R reads identical inputs)
    write_matrix(os.path.join(output_dir, "bcf_basic_X_train.csv"), X_train)
    write_matrix(os.path.join(output_dir, "bcf_basic_Z_train.csv"), Z_train.reshape(-1, 1))
    write_matrix(os.path.join(output_dir, "bcf_basic_y_train.csv"), y_train.reshape(-1, 1))
    write_matrix(os.path.join(output_dir, "bcf_basic_pi_train.csv"), pi_train.reshape(-1, 1))
    write_matrix(os.path.join(output_dir, "bcf_basic_X_test.csv"), X_test)
    write_matrix(os.path.join(output_dir, "bcf_basic_Z_test.csv"), Z_test.reshape(-1, 1))
    write_matrix(os.path.join(output_dir, "bcf_basic_pi_test.csv"), pi_test.reshape(-1, 1))

    model = BCFModel()
    model.sample(
        X_train, Z_train, y_train, pi_train,
        num_gfr=num_gfr,
        num_burnin=num_burnin,
        num_mcmc=num_mcmc,
        general_params={"random_seed": GLOBAL_SEED},
    )

    for split, X_s, Z_s, pi_s in [
        ("train", X_train, Z_train, pi_train),
        ("test", X_test, Z_test, pi_test),
    ]:
        # terms="all" → dict; keys are y_hat, tau_hat, mu_hat, ...
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = model.predict(X_s, Z_s, propensity=pi_s)
        write_matrix(
            os.path.join(output_dir, f"bcf_basic_yhat_{split}.csv"), preds["y_hat"]
        )
        write_matrix(
            os.path.join(output_dir, f"bcf_basic_tau_{split}.csv"), preds["tau_hat"]
        )
        write_matrix(
            os.path.join(output_dir, f"bcf_basic_mu_{split}.csv"), preds["mu_hat"]
        )

    meta = {
        "scenario": "bcf_basic",
        "seed": GLOBAL_SEED,
        "n": n,
        "p": p,
        "n_train": n_train,
        "n_test": n_test,
        "num_gfr": num_gfr,
        "num_burnin": num_burnin,
        "num_mcmc": num_mcmc,
        "params": {"sample_intercept": True, "num_trees_variance": 0},
    }
    write_metadata(os.path.join(output_dir, "bcf_basic.json"), meta)

    print("  bcf_basic — OK")


# ---------------------------------------------------------------------------
# Scenario: bart_rfx
# ---------------------------------------------------------------------------

def scenario_bart_rfx(output_dir: str) -> None:
    """Continuous BART with intercept-only group random effects.

    Group IDs are 1-indexed integers (1..num_groups) so R's factor() conversion
    produces the same integers Python uses.
    Terms written: yhat_train, yhat_test, rfx_train, rfx_test.
    """
    from stochtree import BARTModel

    n = 300
    p = 5
    n_test = 60
    n_train = n - n_test
    num_groups = 8
    num_gfr = 10
    num_burnin = 10
    num_mcmc = 50

    rng = np.random.default_rng(GLOBAL_SEED)
    X = rng.uniform(size=(n, p))
    group_ids = rng.integers(1, num_groups + 1, size=n)  # 1..num_groups
    group_effects = rng.normal(scale=2.0, size=num_groups + 1)  # index 0 unused
    f_x = np.sin(np.pi * X[:, 0]) * X[:, 1]
    y = f_x + group_effects[group_ids] + rng.normal(scale=0.5, size=n)

    X_train, X_test = X[:n_train], X[n_train:]
    g_train, g_test = group_ids[:n_train], group_ids[n_train:]
    y_train = y[:n_train]

    write_matrix(os.path.join(output_dir, "bart_rfx_X_train.csv"), X_train)
    write_matrix(os.path.join(output_dir, "bart_rfx_y_train.csv"), y_train.reshape(-1, 1))
    write_matrix(os.path.join(output_dir, "bart_rfx_group_train.csv"), g_train.reshape(-1, 1))
    write_matrix(os.path.join(output_dir, "bart_rfx_X_test.csv"), X_test)
    write_matrix(os.path.join(output_dir, "bart_rfx_group_test.csv"), g_test.reshape(-1, 1))

    model = BARTModel()
    model.sample(
        X_train, y_train,
        rfx_group_ids_train=g_train,
        num_gfr=num_gfr,
        num_burnin=num_burnin,
        num_mcmc=num_mcmc,
        general_params={"random_seed": GLOBAL_SEED},
        random_effects_params={"model_spec": "intercept_only"},
    )

    for split, X_s, g_s in [("train", X_train, g_train), ("test", X_test, g_test)]:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = model.predict(X_s, rfx_group_ids=g_s.astype(np.int64))
        write_matrix(
            os.path.join(output_dir, f"bart_rfx_yhat_{split}.csv"), preds["y_hat"]
        )
        write_matrix(
            os.path.join(output_dir, f"bart_rfx_rfx_{split}.csv"), preds["rfx_predictions"]
        )

    meta = {
        "scenario": "bart_rfx",
        "seed": GLOBAL_SEED,
        "n": n,
        "p": p,
        "n_train": n_train,
        "n_test": n_test,
        "num_groups": num_groups,
        "num_gfr": num_gfr,
        "num_burnin": num_burnin,
        "num_mcmc": num_mcmc,
        "params": {"model_spec": "intercept_only"},
    }
    write_metadata(os.path.join(output_dir, "bart_rfx.json"), meta)

    print("  bart_rfx — OK")


# ---------------------------------------------------------------------------
# Scenario: bcf_varforest
# ---------------------------------------------------------------------------

def scenario_bcf_varforest(output_dir: str) -> None:
    """BCF with sample_tau_0=True AND a variance forest (num_trees=50).

    Same DGP and data layout as bcf_basic.  Uses a different RNG seed offset
    so the actual data differs from bcf_basic.
    Terms written: yhat, tau, mu, sigma2x (variance forest).
    """
    from stochtree import BCFModel

    SEED = GLOBAL_SEED + 1  # distinct from bcf_basic

    n = 500
    p = 5
    n_test = 100
    n_train = n - n_test
    num_gfr = 10
    num_burnin = 10
    num_mcmc = 50
    num_trees_variance = 50

    rng = np.random.default_rng(SEED)
    X = rng.uniform(size=(n, p))

    pi_x = (
        ((X[:, 0] >= 0.00) & (X[:, 0] < 0.25)) * 0.2
        + ((X[:, 0] >= 0.25) & (X[:, 0] < 0.50)) * 0.4
        + ((X[:, 0] >= 0.50) & (X[:, 0] < 0.75)) * 0.6
        + ((X[:, 0] >= 0.75) & (X[:, 0] <= 1.00)) * 0.8
    )
    Z = rng.binomial(1, pi_x).astype(float)
    mu_x = (
        ((X[:, 0] >= 0.00) & (X[:, 0] < 0.25)) * (-7.5)
        + ((X[:, 0] >= 0.25) & (X[:, 0] < 0.50)) * (-2.5)
        + ((X[:, 0] >= 0.50) & (X[:, 0] < 0.75)) * 2.5
        + ((X[:, 0] >= 0.75) & (X[:, 0] <= 1.00)) * 7.5
    )
    tau_x = (
        ((X[:, 1] >= 0.00) & (X[:, 1] < 0.25)) * 0.5
        + ((X[:, 1] >= 0.25) & (X[:, 1] < 0.50)) * 1.0
        + ((X[:, 1] >= 0.50) & (X[:, 1] < 0.75)) * 1.5
        + ((X[:, 1] >= 0.75) & (X[:, 1] <= 1.00)) * 2.0
    )
    y = mu_x + tau_x * Z + rng.normal(scale=1.0, size=n)

    X_train, X_test = X[:n_train], X[n_train:]
    Z_train, Z_test = Z[:n_train], Z[n_train:]
    pi_train, pi_test = pi_x[:n_train], pi_x[n_train:]
    y_train = y[:n_train]

    write_matrix(os.path.join(output_dir, "bcf_varforest_X_train.csv"), X_train)
    write_matrix(os.path.join(output_dir, "bcf_varforest_Z_train.csv"), Z_train.reshape(-1, 1))
    write_matrix(os.path.join(output_dir, "bcf_varforest_y_train.csv"), y_train.reshape(-1, 1))
    write_matrix(os.path.join(output_dir, "bcf_varforest_pi_train.csv"), pi_train.reshape(-1, 1))
    write_matrix(os.path.join(output_dir, "bcf_varforest_X_test.csv"), X_test)
    write_matrix(os.path.join(output_dir, "bcf_varforest_Z_test.csv"), Z_test.reshape(-1, 1))
    write_matrix(os.path.join(output_dir, "bcf_varforest_pi_test.csv"), pi_test.reshape(-1, 1))

    model = BCFModel()
    model.sample(
        X_train, Z_train, y_train, pi_train,
        num_gfr=num_gfr,
        num_burnin=num_burnin,
        num_mcmc=num_mcmc,
        general_params={"random_seed": SEED},
        variance_forest_params={"num_trees": num_trees_variance},
    )

    for split, X_s, Z_s, pi_s in [
        ("train", X_train, Z_train, pi_train),
        ("test", X_test, Z_test, pi_test),
    ]:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = model.predict(X_s, Z_s, propensity=pi_s)
        write_matrix(
            os.path.join(output_dir, f"bcf_varforest_yhat_{split}.csv"), preds["y_hat"]
        )
        write_matrix(
            os.path.join(output_dir, f"bcf_varforest_tau_{split}.csv"), preds["tau_hat"]
        )
        write_matrix(
            os.path.join(output_dir, f"bcf_varforest_mu_{split}.csv"), preds["mu_hat"]
        )
        write_matrix(
            os.path.join(output_dir, f"bcf_varforest_sigma2x_{split}.csv"),
            preds["variance_forest_predictions"],
        )

    meta = {
        "scenario": "bcf_varforest",
        "seed": SEED,
        "n": n,
        "p": p,
        "n_train": n_train,
        "n_test": n_test,
        "num_gfr": num_gfr,
        "num_burnin": num_burnin,
        "num_mcmc": num_mcmc,
        "params": {
            "sample_intercept": True,
            "num_trees_variance": num_trees_variance,
        },
    }
    write_metadata(os.path.join(output_dir, "bcf_varforest.json"), meta)

    print("  bcf_varforest — OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate cross-language parity fixtures"
    )
    parser.add_argument(
        "--output-dir",
        default="test/cross_language/fixtures",
        help="Directory to write fixture files (default: test/cross_language/fixtures)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Writing fixtures to: {args.output_dir}")

    scenario_bart_basic(args.output_dir)
    scenario_bcf_basic(args.output_dir)
    scenario_bart_rfx(args.output_dir)
    scenario_bcf_varforest(args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
