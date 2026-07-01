#!/usr/bin/env python3
"""Verify the Python<-R cross-load direction.

Companion to verify_predictions.R, which writes ``<scenario>_r_model.json`` for
each scenario. Here we load each R-written model in Python and assert the real
cross-platform serialization guarantee: the foreign forest reconstructs
**bit-exactly** (re-serialize the loaded forest, compare to the original named
forest at full precision). Predictions are only sanity-checked -- cross-language
prediction *equality* is confounded by ~1-ULP differences between R and numpy
CSV float parsing, which flip rare split-boundary observations -- so we bound the
fraction of flipped rows rather than asserting equality.

Usage:
  python test/cross_language/verify_r_models.py [fixture-dir]
"""

import json
import os
import sys

import numpy as np

from stochtree import BARTModel, BCFModel
from stochtree.serialization import JSONSerializer


def read_matrix(path):
    return np.loadtxt(path, delimiter=",", ndmin=2)


def read_text(path):
    with open(path) as f:
        return f.read()


def check_forest_bitexact(label, forest_container, envelope, forest_name):
    s = JSONSerializer()
    s.add_forest(forest_container)
    got = json.loads(s.return_json_string())["forests"]["forest_0"]
    want = envelope["forests"][forest_name]
    if got == want:
        print(f"  PASS  {label:<42} (forest bit-exact across platforms)")
    else:
        print(f"  FAIL  {label:<42} (forest did NOT round-trip bit-exactly)")
        raise SystemExit(f"Cross-load forest round-trip failed: {label}")


def soft_pred_info(label, a, b):
    diff = np.abs(a - b)
    md = float(diff.max())
    frac = float(np.mean(diff.max(axis=1) > 1e-6))
    print(
        f"  INFO  {label:<38} max|diff|={md:.2e}, boundary-flipped rows={100 * frac:.1f}%"
    )
    if frac > 0.15:
        raise SystemExit(
            f"Cross-load: {100 * frac:.1f}% of rows differ for {label} -- likely a real bug"
        )


def has_r_model(fixture_dir, name):
    return os.path.exists(os.path.join(fixture_dir, name + "_r_model.json"))


def scenario_bart_basic(fixture_dir):
    name = "bart_basic"
    if not has_r_model(fixture_dir, name):
        print(f"Skipping {name} -- no R model fixture")
        return
    print(f"--- {name} ---")
    r_path = os.path.join(fixture_dir, name + "_r_model.json")
    envelope = json.loads(read_text(r_path))
    m = BARTModel()
    m.from_json(read_text(r_path))
    check_forest_bitexact(
        f"{name} / py<-R mean_forest", m.forest_container_mean, envelope, "mean_forest"
    )
    X = read_matrix(os.path.join(fixture_dir, f"{name}_X_train.csv"))
    py = read_matrix(os.path.join(fixture_dir, f"{name}_yhat_train.csv"))
    soft_pred_info(f"{name} / py<-R yhat_train", m.predict(X, terms="y_hat"), py)


def scenario_bcf_basic(fixture_dir):
    name = "bcf_basic"
    if not has_r_model(fixture_dir, name):
        print(f"Skipping {name} -- no R model fixture")
        return
    print(f"--- {name} ---")
    r_path = os.path.join(fixture_dir, name + "_r_model.json")
    envelope = json.loads(read_text(r_path))
    m = BCFModel()
    m.from_json(read_text(r_path))
    check_forest_bitexact(
        f"{name} / py<-R prognostic", m.forest_container_mu, envelope, "prognostic_forest"
    )
    check_forest_bitexact(
        f"{name} / py<-R treatment", m.forest_container_tau, envelope, "treatment_forest"
    )
    X = read_matrix(os.path.join(fixture_dir, f"{name}_X_train.csv"))
    Z = read_matrix(os.path.join(fixture_dir, f"{name}_Z_train.csv")).ravel()
    pi = read_matrix(os.path.join(fixture_dir, f"{name}_pi_train.csv")).ravel()
    py = read_matrix(os.path.join(fixture_dir, f"{name}_yhat_train.csv"))
    soft_pred_info(f"{name} / py<-R yhat_train", m.predict(X, Z, pi, terms="y_hat"), py)


def scenario_bart_rfx(fixture_dir):
    name = "bart_rfx"
    if not has_r_model(fixture_dir, name):
        print(f"Skipping {name} -- no R model fixture")
        return
    print(f"--- {name} ---")
    r_path = os.path.join(fixture_dir, name + "_r_model.json")
    envelope = json.loads(read_text(r_path))
    m = BARTModel()
    m.from_json(read_text(r_path))
    check_forest_bitexact(
        f"{name} / py<-R mean_forest", m.forest_container_mean, envelope, "mean_forest"
    )
    X = read_matrix(os.path.join(fixture_dir, f"{name}_X_train.csv"))
    g = read_matrix(os.path.join(fixture_dir, f"{name}_group_train.csv")).ravel().astype(np.int32)
    basis = np.ones((X.shape[0], 1))
    py = read_matrix(os.path.join(fixture_dir, f"{name}_yhat_train.csv"))
    soft_pred_info(
        f"{name} / py<-R yhat_train",
        m.predict(X, rfx_group_ids=g, rfx_basis=basis, terms="y_hat"),
        py,
    )


def scenario_bcf_varforest(fixture_dir):
    name = "bcf_varforest"
    if not has_r_model(fixture_dir, name):
        print(f"Skipping {name} -- no R model fixture")
        return
    print(f"--- {name} ---")
    r_path = os.path.join(fixture_dir, name + "_r_model.json")
    envelope = json.loads(read_text(r_path))
    m = BCFModel()
    m.from_json(read_text(r_path))
    check_forest_bitexact(
        f"{name} / py<-R prognostic", m.forest_container_mu, envelope, "prognostic_forest"
    )
    check_forest_bitexact(
        f"{name} / py<-R treatment", m.forest_container_tau, envelope, "treatment_forest"
    )
    check_forest_bitexact(
        f"{name} / py<-R variance", m.forest_container_variance, envelope, "variance_forest"
    )
    X = read_matrix(os.path.join(fixture_dir, f"{name}_X_train.csv"))
    Z = read_matrix(os.path.join(fixture_dir, f"{name}_Z_train.csv")).ravel()
    pi = read_matrix(os.path.join(fixture_dir, f"{name}_pi_train.csv")).ravel()
    py = read_matrix(os.path.join(fixture_dir, f"{name}_yhat_train.csv"))
    soft_pred_info(f"{name} / py<-R yhat_train", m.predict(X, Z, pi, terms="y_hat"), py)


def main():
    fixture_dir = sys.argv[1] if len(sys.argv) > 1 else "test/cross_language/fixtures"
    scenario_bart_basic(fixture_dir)
    scenario_bcf_basic(fixture_dir)
    scenario_bart_rfx(fixture_dir)
    scenario_bcf_varforest(fixture_dir)
    print("Cross-language (Python<-R) check complete")


if __name__ == "__main__":
    main()
