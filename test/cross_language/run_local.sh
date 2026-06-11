#!/usr/bin/env bash
# run_local.sh — Run cross-language parity tests locally.
#
# Usage:
#   bash test/cross_language/run_local.sh [fixture-dir]
#
# Prerequisites:
#   Python  pip install -e .   (or activate a venv with stochtree installed)
#   R       R CMD INSTALL .    (builds and installs stochtree from source)
#
# For rapid local dev where stochtree is not installed in R, use devtools:
#   STOCHTREE_R_DEV=1 bash test/cross_language/run_local.sh
# This sets STOCHTREE_R_DEV and STOCHTREE_REPO_ROOT so that verify_predictions.R
# calls devtools::load_all() instead of library(stochtree).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FIXTURE_DIR="${1:-$SCRIPT_DIR/fixtures}"

mkdir -p "$FIXTURE_DIR"

echo "==> Generating Python predictions..."
python "$SCRIPT_DIR/generate_predictions.py" --output-dir "$FIXTURE_DIR"

echo "==> Verifying R predictions match Python..."
# cd to repo root so that devtools::load_all(".") resolves correctly in dev mode
cd "$REPO_ROOT"
export STOCHTREE_REPO_ROOT="$REPO_ROOT"
Rscript "$SCRIPT_DIR/verify_predictions.R" "$FIXTURE_DIR"

echo "==> Done"
