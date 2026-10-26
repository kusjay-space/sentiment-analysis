#!/usr/bin/env bash
# Convenience wrapper: run sentiment analysis with the project's virtualenv,
# no need to activate it first.
#
#   ./run.sh predict "This product is great"
#   ./run.sh analyze reviews.csv
#   ./run.sh analyze reviews.csv -o hasil.xlsx
#   ./run.sh inference
#   ./run.sh baseline
#
# Any src.<module> works:
#   ./run.sh train --help
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
    echo "ERROR: virtualenv not found at $REPO_ROOT/.venv" >&2
    echo "" >&2
    echo "Create it first:" >&2
    echo "  cd $REPO_ROOT" >&2
    echo "  python3 -m venv .venv" >&2
    echo "  .venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

if [ $# -eq 0 ]; then
    echo "Usage: ./run.sh <command> [args...]" >&2
    echo "" >&2
    echo "Commands:" >&2
    echo "  predict \"text\"        predict a single review" >&2
    echo "  analyze FILE           batch-analyse a CSV/Excel file" >&2
    echo "  inference              evaluate on the test split" >&2
    echo "  baseline               train the TF-IDF baseline" >&2
    echo "  train                  fine-tune DistilBERT (needs GPU)" >&2
    echo "" >&2
    echo "Example:" >&2
    echo "  ./run.sh analyze examples/reviews_sample.csv" >&2
    exit 1
fi

cd "$REPO_ROOT"
exec "$VENV_PY" -m "src.$@"
