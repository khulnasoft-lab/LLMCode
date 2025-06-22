#!/bin/bash
set -e

echo "Generating requirements.txt from .in files"

# Option 1: Use pip-compile directly if it's in PATH
# pip-compile requirements/requirements.in
# pip-compile requirements/requirements-dev.in

# Option 2: More reliable in all envs
python -m piptools compile requirements/requirements.in
python -m piptools compile requirements/requirements-dev.in

echo "Done."
