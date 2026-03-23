#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 David Wong, University of Oxford
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

walberla_root_arg=""
project_root_arg=""
venv_arg=""
python_arg=""
recreate_venv=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --walberla-root)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --walberla-root requires a path" >&2
        exit 1
      fi
      walberla_root_arg="$2"
      shift 2
      ;;
    --project-root)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --project-root requires a path" >&2
        exit 1
      fi
      project_root_arg="$2"
      shift 2
      ;;
    --venv)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --venv requires a path" >&2
        exit 1
      fi
      venv_arg="$2"
      shift 2
      ;;
    --python)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --python requires an executable name/path" >&2
        exit 1
      fi
      python_arg="$2"
      shift 2
      ;;
    --recreate)
      recreate_venv=1
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage: install_codegen_venv.sh [options]

Options:
  --walberla-root <path>   waLBerla source root (default: script-relative)
  --project-root <path>    project root containing venv-walberla-codegen
  --venv <path>            override venv location
  --python <exe>           python executable (default: python3)
  --recreate               delete and recreate the venv
  -h, --help               show this help
EOF
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument '$1'" >&2
      exit 1
      ;;
  esac
done

DEFAULT_WALBERLA_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WALBERLA_ROOT="${walberla_root_arg:-$DEFAULT_WALBERLA_ROOT}"
PROJECT_ROOT="${project_root_arg:-$(cd "$WALBERLA_ROOT/.." && pwd)}"
VENV="${venv_arg:-$PROJECT_ROOT/venv-walberla-codegen}"
PYTHON_BIN="${python_arg:-python3}"

SWEEPGEN_REQUIREMENTS="$WALBERLA_ROOT/sweepgen/cmake/sweepgen-requirements.txt"
SWEEPGEN_DIR="$WALBERLA_ROOT/sweepgen"

if [[ ! -d "$WALBERLA_ROOT" || ! -f "$WALBERLA_ROOT/CMakeLists.txt" ]]; then
  echo "ERROR: Invalid walberla root: $WALBERLA_ROOT" >&2
  exit 1
fi
if [[ ! -f "$SWEEPGEN_REQUIREMENTS" ]]; then
  echo "ERROR: Missing sweepgen requirements: $SWEEPGEN_REQUIREMENTS" >&2
  exit 1
fi
if [[ ! -d "$SWEEPGEN_DIR" || ! -f "$SWEEPGEN_DIR/pyproject.toml" ]]; then
  echo "ERROR: Invalid sweepgen directory: $SWEEPGEN_DIR" >&2
  exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

echo "waLBerla root: $WALBERLA_ROOT"
echo "project root: $PROJECT_ROOT"
echo "venv path:    $VENV"
echo "python:       $(command -v "$PYTHON_BIN")"

if [[ "$recreate_venv" == "1" && -d "$VENV" ]]; then
  echo "Recreating venv: $VENV"
  rm -rf "$VENV"
fi

if [[ ! -x "$VENV/bin/python" || ! -f "$VENV/bin/activate" ]]; then
  echo "Creating venv: $VENV"
  mkdir -p "$(dirname "$VENV")"
  "$PYTHON_BIN" -m venv "$VENV"
fi

if [[ ! -x "$VENV/bin/python" || ! -f "$VENV/bin/activate" ]]; then
  echo "ERROR: Failed to create/locate venv at $VENV" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV/bin/activate"

echo "Upgrading packaging tools..."
python -m pip install --upgrade pip setuptools wheel

echo "Installing sweepgen codegen requirements..."
python -m pip install -r "$SWEEPGEN_REQUIREMENTS"

echo "Installing sweepgen (editable)..."
python -m pip install -e "$SWEEPGEN_DIR"

echo "Installing additional direct imports used by app build checks..."
python -m pip install numpy sympy jinja2

echo "Validating codegen imports..."
python - <<'PY'
import numpy, sympy, jinja2, lbmpy, pystencils, pystencilssfg, sweepgen
print("OK: codegen imports available")
PY

echo
echo "Codegen venv is ready at: $VENV"
echo "Activate with: source \"$VENV/bin/activate\""
