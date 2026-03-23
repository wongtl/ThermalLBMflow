#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 David Wong, University of Oxford
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYSTENCILS_VERSION="2.0.dev0+170.ga13c5b1"
PYSTENCILS_COMMIT="a13c5b1c71c924b0d58c1a343ac9780015786edd"
PYSTENCILS_URL="git+https://i10git.cs.fau.de/pycodegen/pystencils.git@${PYSTENCILS_COMMIT}#egg=pystencils"

LBMPY_VERSION="1.4+2.gfada994"
LBMPY_COMMIT="fada99477fbd5de9d0f583402269595d77a2b40b"
LBMPY_URL="git+https://i10git.cs.fau.de/pycodegen/lbmpy.git@${LBMPY_COMMIT}#egg=lbmpy"

PYSTENCILSSFG_VERSION="0.1a4+59.gc635a76"
PYSTENCILSSFG_COMMIT="c635a7666bb97fe6c15490011fa03d48a93e2879"
PYSTENCILSSFG_URL="git+https://i10git.cs.fau.de/pycodegen/pystencils-sfg.git@${PYSTENCILSSFG_COMMIT}#egg=pystencilssfg"

NUMPY_VERSION="2.4.2"
SYMPY_VERSION="1.14.0"
JINJA2_VERSION="3.1.6"
PY_CPUINFO_VERSION="9.0.0"

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

SWEEPGEN_DIR="$WALBERLA_ROOT/sweepgen"

if [[ ! -d "$WALBERLA_ROOT" || ! -f "$WALBERLA_ROOT/CMakeLists.txt" ]]; then
  echo "ERROR: Invalid walberla root: $WALBERLA_ROOT" >&2
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

echo "Installing pinned pycodegen stack..."
python -m pip install \
  "$PYSTENCILS_URL" \
  "$LBMPY_URL" \
  "$PYSTENCILSSFG_URL" \
  "numpy==${NUMPY_VERSION}" \
  "sympy==${SYMPY_VERSION}" \
  "Jinja2==${JINJA2_VERSION}" \
  "py-cpuinfo==${PY_CPUINFO_VERSION}"

echo "Installing sweepgen (editable)..."
python -m pip install -e "$SWEEPGEN_DIR"

echo "Validating codegen imports..."
python - <<'PY'
import numpy, sympy, jinja2, lbmpy, pystencils, pystencilssfg, sweepgen
print("OK: codegen imports available")
PY

echo
echo "Codegen venv is ready at: $VENV"
echo "Pinned codegen stack:"
echo "  pystencils==${PYSTENCILS_VERSION} @ ${PYSTENCILS_COMMIT}"
echo "  lbmpy==${LBMPY_VERSION} @ ${LBMPY_COMMIT}"
echo "  pystencilssfg==${PYSTENCILSSFG_VERSION} @ ${PYSTENCILSSFG_COMMIT}"
echo "  numpy==${NUMPY_VERSION}"
echo "  sympy==${SYMPY_VERSION}"
echo "  Jinja2==${JINJA2_VERSION}"
echo "  py-cpuinfo==${PY_CPUINFO_VERSION}"
echo "  sweepgen @ editable:$SWEEPGEN_DIR"
echo "Activate with: source \"$VENV/bin/activate\""
