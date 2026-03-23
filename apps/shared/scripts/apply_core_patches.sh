#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 David Wong, University of Oxford
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_WALBERLA_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WALBERLA_ROOT="${WALBERLA_ROOT:-$DEFAULT_WALBERLA_ROOT}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --walberla-root)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --walberla-root requires a path" >&2
        exit 1
      fi
      WALBERLA_ROOT="$2"
      shift 2
      ;;
    *)
      echo "ERROR: unknown argument '$1'" >&2
      exit 1
      ;;
  esac
done

PATCH_FILE="$SHARED_DIR/patches/genericboundary_nvcc_crtp.patch"

if [[ ! -d "$WALBERLA_ROOT" || ! -f "$WALBERLA_ROOT/CMakeLists.txt" ]]; then
  echo "ERROR: Invalid walberla root: $WALBERLA_ROOT" >&2
  exit 1
fi
if [[ ! -f "$PATCH_FILE" ]]; then
  echo "ERROR: Missing patch file: $PATCH_FILE" >&2
  exit 1
fi
if ! command -v patch >/dev/null 2>&1; then
  echo "ERROR: 'patch' command not found in PATH." >&2
  exit 1
fi

if patch --dry-run -p1 -d "$WALBERLA_ROOT" < "$PATCH_FILE" >/dev/null 2>&1; then
  patch -p1 -d "$WALBERLA_ROOT" < "$PATCH_FILE" >/dev/null
  echo "[core-patches] Applied: $(basename "$PATCH_FILE")"
elif patch --dry-run -R -p1 -d "$WALBERLA_ROOT" < "$PATCH_FILE" >/dev/null 2>&1; then
  echo "[core-patches] Already applied: $(basename "$PATCH_FILE")"
else
  echo "ERROR: Patch does not apply cleanly: $PATCH_FILE" >&2
  echo "       Check walberla tree version and patch contents." >&2
  exit 1
fi
