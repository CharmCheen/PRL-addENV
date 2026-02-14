#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

echo "== Repo Audit =="
echo "root: ${ROOT_DIR}"
echo

echo "== scripts (mr/sim/sum) =="
find scripts/mr scripts/sim scripts/sum -maxdepth 2 -type f | sort
echo

echo "== root scattered files =="
find . -maxdepth 1 -type f \( -name 'PATCH_*.md' -o -name 'logs_*.log' -o -name '*.patch' \) | sort
echo

echo "== artifact dirs =="
find . -maxdepth 2 -type d \( -name 'output' -o -name 'wandb' -o -name '.cache' -o -name '$OUT' \) | sort
echo

echo "== suspicious path tokens =="
rg -n '\$OUT|output/sum|output/sim|output/mr|output\b' scripts docs README.md .gitignore 2>/dev/null || true
echo

echo "== tracked status summary =="
git status --short
