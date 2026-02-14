#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

APPLY=0
if [[ "${1:-}" == "--apply" ]]; then
  APPLY=1
fi

run_cmd() {
  if [[ "${APPLY}" -eq 1 ]]; then
    echo "+ $*"
    eval "$@"
  else
    echo "[dry-run] $*"
  fi
}

run_cmd "mkdir -p docs/repro/patches logs/mr logs/sim logs/sum logs/resume logs/verify repro/legacy"

for f in PATCH_*.md; do
  [[ -e "${f}" ]] || continue
  run_cmd "mv \"${f}\" docs/repro/patches/"
done

for f in logs_*.log; do
  [[ -e "${f}" ]] || continue
  dst="logs/mr"
  case "${f}" in
    logs_resume_*.log) dst="logs/resume" ;;
    logs_verify_*.log) dst="logs/verify" ;;
    logs_sim_*.log) dst="logs/sim" ;;
    logs_sum_*.log) dst="logs/sum" ;;
    logs_mr_*.log) dst="logs/mr" ;;
  esac
  run_cmd "mv \"${f}\" \"${dst}/\""
done

if [[ -d '$OUT' ]]; then
  run_cmd "mv '$OUT' repro/legacy/OUT_literal"
fi

if [[ "${APPLY}" -eq 0 ]]; then
  echo
  echo "No files changed. Re-run with --apply to execute."
fi
