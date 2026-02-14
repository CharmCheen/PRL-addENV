#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# PRL local wrapper for upstream-like scripts/sim/sim_qwen_qwen.sh invocation.
# This file is additive-only and does not modify any existing training/infer code.
# ------------------------------------------------------------------------------

if [[ "${CONDA_DEFAULT_ENV:-}" != "prl_clean" ]]; then
  echo "[ERROR] Expected conda env 'prl_clean', got '${CONDA_DEFAULT_ENV:-<empty>}'"
  exit 1
fi

BASE_OUTPUT_DIR="output/sim"
EXP_NAME="${EXP_NAME:-sim-qwen-qwen}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${BASE_OUTPUT_DIR}/${EXP_NAME}-${TIMESTAMP}"
LOG_FILE="${RUN_DIR}/sim.log"

mkdir -p "${RUN_DIR}"

echo "BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR}"
echo "RUN_DIR=${RUN_DIR}"
echo "LOG_FILE=${LOG_FILE}"

count_visible_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    # CUDA_VISIBLE_DEVICES=0,1,2 -> 3
    local cleaned
    cleaned="$(echo "${CUDA_VISIBLE_DEVICES}" | tr -d ' ')"
    if [[ -z "${cleaned}" ]]; then
      echo "1"
      return
    fi
    awk -F',' '{print NF}' <<<"${cleaned}"
    return
  fi

  # Fallback to torch device count; if unavailable return 1.
  python - <<'PY'
try:
    import torch
    n = int(torch.cuda.device_count())
    print(n if n > 0 else 1)
except Exception:
    print(1)
PY
}

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  NPROC_PER_NODE="$(count_visible_gpus)"
fi
if [[ -z "${MASTER_PORT:-}" ]]; then
  MASTER_PORT="29500"
fi

if ! [[ "${NPROC_PER_NODE}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] NPROC_PER_NODE must be an integer, got '${NPROC_PER_NODE}'"
  exit 1
fi
if [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
  echo "[ERROR] NPROC_PER_NODE must be >= 1, got '${NPROC_PER_NODE}'"
  exit 1
fi

# Candidate upstream entry wrappers in this repo.
SELF_PATH="scripts/sim/sim_qwen_qwen_prl.sh"
CANDIDATES=(
  "scripts/sim/sim_qwen_qwen.sh"
  "scripts/mr/mr_qwen_qwen.sh"
)

UNDERLYING_SCRIPT="${UNDERLYING_SCRIPT:-}"
if [[ -z "${UNDERLYING_SCRIPT}" ]]; then
  for c in "${CANDIDATES[@]}"; do
    if [[ "${c}" != "${SELF_PATH}" && -f "${c}" ]]; then
      UNDERLYING_SCRIPT="${c}"
      break
    fi
  done
fi

if [[ -z "${UNDERLYING_SCRIPT}" || ! -f "${UNDERLYING_SCRIPT}" ]]; then
  echo "[ERROR] Cannot locate an underlying qwen/qwen script."
  echo "[ERROR] Checked: ${CANDIDATES[*]}"
  echo "[ERROR] You can set UNDERLYING_SCRIPT=/path/to/existing/script.sh"
  exit 1
fi

extract_entry_cmd() {
  local file="$1"
  local line
  local cmd=""
  local in_cmd="0"

  while IFS= read -r line || [[ -n "${line}" ]]; do
    # Skip comments/empty lines before command.
    if [[ "${in_cmd}" == "0" ]]; then
      [[ -z "${line// }" ]] && continue
      [[ "${line}" =~ ^[[:space:]]*# ]] && continue
      if [[ "${line}" =~ ^[[:space:]]*(python|python3|swift|torchrun)[[:space:]] ]]; then
        in_cmd="1"
      else
        continue
      fi
    fi

    cmd+="${line}"$'\n'
    # Continue while current line ends with backslash.
    if [[ "${line}" =~ \\[[:space:]]*$ ]]; then
      continue
    fi
    break
  done < "${file}"

  echo "${cmd}"
}

RAW_CMD="$(extract_entry_cmd "${UNDERLYING_SCRIPT}")"
if [[ -z "${RAW_CMD// }" ]]; then
  echo "[ERROR] Failed to extract entry command from '${UNDERLYING_SCRIPT}'."
  echo "[ERROR] Expected first command to start with python/python3/swift/torchrun."
  exit 1
fi

# Flatten multi-line command for controlled rewriting/execution.
FLAT_CMD="$(echo "${RAW_CMD}" | sed -E ':a;N;$!ba;s/\\[[:space:]]*\n/ /g' | tr '\n' ' ')"
FLAT_CMD="$(echo "${FLAT_CMD}" | sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//')"

# Force output_dir to RUN_DIR while keeping all other semantics intact.
if [[ "${FLAT_CMD}" =~ --output_dir[[:space:]]+[^[:space:]]+ ]]; then
  FLAT_CMD="$(echo "${FLAT_CMD}" | sed -E "s#--output_dir[[:space:]]+[^[:space:]]+#--output_dir ${RUN_DIR}#")"
else
  FLAT_CMD="${FLAT_CMD} --output_dir ${RUN_DIR}"
fi

build_torchrun_cmd() {
  local base="$1"
  if [[ "${base}" =~ ^torchrun[[:space:]]+ ]]; then
    # Remove existing nproc/master_port so wrapper controls them.
    base="$(echo "${base}" | sed -E 's/--nproc_per_node[[:space:]]+[^[:space:]]+[[:space:]]*//g')"
    base="$(echo "${base}" | sed -E 's/--master_port[[:space:]]+[^[:space:]]+[[:space:]]*//g')"
    echo "${base} --nproc_per_node ${NPROC_PER_NODE} --master_port ${MASTER_PORT}"
  elif [[ "${base}" =~ ^python3?[[:space:]]+-m[[:space:]]+ ]]; then
    # python -m pkg.mod ... -> torchrun ... -m pkg.mod ...
    local rest
    rest="$(echo "${base}" | sed -E 's/^python3?[[:space:]]+-m[[:space:]]+//')"
    echo "torchrun --nproc_per_node ${NPROC_PER_NODE} --master_port ${MASTER_PORT} -m ${rest}"
  elif [[ "${base}" =~ ^python3?[[:space:]]+ ]]; then
    # python train.py ... -> torchrun ... train.py ...
    local rest
    rest="$(echo "${base}" | sed -E 's/^python3?[[:space:]]+//')"
    echo "torchrun --nproc_per_node ${NPROC_PER_NODE} --master_port ${MASTER_PORT} ${rest}"
  elif [[ "${base}" =~ ^swift[[:space:]]+ ]]; then
    # swift CLI script is python-backed; keep args and prepend torchrun.
    local rest
    rest="$(echo "${base}" | sed -E 's/^swift[[:space:]]+//')"
    local swift_bin
    swift_bin="$(command -v swift || true)"
    if [[ -z "${swift_bin}" ]]; then
      echo "torchrun --nproc_per_node ${NPROC_PER_NODE} --master_port ${MASTER_PORT} swift ${rest}"
    else
      echo "torchrun --nproc_per_node ${NPROC_PER_NODE} --master_port ${MASTER_PORT} ${swift_bin} ${rest}"
    fi
  else
    # Fallback for uncommon wrappers.
    echo "torchrun --nproc_per_node ${NPROC_PER_NODE} --master_port ${MASTER_PORT} ${base}"
  fi
}

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  FINAL_CMD="$(build_torchrun_cmd "${FLAT_CMD}")"
  echo "[INFO] MODE=multi-gpu"
else
  FINAL_CMD="${FLAT_CMD}"
  echo "[INFO] MODE=single-process"
fi

echo "[INFO] UNDERLYING_SCRIPT=${UNDERLYING_SCRIPT}"
echo "[INFO] NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "[INFO] MASTER_PORT=${MASTER_PORT}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[INFO] FINAL_CMD=${FINAL_CMD}"

# Keep full launcher output in RUN_DIR/sim.log
bash -lc "${FINAL_CMD}" 2>&1 | tee "${LOG_FILE}"

