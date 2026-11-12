#!/usr/bin/env bash
# 默认仅输出绑定 medium 配置的计划；--execute 仍需明确动态授权和预算准入。
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../config/models.local.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$RSI_PROJECT_ROOT/core/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$RSI_RPENT_PYTHON" -m rsi_vla.runtime --root "$RSI_PROJECT_ROOT" "$@"
