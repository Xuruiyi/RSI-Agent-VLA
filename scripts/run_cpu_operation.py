#!/usr/bin/env python3
# Copyright 2026 RSI-Agent-VLA Authors.
"""Consume a registered Host CPU permit without model, GPU or activation calls."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core/src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "RPent"))

from rsi_vla.cpu_operation import execute

if __name__ == "__main__":
    execute(os.environ["RSI_CPU_PERMIT_REF"], os.environ["RSI_OPERATION_INPUT_REF"])
