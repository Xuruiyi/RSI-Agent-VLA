# Copyright 2026 The RPent Authors.
"""Trusted task files through the existing RLinf LIBERO environment lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def load_materialized_task(task_ref: str | Path) -> dict[str, Any]:
    """Read a task whose identity is bound by its independent materializer."""
    path = Path(task_ref).resolve(strict=True)
    task = json.loads(path.read_text())
    receipt_ref = Path(task["materialization_ref"]).resolve(strict=True)
    receipt = json.loads(receipt_ref.read_text())
    if (
        receipt.get("schema") != "rsi-materialized-goal-v1"
        or receipt.get("task") != task
        or Path(receipt["task_ref"]).resolve() != path
        or task.get("suite") != "materialized"
        or task.get("task_id") != 0
        or task.get("native_success") != "bddl_goal"
    ):
        raise ValueError("materialized task differs from its trusted identity")
    for field, registered in (
        ("bddl_file", "bddl_ref"),
        ("initial_state_ref", "initial_state_ref"),
    ):
        artifact = Path(task[field]).resolve(strict=True)
        if (
            artifact != Path(receipt[registered]).resolve()
            or artifact.parent != receipt_ref.parent
            or not artifact.is_file()
        ):
            raise ValueError("materialized task artifact escapes its registered domain")
    if not task.get("instruction") or not task.get("task_key"):
        raise ValueError("materialized task requires instruction and identity")
    return task


def runtime_binding(task_ref: str | Path, *, seed: int) -> dict[str, Any]:
    """Bind task files without initializing a simulator or reading initial states."""
    if type(seed) is not int or seed < 0:
        raise ValueError("materialized task seed must be nonnegative")
    task = load_materialized_task(task_ref)
    identity = {"suite": task["suite"], "task": task["task_id"], "seed": seed}
    identity.update(
        {
            key: task[key]
            for key in (
                "task_key",
                "bddl_file",
                "initial_state_ref",
                "materialization_ref",
                "native_success",
            )
        }
    )
    return {
        "task_binding": identity,
        "instruction": task["instruction"],
        "runtime_arguments": [
            "--suite",
            "materialized",
            "--task",
            "0",
            "--seed",
            str(seed),
            "--materialized-task",
            str(Path(task_ref).resolve()),
        ],
        "environment_initialization": "unmeasured",
        "physical_feasibility": "unmeasured",
    }


def install_materialized_suite(benchmark: Any, task: dict[str, Any]) -> None:
    """Register one process-local task in LIBERO's existing benchmark registry."""
    existing = benchmark.BENCHMARK_MAPPING.get("materialized")
    if existing:
        if getattr(existing, "materialized_identity", None) != task:
            raise ValueError(
                "environment process already binds another materialized task"
            )
        return

    class MaterializedSuite:
        materialized_identity = task
        name = "materialized"
        n_tasks = 1

        def get_num_tasks(self):
            return 1

        def get_task(self, task_id):
            if task_id != 0:
                raise ValueError("materialized environment has one bound task")
            return SimpleNamespace(
                name=task["task_key"],
                language=task["instruction"],
                problem_folder="",
                bddl_file=task["bddl_file"],
                init_states_file=Path(task["initial_state_ref"]).name,
            )

        def get_task_init_states(self, task_id):
            self.get_task(task_id)
            # Same trusted native initial-state serialization consumed by RLinf.
            import torch

            return torch.load(task["initial_state_ref"], weights_only=False)

    benchmark.BENCHMARK_MAPPING["materialized"] = MaterializedSuite


def materialized_libero_class(base: type, task: dict[str, Any]) -> type:
    """Override only task file selection; RLinf retains reset/action/success code."""

    class MaterializedLibero(base):
        def get_env_fn_params(self, env_idx=None):
            from omegaconf import OmegaConf

            parameters = OmegaConf.to_container(self.cfg.init_params, resolve=True)
            self.task_descriptions = [task["instruction"]] * self.num_envs
            # Prevent Pro's perturbation-folder loader from replacing the bound init.
            self._pert_init_folders = []
            return [
                {**parameters, "bddl_file_name": task["bddl_file"], "seed": self.seed}
                for _ in range(self.num_envs)
            ]

    return MaterializedLibero
