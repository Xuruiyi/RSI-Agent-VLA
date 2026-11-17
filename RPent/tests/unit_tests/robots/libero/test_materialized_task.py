# Copyright 2026 The RPent Authors.
"""Materialized task identities survive the CLI, daemon and native Env adapter."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from robots.libero.materialized_task import (
    install_materialized_suite,
    load_materialized_task,
    materialized_libero_class,
    runtime_binding,
)


@pytest.fixture
def materialized(tmp_path):
    task_ref = tmp_path / "task.json"
    receipt_ref = tmp_path / "materialization.json"
    bddl = tmp_path / "joint.bddl"
    bddl.write_text("(define (problem test) (:goal (And (Open drawer_1))))")
    initial = tmp_path / "joint.pruned_init"
    initial.write_text("synthetic initial-state boundary; no simulator")
    task = {
        "task_key": "joint",
        "suite": "materialized",
        "task_id": 0,
        "instruction": "open the drawer",
        "native_success": "bddl_goal",
        "bddl_file": str(bddl),
        "initial_state_ref": str(initial),
        "materialization_ref": str(receipt_ref),
    }
    task_ref.write_text(json.dumps(task))
    receipt_ref.write_text(
        json.dumps(
            {
                "schema": "rsi-materialized-goal-v1",
                "task": task,
                "task_ref": str(task_ref),
                "bddl_ref": str(bddl),
                "initial_state_ref": str(initial),
            }
        )
    )
    return task_ref, task


def test_cli_and_environment_daemon_preserve_materialized_task(
    materialized, tmp_path, monkeypatch
):
    from robots.libero import robot_spec
    from rpent.cli.main import _build_argparser
    from rpent.session import runtime

    task_ref, task = materialized
    parser = _build_argparser()
    robot_spec._add_cli_args(parser, use_dashboard=False)
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "MEMORY.md").write_text("legal development memory")
    args = parser.parse_args(
        [
            "--robot",
            "libero",
            "--planner",
            "codex",
            "--suite",
            "materialized",
            "--task",
            "0",
            "--seed",
            "2",
            "--materialized-task",
            str(task_ref),
            "--memory-profile",
            "local",
            "--memory-dir",
            str(memory),
        ]
    )
    monkeypatch.setattr(runtime, "runtime_context", lambda: None)
    config = robot_spec._parse_config(args)
    assert config.task_desc["materialization_ref"] == task["materialization_ref"]
    assert config.task_desc["bddl_file"] == task["bddl_file"]
    assert config.task_desc["native_success"] == "bddl_goal"
    daemon = Mock()
    constructor = Mock(return_value=daemon)
    monkeypatch.setattr(robot_spec, "ProcessDaemon", constructor)
    monkeypatch.setattr(robot_spec, "pick_free_port", lambda: 8123)
    monkeypatch.setattr(robot_spec, "HttpRpcClient", lambda endpoint: endpoint)
    monkeypatch.setattr(robot_spec, "resolve_cuda_device", lambda *_: None)
    robot_spec._spawn_env_server(args, tmp_path / "run")
    command = constructor.call_args.kwargs["cmd"]
    assert command[command.index("--materialized-task") + 1] == str(task_ref)
    assert constructor.call_args.kwargs["env_overrides"]["LIBERO_TYPE"] == "standard"
    daemon.start.assert_called_once()


def test_native_task_file_selection_retains_rlinf_lifecycle(materialized):
    task_ref, task = materialized

    class ExistingEnv:
        def reset(self):
            return "native-reset"

    adapter = materialized_libero_class(ExistingEnv, load_materialized_task(task_ref))
    env = adapter()
    env.cfg = SimpleNamespace(init_params=OmegaConf.create({"camera_depths": True}))
    env.num_envs = 1
    env.seed = 4
    parameters = env.get_env_fn_params()
    assert parameters == [
        {"camera_depths": True, "bddl_file_name": task["bddl_file"], "seed": 4}
    ]
    assert env.reset() == "native-reset"
    assert env.task_descriptions == [task["instruction"]]
    assert env._pert_init_folders == []
    registry = SimpleNamespace(BENCHMARK_MAPPING={})
    install_materialized_suite(registry, task)
    assert (
        registry.BENCHMARK_MAPPING["materialized"]().get_task(0).language
        == task["instruction"]
    )
    with pytest.raises(ValueError, match="another materialized task"):
        install_materialized_suite(registry, {**task, "task_key": "other"})


@pytest.mark.parametrize(
    "field,value",
    [
        ("task_id", 1),
        ("native_success", "caller_boolean"),
        ("bddl_file", "/etc/passwd"),
    ],
)
def test_modified_task_identity_is_rejected(materialized, field, value):
    ref, task = materialized
    ref.write_text(json.dumps({**task, field: value}))
    with pytest.raises(ValueError, match="trusted identity"):
        runtime_binding(ref, seed=0)
