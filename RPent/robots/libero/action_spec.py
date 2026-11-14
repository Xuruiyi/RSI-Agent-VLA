# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native action metadata exported by each pinned-task RLinf worker."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

from rpent.robots.components.action_spec import box_action_spec


def _native_action_spec(env: Any) -> dict[str, Any]:
    """Describe the actual robosuite controller without sending it across RPC."""
    robot_env = env.env
    low, high = robot_env.action_spec
    robot_config = robot_env.robot_configs[0]
    # Standard LIBERO uses a composite controller; LIBERO-plus uses
    # robosuite 1.4's single arm controller configuration.
    if "composite_controller_config" in robot_config:
        arm = robot_config["composite_controller_config"]["body_parts"]["right"]
        frame = arm.get("input_ref_frame", "base")
        mode = arm.get("input_type", "delta")
    else:
        arm = robot_config["controller_config"]
        frame = "world"
        mode = "delta" if arm.get("control_delta", True) else "absolute"
    impedance = arm.get("impedance_mode", "fixed")
    gains = {
        "fixed": "",
        "variable_kp": "6 stiffness gains, then ",
        "variable": "6 damping ratios, 6 stiffness gains, then ",
    }[impedance]
    description = (
        f"Native {arm['type']} action: {gains}"
        "[x, y, z, rx, ry, rz], then gripper. "
        f"Arm commands use the {frame} reference frame and {mode} control; "
        "rotation is an axis-angle vector, not Euler angles. "
    )
    if mode == "delta":
        description += (
            "Arm inputs are normalized controller commands, linearly mapped "
            f"from input_min={arm['input_min']} / input_max={arm['input_max']} "
            f"to output_min={arm['output_min']} / output_max={arm['output_max']} "
            "(xyz in metres per action; rotation in radians per action). "
        )
    else:
        description += (
            "Arm inputs are absolute xyz in metres and axis-angle in radians. "
        )
    description += "Gripper: -1 opens, +1 closes, 0 keeps the current command."
    return {"default": box_action_spec(low, high, description)}


def _with_action_spec(factory: Callable[[], Any]) -> Any:
    env = factory()
    env.rpent_action_spec = _native_action_spec(env)
    return env


def action_spec_libero_class(base: type) -> type:
    """Export a serializable contract through RLinf's supported getattr channel.

    RPent pins one task and controller for the worker lifetime. Native resets
    preserve the metadata; task reconfiguration is outside this facade's scope.
    """

    class ActionSpecLiberoEnv(base):
        def get_env_fns(self) -> list[Callable[[], Any]]:
            return [
                partial(_with_action_spec, factory) for factory in super().get_env_fns()
            ]

    return ActionSpecLiberoEnv
