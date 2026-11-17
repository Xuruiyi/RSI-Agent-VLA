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

"""Fixed native drawer-joint metadata read inside each pinned LIBERO worker."""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import partial
from math import isfinite
from numbers import Integral
from typing import Any

DRAWER_PROPERTY = "rpent_drawer_joint_observation"
DRAWER_SCHEMA = "rsi-private-drawer-joint-observation-v1"
DRAWER_REGION = "wooden_cabinet_1_middle_region"
DRAWER_FIXTURE = "wooden_cabinet_1"
DRAWER_JOINT = "wooden_cabinet_1_middle_level"


def unknown_drawer_observation(reason: str) -> dict[str, Any]:
    """Describe unavailable measurement without manufacturing a zero joint state."""
    return {
        "schema": DRAWER_SCHEMA,
        "status": "unknown",
        "region": DRAWER_REGION,
        "fixture": None,
        "joint": None,
        "joints": None,
        "joint_type": None,
        "joint_type_id": None,
        "joint_range": None,
        "qpos": None,
        "qpos_address": None,
        "measured_at_ns": time.time_ns(),
        "missing": [reason],
        "source": "native_worker.object_sites_dict/sim.model.get_joint_qpos_addr/sim.data.qpos",
    }


def _native_drawer_observation(worker_env: Any) -> dict[str, Any]:
    """Read the actual region mapping and current scalar qpos; never step the Env."""
    import numpy as np

    result = unknown_drawer_observation("")
    result["missing"] = []
    try:
        native = worker_env.env
        site = native.object_sites_dict[DRAWER_REGION]
        result["fixture"] = site.parent_name
        if site.parent_name != DRAWER_FIXTURE:
            raise ValueError("drawer region parent differs from the registered fixture")
        joints = list(site.joints)
        if any(not isinstance(name, str) or not name for name in joints):
            raise ValueError("drawer region joint names are unavailable")
        result["joints"] = joints
        if len(joints) != 1:
            raise ValueError("drawer region must resolve to exactly one joint")
        joint = result["joint"] = joints[0]
        if joint != DRAWER_JOINT:
            raise ValueError("drawer region maps to a different joint")
        model = native.sim.model
        joint_id = model.joint_name2id(joint)
        if (
            isinstance(joint_id, bool)
            or not isinstance(joint_id, Integral)
            or joint_id < 0
        ):
            raise ValueError("native drawer joint id is unavailable")
        joint_type = int(model.jnt_type[joint_id])
        result["joint_type_id"] = joint_type
        # MuJoCo mjtJoint: FREE=0, BALL=1, SLIDE=2, HINGE=3.
        result["joint_type"] = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}.get(
            joint_type
        )
        joint_range = np.asarray(model.jnt_range[joint_id])
        if joint_range.shape != (2,) or not all(
            isfinite(float(value)) for value in joint_range
        ):
            raise ValueError("native drawer joint range is unavailable")
        result["joint_range"] = [float(value) for value in joint_range]
        if joint_type != 2:
            raise ValueError("registered drawer joint is not a native slide joint")
        address = model.get_joint_qpos_addr(joint)
        if (
            isinstance(address, bool)
            or not isinstance(address, Integral)
            or address < 0
        ):
            raise ValueError("native drawer qpos address is not a scalar joint address")
        result["qpos_address"] = int(address)
        qpos = np.asarray(native.sim.data.qpos[address])
        if qpos.shape != () or not isfinite(float(qpos)):
            raise ValueError("native drawer qpos is not a finite scalar")
        result.update(
            status="recorded", qpos=float(qpos), measured_at_ns=time.time_ns()
        )
    except (
        AttributeError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:
        result["missing"].append(f"{type(exc).__name__}: {exc}")
    return result


class _DrawerObservationWorker:
    """Delegate the native worker API and add one dynamic, read-only property."""

    def __init__(self, env: Any) -> None:
        self._native_env = env

    def __getattr__(self, name: str) -> Any:
        return getattr(self._native_env, name)

    @property
    def rpent_drawer_joint_observation(self) -> dict[str, Any]:
        return _native_drawer_observation(self._native_env)


def _with_drawer_observation(factory: Callable[[], Any]) -> Any:
    return _DrawerObservationWorker(factory())


def drawer_observation_libero_class(base: type) -> type:
    """Use the supported worker factory/getattr channel; import no simulator."""

    class DrawerObservationLiberoEnv(base):
        def get_env_fns(self) -> list[Callable[[], Any]]:
            return [
                partial(_with_drawer_observation, factory)
                for factory in super().get_env_fns()
            ]

    return DrawerObservationLiberoEnv
