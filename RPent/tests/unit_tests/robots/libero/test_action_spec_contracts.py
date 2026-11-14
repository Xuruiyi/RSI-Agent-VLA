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

"""Native worker metadata is queried through the real facade RPC boundary."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from robots.libero.action_spec import action_spec_libero_class
from robots.libero.env_server import LiberoEnvFacade


@pytest.mark.parametrize("composite", [True, False])
def test_actual_worker_getattr_contract_preserves_native_bounds_and_units(composite):
    arm = {
        "type": "OSC_POSE",
        "input_min": -1,
        "input_max": 1,
        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        "impedance_mode": "fixed",
    }
    if composite:
        arm.update(input_ref_frame="base", input_type="delta")
        config = {"composite_controller_config": {"body_parts": {"right": arm}}}
    else:
        config = {"controller_config": {**arm, "control_delta": True}}
    native = SimpleNamespace(
        env=SimpleNamespace(
            action_spec=(-np.ones(7), np.ones(7)), robot_configs=[config]
        )
    )

    class Backend:
        def get_env_fns(self):
            return [lambda: native]

    factory = action_spec_libero_class(Backend)().get_env_fns()[0]
    exported = factory()
    assert exported is native
    queried = []

    class Worker:
        # This is the installed RLinf interface. There is deliberately no env_call.
        def get_env_attr(self, name):
            queried.append(name)
            return getattr(exported, name, None)

    facade = LiberoEnvFacade(
        SimpleNamespace(env=SimpleNamespace(workers=[Worker()])), meta={}
    )
    spec = facade._dispatch("env.get_action_spec", (), {})
    assert queried == ["rpent_action_spec"]
    assert spec == json.loads(json.dumps(spec))
    assert spec["default"]["low"] == [-1.0] * 7
    assert spec["default"]["high"] == [1.0] * 7
    assert ("base" if composite else "world") in spec["default"]["description"]
    assert "metres per action" in spec["default"]["description"]
    assert "radians per action" in spec["default"]["description"]


def test_missing_worker_metadata_is_an_explicit_error():
    worker = SimpleNamespace(get_env_attr=lambda name: None)
    facade = LiberoEnvFacade(
        SimpleNamespace(env=SimpleNamespace(workers=[worker])), meta={}
    )
    with pytest.raises(RuntimeError, match="metadata is unavailable"):
        facade.get_action_spec()
