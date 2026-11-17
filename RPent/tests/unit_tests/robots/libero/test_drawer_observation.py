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

"""Private drawer snapshots use native worker joint lookup, never raw-observation keys."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import cloudpickle
import numpy as np
import pytest

from robots.libero.action_spec import action_spec_libero_class
from robots.libero.drawer_observation import (
    DRAWER_FIXTURE,
    DRAWER_JOINT,
    DRAWER_PROPERTY,
    DRAWER_REGION,
    drawer_observation_libero_class,
)
from robots.libero.env_client import LiberoEnvClient
from robots.libero.env_server import LiberoEnvFacade

SIGNATURE = {
    "theta_ref": "theta",
    "processor_ref": "processor",
    "phi_ref": "phi",
    "code_ref": "code",
}
CRITERION = {
    "schema": "rsi-skill-drawer-open-criterion-v1",
    "criterion_ref": "fixture-criterion",
    "region": DRAWER_REGION,
    "fixture": DRAWER_FIXTURE,
    "joint": DRAWER_JOINT,
    "joint_type": "slide",
    "joint_range": [-0.16, 0.01],
    "open_threshold": -0.14,
}


class Journal:
    def __init__(self, criterion=None, worker="pi0_doubled"):
        self.context = {
            **SIGNATURE,
            "relation_control": {
                "schema": "rsi-skill-routing-v1",
                "rho_ref": "fixture-rho",
                "version_signature": SIGNATURE,
                "routes": [
                    {
                        "route_id": "drawer",
                        "task_key": "old_drawer",
                        "state": "old_drawer",
                        "max_chunks": 24,
                        "selected_worker": worker,
                        "criterion": criterion or CRITERION,
                    }
                ],
            },
        }
        self.events = []

    def identity(self):
        return {"run_id": "r", "episode_id": "e", "bundle_id": "b", "protocol_id": "p"}

    def emit(self, kind, **fields):
        sequence = len(self.events)
        self.events.append(
            {**self.identity(), "kind": kind, "sequence": sequence, **fields}
        )
        return sequence


def fixture_env():
    model = SimpleNamespace(
        joint_name2id=Mock(return_value=np.int32(1)),
        get_joint_qpos_addr=Mock(return_value=np.int64(7)),
        jnt_type=np.asarray([3, 2]),
        jnt_range=np.asarray([[0.0, 0.0], [-0.16, 0.01]]),
    )
    data = SimpleNamespace(
        qpos=np.asarray([99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 0.0, 99.0])
    )
    site = SimpleNamespace(parent_name=DRAWER_FIXTURE, joints=[DRAWER_JOINT])
    arm = {
        "type": "OSC_POSE",
        "input_min": -1,
        "input_max": 1,
        "output_min": [-0.05] * 6,
        "output_max": [0.05] * 6,
        "impedance_mode": "fixed",
        "control_delta": True,
    }
    native = SimpleNamespace(
        object_sites_dict={DRAWER_REGION: site},
        sim=SimpleNamespace(model=model, data=data),
        action_spec=(-np.ones(7), np.ones(7)),
        robot_configs=[{"controller_config": arm}],
        check_success=Mock(
            side_effect=AssertionError("task success is not joint feedback")
        ),
    )
    wrapped = SimpleNamespace(
        env=native,
        step=Mock(return_value="native step"),
        reset=Mock(return_value="native reset"),
    )

    class Backend:
        def get_env_fns(self):
            return [lambda: wrapped]

    env_type = drawer_observation_libero_class(action_spec_libero_class(Backend))
    exported = env_type().get_env_fns()[0]()
    return exported, wrapped, native, site, model, data


def evidence(journal, phase="start"):
    return {
        **journal.identity(),
        "route_id": "drawer",
        "worker": journal.context["relation_control"]["routes"][0]["selected_worker"],
        "phase": phase,
        "client_route_sequence": 4,
    }


def facade_with(exported, journal):
    queried = []
    worker = SimpleNamespace(
        get_env_attr=lambda key: queried.append(key) or getattr(exported, key, None)
    )
    backend = SimpleNamespace(
        _rsi_journal=journal, env=SimpleNamespace(workers=[worker])
    )
    facade = LiberoEnvFacade(backend, meta={})
    facade.raw_obs = Mock(
        side_effect=AssertionError(
            "drawer observation must not guess raw observation keys"
        )
    )
    return facade, worker, queried


def test_worker_property_reads_current_scalar_with_native_name_and_address():
    exported, wrapped, native, site, model, data = fixture_env()
    first = getattr(exported, DRAWER_PROPERTY)
    assert first["status"] == "recorded" and first["qpos"] == 0.0
    data.qpos[7] = -0.15
    second = getattr(exported, DRAWER_PROPERTY)
    assert second["qpos"] == -0.15 and second["qpos_address"] == 7
    assert second["joint"] == DRAWER_JOINT and second["joint_type"] == "slide"
    assert second["joint_range"] == [-0.16, 0.01] and second["missing"] == []
    assert model.get_joint_qpos_addr.call_args.args == (DRAWER_JOINT,)
    assert model.joint_name2id.call_args.args == (DRAWER_JOINT,)
    assert second["measured_at_ns"] >= first["measured_at_ns"]
    assert second == json.loads(json.dumps(second))
    native.check_success.assert_not_called()
    wrapped.step.assert_not_called()
    wrapped.reset.assert_not_called()
    with pytest.raises(AttributeError):
        setattr(exported, DRAWER_PROPERTY, {"qpos": -0.15})


def test_factory_pickle_and_native_worker_api_are_preserved():
    exported, wrapped, native, _, model, _ = fixture_env()
    model.joint_name2id = lambda _: 1
    model.get_joint_qpos_addr = lambda _: 7
    wrapped.step = lambda: "native step"
    wrapped.reset = lambda: "native reset"

    def no_task_success():
        raise AssertionError("joint getter cannot query task success")

    native.check_success = no_task_success

    class Backend:
        def get_env_fns(self):
            return [lambda: wrapped]

    factory = drawer_observation_libero_class(
        action_spec_libero_class(Backend)
    )().get_env_fns()[0]
    restored = cloudpickle.loads(cloudpickle.dumps(factory))()
    assert getattr(restored, DRAWER_PROPERTY)["status"] == "recorded"
    assert restored.rpent_action_spec["default"]["low"] == [-1.0] * 7
    assert restored.step() == "native step" and restored.reset() == "native reset"


@pytest.mark.parametrize(
    "fault",
    [
        "absent_region",
        "wrong_parent",
        "no_joint",
        "multiple_joints",
        "wrong_joint",
        "vector_address",
        "wrong_type",
        "nonfinite",
        "bad_range",
        "negative_joint_id",
    ],
)
def test_missing_or_ambiguous_native_joint_stays_unknown(fault):
    exported, _, native, site, model, data = fixture_env()
    if fault == "absent_region":
        native.object_sites_dict = {}
    elif fault == "wrong_parent":
        site.parent_name = "another_cabinet"
    elif fault == "no_joint":
        site.joints = []
    elif fault == "multiple_joints":
        site.joints = [DRAWER_JOINT, "other_joint"]
    elif fault == "wrong_joint":
        site.joints = ["wooden_cabinet_1_top_level"]
    elif fault == "vector_address":
        model.get_joint_qpos_addr.return_value = (7, 8)
    elif fault == "wrong_type":
        model.jnt_type[1] = 3
    elif fault == "nonfinite":
        data.qpos[7] = np.nan
    elif fault == "bad_range":
        model.jnt_range[1, 0] = np.nan
    elif fault == "negative_joint_id":
        model.joint_name2id.return_value = -1
    snapshot = getattr(exported, DRAWER_PROPERTY)
    assert (
        snapshot["status"] == "unknown"
        and snapshot["qpos"] is None
        and snapshot["missing"]
    )


def test_private_start_and_handoff_joint_values_never_return_to_policy():
    exported, _, _, _, _, data = fixture_env()
    journal = Journal()
    facade, _, queried = facade_with(exported, journal)
    before = facade.rsi_skill_observation(evidence=evidence(journal))
    data.qpos[7] = -0.15
    after = facade.rsi_skill_observation(evidence=evidence(journal, "handoff"))
    assert before == {"source_sequence": 0, "status": "recorded"}
    assert after == {"source_sequence": 1, "status": "recorded"}
    assert queried == [DRAWER_PROPERTY, DRAWER_PROPERTY]
    assert [item["measurements"]["joint"]["qpos"] for item in journal.events] == [
        0.0,
        -0.15,
    ]
    assert [item["phase"] for item in journal.events] == ["start", "handoff"]
    assert all(item["criterion_ref"] == "fixture-criterion" for item in journal.events)
    assert all(
        "joint" not in receipt and "qpos" not in receipt for receipt in [before, after]
    )
    facade.raw_obs.assert_not_called()


@pytest.mark.parametrize("fault", ["identity", "phase", "route", "version"])
def test_rejected_request_does_not_read_worker_state(fault):
    exported, *_ = fixture_env()
    journal = Journal()
    facade, worker, queried = facade_with(exported, journal)
    request = evidence(journal)
    if fault == "identity":
        request["run_id"] = "foreign"
    elif fault == "phase":
        request["phase"] = "unregistered"
    elif fault == "route":
        request["route_id"] = "foreign"
    else:
        journal.context["code_ref"] = "changed"
    with pytest.raises(ValueError):
        facade.rsi_skill_observation(evidence=request)
    assert not queried and not journal.events


def test_worker_lookup_failure_records_unknown_without_zero_or_success_fallback():
    exported, *_ = fixture_env()
    journal = Journal()
    facade, worker, _ = facade_with(exported, journal)
    worker.get_env_attr = Mock(side_effect=RuntimeError("fixture worker unavailable"))
    receipt = facade.rsi_skill_observation(evidence=evidence(journal))
    assert receipt == {"source_sequence": 0, "status": "unknown"}
    joint = journal.events[0]["measurements"]["joint"]
    assert joint["qpos"] is None and "worker unavailable" in str(joint["missing"])


def test_lift_path_keeps_existing_xyz_and_never_queries_drawer_property():
    exported, *_ = fixture_env()
    criterion = {
        "schema": "rsi-skill-lift-criterion-v1",
        "criterion_ref": "lift",
        "object_position_key": "target_pos",
    }
    journal = Journal(criterion, worker="pi0_pick")
    facade, _, queried = facade_with(exported, journal)
    facade.raw_obs = Mock(
        return_value={
            "target_pos": [0.0, 0.0, 0.1],
            "robot0_eef_pos": [0.0, 0.0, 0.2],
            "robot0_gripper_qpos": [0.02, -0.02],
        }
    )
    assert (
        facade.rsi_skill_observation(evidence=evidence(journal))["status"] == "recorded"
    )
    assert queried == [] and journal.events[0]["measurements"]["object_pos"] == [
        0.0,
        0.0,
        0.1,
    ]


def test_client_returns_only_private_receipt_even_if_server_adds_measurements():
    client = object.__new__(LiberoEnvClient)
    client._client = SimpleNamespace(
        call=Mock(
            return_value={
                "source_sequence": 3,
                "status": "recorded",
                "qpos": -0.15,
                "measurements": {"joint": {"qpos": -0.15}},
            }
        )
    )
    assert client.rsi_skill_observation(evidence={"route_id": "drawer"}) == {
        "source_sequence": 3,
        "status": "recorded",
    }
