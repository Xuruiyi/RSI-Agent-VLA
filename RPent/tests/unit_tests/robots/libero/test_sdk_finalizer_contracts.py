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

"""Keep raw SDK diagnostics while closing only bound local budget failures."""

import json
from copy import deepcopy

import pytest

from robots.libero.result import finalize_run
from rpent.evaluation import RunFinalizationContext
from rpent.session.runtime import RuntimeJournal


@pytest.mark.parametrize(
    "receipt", ["bound", "missing", "wrong_identity", "unforwarded_unproven"]
)
def test_failed_sdk_budget_turn_requires_local_receipt(tmp_path, receipt):
    output = tmp_path / "run"
    journal = RuntimeJournal(
        {
            "run_id": "r",
            "episode_id": "e",
            "bundle_id": "b",
            "protocol_id": "p",
            "output_dir": str(output),
            "limits": {"model_requests": 20},
        }
    )
    identity = journal.identity()
    journal.mark_budget_stop("model_requests", limit=20, observed=20)
    marker = journal.budget_stop()
    sequence = journal.emit("planner_budget_stop", budget_stop=marker)
    local = {
        "schema_version": "1.0",
        "identity": identity,
        "reason": "budget_exhausted",
        "forwarded": False,
        "marker_ref": str(journal.budget_stop_path),
        "marker_sequence": marker["sequence"],
    }
    path = output / "local_model_stop.json"
    if receipt == "wrong_identity":
        local = {**local, "identity": {**identity, "run_id": "other-run"}}
    if receipt == "unforwarded_unproven":
        local = {**local, "forwarded": None}
    if receipt != "missing":
        path.write_text(json.dumps(local))
    stats = {
        "attempt_closed": True,
        "planner_turn_status": "failed",
        "sdk_errors": [
            {
                "error": {"message": "exceeded retry limit", "http_status": 429},
                "local_rejection_candidate": True,
            }
        ],
        "planner_budget_stop": {
            **marker,
            "budget_stop_ref": str(journal.budget_stop_path),
            "planner_event_sequence": sequence,
            "planner_events_ref": str(journal.events_path),
            "local_model_stop": local,
            "local_model_stop_ref": str(path),
        },
    }
    original_stats = deepcopy(stats)
    service = output / "service.json"
    service.write_text(
        json.dumps(
            {
                "identity": identity,
                "service_closed": True,
                "pending_action": False,
                "environment_steps_unknown": 0,
                "last_native_success": False,
                "last_feedback_sequence": 7,
                "native_success_source": "worker-native-check",
            }
        )
    )
    result = json.loads(
        finalize_run(
            RunFinalizationContext(
                output_dir=output,
                robot_name="libero",
                task_desc={"seed": 3},
                environment_success=None,
                agent_error=None,
                elapsed_s=1,
                planner="codex",
                model="gpt-6.1-sol",
                reasoning_effort="medium",
                max_turns=20,
                planner_timeout_s=600,
                finish_result=None,
                stats=stats,
                environment_metrics={
                    "terminated": False,
                    "truncated": False,
                    "runtime": {
                        "identity": identity,
                        "action_events_ref": str(journal.events_path),
                        "environment_service": {"costs_ref": str(service)},
                    },
                },
            )
        ).read_text()
    )
    assert result["planner_diagnostics"]["sdk_errors"] == original_stats["sdk_errors"]
    assert result["planner_diagnostics"]["planner_turn_status"] == "failed"
    assert result["verified_budget_exhaustion"] is (receipt == "bound")
    assert result["benchmark_success"] is (False if receipt == "bound" else None)
    assert stats == original_stats
