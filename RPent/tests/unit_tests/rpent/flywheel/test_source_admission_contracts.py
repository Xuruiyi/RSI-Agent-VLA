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

"""Preserve trajectory provenance at the real export admission boundary."""

import numpy as np
import pytest

from rpent.flywheel.episode import EpisodeWriter
from rpent.flywheel.export import _successful_episodes, episode_source


@pytest.mark.parametrize(
    "source",
    [
        "autonomous_execution",
        "public_demonstration",
        "teacher_data",
        "authorized_demonstration",
        "independent_evaluation",
    ],
)
def test_sources_reach_actual_export_admission_without_success_relabeling(
    tmp_path, source
):
    spec = {
        "arrays": {
            "states": {"shape": (1,), "dtype": "float32"},
            "actions": {"shape": (1,), "dtype": "float32"},
        },
        "success_mask": lambda data: data["terminated"],
    }
    obs = {"states": np.zeros(1)}
    writer = EpisodeWriter(
        tmp_path,
        metadata={"source_kind": source, "task_language": "move"},
        spec=spec,
        initial_observation=obs,
    )
    writer.add_transition(np.zeros(1), obs, 1, True, False)
    path = writer.finalize()
    if source in {"autonomous_execution", "independent_evaluation"}:
        with pytest.raises(ValueError, match="separate training admission"):
            _successful_episodes([path], spec=spec, expected_metadata={})
    else:
        assert (
            _successful_episodes([path], spec=spec, expected_metadata={})[0][1][
                "source_kind"
            ]
            == source
        )
    diagnosed = _successful_episodes(
        [path], spec=spec, expected_metadata={}, purpose="diagnostic"
    )
    assert episode_source(diagnosed[0][1]) == source


def test_evaluation_scope_preserves_independent_origin_despite_autonomous_source_label():
    assert (
        episode_source(
            {"source_kind": "autonomous_execution", "information_scope": "release"}
        )
        == "independent_evaluation"
    )
    assert episode_source({}) == "legacy_unspecified"
