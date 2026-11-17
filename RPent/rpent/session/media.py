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

"""Opt-in rhoA selection at the optional development-media persistence outlet."""

from __future__ import annotations

from pathlib import Path
from typing import Any

OPTIONAL_DERIVATIVES = frozenset({"agentview_world_high.npz", "wrist_world_high.npz"})


class MediaPersistencePolicy:
    """Keep control, audit and evaluation artifacts outside research selection.

    Decisions happen before encoding. The byte cap is checked against the actual
    encoded temporary file before publication. Temporary writes remain costs.
    Whole-process accounting remains the authority for the complete output tree.
    """

    def __init__(self, journal: Any, config: dict[str, Any]) -> None:
        self.journal = journal
        context = journal.context
        binding = context["execution_binding"]
        self.rho_ref = binding["bundle"]["rho_ref"]
        if (
            config.get("schema") != "rsi-l1-media-policy-v1"
            or config.get("rho_ref") != self.rho_ref
        ):
            raise ValueError("L1 policy differs from the frozen rho binding")
        self.optional = set(config["optional_artifacts"])
        if not self.optional <= OPTIONAL_DERIVATIVES:
            raise ValueError(
                "L1 policy may select only optional high-world derivatives"
            )
        self.optional -= set(config.get("protected_artifacts", []))
        self.limit = config.get("max_persisted_bytes")
        if type(self.limit) is not int or self.limit < 0:
            raise ValueError("L1 outlet needs an explicit nonnegative byte budget")
        self.scope = context["information_scope"]
        self.protected_steps = set(config.get("protected_steps", []))
        self.unused_by_control = (
            context.get("execution_arm", {}).get("arm") in {"D", "S", "R"}
            and context.get("planner", {}).get("backend") == "flash"
            and not context.get("relation_control")
        )
        policy = binding["records"]["rho_ref"]["rhoA"]
        if policy.get("type") == "fixed":
            if (
                policy.get("retain_l0") != "all"
                or policy.get("media") != "all_available"
            ):
                raise ValueError("unsupported fixed L1 policy")
            self.action = "all_available"
        elif policy.get("type") == "rsi-relation-feedback-v1":
            values = policy.get("values", {}).get("media", {})
            measured = [
                name
                for name in ("all_available", "one_per_event")
                if values.get(name, {}).get("count", 0)
            ]
            self.action = (
                max(
                    measured,
                    key=lambda name: values[name]["total"] / values[name]["count"],
                )
                if measured
                else "all_available"
            )
        else:
            raise ValueError("unsupported L1 rhoA policy")
        self.persisted: dict[str, int] = {}
        for name in self.optional:
            for path in (journal.output_dir / name).glob("*"):
                if path.is_file():
                    self.persisted[str(path)] = path.stat().st_size
        self.cumulative = 0
        self.temporary_peak = 0

    def protect_step(self, step: int) -> None:
        """Keep terminal observations and explicitly required audit states."""
        self.protected_steps.add(step)

    def _optional(self, name: str, step: int | None) -> bool:
        return (
            self.scope == "development"
            and self.unused_by_control
            and name in self.optional
            and step not in self.protected_steps
        )

    def select(self, name: str, step: int | None) -> bool:
        """Record a deterministic retention mechanism without selecting L0."""
        optional = self._optional(name, step)
        selected = (
            not optional or self.action == "all_available" or name == min(self.optional)
        )
        self.journal.emit(
            "media_retention_selection",
            artifact=name,
            step=step,
            rho_ref=self.rho_ref,
            information_scope=self.scope,
            protected=not optional,
            selected=selected,
            mechanism=self.action if optional else "protected",
            probability=float(selected),
            probability_basis="deterministic_conditional_on_bound_policy",
            population_coverage="selected_outlet_only; zero coverage cannot recover unrecorded media",
            selection_target="L1_persistence",
            outcome_label_probability=None,
            repair_selection_probability=None,
        )
        return selected

    def admit(
        self, name: str, step: int | None, temporary: Path, destination: Path
    ) -> bool:
        """Measure actual temporary bytes, then enforce optional publication cap."""
        if not self._optional(name, step):
            return True
        size = temporary.stat().st_size
        self.temporary_peak = max(self.temporary_peak, size)
        self.cumulative += size
        current = sum(self.persisted.values())
        admitted = (
            current - self.persisted.get(str(destination), 0) + size <= self.limit
        )
        self.journal.emit(
            "media_persistence_admission",
            artifact=name,
            step=step,
            rho_ref=self.rho_ref,
            admitted=admitted,
            encoded_bytes=size,
            temporary_bytes=size,
            persistent_bytes=current,
            sampled_temporary_peak_bytes=self.temporary_peak,
            cumulative_new_bytes_lower_bound=self.cumulative,
            max_persisted_bytes=self.limit,
            probability=float(admitted),
            mechanism="actual_encoded_bytes_within_optional_outlet_budget",
            deletion_refunds_cumulative_cost=False,
        )
        return admitted

    def materialized(self, name: str, step: int | None, destination: Path) -> None:
        """Publish the physical artifact receipt after the atomic write."""
        if self._optional(name, step):
            self.persisted[str(destination)] = destination.stat().st_size
        self.journal.emit(
            "media_materialized",
            artifact=name,
            step=step,
            rho_ref=self.rho_ref,
            path=str(destination),
            persisted_bytes=destination.stat().st_size,
            optional_persistent_bytes=sum(self.persisted.values()),
            protected=not self._optional(name, step),
            cumulative_new_bytes_lower_bound=self.cumulative,
        )


def configure_media_policy(state: Any, journal: Any) -> None:
    """Enable a launcher-bound outlet policy; ordinary runs keep all media."""
    config = journal.context.get("l1_media_policy")
    if config is not None:
        state.media_policy = MediaPersistencePolicy(journal, config)
