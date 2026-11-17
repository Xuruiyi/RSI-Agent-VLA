"""Serializable offline contracts. References identify artifacts, not inferred abilities."""
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunBundle:
    bundle_id: str
    theta_ref: str
    processor_ref: str
    phi_ref: str
    rho_ref: str
    psi_ref: str
    memory_ref: str
    protocol_id: str
    backend_ref: str
    code_ref: str
    parent_bundle_id: str | None = None
    diff_ref: str | None = None


@dataclass(frozen=True)
class ExecutionEvent:
    event_id: str
    run_id: str
    episode_id: str
    sequence: int
    active_bundle: str
    kind: str
    source: str
    timestamps: dict[str, Any] = field(default_factory=dict)
    parent_decision: str | None = None
    proposal: Any = None
    sent: Any = None
    applied: Any = None
    feedback: Any = None
    execution_mask: list[bool] | None = None
    outcome: str = "unknown"
    outcome_source: str | None = None
    tool_status: str = "unknown"
    subgoal_outcome: str = "unknown"
    flags: dict[str, Any] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    media: list[str] = field(default_factory=list)
    probabilities: dict[str, float | None] = field(default_factory=lambda: {
        "media_retention": None, "outcome_label": None, "repair_selection": None})

    def __post_init__(self):
        if self.outcome not in {"success", "failure", "unknown"}:
            raise ValueError("outcome must be success/failure/unknown")
        if self.subgoal_outcome not in {"success", "failure", "unknown"}:
            raise ValueError("invalid subgoal outcome")
        if self.sequence < 0:
            raise ValueError("negative event sequence")


@dataclass(frozen=True)
class EvidenceView:
    view_id: str
    bundle_id: str
    l0: list[dict[str, Any]]
    selected_media: list[str]
    policy_ref: str
    selection: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentResult:
    result_id: str
    bundle_id: str
    protocol_id: str
    benchmark_id: str
    role: str
    evidence_level: str
    intervention: dict[str, Any]
    data_refs: list[str]
    criteria_ref: str
    metrics: dict[str, float | None]
    hard_constraints: dict[str, bool | None]
    costs: dict[str, float]
    candidate_id: str | None = None
    unknown_count: int = 0
    invalid_count: int = 0
    assigned_count: int = 0
    outcome: str = "unknown"
    notes: list[str] = field(default_factory=list)
    assignments: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    base_bundle_id: str
    bundle_id: str
    protocol_id: str
    benchmark_id: str
    changes: dict[str, str]
    artifact_refs: list[str]
    development_results: list[str]
    costs: dict[str, float]
    compatibility: dict[str, Any]
    approval_scope: str = "offline"
    novelty_type: str = "configuration"
    approval_ref: str | None = None


@dataclass(frozen=True)
class GateDecision:
    decision_id: str
    candidate_id: str
    base_bundle_id: str
    bundle_id: str
    protocol_id: str
    benchmark_id: str
    verdict: str
    reasons: list[str]
    release_eligible: bool
    scope: str
    scores: dict[str, float | None]
    result_refs: list[str]


def record(value):
    return asdict(value)
