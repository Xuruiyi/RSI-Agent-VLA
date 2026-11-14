"""Independent deployment qualification from frozen comparison evidence mappings.

Raw comparison identities and decisions remain immutable. The registration is a
protected operator input; each consumer also checks its mounted OS write domain.
"""

import difflib
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from . import qualification_import as trust
from .contracts import Candidate, ExperimentResult, RunBundle
from .evaluation import EvaluationStore, Gate
from .storage import ResearchStore

BEHAVIOR_FIELDS = (
    "model",
    "processor",
    "phi",
    "legal_memory",
    "planner_input",
    "actions",
    "action_timing",
    "budget_stop",
    "collection",
)
CRITERIA_FIELDS = (
    "scope",
    "benchmark_id",
    "tasks",
    "metrics",
    "hard_constraints",
    "min_gain",
    "anchor_max_drop",
    "cost_weights",
    "conditional_objective",
    "initial_requirements",
    "recovery",
    "outcomes",
    "state_evaluation",
    "asset_binding",
)


def _read(ref):
    return json.loads(Path(ref).read_text(encoding="utf-8"))


def _time(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("upgrade timestamps require a timezone")
    return parsed


def _protected(ref, registration, *, operator=False):
    path = Path(ref).resolve(strict=True)
    if not trust._readonly(path) or (
        operator
        and not path.is_relative_to(
            Path(registration["role_domains"]["operator"]).resolve()
        )
    ):
        raise ValueError("upgrade input must be protected: " + str(path))
    return path


def _pointer(value, parts):
    for part in parts:
        value = value[part]
    return value


def upgrade_authorization(reference, *, role, bundle_id=None, protocol_id=None):
    """Consume exact, current operator authority without granting a JSON role."""
    value = _read(reference)
    if (
        value.get("version") != "candidate-upgrade-registration-v1"
        or value.get("approved") is not True
        or value.get("formal_results_observed") != 0
        or value.get("publication_authorized") is not True
        or value.get("inheritance_authorized") is not True
        or not value.get("scope")
        or _time(value["registered_at"]) >= _time(value["expires_at"])
        or _time(value["expires_at"]) <= datetime.now(timezone.utc)
    ):
        raise ValueError("candidate upgrade authorization missing or expired")
    trust.require_role(value, role)
    _protected(reference, value, operator=True)
    if (bundle_id is not None and bundle_id != value["target_bundle_id"]) or (
        protocol_id is not None
        and protocol_id != value["target_protocol_snapshot"]["protocol_id"]
    ):
        raise ValueError("candidate upgrade authorization target mismatch")
    return value


def _validate_projection(source, registration):
    projection = source.get("runtime_projection")
    if projection is None:
        return
    from .improvement_protocol import runtime_projection

    frozen_ref = _protected(projection["source_frozen_protocol_ref"], registration)
    authority_ref = _protected(
        projection["authorization_ref"], registration, operator=True
    )
    frozen = _read(frozen_ref)
    if source != runtime_projection(
        frozen, frozen_ref=frozen_ref, authorization_ref=authority_ref
    ):
        raise ValueError(
            "upgrade comparison protocol is not the exact authorized frozen projection"
        )


def _mapping(registration):
    source = registration["source_protocol_snapshot"]
    target = registration["target_protocol_snapshot"]
    _validate_projection(source, registration)
    for name in ("source", "target"):
        ref = _protected(
            registration[name + "_protocol_ref"],
            registration,
            operator=name == "target",
        )
        if _read(ref) != registration[name + "_protocol_snapshot"]:
            raise ValueError("upgrade protocol snapshot mismatch")
    if (
        source.get("comparison_only") is not True
        or source.get("status") != "frozen"
        or target.get("status") != "frozen"
        or target.get("comparison_only") is not False
        or source["protocol_id"] == target["protocol_id"]
        or source["anchor_bundle_id"] != registration["source_baseline_bundle_id"]
        or target["anchor_bundle_id"] != registration["active_snapshot"]["bundle_id"]
        or target.get("deployment_derivation", {}).get("kind") != "candidate_upgrade"
        or target.get("candidate_upgrade_derivation", {}).get("source_protocol_id")
        != source["protocol_id"]
    ):
        raise ValueError("upgrade protocol identity mapping mismatch")
    for key in CRITERIA_FIELDS:
        if source.get(key) != target.get(key):
            raise ValueError("upgrade changes frozen criterion: " + key)
    for key in ("runtime_config",):
        ignored = {"profile_reference", "authority"}
        if {k: v for k, v in source.get(key, {}).items() if k not in ignored} != {
            k: v for k, v in target.get(key, {}).items() if k not in ignored
        }:
            raise ValueError("upgrade changes evaluated runtime behavior")
    if source.get("budget", {}).get("per_attempt_proposal") != target.get(
        "budget", {}
    ).get("per_attempt_proposal"):
        raise ValueError("upgrade changes per-attempt budget/stop rules")
    for split, original in source.get("task_sets", {}).items():
        derived = target.get("task_sets", {}).get(split, {})
        if {k: v for k, v in original.items() if k != "authorized"} != {
            k: v for k, v in derived.items() if k != "authorized"
        }:
            raise ValueError("upgrade changes fixed assignment matrix")
    if (
        any(
            registration.get("applicability", {}).get(key) is not True
            for key in (
                "hard_constraints",
                "score_criteria",
                "current_active_qualification",
                "candidate_behavior_preserved",
            )
        )
        or not registration.get("source_manifests")
        or not registration.get("execution_sources")
    ):
        raise ValueError("upgrade applicability/source review missing")
    for manifest in registration["source_manifests"]:
        if _read(_protected(manifest["ref"], registration)) != manifest["snapshot"]:
            raise ValueError("upgrade source manifest changed")
    for key in BEHAVIOR_FIELDS:
        bindings = registration.get("behavior_review", {}).get(key, [])
        if not bindings:
            raise ValueError("upgrade behavior review missing: " + key)
        for binding in bindings:
            if not binding.get("reason"):
                raise ValueError("upgrade behavior applicability reason missing")
            for side in ("source", "target"):
                actual = _pointer(
                    _read(_protected(binding[side + "_ref"], registration)),
                    binding[side + "_path"],
                )
                if actual != binding[side + "_value"]:
                    raise ValueError("upgrade behavior binding changed: " + key)
    research = ResearchStore(registration["role_domains"]["research"])
    evaluated = research.get("bundles", registration["source_candidate_bundle_id"])
    evaluated_ref = Path(evaluated["backend_ref"]).resolve()
    evaluated_profile = _read(evaluated_ref)
    profile_targets = {
        Path(binding["target_ref"]).resolve()
        for binding in registration["behavior_review"]["planner_input"]
        if Path(binding["source_ref"]).resolve() == evaluated_ref
        and binding["source_path"] == ["planner"]
        and binding["target_path"] == ["planner"]
        and _read(binding["target_ref"]).get("profile_id")
        == registration["target_profile_id"]
        and _read(binding["target_ref"]).get("authorization_reference")
        == registration["registration_ref"]
    }
    for field in ("relation_routes", "relation_skill_starts"):
        targets = {ref: _read(ref) for ref in profile_targets}
        if field not in evaluated_profile and not any(
            field in value for value in targets.values()
        ):
            continue
        if not targets or field not in evaluated_profile:
            raise ValueError("upgrade relation profile mapping missing: " + field)
        for ref, profile in targets.items():
            bindings = [
                binding
                for binding in registration["behavior_review"].get(field, [])
                if Path(binding["source_ref"]).resolve() == evaluated_ref
                and Path(binding["target_ref"]).resolve() == ref
                and binding["source_path"] == binding["target_path"] == [field]
            ]
            if len(bindings) != 1 or field not in profile:
                raise ValueError("upgrade relation profile mapping missing: " + field)
            binding = bindings[0]
            before = _read(_protected(evaluated_ref, registration))[field]
            after = _read(_protected(ref, registration))[field]
            if (
                not binding.get("reason")
                or before != after
                or before != binding["source_value"]
                or after != binding["target_value"]
            ):
                raise ValueError("upgrade changes evaluated relation profile: " + field)
    for entry in registration["execution_sources"]:
        before = _protected(entry["source_ref"], registration).read_text(
            encoding="utf-8"
        )
        after = _protected(entry["target_ref"], registration).read_text(
            encoding="utf-8"
        )
        delta = "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=entry["source_ref"],
                tofile=entry["target_ref"],
            )
        )
        if not entry.get("reason") or delta != entry.get("unified_diff"):
            raise ValueError("upgrade execution source difference is not reviewed")


def _stores(registration):
    from .release import ReleaseManager

    domains = registration["role_domains"]
    research = ResearchStore(domains["research"])
    evaluation = EvaluationStore(domains["evaluation"])
    return ReleaseManager(domains["release"], research, evaluation)


def _derived(registration, research):
    source = Candidate(
        **research.get("candidates", registration["source_candidate_id"])
    )
    original = research.get("bundles", source.bundle_id)
    if (
        source.bundle_id != registration["source_candidate_bundle_id"]
        or source.base_bundle_id != registration["source_baseline_bundle_id"]
        or source.protocol_id != registration["source_protocol_snapshot"]["protocol_id"]
    ):
        raise ValueError("upgrade source candidate mapping mismatch")
    profile = _read(original["backend_ref"])
    profile.update(
        profile_id=registration["target_profile_id"],
        dynamic_authorized=True,
        release_authorized=False,
        deployment_authorized=True,
        authorization_reference=registration["registration_ref"],
        changed_configuration="Candidate deployment identity; evaluated behavior preserved",
    )
    profile_ref = research.path(
        "deployment_profiles", registration["target_profile_id"]
    )
    bundle = {
        **original,
        "bundle_id": registration["target_bundle_id"],
        "protocol_id": registration["target_protocol_snapshot"]["protocol_id"],
        "parent_bundle_id": registration["active_snapshot"]["bundle_id"],
        "backend_ref": str(profile_ref.resolve()),
    }
    base = research.get("bundles", registration["active_snapshot"]["bundle_id"])
    fields = (
        "theta_ref",
        "processor_ref",
        "phi_ref",
        "rho_ref",
        "psi_ref",
        "memory_ref",
        "backend_ref",
        "code_ref",
        "diff_ref",
    )
    changes = {
        key: bundle.get(key) for key in fields if bundle.get(key) != base.get(key)
    }
    candidate = replace(
        source,
        candidate_id=registration["target_candidate_id"],
        base_bundle_id=base["bundle_id"],
        bundle_id=bundle["bundle_id"],
        protocol_id=bundle["protocol_id"],
        changes=changes,
        compatibility={
            **source.compatibility,
            "upgrade_registration_ref": registration["registration_ref"],
            "source_candidate_id": source.candidate_id,
        },
    )
    return bundle, profile, asdict(candidate)


def _registration(reference, role):
    registration = upgrade_authorization(reference, role=role)
    if registration.get("registration_ref") != str(Path(reference).resolve()):
        raise ValueError("upgrade registration self-reference mismatch")
    for key in ("deployment_derivation", "candidate_upgrade_derivation"):
        if (
            registration["target_protocol_snapshot"]
            .get(key, {})
            .get(
                "authorization_ref"
                if key == "deployment_derivation"
                else "registration_ref"
            )
            != registration["registration_ref"]
        ):
            raise ValueError("upgrade target authorization provenance mismatch")
    _mapping(registration)
    return registration


def prepare_candidate_deployment(research_store, *, registration_ref):
    """Research materializes the frozen identity transformation in its own domain."""
    registration = _registration(registration_ref, "research")
    if (
        research_store.root.resolve()
        != Path(registration["role_domains"]["research"]).resolve()
    ):
        raise ValueError("candidate preparation store is outside research domain")
    bundle, profile, candidate = _derived(registration, research_store)
    research_store.put(
        "deployment_profiles", registration["target_profile_id"], profile
    )
    research_store.save_bundle(RunBundle(**bundle))
    research_store.save_candidate(Candidate(**candidate))
    return {"bundle": bundle, "candidate": candidate, "profile": profile}


def _check_evidence(registration, evaluation):
    source = Candidate(
        **_stores(registration).research.get(
            "candidates", registration["source_candidate_id"]
        )
    )
    results = {}
    for side, identity in registration["source_result_ids"].items():
        path = _protected(evaluation.path("results", identity), registration)
        if path.stat().st_mtime < _time(registration["registered_at"]).timestamp():
            raise ValueError("upgrade mapping was not frozen before formal results")
        result = ExperimentResult(**evaluation.get("results", identity))
        for ref in result.data_refs + [
            row["evidence_ref"] for row in result.assignments if row.get("evidence_ref")
        ]:
            evidence_path = _protected(ref, registration)
            if (
                evidence_path.stat().st_mtime
                < _time(registration["registered_at"]).timestamp()
            ):
                raise ValueError(
                    "upgrade mapping was not frozen before original evidence"
                )
        if not result.data_refs:
            raise ValueError("upgrade independent evidence is missing")
        results[side] = result
    canonical_gate = _protected(registration["source_decision_ref"], registration)
    if canonical_gate.parent == (evaluation.root / "decisions").resolve():
        decision_id = canonical_gate.stem
        resolved_gate = _protected(evaluation.path("decisions", decision_id), registration)
        saved_gate = evaluation.get("decisions", decision_id)
        if resolved_gate != canonical_gate:
            pointer = _read(_protected(evaluation.root / "gate_evaluation_current" / (decision_id + ".json"), registration))
            receipt = _read(_protected(pointer["receipt_ref"], registration))
            if (receipt.get("protocol_snapshot") != registration["source_protocol_snapshot"]
                    or receipt.get("candidate_snapshot") != asdict(source)
                    or receipt.get("result_ids") != registration["source_result_ids"]
                    or receipt.get("current_bundle_id") != source.base_bundle_id
                    or receipt.get("result_version_refs") != {
                        side: str(evaluation.path("results", identity).resolve())
                        for side, identity in registration["source_result_ids"].items()}):
                raise ValueError("upgrade Gate revision differs from frozen mapping/current evidence")
    else:
        # Existing frozen registrations may point directly to a standalone raw Gate.
        saved_gate = _read(canonical_gate)
    gate = Gate(registration["source_protocol_snapshot"]).evaluate(
        saved_gate["decision_id"],
        source,
        current_bundle_id=source.base_bundle_id,
        candidate_result=results["candidate"],
        baseline_result=results["baseline"],
        anchor_result=results["anchor"],
    )
    if asdict(gate) != saved_gate or gate.verdict != "accept" or gate.release_eligible:
        raise ValueError("upgrade requires independently reproduced comparison accept")
    return results, saved_gate


def _gate_resolution(registration, evaluation):
    canonical = Path(registration["source_decision_ref"]).resolve()
    pointer_ref = evaluation.root / "gate_evaluation_current" / canonical.name
    pointer = _read(_protected(pointer_ref, registration))
    receipt_ref = _protected(pointer["receipt_ref"], registration)
    return {"canonical_ref": str(canonical),
            "version_ref": str(_protected(evaluation.path("decisions", canonical.stem), registration)),
            "current_ref": str(pointer_ref.resolve()), "current_snapshot": pointer,
            "receipt_ref": str(receipt_ref), "receipt_snapshot": _read(receipt_ref)}


def _source_execution_bindings(registration, manager, results):
    from .runtime import execution_binding

    source = manager.research.get("candidates", registration["source_candidate_id"])
    original = manager.research.get("bundles", source["bundle_id"])
    base = manager.research.get("bundles", source["base_bundle_id"])
    fields = (
        "theta_ref",
        "processor_ref",
        "phi_ref",
        "rho_ref",
        "psi_ref",
        "memory_ref",
        "backend_ref",
        "code_ref",
        "diff_ref",
    )
    if (
        source["changes"]
        != {key: original[key] for key in fields if original[key] != base[key]}
        or original["parent_bundle_id"] != base["bundle_id"]
    ):
        raise ValueError(
            "upgrade source Candidate changes/references differ from evaluated bundles"
        )
    bindings = {}
    for side, result in results.items():
        bundle = manager.research.get("bundles", result.bundle_id)
        expected = execution_binding(
            manager.research, bundle, _read(bundle["backend_ref"])
        )
        if side == "candidate" and expected["candidates"] != [source]:
            raise ValueError("upgrade source Candidate identity is not unique")
        if not result.assignments:
            raise ValueError("upgrade evaluated execution binding is missing")
        for row in result.assignments:
            native = _protected(row["evidence_ref"], registration)
            context = _read(
                _protected(native.parent / "runtime_context.json", registration)
            )
            if (
                context.get("bundle_id") != result.bundle_id
                or context.get("protocol_id") != result.protocol_id
                or context.get("execution_binding") != expected
            ):
                raise ValueError(
                    "upgrade source inputs differ from evaluated execution binding"
                )
        bindings[side] = expected
    return bindings


def _active_qualification_sources(registration, *, _ancestry=()):
    """Locate a current initial or published-upgrade qualification and its inputs."""
    active = registration["active_snapshot"]
    root = Path(registration["role_domains"]["release"])
    activation = active["activation"]
    if active.get("qualification_ref"):
        identity = active["qualification_ref"]
    else:
        link = _read(root / "upgrade_decisions" / (activation + ".json"))
        identity = link["qualification_id"]
    reference = root / "qualifications" / (identity + ".json")
    qualification = _read(reference)
    domains = qualification.get("authorization_snapshot", {}).get("role_domains")
    if domains:
        inputs = [
            Path(domains[name]) for name in ("operator", "research", "evaluation")
        ]
    else:
        evaluation = EvaluationStore(registration["role_domains"]["evaluation"])
        inputs = [evaluation.path("protocols", qualification["protocol_id"])]
        inputs.extend(
            evaluation.path("results", result)
            for result in qualification["result_refs"]
        )
    if qualification["kind"] == "evaluated_candidate_upgrade":
        research = ResearchStore(registration["role_domains"]["research"])
        current = qualification["bundle_snapshot"]
        inputs.append(research.path("bundles", current["bundle_id"]))
        inputs.append(Path(current["backend_ref"]))
        inputs.extend(
            research.path(kind, current[field])
            for kind, field in (
                ("rho", "rho_ref"),
                ("psi", "psi_ref"),
                ("memory", "memory_ref"),
            )
        )
        inputs.extend(
            root / kind / (activation + ".json")
            for kind in (
                "upgrade_decisions",
                "upgrade_publications",
                "activations",
                "decisions",
                "decision_bases",
            )
        )
        evaluation = EvaluationStore(registration["role_domains"]["evaluation"])
        inputs.append(
            evaluation.path(
                "candidate_evidence_imports", qualification["upgrade_import_id"]
            )
        )
        inputs.append(Path(qualification["registration_ref"]))
        inputs.extend(
            isolation_paths(
                {},
                {
                    "action": "candidate_evidence_import",
                    "registration_ref": qualification["registration_ref"],
                },
                _ancestry=_ancestry,
            )
        )
    return reference, inputs


def _original_active_qualification(registration, manager, *, role="evaluation"):
    """Verify the existing qualification under its original independent criteria."""
    from .release import ReleaseManager, _validate_initial_result

    reference, inputs = _active_qualification_sources(registration)
    qualification = _read(_protected(reference, registration))
    for path in inputs:
        _protected(path, registration)
    active = registration["active_snapshot"]
    bundle = manager.research.get("bundles", active["bundle_id"])
    if (
        qualification["qualification_id"] != reference.stem
        or qualification["bundle_id"] != active["bundle_id"]
        or qualification["bundle_snapshot"] != bundle
        or qualification["protocol_id"] != bundle["protocol_id"]
        or qualification["scope"] != active["scope"]
        or qualification["kind"]
        not in {"initial_effective", "evaluated_candidate_upgrade"}
    ):
        raise ValueError("original active qualification identity/snapshot mismatch")
    if qualification["kind"] == "evaluated_candidate_upgrade":
        previous = validate_upgrade_qualification(manager, qualification, role=role)
        identity = active["activation"]
        link = manager.records.get("upgrade_decisions", identity)
        publication = manager.records.get("upgrade_publications", identity)
        imported = manager.evaluations.get(
            "candidate_evidence_imports", qualification["upgrade_import_id"]
        )
        activation = manager.records.get("activations", identity)
        expected_active = dict(
            bundle_id=qualification["bundle_id"],
            scope=qualification["scope"],
            activation=identity,
            revision=previous["active_snapshot"]["revision"] + 1,
        )
        expected_link = dict(
            registration_ref=qualification["registration_ref"],
            import_id=qualification["upgrade_import_id"],
            qualification_id=qualification["qualification_id"],
        )
        expected_activation = dict(
            bundle_id=active["bundle_id"],
            previous=previous["active_snapshot"]["bundle_id"],
            decision_id=identity,
            scope=active["scope"],
            safe_point=activation["safe_point"],
        )
        if (
            active != expected_active
            or qualification["decision_ref"] != identity
            or link != expected_link
            or publication["decision_id"] != identity
            or publication["active"] != active
            or publication["previous_active"] != previous["active_snapshot"]
            or publication["import_snapshot"] != imported
            or manager.records.get("decision_bases", identity)
            != previous["active_snapshot"]
            or manager.records.get("decisions", identity)
            != {**imported["deployment_gate"], "decision_id": identity}
            or activation != expected_activation
            or activation["safe_point"] not in {"task_boundary", "approved_safe_point"}
        ):
            raise ValueError(
                "current upgrade publication/qualification ancestry mismatch"
            )
        return {
            "qualification_ref": str(reference),
            "qualification_snapshot": qualification,
        }
    domains = qualification.get("authorization_snapshot", {}).get("role_domains")
    if qualification.get("evidence_import_id"):
        historical = ReleaseManager(
            domains["release"],
            ResearchStore(domains["research"]),
            EvaluationStore(domains["evaluation"]),
        )
        if historical.records.root.resolve() != manager.records.root.resolve():
            raise ValueError("original qualification release store mismatch")
        # The original permission is immutable historical evidence. This task's
        # registration supplies current authority; no old campaign is reopened.
        historical._validate_imported_qualification(
            qualification, role="evaluation", renewal=registration
        )
    else:
        if len(qualification["result_refs"]) != 1:
            raise ValueError(
                "original initial qualification requires one independent result"
            )
        original = ExperimentResult(
            **manager.evaluations.get("results", qualification["result_refs"][0])
        )
        original_protocol = manager.evaluations.get(
            "protocols", qualification["protocol_id"]
        )
        requirements = _validate_initial_result(original_protocol, bundle, original)
        if qualification["requirements"] != requirements:
            raise ValueError("original active qualification requirements changed")
    return {
        "qualification_ref": str(reference),
        "qualification_snapshot": qualification,
    }


def _current_applicability(registration, manager, baseline, *, role="evaluation"):
    """Preserve original qualification and establish fresh comparison applicability."""
    from .release import inspect_initial_comparison_result

    original = _original_active_qualification(registration, manager, role=role)
    source = registration["source_protocol_snapshot"]
    source_bundle = manager.research.get(
        "bundles", registration["source_baseline_bundle_id"]
    )
    comparison = inspect_initial_comparison_result(source, source_bundle, baseline)
    if comparison["passed"] is not True:
        raise ValueError(
            "current active comparison applicability missing: " + comparison["reason"]
        )
    return {
        "passed": True,
        **original,
        "comparison": comparison,
        "reinitialized": False,
    }


def _build_import(registration, manager, import_id, *, role="evaluation"):
    from .evaluation import assignment_metrics
    from .release import _validate_initial_capability

    bundle, profile, candidate = _derived(registration, manager.research)
    if (
        manager.research.get("bundles", bundle["bundle_id"]) != bundle
        or manager.research.get("candidates", candidate["candidate_id"]) != candidate
        or _read(bundle["backend_ref"]) != profile
    ):
        raise ValueError(
            "upgrade deployment behavior differs from frozen transformation"
        )
    results, source_gate = _check_evidence(registration, manager.evaluations)
    source_bindings = _source_execution_bindings(registration, manager, results)
    protocol = registration["target_protocol_snapshot"]
    mapped = {}
    for side, result in results.items():
        mapped[side] = replace(
            result,
            result_id=import_id + "-" + side,
            bundle_id=bundle["bundle_id"]
            if side == "candidate"
            else registration["active_snapshot"]["bundle_id"],
            protocol_id=protocol["protocol_id"],
            candidate_id=candidate["candidate_id"] if side == "candidate" else None,
            notes=result.notes
            + [
                "Derived evaluation version; original result identity preserved: "
                + result.result_id
            ],
        )
    base = manager.research.get("bundles", registration["active_snapshot"]["bundle_id"])
    applicability = _current_applicability(
        registration, manager, results["baseline"], role=role
    )
    # Preserve every frozen capability requirement for the new deployment.
    requirements = protocol.get("initial_requirements")
    if not requirements:
        raise ValueError(
            "candidate deployment requires frozen initial capability requirements"
        )
    _validate_initial_capability(
        requirements,
        assignment_metrics(protocol, mapped["candidate"]),
        mapped["candidate"],
    )
    gate = Gate(protocol).evaluate(
        import_id + "-deployment-gate",
        Candidate(**candidate),
        current_bundle_id=base["bundle_id"],
        candidate_result=mapped["candidate"],
        baseline_result=mapped["baseline"],
        anchor_result=mapped["anchor"],
    )
    if gate.verdict != "accept" or not gate.release_eligible:
        raise ValueError("independent deployment qualification failed")
    return {
        "import_id": import_id,
        "registration_ref": registration["registration_ref"],
        "registration_snapshot": registration,
        "source_gate": source_gate,
        **({"source_gate_resolution": _gate_resolution(registration, manager.evaluations)}
           if Path(registration["source_decision_ref"]).resolve().parent
           == (manager.evaluations.root / "decisions").resolve()
           and (manager.evaluations.root / "gate_evaluation_current" / (source_gate["decision_id"] + ".json")).is_file()
           else {}),
        "source_execution_bindings": source_bindings,
        "source_results": {k: asdict(v) for k, v in results.items()},
        "derived_results": {k: asdict(v) for k, v in mapped.items()},
        "deployment_gate": asdict(gate),
        "current_active_applicability": applicability,
        "candidate_initial_requirements": {
            "passed": True,
            "requirements": requirements,
        },
        "bundle_snapshot": bundle,
        "candidate_snapshot": candidate,
        "profile_snapshot": profile,
        "original_results_relabelled": False,
    }


def import_candidate_evidence(evaluation_store, *, import_id, registration_ref):
    """Evaluation independently imports read-only facts into new evaluation versions."""
    registration = _registration(registration_ref, "evaluation")
    manager = _stores(registration)
    if evaluation_store.root.resolve() != manager.evaluations.root.resolve():
        raise ValueError("candidate evidence store is outside evaluation domain")
    if manager.active() != registration["active_snapshot"]:
        raise ValueError("upgrade effective active/revision changed")
    value = _build_import(registration, manager, import_id)
    evaluation_store.put(
        "protocols",
        registration["target_protocol_snapshot"]["protocol_id"],
        registration["target_protocol_snapshot"],
    )
    for result in value["derived_results"].values():
        evaluation_store.save_result(ExperimentResult(**result))
    evaluation_store.put("candidate_evidence_imports", import_id, value)
    return value


def _validate_import(manager, reference, import_id, *, role):
    registration = _registration(reference, role)
    actual = _stores(registration)
    if any(
        getattr(manager, name).root.resolve() != getattr(actual, name).root.resolve()
        for name in ("records", "research", "evaluations")
    ):
        raise ValueError(
            "upgrade consumer stores do not match registered write domains"
        )
    imported = manager.evaluations.get("candidate_evidence_imports", import_id)
    if imported != _build_import(registration, manager, import_id, role=role):
        raise ValueError("upgrade independent import facts changed")
    for value in imported["derived_results"].values():
        if manager.evaluations.get("results", value["result_id"]) != value:
            raise ValueError("upgrade derived evaluation changed")
    if (
        manager.evaluations.get(
            "protocols", registration["target_protocol_snapshot"]["protocol_id"]
        )
        != registration["target_protocol_snapshot"]
    ):
        raise ValueError("upgrade deployment protocol changed")
    return registration, imported


def qualify_upgrade(
    manager, *, qualification_id, decision_id, import_id, registration_ref
):
    """Release establishes fresh qualifications and a decision for the exact active."""
    from .runtime import dynamic_job_lock

    with dynamic_job_lock():
        registration, imported = _validate_import(
            manager, registration_ref, import_id, role="release"
        )
        if manager.active() != registration["active_snapshot"]:
            raise ValueError("upgrade effective active/revision changed")
        gate = {**imported["deployment_gate"], "decision_id": decision_id}
        base = manager.research.get("bundles", gate["base_bundle_id"])
        common = {
            "protocol_id": gate["protocol_id"],
            "scope": gate["scope"],
            "upgrade_import_id": import_id,
            "registration_ref": registration_ref,
            "decision_ref": decision_id,
            "relative_gain_claimed": True,
        }
        manager.records.put(
            "qualifications",
            qualification_id + "-active",
            {
                **common,
                "qualification_id": qualification_id + "-active",
                "bundle_id": base["bundle_id"],
                "bundle_snapshot": base,
                "kind": "upgrade_current_applicability",
                "relative_gain_claimed": False,
                "original_qualification_ref": imported["current_active_applicability"][
                    "qualification_ref"
                ],
                "result_refs": [imported["derived_results"]["baseline"]["result_id"]],
            },
        )
        qualification = {
            **common,
            "qualification_id": qualification_id,
            "bundle_id": gate["bundle_id"],
            "bundle_snapshot": imported["bundle_snapshot"],
            "kind": "evaluated_candidate_upgrade",
            "result_refs": gate["result_refs"],
        }
        manager.records.put("qualifications", qualification_id, qualification)
        manager.records.put("decisions", decision_id, gate)
        manager.records.put(
            "decision_bases", decision_id, registration["active_snapshot"]
        )
        manager.records.put(
            "upgrade_decisions",
            decision_id,
            {
                "registration_ref": registration_ref,
                "import_id": import_id,
                "qualification_id": qualification_id,
            },
        )
        return qualification


def validate_upgrade_qualification(manager, value, *, role):
    registration, imported = _validate_import(
        manager, value["registration_ref"], value["upgrade_import_id"], role=role
    )
    bundle_id = value["bundle_id"]
    expected = (
        imported["bundle_snapshot"]
        if value["kind"] == "evaluated_candidate_upgrade"
        else manager.research.get(
            "bundles", registration["active_snapshot"]["bundle_id"]
        )
    )
    if (
        value["bundle_snapshot"] != expected
        or bundle_id != expected["bundle_id"]
        or value["protocol_id"]
        != registration["target_protocol_snapshot"]["protocol_id"]
    ):
        raise ValueError("upgrade qualification target mismatch")
    return registration


def activate_upgrade(manager, decision_id, *, safe_point, queues_empty):
    """Use the normal locked activation, preserving exact active/revision checks."""
    from .runtime import dynamic_job_lock

    with dynamic_job_lock():
        link = manager.records.get("upgrade_decisions", decision_id)
        registration, imported = _validate_import(
            manager, link["registration_ref"], link["import_id"], role="release"
        )
        active = manager.active()
        expected = {
            "bundle_id": imported["bundle_snapshot"]["bundle_id"],
            "scope": registration["active_snapshot"]["scope"],
            "activation": decision_id,
            "revision": registration["active_snapshot"].get("revision", 0) + 1,
        }
        if (
            safe_point not in {"task_boundary", "approved_safe_point"}
            or not queues_empty
        ):
            raise ValueError("activation needs a safe point and cleared queues")
        if active == expected:
            receipt = manager.records.get("activations", decision_id)
            if receipt.get("previous") != registration["active_snapshot"]["bundle_id"]:
                raise ValueError("upgrade activation receipt ancestry mismatch")
            return active
        if active != registration["active_snapshot"]:
            raise ValueError("upgrade effective active/revision changed")
        from .runtime import execution_binding

        base = manager.research.get("bundles", active["bundle_id"])
        rollback_qualification = manager.qualification(
            base["bundle_id"],
            scope=active["scope"],
            protocol_id=registration["target_protocol_snapshot"]["protocol_id"],
        )
        manager.records.put(
            "upgrade_publications",
            decision_id,
            {
                "decision_id": decision_id,
                "previous_active": active,
                "active": expected,
                "import_snapshot": imported,
                "rollback_qualification": rollback_qualification,
                "rollback_binding": execution_binding(
                    manager.research, base, _read(base["backend_ref"])
                ),
            },
        )
        return manager._activate(
            decision_id,
            safe_point=safe_point,
            queues_empty=queues_empty,
            scope=registration["active_snapshot"]["scope"],
        )


def validate_upgrade_rollback(manager, target_bundle_id, *, reason):
    """Recover from an accepted publication without rebuilding faulty candidate inputs."""
    from .runtime import execution_binding

    active = manager.active()
    decision_id = active["activation"]
    link = manager.records.get("upgrade_decisions", decision_id)
    registration = upgrade_authorization(link["registration_ref"], role="release")
    actual = _stores(registration)
    if any(
        getattr(manager, name).root.resolve() != getattr(actual, name).root.resolve()
        for name in ("records", "research", "evaluations")
    ):
        raise ValueError(
            "upgrade rollback stores do not match registered write domains"
        )
    receipt = manager.records.get("upgrade_publications", decision_id)
    imported = manager.evaluations.get("candidate_evidence_imports", link["import_id"])
    activation = manager.records.get("activations", decision_id)
    expected_activation = {
        "bundle_id": receipt["active"]["bundle_id"],
        "previous": receipt["previous_active"]["bundle_id"],
        "decision_id": decision_id,
        "scope": active["scope"],
        "safe_point": activation["safe_point"],
    }
    if (
        receipt["decision_id"] != decision_id
        or receipt["active"] != active
        or receipt["previous_active"] != registration["active_snapshot"]
        or imported != receipt["import_snapshot"]
        or imported["registration_snapshot"] != registration
        or imported["registration_ref"] != link["registration_ref"]
        or activation != expected_activation
        or activation["safe_point"] not in {"task_boundary", "approved_safe_point"}
        or manager.records.get("decision_bases", decision_id)
        != receipt["previous_active"]
        or manager.records.get("decisions", decision_id)
        != {**imported["deployment_gate"], "decision_id": decision_id}
        or imported["deployment_gate"]["verdict"] != "accept"
        or imported["deployment_gate"]["release_eligible"] is not True
    ):
        raise ValueError("upgrade rollback publication receipt/active revision changed")
    policy = registration.get("rollback", {})
    if (
        target_bundle_id != receipt["previous_active"]["bundle_id"]
        or target_bundle_id not in policy.get("allowed_ancestors", [])
        or reason not in policy.get("triggers", [])
    ):
        raise ValueError("upgrade rollback target/trigger is not authorized")
    target = manager.research.get("bundles", target_bundle_id)
    qualification = receipt["rollback_qualification"]
    if (
        manager.records.get("qualifications", qualification["qualification_id"])
        != qualification
        or qualification["bundle_snapshot"] != target
        or qualification["bundle_id"] != target_bundle_id
        or qualification["upgrade_import_id"] != link["import_id"]
        or execution_binding(manager.research, target, _read(target["backend_ref"]))
        != receipt["rollback_binding"]
        or manager.evaluations.get("protocols", qualification["protocol_id"])
        != registration["target_protocol_snapshot"]
        or _current_applicability(
            registration,
            manager,
            ExperimentResult(**imported["source_results"]["baseline"]),
            role="release",
        )
        != imported["current_active_applicability"]
    ):
        raise ValueError(
            "upgrade rollback target no longer matches accepted independent qualification"
        )
    return receipt


def role_action(config, task):
    """Fixed production role actions; no model-selected consumer or destination."""
    registration_ref = task["registration_ref"]
    manager = _stores(_read(registration_ref))
    action = task["action"]
    if action == "candidate_deployment_prepare":
        return prepare_candidate_deployment(
            manager.research, registration_ref=registration_ref
        )
    if action == "candidate_evidence_import":
        return import_candidate_evidence(
            manager.evaluations,
            import_id=task["import_id"],
            registration_ref=registration_ref,
        )
    if action == "candidate_upgrade_qualify":
        return qualify_upgrade(
            manager,
            qualification_id=task["qualification_id"],
            decision_id=task["decision_id"],
            import_id=task["import_id"],
            registration_ref=registration_ref,
        )
    if action == "candidate_upgrade_activate":
        return activate_upgrade(
            manager,
            task["decision_id"],
            safe_point=task["safe_point"],
            queues_empty=task["queues_empty"],
        )
    raise ValueError("unsupported upgrade bridge action")


def isolation_paths(config, task, *, _ancestry=()):
    """Pin reviewed inputs, and original facts, while writing new records nearby."""
    reference = str(Path(task["registration_ref"]).resolve())
    if reference in _ancestry:
        raise ValueError("cyclic current qualification ancestry")
    registration = _read(reference)
    ancestry = (*_ancestry, reference)
    paths = {
        Path(registration[key])
        for key in ("source_protocol_ref", "target_protocol_ref")
    }
    original_qualification, original_inputs = _active_qualification_sources(
        registration, _ancestry=ancestry
    )
    paths.add(original_qualification)
    paths.update(original_inputs)
    paths.update(Path(item["ref"]) for item in registration["source_manifests"])
    for bindings in registration["behavior_review"].values():
        for binding in bindings:
            paths.update(Path(binding[side + "_ref"]) for side in ("source", "target"))
    for item in registration["execution_sources"]:
        paths.update(Path(item[side + "_ref"]) for side in ("source", "target"))
    if task["action"] != "candidate_deployment_prepare":
        root = Path(registration["role_domains"]["evaluation"])
        evaluation = EvaluationStore(root)
        for identity in registration["source_result_ids"].values():
            paths.add(root / "results" / (identity + ".json"))
            paths.add(evaluation.path("results", identity))
            paths.update(
                Path(ref)
                for ref in evaluation.get("results", identity)["data_refs"]
                if Path(ref).is_file()
                and Path(ref).resolve().is_relative_to(root.resolve())
            )
            pointer = root / "result_evaluation_current" / (identity + ".json")
            if pointer.exists():
                paths.add(pointer)
                paths.add(Path(_read(pointer)["receipt_ref"]))
        paths.add(Path(registration["source_decision_ref"]))
        canonical = Path(registration["source_decision_ref"]).resolve()
        if canonical.parent == (evaluation.root / "decisions").resolve():
            from .gate_versions import decision_inputs
            paths.update(decision_inputs(evaluation, canonical.stem))
    return tuple(sorted(paths))
