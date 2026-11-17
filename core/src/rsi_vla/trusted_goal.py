"""Trusted bounded BDDL composition, task materialization and protocol creation."""

import json
from copy import deepcopy
from pathlib import Path

from .goal_value import _atoms, _consistent, _ground, _read, _registry, _scene, _sexp
from .storage import atomic_json, identifier


def composition_evidence(protocol: dict, opportunity: dict, *, policy: dict) -> dict:
    """Apply an external compositional utility rule to a grounded new conjunction.

    The rule authorizes functional objectives; physical reachability and learned
    competence remain unmeasured until independent environment evaluation.
    """
    refs = set()
    result = {
        "schema": "rsi-goal-value-evidence-v1",
        "protocol_id": protocol["protocol_id"],
        "opportunity_id": opportunity["opportunity_id"],
        "target": deepcopy(opportunity["target"]),
        "criterion_ref": policy["value_criterion_ref"],
        "test_ref": policy["independent_value_test_ref"],
        "information_scope": "development",
        "evaluation_kind": "fixed_compositional_functional_value",
        "capability_gain": None,
        "task_success": None,
        "physical_feasibility": "unmeasured",
        "release_eligible": False,
    }

    def finish(value, reason):
        return {
            **result,
            "validity": "unknown" if value is None else "valid",
            "metrics": {policy["metric"]: value},
            "reason": reason,
            "evidence_refs": sorted(refs),
            "checks": {"external_composition_rule": value == 1},
        }

    try:
        target = opportunity["target"]
        if (
            opportunity["protocol_id"] != protocol["protocol_id"]
            or opportunity["source"] != "new_goal"
            or target["domain"] != policy["domain"]
            or target["kind"] not in policy["allowed_goal_kinds"]
            or target["value_criterion_ref"] != policy["value_criterion_ref"]
            or target["independent_value_test_ref"]
            != policy["independent_value_test_ref"]
            or opportunity["goal_review"]["policy_ref"] != policy["policy_ref"]
        ):
            raise ValueError("composition policy/opportunity binding mismatch")
        # Re-read the owning external file: a research-supplied dictionary cannot
        # silently replace a registered value rule.
        saved = json.loads(_read(policy["policy_ref"], refs))
        if saved != policy:
            raise ValueError("external composition policy changed")
        rules = policy["composition_rules"]
        if rules.get("schema") != "rsi-bddl-composition-rules-v1" or not rules.get(
            "functional_value_basis"
        ):
            raise ValueError("fixed functional value rule required")
        description = json.loads(target["description"])
        if set(description) - {"scene_task_key", "all_of", "value_claim"}:
            raise ValueError("unsupported goal description fields")
        atoms = _atoms(description["all_of"])
        key = description["scene_task_key"]
        if key not in rules["scene_task_keys"] or len(atoms) > rules["max_atoms"]:
            return finish(0, "goal exceeds registered scene/conjunction scope")
        tasks = {task["task_key"]: task for task in protocol["tasks"]}
        root = Path(policy["catalog_root"]).resolve(strict=True)

        def scene(task):
            path = (root / task["bddl_relative_path"]).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError("scene escapes registered root")
            return _scene(_read(path, refs))

        selected = scene(tasks[key])
        valid, reason = _ground(
            atoms, selected, _registry(_read(policy["predicate_registry_ref"], refs))
        )
        if valid is not True:
            return finish(None if valid is None else 0, reason)
        if not _consistent(atoms, selected):
            return finish(0, "contradictory goal")
        if any(
            atom[0] not in rules["predicates"]
            or any(symbol not in rules["allowed_symbols"][key] for symbol in atom[1:])
            for atom in atoms
        ):
            return finish(0, "goal exceeds registered predicate/object scope")
        if any(atoms == _atoms(scene(task)["goal"]) for task in tasks.values()):
            return finish(0, "goal repeats a current protocol objective")
        # Destination capacity is an external rule, never inferred from the
        # number of symbolic atoms or the research utility claim.
        placements = {}
        for atom in atoms:
            if atom[0] in {"in", "on"}:
                placements.setdefault(atom[2], set()).add(atom[1])
        if any(
            len(subjects) > rules.get("destination_capacity", {}).get(destination, 0)
            for destination, subjects in placements.items()
        ):
            return finish(0, "goal exceeds externally registered destination capacity")
        result["functional_value_basis"] = rules["functional_value_basis"]
        return finish(
            1, "grounded new combination satisfies fixed external functional rule"
        )
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as exc:
        return finish(None, "composition metadata unavailable: " + str(exc))


def _render(value):
    return (
        "("
        + " ".join(_render(item) if isinstance(item, list) else item for item in value)
        + ")"
    )


def _immutable_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != text:
            raise ValueError("conflicting immutable goal artifact: " + str(path))
    else:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)


def materialize_goal(
    protocol: dict,
    opportunity: dict,
    *,
    policy: dict,
    output_root: str | Path,
    new_protocol_id: str,
) -> dict:
    """Create a runnable task, copied initial states and a new frozen protocol.

    Called in the independent evaluation role after operator policy validation.
    This performs no environment initialization or physical feasibility claim.
    """
    evidence = composition_evidence(protocol, opportunity, policy=policy)
    if evidence["metrics"][policy["metric"]] != 1:
        raise ValueError("new goal not admitted: " + evidence["reason"])
    backend = policy.get("materialization", {})
    if backend.get("backend") != "libero_bddl":
        raise ValueError("new goal has no supported task materialization backend")
    new_protocol_id = identifier(new_protocol_id)
    if new_protocol_id == protocol["protocol_id"]:
        raise ValueError("new goal requires a new protocol identity")
    description = json.loads(opportunity["target"]["description"])
    scene_key = description["scene_task_key"]
    original = next(task for task in protocol["tasks"] if task["task_key"] == scene_key)
    catalog = Path(policy["catalog_root"]).resolve(strict=True)
    scene_path = (catalog / original["bddl_relative_path"]).resolve(strict=True)
    if not scene_path.is_relative_to(catalog):
        raise ValueError("scene escapes catalog")
    initial = Path(backend["initial_states"][scene_key]).resolve(strict=True)
    if not initial.is_file():
        raise ValueError("registered scene initial states missing")
    task_id = identifier(opportunity["opportunity_id"])
    root = Path(output_root).resolve() / new_protocol_id
    bddl_ref = root / (task_id + ".bddl")
    init_ref = root / (task_id + initial.suffix)
    task_ref = root / "task.json"
    protocol_ref = root / "protocol.json"
    receipt_ref = root / "materialization.json"
    private_receipt_ref = root / "evaluation-materialization.json"
    expression = _sexp(scene_path.read_text())
    atoms = _atoms(description["all_of"])
    goal = [":goal", ["And", *[list(atom) for atom in atoms]]]
    for index, item in enumerate(expression):
        if isinstance(item, list) and item and item[0] == ":goal":
            expression[index] = goal
        elif isinstance(item, list) and item and item[0] == ":language":
            expression[index] = [
                ":language",
                *("achieve " + " and ".join(" ".join(atom) for atom in atoms)).split(),
            ]
    # Preserve the simulator's registered problem class; task identity is carried
    # separately because LIBERO uses the problem name to select the environment.
    _immutable_text(bddl_ref, _render(expression) + "\n")
    root.mkdir(parents=True, exist_ok=True)
    data = initial.read_bytes()
    if init_ref.exists() and init_ref.read_bytes() != data:
        raise ValueError("conflicting immutable goal initial states")
    if not init_ref.exists():
        with init_ref.open("xb") as stream:
            stream.write(data)
    task = {
        "task_key": task_id,
        "group": "new",
        "suite": "materialized",
        "task_id": 0,
        "instruction": "achieve " + " and ".join(" ".join(atom) for atom in atoms),
        "bddl_file": str(bddl_ref),
        "initial_state_ref": str(init_ref),
        "materialization_ref": str(receipt_ref),
        "native_success": "bddl_goal",
        "scene_task_key": scene_key,
    }
    changed = deepcopy(protocol)
    changed.update(
        protocol_id=new_protocol_id,
        benchmark_id=new_protocol_id + "-release",
        parent_protocol_id=protocol["protocol_id"],
        status="frozen",
    )
    changed["tasks"].append(task)
    for role, split in changed["task_sets"].items():
        if role in {"development", "release"}:
            split["task_keys"].append(task_id)
            split["benchmark_id"] = new_protocol_id + "-" + role
    changed["objective_comparison"] = {
        "schema": "rsi-objective-comparison-v1",
        "objective": "new_capability",
        "task_keys": [task_id],
        "minimum_gain": backend.get("minimum_gain", 0.1),
        "anchor_max_drop": 0,
        "statistics": backend.get("statistics"),
    }
    receipt = {
        "schema": "rsi-materialized-goal-v1",
        "protocol_id": new_protocol_id,
        "parent_protocol_id": protocol["protocol_id"],
        "opportunity_id": opportunity["opportunity_id"],
        "policy_ref": policy["policy_ref"],
        "evidence": evidence,
        "task": task,
        "protocol": changed,
        "task_ref": str(task_ref),
        "protocol_ref": str(protocol_ref),
        "private_receipt_ref": str(private_receipt_ref),
        "public_receipt_ref": str(receipt_ref),
        "bddl_ref": str(bddl_ref),
        "initial_state_ref": str(init_ref),
        "initial_state_source_ref": str(initial),
        "native_success": "bddl_goal",
        "physical_feasibility": "unmeasured",
        "release_eligible": False,
    }
    public = {
        key: deepcopy(receipt[key])
        for key in (
            "schema",
            "protocol_id",
            "parent_protocol_id",
            "opportunity_id",
            "task",
            "task_ref",
            "bddl_ref",
            "initial_state_ref",
            "native_success",
            "physical_feasibility",
            "release_eligible",
        )
    }
    for path, value in (
        (task_ref, task),
        (protocol_ref, changed),
        (private_receipt_ref, receipt),
        (receipt_ref, public),
    ):
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError("conflicting immutable goal registration")
        if not path.exists():
            atomic_json(path, value)
    return receipt
