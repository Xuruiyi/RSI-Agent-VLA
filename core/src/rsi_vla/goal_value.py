"""Static functional-value evidence from public catalog goals, without an Env.

A matching catalog goal establishes an independently specified use for a symbolic
objective. It does not establish physical feasibility, policy competence, task
success, or a capability gain.
"""

import ast
from copy import deepcopy
import json
from pathlib import Path
import re


_KIND = "bddl_conjunction"
_METRIC = "domain_functional_value"
_ARITY = {"in": 2, "on": 2, "open": 1, "close": 1, "turnon": 1, "turnoff": 1}
_MAX_ATOMS = 6
_MAX_MATCH_STEPS = 4096


class _MetadataError(ValueError):
    pass


def _read(path, refs):
    path = Path(path).resolve(strict=True)
    refs.add(str(path))
    return path.read_text(encoding="utf-8")


def _sexp(text):
    """Read BDDL lists strictly; never evaluate expressions or use fallback tasks."""
    tokens = re.findall(r"\(|\)|[^\s()]+", re.sub(r";[^\n]*", "", text))
    stack, roots = [], []
    for token in tokens:
        if token == "(":
            value = []
            if stack:
                stack[-1].append(value)
            else:
                roots.append(value)
            stack.append(value)
        elif token == ")":
            if not stack:
                raise _MetadataError("unbalanced BDDL")
            stack.pop()
        else:
            if not stack:
                raise _MetadataError("token outside BDDL expression")
            stack[-1].append(token)
    if stack or len(roots) != 1 or not roots[0] or roots[0][0].lower() != "define":
        raise _MetadataError("invalid BDDL definition")
    return roots[0]


def _typed(section):
    result, pending, index = {}, [], 1
    while index < len(section):
        item = section[index]
        if not isinstance(item, str):
            raise _MetadataError("invalid typed object declaration")
        if item == "-":
            if not pending or index + 1 >= len(section):
                raise _MetadataError("missing object type")
            category = section[index + 1]
            if not isinstance(category, str):
                raise _MetadataError("invalid object type")
            for name in pending:
                if name in result:
                    raise _MetadataError("duplicate scene object")
                result[name] = category
            pending = []
            index += 2
        else:
            pending.append(item)
            index += 1
    if pending:
        raise _MetadataError("untyped scene object")
    return result


def _scene(text):
    sections = {}
    for item in _sexp(text)[1:]:
        if (
            isinstance(item, list)
            and item
            and isinstance(item[0], str)
            and item[0].startswith(":")
        ):
            key = item[0].lower()
            if key in sections:
                raise _MetadataError("duplicate BDDL section")
            sections[key] = item
    objects = _typed(sections.get(":objects", [":objects"]))
    fixtures = _typed(sections.get(":fixtures", [":fixtures"]))
    if set(objects) & set(fixtures):
        raise _MetadataError("object/fixture name collision")
    entities = {name: ("object", category, None) for name, category in objects.items()}
    entities.update(
        {name: ("fixture", category, None) for name, category in fixtures.items()}
    )
    unresolved_regions = {}
    for region in sections.get(":regions", [":regions"])[1:]:
        if not isinstance(region, list) or not region or not isinstance(region[0], str):
            raise _MetadataError("invalid region")
        targets = [
            item[1]
            for item in region[1:]
            if isinstance(item, list) and len(item) == 2 and item[0] == ":target"
        ]
        if len(targets) != 1 or not isinstance(targets[0], str):
            raise _MetadataError("region requires one named parent")
        parent = targets[0]
        name = parent + "_" + region[0]
        if name in entities or name in unresolved_regions:
            raise _MetadataError("duplicate scene region")
        if parent not in entities:
            unresolved_regions[name] = {"parent": parent, "region": region[0]}
            continue
        entities[name] = ("site", region[0], parent)
    goal = sections.get(":goal")
    if not goal or len(goal) != 2 or not isinstance(goal[1], list):
        raise _MetadataError("missing goal conjunction")
    expression = goal[1]
    atoms = (
        expression[1:]
        if expression and expression[0].lower() == "and"
        else [expression]
    )
    return {
        "entities": entities,
        "objects": set(objects),
        "goal": atoms,
        "unresolved_regions": unresolved_regions,
    }


def _atoms(raw):
    if not isinstance(raw, list) or not raw or len(raw) > _MAX_ATOMS:
        raise ValueError("goal must contain one to six atomic conjuncts")
    atoms = []
    for atom in raw:
        if (
            not isinstance(atom, list)
            or not atom
            or any(not isinstance(v, str) or not v for v in atom)
        ):
            raise ValueError("goal conjunct must be a nonempty string list")
        predicate = atom[0].lower()
        if predicate not in _ARITY or len(atom) != _ARITY[predicate] + 1:
            raise ValueError("unsupported predicate or arity")
        atoms.append((predicate, *atom[1:]))
    return tuple(sorted(set(atoms)))


def _ground(atoms, scene, registry):
    entities = scene["entities"]
    for atom in atoms:
        if atom[0] not in registry:
            return False, "predicate is absent from the installed registry"
        if any(name not in entities for name in atom[1:]):
            return None, "goal symbol is not grounded in the selected scene"
        if atom[0] in {"in", "on"}:
            if atom[1] not in scene["objects"]:
                return False, "placement subject is not a movable scene object"
            if atom[0] == "in" and entities[atom[2]][0] != "site":
                return None, "containment needs a declared destination region"
            if atom[1] == atom[2] or entities[atom[2]][2] == atom[1]:
                return False, "self placement/containment"
    return True, None


def _consistent(atoms, scene):
    values = set(atoms)
    for predicate, opposite in [("open", "close"), ("turnon", "turnoff")]:
        if any(
            atom[0] == predicate and (opposite, atom[1]) in values for atom in atoms
        ):
            return False
    # A movable object cannot contain itself through a chain of destination sites.
    parents = {}
    for atom in atoms:
        if atom[0] == "in":
            parents.setdefault(atom[1], set()).add(scene["entities"][atom[2]][2])

    def cyclic(node, visiting):
        if node in visiting:
            return True
        return any(
            cyclic(parent, visiting | {node}) for parent in parents.get(node, ())
        )

    return not any(cyclic(node, set()) for node in parents)


def _match(atoms, scene, other_atoms, other):
    """Match complete goals, preserving types, region semantics and shared roles."""
    if len(atoms) != len(other_atoms):
        return None
    steps = 0

    def bind(left, right, forward, backward):
        left_type = scene["entities"][left]
        right_type = other["entities"][right]
        if left_type[:2] != right_type[:2]:
            return False
        if left in forward or right in backward:
            return forward.get(left) == right and backward.get(right) == left
        forward[left], backward[right] = right, left
        if left_type[0] == "site":
            return bind(left_type[2], right_type[2], forward, backward)
        return True

    def walk(index, available, forward, backward):
        nonlocal steps
        if index == len(atoms):
            return forward
        for other_index in available:
            steps += 1
            if steps > _MAX_MATCH_STEPS:
                raise _MetadataError("catalog goal matching exceeded its finite bound")
            left, right = atoms[index], other_atoms[other_index]
            if left[0] != right[0]:
                continue
            new_forward, new_backward = dict(forward), dict(backward)
            if all(
                bind(a, b, new_forward, new_backward)
                for a, b in zip(left[1:], right[1:])
            ):
                found = walk(
                    index + 1, available - {other_index}, new_forward, new_backward
                )
                if found is not None:
                    return found
        return None

    return walk(0, set(range(len(other_atoms))), {}, {})


def _literal_assignment(text, name):
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise _MetadataError("catalog declaration is missing")


def _registry(text):
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "VALIDATE_PREDICATE_FN_DICT"
            for t in node.targets
        ):
            if isinstance(node.value, ast.Dict):
                return {
                    key.value
                    for key in node.value.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
    raise _MetadataError("predicate registry is missing")


def _primitive_names(text):
    for node in ast.parse(text).body:
        if isinstance(node, ast.ClassDef) and node.name == "LiberoPrimitives":
            return {
                method.name
                for method in node.body
                if isinstance(method, ast.FunctionDef)
                and any(
                    isinstance(decorator, ast.Name) and decorator.id == "tool"
                    for decorator in method.decorator_list
                )
            }
    raise _MetadataError("runtime primitive declarations are missing")


def evaluate_goal_value(protocol, opportunity, *, policy):
    """Return independent static measurement evidence for an unlisted objective.

    Only public BDDL, the literal catalog index and predicate registry are read.
    Catalog goal matching is a functional specification witness. Joint-destination
    capacity is accepted only under the explicitly registered catalog basis;
    neither this measurement nor its numeric value reports physical success.
    """
    target = opportunity["target"]
    if (
        opportunity.get("source") != "new_goal"
        or opportunity.get("protocol_id") != protocol["protocol_id"]
        or target.get("domain") != policy["domain"]
        or target.get("kind") != _KIND
        or policy["allowed_goal_kinds"] != [_KIND]
        or target.get("value_criterion_ref") != policy["value_criterion_ref"]
        or target.get("independent_value_test_ref")
        != policy["independent_value_test_ref"]
        or opportunity["goal_review"]["policy_ref"] != policy["policy_ref"]
        or policy["metric"] != _METRIC
        or type(policy["minimum"]) not in (int, float)
        or policy["minimum"] != 1
    ):
        raise ValueError("static goal value policy/opportunity binding mismatch")
    refs, checks = {str(Path(__file__).resolve())}, {}
    result = dict(
        schema="rsi-goal-value-evidence-v1",
        protocol_id=protocol["protocol_id"],
        opportunity_id=opportunity["opportunity_id"],
        target=deepcopy(target),
        criterion_ref=policy["value_criterion_ref"],
        test_ref=policy["independent_value_test_ref"],
        information_scope="development",
        evaluation_kind="static_catalog_functional_value",
        capability_gain=None,
        task_success=None,
        physical_feasibility="unmeasured",
        release_eligible=False,
    )

    def finish(value, reason):
        return {
            **result,
            "validity": "unknown" if value is None else "valid",
            "metrics": {_METRIC: value},
            "checks": checks,
            "reason": reason,
            "evidence_refs": sorted(refs),
        }

    try:
        description = json.loads(target["description"])
        if not isinstance(description, dict) or set(description) - {
            "scene_task_key",
            "all_of",
            "value_claim",
        }:
            raise ValueError("description must contain scene_task_key and all_of")
        atoms = _atoms(description["all_of"])
    except (KeyError, TypeError, ValueError) as exc:
        return finish(0, "invalid goal description: " + str(exc))
    try:
        root = Path(policy["catalog_root"]).resolve(strict=True)
        suites = policy["catalog_suites"]
        if (
            not isinstance(suites, list)
            or not suites
            or any(not isinstance(s, str) or Path(s).name != s for s in suites)
        ):
            raise _MetadataError("invalid public catalog suites")
        registry = _registry(_read(policy["predicate_registry_ref"], refs))
        catalog = _literal_assignment(
            _read(policy["catalog_index_ref"], refs), "libero_task_map"
        )
        if any(suite not in catalog for suite in suites):
            raise _MetadataError("public catalog suite is not indexed")
        tasks = {task["task_key"]: task for task in protocol["tasks"]}
        if description.get("scene_task_key") not in tasks:
            return finish(0, "scene is outside the current protocol task set")

        def path(relative):
            selected = (root / relative).resolve(strict=True)
            if not selected.is_relative_to(root) or selected.suffix != ".bddl":
                raise _MetadataError("scene reference escapes public BDDL catalog")
            return selected

        scene_ref = path(tasks[description["scene_task_key"]]["bddl_relative_path"])
        scene = _scene(_read(scene_ref, refs))
        grounded, reason = _ground(atoms, scene, registry)
        checks["grounded"] = grounded
        if grounded is not True:
            return finish(None if grounded is None else 0, reason)
        checks["consistent"] = _consistent(atoms, scene)
        if not checks["consistent"]:
            return finish(0, "goal contains contradictory terminal states")
        # Compare literal goals on their actual shared instances, not renamed object
        # types: changing the manipulated object is allowed to be a new objective.
        for task in tasks.values():
            prior = _scene(_read(path(task["bddl_relative_path"]), refs))
            if atoms == _atoms(prior["goal"]):
                checks["outside_original_goals"] = False
                return finish(0, "goal repeats a current protocol objective")
        checks["outside_original_goals"] = True
        if not policy.get("runtime_primitives_ref"):
            return finish(None, "runtime actionability basis is missing")
        primitives = _primitive_names(_read(policy["runtime_primitives_ref"], refs))
        required = {"move_to", "set_gripper", "pi0_pick"}
        if any(atom[0] in {"in", "on"} for atom in atoms):
            required.add("release")
        if not required <= primitives:
            checks["symbolic_actionability"] = None
            return finish(
                None, "runtime primitives do not establish symbolic actionability"
            )
        checks["runtime_primitives"] = sorted(required)
        incomplete = False
        for suite in suites:
            for name in catalog[suite]:
                if not isinstance(name, str) or Path(name).name != name:
                    raise _MetadataError("invalid catalog task identity")
                try:
                    witness_ref = path(
                        Path(policy.get("catalog_directory", ""))
                        / suite
                        / (name + ".bddl")
                    )
                    witness = _scene(_read(witness_ref, refs))
                    witness_atoms = _atoms(witness["goal"])
                    valid, _ = _ground(witness_atoms, witness, registry)
                    if valid is not True or not _consistent(witness_atoms, witness):
                        incomplete = True
                        continue
                    mapping = _match(atoms, scene, witness_atoms, witness)
                except (OSError, ValueError, SyntaxError):
                    incomplete = True
                    continue
                if mapping is None:
                    continue
                grouped = {}
                for atom in atoms:
                    if atom[0] in {"in", "on"}:
                        destination = scene["entities"][atom[2]]
                        owner = destination[2] if destination[0] == "site" else atom[2]
                        grouped.setdefault(owner, set()).add(atom[1])
                joint = any(len(subjects) > 1 for subjects in grouped.values())
                if joint and policy.get("capacity_basis") != "exact_catalog_joint_goal":
                    checks["capacity_basis"] = None
                    return finish(
                        None,
                        "joint destination capacity has no registered independent basis",
                    )
                checks.update(
                    functional_witness=True,
                    symbolic_actionability=True,
                    capacity_basis="exact_catalog_joint_goal"
                    if joint
                    else "exact_catalog_goal",
                )
                result["catalog_witness"] = dict(
                    ref=str(witness_ref),
                    symbol_mapping=mapping,
                    goal=[list(atom) for atom in witness_atoms],
                    basis="complete_typed_goal_graph",
                )
                return finish(
                    1, "independent catalog specifies this functional objective"
                )
        checks["functional_witness"] = None
        return finish(
            None,
            "independent functional/actionability witness missing"
            + ("; some catalog metadata unavailable" if incomplete else ""),
        )
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as exc:
        return finish(None, "static metadata unavailable: " + str(exc))
