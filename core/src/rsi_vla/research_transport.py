"""Reversible JSON factoring for complete development requests, not summarization."""

import json
from collections import Counter


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def pack(value):
    """Share repeated subtrees/constants and dictionary field names; keep all values."""
    counts = Counter()

    def count(item):
        if isinstance(item, (dict, list, str)):
            counts[compact(item)] += 1
        if isinstance(item, dict):
            for child in item.values():
                count(child)
        elif isinstance(item, list):
            for child in item:
                count(child)

    count(value)
    fields, shapes, shared, ids = [], {}, [], {}

    def encode(item, defining=False):
        key = compact(item) if isinstance(item, (dict, list, str)) else None
        if key is not None and counts[key] > 1 and len(key) > 32 and not defining:
            if key not in ids:
                ids[key] = len(shared)
                shared.append(None)
                shared[ids[key]] = encode(item, True)
            return {"r": ids[key]}
        if isinstance(item, dict):
            names = tuple(item)
            if names not in shapes:
                shapes[names] = len(fields)
                fields.append(list(names))
            return {"d": [shapes[names], [encode(v) for v in item.values()]]}
        if isinstance(item, list):
            return [encode(v) for v in item]
        return item

    root = encode(value)
    return {
        "schema": "rsi-lossless-json-v1",
        "decoding": "All facts are included. Recursively decode plain arrays elementwise; {r:i} is shared[i]; "
        "{d:[i,values]} zips fields[i] with decoded values into a dictionary. Other scalars are literal. "
        "All supplied values are retained. Any input selection is declared in evidence_input. "
        "Missing/unknown and source references retain their original meaning.",
        "fields": fields,
        "shared": shared,
        "root": root,
    }


def unpack(document):
    """Restore the exact JSON value represented by this finite transport codec."""
    if document.get("schema") != "rsi-lossless-json-v1":
        raise ValueError("unknown research transport")

    def decode(item):
        if isinstance(item, list):
            return [decode(v) for v in item]
        if isinstance(item, dict):
            if set(item) == {"r"}:
                return decode(document["shared"][item["r"]])
            if set(item) == {"d"}:
                index, values = item["d"]
                names = document["fields"][index]
                if len(names) != len(values):
                    raise ValueError("transport field count mismatch")
                return dict(zip(names, [decode(v) for v in values]))
            raise ValueError("unknown transport node")
        return item

    return decode(document["root"])


def model_prompt(transport):
    """This exact representation is consumed by the real Research SDK path."""
    request = unpack(transport)
    overview = request.get("experience_overview", {})
    semantic = {
        "meaning": "Deterministic navigation only; all complete L0 remains in the codec below. No causal claims.",
        "runs": [{
            **{k:r[k] for k in ("result_id","bundle_id","outcome","assignments","invalid_count",
                                "unknown_count","segments","costs","evidence_refs")},
            "independent_fact":r.get("independent_fact"),
            "cost_missing":(r.get("cost_details") or {}).get("missing"),
            "execution_statistics":r.get("execution_statistics"),
            "fault_or_missing_event_count":len(r['faults']),
            "confirmed_errors":[f for f in r['faults'] if f['error']],
            "missing_counts":dict(Counter(name for f in r['faults'] for name in f['missing'])),
            "fault_details":"all event-specific gaps/errors and refs remain in complete codec; counts are navigation",
        } for r in overview.get("facts", [])],
        "available_operators": request.get("allowed_operators", []),
        "operator_preflight": request.get("domain_policy", {}).get("operator_preflight", "not supplied; execute requires fresh preflight"),
        "processed_findings": overview.get("findings", []), "decisions": overview.get("decisions", []),
    }
    selection = request.get("evidence_input")
    if selection:
        return (
            "Review the selected legal development input below using its reversible codec. "
            "Selection is not complete L0; omitted events remain in the authoritative store. "
            "Use evidence_input to identify gaps and defer when needed. A host can read exact "
            "development events through read_development_events and prepare a separately budgeted request. "
            "No external tools or experiments are available in this call. Retain uncertainty and "
            "competing explanations. Return the bounded response schema.\n"
            + "Selected reversible evidence: " + compact(transport)
        )
    if request.get("research_mode") == "exploratory":
        return (
            "Review the registered hypothesis and legal development input below. "
            "Prior trigger evidence may be absent; absence is not evidence of benefit. "
            "Choose one allowed exploratory intervention or defer. Retain competing "
            "explanations and the registered slots, budget and stop condition. "
            "No external tools, experiments or release actions are available in this call. "
            "Return the bounded response schema.\nComplete reversible input: " + compact(transport)
        )
    return (
        "Review the complete legal development request below using its reversible codec. "
        "Retain uncertainty, fault validity and competing explanations. Choose one allowed "
        "evidenced intervention or defer. Do not inspect release/sealed sources. "
        "No external tools or experiments are available. Return the bounded response schema.\n"
        + "Semantic overview: " + compact(semantic) + "\nComplete reversible evidence: " + compact(transport)
    )


def select_input(tools, request):
    """Project only model input; saved L0 and source identities remain authoritative."""
    from copy import deepcopy

    from .task_policy import task_policy

    registered = task_policy(tools.protocol)
    if "evidence_input" in tools.domain_policy:
        if not registered:
            raise ValueError("request-specific evidence selection requires a registered new-task policy")
        policy = tools.domain_policy["evidence_input"]
    else:
        policy = registered.get("evidence_input")
    if policy is None:
        return request
    if policy.get("mode") != "selected_l0" or not policy.get("reason"):
        raise ValueError("selected input needs a selection reason and supported mode")
    ids = policy["event_ids"]
    legal = {r["result_id"] for r in request["evidence"]}
    if set(ids) != legal:
        raise ValueError("selection must explicitly cover every supplied development result")
    selected = deepcopy(request)
    views, manifest, wanted = {}, [], set()
    routed = {view["view_id"]: view for view in request.get("evidence_views", [])}
    for result in request["evidence"]:
        result_id = result["result_id"]
        read = tools.read_development_events(result_id, ids[result_id])
        view_id = read["view_id"]
        events = read["events"]
        wanted.update(e["event_id"] for e in events)
        original = tools.store.get("evidence_views", view_id)
        if view_id in views and views[view_id]["l0"] != events:
            raise ValueError("conflicting selections for a shared L0 view")
        views[view_id] = {**routed.get(view_id, original), "l0": events}
        manifest.append({"result_id": result_id, "view_id": view_id,
                         "available_event_count": read["available_event_count"],
                         "selected_event_ids": [e["event_id"] for e in events],
                         "source_refs": result["data_refs"]})
    selected["evidence_views"] = list(views.values())
    # Avoid duplicating event payloads from omitted L0 in the navigation view.
    for row in selected.get("experience_overview", {}).get("facts", []):
        for key in ("segments", "faults", "skill_observations"):
            row[key] = [e for e in row.get(key, []) if e.get("event_ref") in wanted]
    selected["evidence_input"] = {"mode": "selected_l0", "reason": policy["reason"],
                                  "complete_l0_in_prompt": False, "views": manifest,
                                  "read_interface": "read_development_events(result_id) lists event IDs; supply event_ids to read exact events"}
    return selected
