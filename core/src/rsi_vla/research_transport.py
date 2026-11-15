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
    request = _sdk_projection(request)
    if request.get("readable_evidence") is not None:
        return (
            "Review this readable legal development request. "
            + ("The sdk_projection manifest lists omitted cumulative fields; remaining event fields retain original values. "
               "Use evidence_requests with the listed event IDs to obtain complete events in a bounded follow-up. "
               if request.get("sdk_projection") else "Event payloads are original facts; ")
            +
            "overview entries are navigation. Selection, omissions, missing fields and source references "
            "are explicit. Media paths are references only: this call is text-only and visual content "
            "is unknown. Request exact legal development events through evidence_requests when needed; "
            "the host can prepare a separately budgeted follow-up. Do not use tools or access files, "
            "networks, release or sealed sources. Return the bounded response schema.\n"
            + json.dumps(request, ensure_ascii=False, indent=2, allow_nan=False)
        )
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


def _sdk_projection(request):
    """Project only the SDK copy; archives and exact legal event reads stay full."""
    from copy import deepcopy

    mode = request.get("evidence_input", {}).get("sdk_projection")
    if mode is None:
        return request
    if mode == "terminal_history_catalog_v2":
        return _sdk_catalog_projection(request)
    if mode != "terminal_history_v1" or request.get("readable_evidence") is None:
        raise ValueError("unsupported explicit SDK field projection")
    if request.get("evidence_followup") is not None:
        return request
    projected = deepcopy(request)
    source_views = {
        row["view_id"]: row.get("source_view_ref")
        or row.get("media_selection", {}).get("source_view_ref")
        for row in request["evidence_input"]["views"]
    }
    omitted = []
    for vi, view in enumerate(projected.get("evidence_views", [])):
        for ei, event in enumerate(view.get("l0", [])):
            if event.get("kind") != "task_result":
                continue
            independent = event.get("flags", {}).get("independent_result")
            episode = (
                independent.get("episode_execution")
                if isinstance(independent, dict)
                else None
            )
            if not isinstance(episode, dict):
                continue
            for field in ("decisions", "segments"):
                if not isinstance(episode.get(field), list):
                    continue
                value = episode.pop(field)
                omitted.append(
                    {
                        "field_path": f"/evidence_views/{vi}/l0/{ei}/flags/independent_result/episode_execution/{field}",
                        "event_id": event["event_id"],
                        "view_id": view["view_id"],
                        "source_view_ref": source_views.get(view["view_id"]),
                        "omitted_item_count": len(value),
                        "omitted_compact_utf8_bytes": len(
                            compact(value).encode("utf-8")
                        ),
                    }
                )
    readable = {
        row["view_id"]: (index, row)
        for index, row in enumerate(projected["readable_evidence"]["views"])
        if "view_id" in row
    }
    for index, row in enumerate(projected["evidence_input"]["views"]):
        readable_index, detail = readable[row["view_id"]]
        chosen = set(detail["selected_event_ids"])
        derived = [
            event["event_id"]
            for event in detail["event_index"]
            if event["event_id"] not in chosen
        ]
        if (
            row["selected_event_ids"] != detail["selected_event_ids"]
            or row.get("omitted_event_ids") != derived
            or detail.get("omitted_event_ids") != derived
        ):
            continue
        for location, value in (
            (f"/evidence_input/views/{index}", row),
            (f"/readable_evidence/views/{readable_index}", detail),
        ):
            removed = value.pop("omitted_event_ids")
            value["omitted_event_count"] = len(removed)
            value["omitted_event_ids_derivation"] = (
                "event_index event IDs minus selected_event_ids; preserve event_index order"
            )
            value["source_view_ref"] = source_views[row["view_id"]]
            omitted.append(
                {
                    "field_path": location + "/omitted_event_ids",
                    "view_id": row["view_id"],
                    "source_view_ref": source_views[row["view_id"]],
                    "omitted_item_count": len(removed),
                    "omitted_compact_utf8_bytes": len(compact(removed).encode("utf-8")),
                    "derivation": value["omitted_event_ids_derivation"],
                }
            )
    projected["sdk_projection"] = {
        "schema": "rsi-sdk-field-projection-v1",
        "mode": mode,
        "omitted_fields": omitted,
        "original_selected_events_in_business_transport": True,
        "full_event_read_interface": "evidence_requests(result_id,event_ids); follow-up sends full legal events",
        "media_consumption": "text_only; image_bytes_sent=false; visual_content=unknown",
    }
    return projected



def _sdk_catalog_projection(request):
    """Factor catalog IDs losslessly while retaining the selected event payloads."""
    from copy import deepcopy
    from os.path import commonprefix

    original = deepcopy(request)
    original["evidence_input"]["sdk_projection"] = "terminal_history_v1"
    projected = _sdk_projection(original)
    projected["evidence_input"]["sdk_projection"] = "terminal_history_catalog_v2"
    if "sdk_projection" in projected:
        projected["sdk_projection"]["mode"] = "terminal_history_catalog_v2"
    manifests = []
    for view in projected["readable_evidence"]["views"]:
        catalog = view["event_index"]
        if any(set(row) != {"event_id", "sequence", "kind"} for row in catalog):
            raise ValueError("SDK catalog encoding requires the complete known index schema")
        prefix = commonprefix([row["event_id"] for row in catalog]) if catalog else ""
        rows = [[row["event_id"][len(prefix):], row["sequence"], row["kind"]] for row in catalog]
        encoded = {
            "encoding": "prefixed-json-rows-v1",
            "event_id_prefix": prefix,
            "columns": ["event_id_suffix", "sequence", "kind"],
            "rows_json": compact(rows),
        }
        restored = [
            {"event_id": prefix + row[0], "sequence": row[1], "kind": row[2]}
            for row in json.loads(encoded["rows_json"])
        ]
        if restored != catalog:
            raise ValueError("SDK catalog facts failed lossless reconstruction")
        manifests.append({
            "view_id": view["view_id"], "event_count": len(catalog),
            "original_compact_utf8_bytes": len(compact(catalog).encode("utf-8")),
            "encoded_compact_utf8_bytes": len(compact(encoded).encode("utf-8")),
        })
        view["event_index"] = encoded
    projected["sdk_catalog_encoding"] = {
        "schema": "rsi-sdk-catalog-encoding-v1", "lossless": True,
        "decode": "Parse event_index.rows_json as JSON rows. Each complete event ID is event_id_prefix + row[0]; row[1] is sequence and row[2] is kind. Row order is original index order. No index entries are omitted.",
        "views": manifests,
        "selected_event_payloads": "same terminal-history selection; follow-up requested events remain complete",
    }
    return projected


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
    media_mode = policy.get("media_refs")
    if media_mode not in (None, "selected_events"):
        raise ValueError("unsupported selected input media reference policy")
    projection = policy.get("sdk_projection")
    if projection not in (None, "terminal_history_v1", "terminal_history_catalog_v2"):
        raise ValueError("unsupported explicit SDK field projection")
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
        routed_view = routed.get(view_id, original)
        views[view_id] = {**routed_view, "l0": events}
        manifest.append({"result_id": result_id, "view_id": view_id,
                         "available_event_count": read["available_event_count"],
                         "selected_event_ids": [e["event_id"] for e in events],
                         "omitted_event_ids": [e["event_id"] for e in original["l0"]
                                               if e["event_id"] not in {v["event_id"] for v in events}],
                         "source_refs": result["data_refs"]})
        if media_mode == "selected_events":
            event_media = {ref for event in events for ref in event.get("media", [])}
            legal_media = set(original.get("selected_media", []))
            previous = routed_view.get("selected_media", [])
            chosen_media = list(dict.fromkeys(ref for ref in previous if ref in event_media and ref in legal_media))
            views[view_id]["selected_media"] = chosen_media
            manifest[-1]["media_selection"] = {
                "mode": "selected_events", "source_view_ref": str(tools.store.path("evidence_views", view_id).resolve()),
                "original_view_media_count": len(original.get("selected_media", [])),
                "input_media_count_before": len(previous), "selected_media_count": len(chosen_media),
                "omitted_media_count": len(previous) - len(chosen_media),
                "image_bytes_sent": False, "visual_content": "unknown",
            }
        if projection is not None:
            manifest[-1]["source_view_ref"] = str(tools.store.path("evidence_views", view_id).resolve())
    selected["evidence_views"] = list(views.values())
    # Avoid duplicating event payloads from omitted L0 in the navigation view.
    for row in selected.get("experience_overview", {}).get("facts", []):
        for key in ("segments", "faults", "skill_observations"):
            row[key] = [e for e in row.get(key, []) if e.get("event_ref") in wanted]
    selected["evidence_input"] = {"mode": "selected_l0", "reason": policy["reason"],
                                  "complete_l0_in_prompt": False, "views": manifest,
                                  "read_interface": "read_development_events(result_id) lists event IDs; supply event_ids to read exact events"}
    if media_mode is not None:
        selected["evidence_input"]["media_refs"] = media_mode
    if projection is not None:
        selected["evidence_input"]["sdk_projection"] = projection
    return selected



def readable_input(tools, request):
    """Expose selected original facts and their omissions without decoding a codec."""
    from .task_policy import task_policy
    policy = task_policy(tools.protocol).get("research_input")
    if policy is None:
        return request
    if (policy.get("mode") != "readable_v1"
            or type(policy.get("max_read_requests")) is not int
            or not 1 <= policy["max_read_requests"] <= 16
            or type(policy.get("max_events_per_read")) is not int
            or not 1 <= policy["max_events_per_read"] <= 256):
        raise ValueError("readable Research input needs finite legal-event read bounds")
    rows = []
    selected = {v["view_id"]: v for v in request.get("evidence_views", [])}
    for result in request["evidence"]:
        view_id = result["intervention"].get("evidence_view_id")
        if view_id is None:
            rows.append({"result_id": result["result_id"], "missing": ["development L0 view"]})
            continue
        original = tools.store.get("evidence_views", view_id)
        view = selected.get(view_id, {"l0": [], "selected_media": []})
        selected_ids = {e["event_id"] for e in view["l0"]}
        rows.append({"result_id": result["result_id"], "view_id": view_id,
            "source_refs": result["data_refs"],
            "selected_event_ids": [e["event_id"] for e in view["l0"]],
            "omitted_event_ids": [e["event_id"] for e in original["l0"] if e["event_id"] not in selected_ids],
            "event_index": [{k: e.get(k) for k in ("event_id", "sequence", "kind")} for e in original["l0"]],
            "selected_media_references": view.get("selected_media", []),
            "missing": [{"event_id": e["event_id"], "fields": e.get("missing", [])} for e in view["l0"]
                        if e.get("missing")],
            "media_consumption": {"mode": "text_only", "image_bytes_sent": False,
                                  "visual_content": "unknown"}})
    return {**request, "readable_evidence": {"schema": "rsi-readable-development-v1", "views": rows,
            "facts_location": "evidence_views[].l0", "overview_role": "navigation_only",
            "rhoA_objective": "recorded_event_coverage_minus_reference_cost",
            "payload_bytes": None, "persisted_media_bytes": None, "diagnostic_quality": None},
            "evidence_read_policy": policy}


def sdk_input(transport, output_schema):
    """Exact public SDK arguments; HTTP serialization and media consumption are separate facts."""
    request = unpack(transport)
    return {"schema": "rsi-research-sdk-input-v1", "request_id": request["request_id"],
        "job_id": request["job_id"], "bundle_id": request["bundle_id"],
        "business_request_ref": "request.json", "business_transport_schema": transport["schema"],
        "prompt": model_prompt(transport), "model": "gpt-6.1-sol", "reasoning_effort": "high",
        "developer_instructions": "Use only supplied evidence. Do not call tools, read files, access networks, or launch experiments.",
        "output_schema": output_schema, "media_mode": "text_only", "image_bytes_sent": False,
        "http_serialization": "unknown", "model_attention": "unknown"}
