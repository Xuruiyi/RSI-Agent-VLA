"""Replay a registered development-only denominator into isolated rho snapshots."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from rsi_vla.contracts import RunBundle
from rsi_vla.relation_learning import freeze_relations, learn_relations
from rsi_vla.research import ResearchJob
from rsi_vla.research_tools import ResearchTools
from rsi_vla.storage import ResearchStore, atomic_json


RESULT_IDS = ["completion11-base-dev-result",
              "completion11-research02-experiment-bound-result",
              "completion11-research02-experiment-base_salad-s2-bound-result"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope-root", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("replay output already exists; inspect its receipt before another run")
    root = args.scope_root.resolve()
    registration = json.loads(args.registration.read_text())
    registered = {row["run_id"]: Path(row["output_ref"]).resolve() for row in registration["denominator"]}
    if len(registered) != 3:
        raise ValueError("this replay needs the registered three-run denominator")
    original = ResearchStore(root / "state/research")
    protocol = json.loads((root / "operator/frozen-protocol.json").read_text())
    source = ResearchStore(args.output / "inputs")
    derived = ResearchStore(args.output / "learned")
    results, seen, input_refs = [], set(), []
    for key in RESULT_IDS:
        result = original.get("development_results", key)
        identity = result["intervention"]["identity"]
        run_id = identity["run_id"]
        if run_id not in registered or run_id in seen:
            raise ValueError("replay denominator differs from registration")
        seen.add(run_id)
        if result["role"] != "development" or result["intervention"]["information_scope"] != "development":
            raise ValueError("formal/sealed evidence cannot enter relation learning")
        # References identify the admitted development capture. No payload file is opened.
        context_path = registered[run_id] / "runtime_context.json"
        context = json.loads(context_path.read_text())
        if context.get("information_scope") != "development" or any(
            context[name] != identity[name] for name in ("run_id", "episode_id", "bundle_id", "protocol_id")
        ):
            raise ValueError("registered development execution identity mismatch")
        bundle = original.get("bundles", result["bundle_id"])
        for kind, ref in (("rho", bundle["rho_ref"]), ("psi", bundle["psi_ref"]), ("memory", bundle["memory_ref"])):
            source.put(kind, ref, original.get(kind, ref))
        source.save_bundle(RunBundle(**bundle))
        view_id = result["intervention"]["evidence_view_id"]
        refs = [Path(ref) for ref in result["data_refs"] if Path(ref).name == view_id + ".json"]
        if len(refs) != 1:
            raise ValueError("one exact development view reference required")
        view_path = refs[0].resolve()
        allowed = [root / "state/research/evidence_views", root / "state/evaluation/development"]
        if not any(view_path.is_relative_to(directory) for directory in allowed):
            raise ValueError("view reference outside admitted development namespaces")
        view = json.loads(view_path.read_text())
        source.put("evidence_views", view_id, view)
        source.put("development_results", key, result)
        results.append(result)
        input_refs.append({"result_ref": str(original.path("development_results", key)),
                           "context_ref": str(context_path), "view_ref": str(view_path)})
    if seen != set(registered):
        raise ValueError("registered denominator incomplete")
    base = RunBundle(**source.get("bundles", "initial-deployment09-completion11"))
    policy = {"historical_development_bundle_ids": sorted({r["bundle_id"] for r in results})}
    tools = ResearchTools(source, ResearchJob(source, "offline-replay", base.bundle_id, {"offline_steps": 10}),
                          protocol, domain_policy=policy)
    before = source.get("rho", base.rho_ref)
    state = learn_relations(tools, derived, "development-metadata", RESULT_IDS, max_feedback=64)
    target = freeze_relations(tools, derived, "development-metadata",
                              rho_ref="rho-development-metadata-v1", bundle_id="rho-development-metadata-v1")
    for bundle_id in policy["historical_development_bundle_ids"]:
        bundle = RunBundle(**source.get("bundles", bundle_id))
        derived.save_bundle(bundle)
    for result in results:
        key = result["result_id"]
        derived.put("development_results", key, result)
        view_id = result["intervention"]["evidence_view_id"]
        derived.put("evidence_views", view_id, source.get("evidence_views", view_id))
    inherited = ResearchTools(derived, ResearchJob(derived, "offline-inherited", target.bundle_id,
                                                  {"offline_steps": 10}), protocol, domain_policy=policy)
    rows = []
    for result in results:
        key = result["result_id"]
        task = result["assignments"][0]["task_key"]
        step = inherited.prepare_relation_step([key], requested_worker="vla", allowed_workers=["vla", "observe"],
            state=task, inspected=True, legal_repairs=["planner_task_hint"], experiment_permitted=True)
        view = source.get("evidence_views", result["intervention"]["evidence_view_id"])
        after = step["evidence_views"][0]
        assert view["l0"] == after["l0"]
        rows.append({"result_id": key, "run_id": result["intervention"]["identity"]["run_id"],
                     "l0_event_count": len(view["l0"]), "l0_unchanged": True,
                     "original_l1_references": len(view["selected_media"]),
                     "learned_l1_references": len(after["selected_media"]),
                     "worker": step["worker"], "research_step": step["research_step"]})
    second = ResearchStore(args.output / "inherited-round")
    inherited_state = learn_relations(inherited, second, "next-round", [], max_feedback=64)
    next_target = freeze_relations(inherited, second, "next-round",
                                   rho_ref="rho-development-inherited-v1", bundle_id="rho-development-inherited-v1")
    assert inherited_state["feedback"] == state["feedback"]
    assert source.get("rho", base.rho_ref) == before
    report = {"schema": "rsi-relation-metadata-replay-v1", "registration_ref": str(args.registration.resolve()),
        "input_refs": input_refs, "denominator": rows, "feedback": state["feedback"],
        "missing": state["missing"], "replacement_bundle": asdict(target),
        "inherited_bundle": asdict(next_target), "sample_denominator_preserved": True,
        "native_success_attribution": False, "causal_claim": False, "runtime_validation": False,
        "real_research_calls": 0, "actual_autonomous_candidates_created": 0,
        "source_rho_unchanged": True, "active_modified": False,
        "payload_files_opened": [], "limit": "A metadata coverage only; B lacks independent route feedback; C observed screening only"}
    atomic_json(args.output / "report.json", report)
    print(json.dumps({"report_ref": str((args.output / "report.json").resolve()),
                      "denominator": rows, "feedback_count": len(state["feedback"]),
                      "missing": state["missing"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
