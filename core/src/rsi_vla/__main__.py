"""Bounded offline command line, no network or execution backend."""
import argparse
import json
from pathlib import Path
from . import EventLedger, ResearchStore, RunBundle, initialize, load_protocol
from .adapters.rpent import read_trajectory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="persist initial fixed policies and legal external-prior memory")
    init.add_argument("--artifacts", required=True)
    init.add_argument("--profile", required=True, help="config/run_profile_v6.json for new medium runs")
    init.add_argument("--protocol", required=True)
    init.add_argument("--bundle", help="defaults to the protocol's fixed anchor identity")
    imp = commands.add_parser("import-rpent", help="read existing states/transcript; does not execute RPent")
    imp.add_argument("--artifacts", required=True)
    imp.add_argument("--states", required=True)
    imp.add_argument("--transcript", required=True)
    imp.add_argument("--result")
    imp.add_argument("--action-events", help="runtime action_events.jsonl; identity and sequence are checked")
    imp.add_argument("--protocol-id", help="require runtime-recorded protocol identity")
    imp.add_argument("--bundle", default="initial")
    imp.add_argument("--run", required=True)
    imp.add_argument("--episode", required=True)
    dev = commands.add_parser("import-development", help="versioned legal development input; no execution or charge")
    dev.add_argument("--artifacts", required=True)
    dev.add_argument("--protocol", required=True)
    dev.add_argument("--run-dir", required=True)
    dev.add_argument("--result-id", required=True)
    check = commands.add_parser("check-protocol")
    check.add_argument("path")
    args = parser.parse_args(argv)
    if args.command == "check-protocol":
        protocol = load_protocol(args.path)
        print(json.dumps({"protocol_id": protocol["protocol_id"], "scope": protocol["scope"],
                          "dynamic_authorized": protocol.get("dynamic_authorized", False)}))
        return
    store = ResearchStore(Path(args.artifacts) / "research")
    if args.command == "init":
        profile_path = str(Path(args.profile).resolve())
        profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
        protocol = load_protocol(args.protocol)
        memory = profile["memory"]
        bundle_id = args.bundle or protocol["anchor_bundle_id"]
        store.put("profiles", bundle_id, profile)
        frozen_profile = str(store.path("profiles", bundle_id).resolve())
        bundle = RunBundle(bundle_id, frozen_profile + "#/model", frozen_profile + "#/model/normalization",
                           frozen_profile + "#/control", "rho-" + bundle_id, "psi-" + bundle_id, "memory-" + bundle_id,
                           protocol["protocol_id"], frozen_profile, profile.get("code_ref", "aggregate-core-v0.4.0"))
        initialize(store, bundle, psi={"type": "fixed", "version": "initial", "learned": False,
                   "strategy": ["inspect_legal_evidence", "bounded_experiment", "submit_candidate_or_defer"],
                   "profile_ref": frozen_profile, "source_profile": profile_path},
                   memory={"provenance": "external_upstream_prior", "profile_ref": frozen_profile + "#/memory",
                           "upstream": memory, "legal_information": "existing public upstream runtime memory; no sealed evaluation labels",
                           "project_learning": False})
        print(json.dumps({"bundle_id": bundle_id, "artifact_root": str(store.root),
                          "evidence_level": "offline", "runtime_activated": False}))
        return
    if args.command == "import-development":
        from .development import import_development
        result = import_development(store, load_protocol(args.protocol), args.run_dir, result_id=args.result_id)
        print(json.dumps({"result_id": result.result_id, "outcome": result.outcome, "charged": False}))
        return
    ledger = EventLedger(store)
    events = read_trajectory(args.states, args.transcript, bundle_id=args.bundle,
                             run_id=args.run, episode_id=args.episode,
                             independent_result_path=args.result, action_events_path=args.action_events,
                             protocol_id=args.protocol_id)
    added = sum(ledger.append(event) for event in events)
    print(json.dumps({"events": len(events), "added": added, "outcome": events[-1].outcome,
                      "outcome_source": events[-1].outcome_source, "missing": events[-1].missing}))


if __name__ == "__main__":
    main()
