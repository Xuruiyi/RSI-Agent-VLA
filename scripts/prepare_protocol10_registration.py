#!/usr/bin/env python3
# ruff: noqa: UP017, FURB162 - Python 3.10 support
"""Materialize a reviewed protocol10 registration after affirmative authorization.

No permit is issued and no Runtime or Research request is dispatched.
"""

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core/src"))

from rsi_vla.contracts import RunBundle
from rsi_vla.improvement_protocol import (
    BASELINE,
    admit_resources,
    load_frozen_protocol,
    runtime_projection,
)
from rsi_vla.research import ResearchJob
from rsi_vla.storage import ResearchStore, atomic_json


def prepare(
    configuration_ref,
    authorization_ref,
    source_research,
    *,
    offline_seconds,
    offline_bytes,
):
    """Create deterministic preparation artifacts only within approved authority."""
    config = json.loads(Path(configuration_ref).read_text())
    protocol = load_frozen_protocol(
        config["protocol_ref"], config["frozen_protocol_ref"]
    )
    authority = json.loads(Path(authorization_ref).read_text())
    if authority.get("authorized") is not True:
        raise ValueError("explicit approved administrative authorization required")
    expiry = datetime.fromisoformat(
        str(authority.get("expires_at")).replace("Z", "+00:00")
    )
    if expiry.tzinfo is None or datetime.now(timezone.utc) >= expiry:
        raise ValueError("current unexpired administrative authority required")
    projected = runtime_projection(
        protocol,
        frozen_ref=config["frozen_protocol_ref"],
        authorization_ref=authorization_ref,
    )
    if (
        Path(authority["phase_snapshot_ref"]).resolve()
        != Path(config["phase_ref"]).resolve()
    ):
        raise ValueError(
            "administrative authority must name the unique registered phase ledger"
        )
    phase = json.loads(Path(config["phase_ref"]).read_text())
    admission = admit_resources(
        protocol,
        phase,
        offline_used_seconds=offline_seconds,
        offline_used_bytes=offline_bytes,
    )
    if not admission["resource_fit"]:
        raise ValueError("complete remaining frozen envelope is not admitted")
    source = ResearchStore(source_research)
    original = RunBundle(**source.get("bundles", BASELINE))
    if original.protocol_id != protocol["protocol_id"]:
        raise ValueError("source baseline protocol mismatch")
    profile = json.loads(Path(original.backend_ref).read_text())
    if profile.get("dynamic_authorized") is not False:
        raise ValueError("source must be the frozen unissued baseline profile")
    profile["dynamic_authorized"] = True
    target = ResearchStore(config["research"]["store_ref"])
    profile_ref = str(Path(config["profile_ref"]).resolve())
    # Load all shared inputs before writing the prepared store.
    snapshots = [
        (kind, ref, source.get(kind, ref))
        for kind, ref in [
            ("psi", original.psi_ref),
            ("memory", original.memory_ref),
            ("rho", original.rho_ref),
        ]
    ]
    bundle = replace(
        original,
        backend_ref=profile_ref,
        theta_ref=profile_ref + "#/model",
        processor_ref=profile_ref + "#/model/normalization",
        phi_ref=profile_ref + "#/control",
        code_ref=config.get("execution_code_ref", original.code_ref),
    )
    for destination, value in [
        (Path(config["runtime_protocol_ref"]), projected),
        (Path(profile_ref), profile),
    ]:
        if destination.exists() and json.loads(destination.read_text()) != value:
            raise ValueError(
                "prepared artifact already exists with conflicting content: "
                + str(destination)
            )
    atomic_json(config["runtime_protocol_ref"], projected)
    atomic_json(profile_ref, profile)
    for kind, ref, value in snapshots:
        target.put(kind, ref, value)
    target.save_bundle(bundle)
    job = ResearchJob(
        target, config["research"]["job_id"], BASELINE, config["research"]["job_limits"]
    )
    return {
        "dispatch": False,
        "prepared": True,
        "configuration_ref": str(Path(configuration_ref).resolve()),
        "job_id": job.state["job_id"],
        "admission": admission,
        "permits_issued": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("configuration", "administrative-authorization", "source-research"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--offline-used-seconds", type=float, required=True)
    parser.add_argument("--offline-used-bytes", type=int, required=True)
    args = parser.parse_args()
    try:
        result = prepare(
            args.configuration,
            args.administrative_authorization,
            args.source_research,
            offline_seconds=args.offline_used_seconds,
            offline_bytes=args.offline_used_bytes,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"dispatch": False, "prepared": False, "error": str(exc)}))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
