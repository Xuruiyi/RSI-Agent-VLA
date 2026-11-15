#!/usr/bin/env python3
# Copyright 2026 RSI-Agent-VLA Authors.
"""Plan or run one host-supervised pinned public demonstration CPU operation."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core/src"))
from rsi_vla.adapters.libero_public_data import (
    acquire_registered_files,
    materialize_registered_data,
    registered_plan,
    require_data_permit,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    parser.add_argument(
        "--action",
        choices=["plan", "acquire", "materialize"],
        default="plan",
    )
    parser.add_argument("--permit")
    parser.add_argument("--output")
    parser.add_argument("--source-root")
    args = parser.parse_args(argv)
    registration_ref = Path(args.registration).resolve()
    registration = json.loads(registration_ref.read_text())
    if args.action == "plan":
        print(json.dumps(registered_plan(registration), indent=2))
        return 0
    if not args.permit or not args.output:
        raise PermissionError(
            "public data operation requires its host permit and exact output root"
        )
    output = Path(args.output).resolve()
    permit = json.loads(Path(args.permit).read_text())
    registration, plan = require_data_permit(
        registration_ref, permit, action=args.action, output_root=output
    )
    if output.exists():
        raise FileExistsError(
            "data operation output already exists; reconcile its receipt"
        )
    identity = {key: permit[key] for key in ("scope_id", "operation_id", "owner_uid")}
    result = {}
    error = None
    try:
        if args.action == "materialize" and (
            not args.source_root
            or Path(args.source_root).resolve() != Path(permit["source_root"]).resolve()
        ):
            raise PermissionError(
                "materialization source root differs from host permit"
            )
        os.environ.update(
            HF_HUB_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
            CUDA_VISIBLE_DEVICES="",
            JAX_PLATFORMS="cpu",
            HF_DATASETS_CACHE=str(output.parent / "datasets-cache"),
        )
        if args.action == "acquire":
            report = acquire_registered_files(
                registration,
                output,
                max_download_bytes=permit["max_download_bytes"],
                operation_identity=identity,
            )
            result = {
                "source_ref": str(output / "source.json"),
                "download_response_payload_bytes": report[
                    "downloaded_demonstration_bytes"
                ],
            }
        else:
            result = materialize_registered_data(registration, args.source_root, output)
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        output.mkdir(parents=True, exist_ok=True)
        meter_ref = output / "model-meter.json"
        meter = {
            "schema": "rsi-public-data-model-meter-v1",
            **identity,
            "status": "closed",
            "project_model_requests": 0,
            "model_clients_created": [],
            "model_transports_created": [],
            "producer": "registered public-data CPU and pinned public download I/O; no model component constructed",
        }
        meter_ref.write_text(json.dumps(meter, indent=2) + "\n")
        download = (
            _download_result(output, identity)
            if args.action == "acquire"
            else {
                "download_operation": False,
                "download_complete": None,
                "download_meter_closed": None,
                "download_ledger_ref": None,
                "observed_download_payload_bytes_lower_bound": None,
                "download_response_payload_bytes": None,
            }
        )
        result.update(
            schema="rsi-libero-public-data-result-v1",
            **identity,
            status="failed" if error else "completed",
            error=error,
            action=args.action,
            registration_ref=str(registration_ref),
            source_manifest_ref=permit["source_manifest_ref"],
            model_meter_ref=str(meter_ref),
            training_admission="not_admitted",
            **download,
        )
        (output / "job-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 1 if error else 0


def _download_result(output, identity):
    ledger_ref = output / "download-ledger.jsonl"
    meter_ref = output / "download-meter.json"
    lower = None
    valid = True
    if ledger_ref.is_file():
        lower = 0
        for line in ledger_ref.read_text().splitlines():
            try:
                row = json.loads(line)
                if (
                    any(row.get(key) != value for key, value in identity.items())
                    or type(row.get("response_payload_bytes")) is not int
                    or row["response_payload_bytes"] < 0
                ):
                    valid = False
                    break
                lower += row["response_payload_bytes"]
            except (ValueError, TypeError):
                valid = False
                break
    try:
        meter = json.loads(meter_ref.read_text())
    except (OSError, ValueError):
        meter = {}
    closed = bool(
        valid
        and lower is not None
        and meter.get("schema") == "rsi-public-data-download-meter-v1"
        and all(meter.get(key) == value for key, value in identity.items())
        and meter.get("status") == "closed"
        and meter.get("download_meter_closed") is True
        and meter.get("download_response_payload_bytes") == lower
    )
    return {
        "download_operation": True,
        "download_complete": meter.get("download_complete", False),
        "download_meter_closed": closed,
        "download_meter_ref": str(meter_ref) if meter_ref.is_file() else None,
        "download_ledger_ref": str(ledger_ref) if ledger_ref.is_file() else None,
        "observed_download_payload_bytes_lower_bound": lower,
        "download_response_payload_bytes": lower if closed else None,
    }


if __name__ == "__main__":
    raise SystemExit(main())
