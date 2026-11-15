"""Prepare the registered one-call boundary before any registry or request dispatch."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rsi_vla.research_supervision import prepare_research_storage_recovery
from rsi_vla.storage import atomic_json


class RequestLimitTests(unittest.TestCase):
    def prepare(self, limit=None):
        import rsi_vla.midterm_host

        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        repo = Path(rsi_vla.midterm_host.__file__).parents[3]
        scope = root / "scope.json"
        atomic_json(scope, {"scope_id": "scope", "write_root": str(root)})
        protocol = root / "protocol.json"
        atomic_json(
            protocol,
            {
                "task_policy": {
                    "research_per_request": {
                        "wall_seconds": 180,
                        "storage_bytes": 16 * 1024**2,
                    }
                }
            },
        )
        source = root / "old-source/manifest.json"
        atomic_json(
            source,
            {
                "version": "5a-assets-offline",
                "files": ["core/src/rsi_vla/research_supervision.py"],
            },
        )
        inputs = root / "inputs.json"
        atomic_json(inputs, {"authorization": {"binding": {}, "operation_ids": []}})
        requests = []
        operations = []
        nodes = {}
        for i, name in enumerate(("research-one", "research-two"), 1):
            domain = root / (name + "-domain.json")
            atomic_json(
                domain,
                {
                    "purpose": "drawer",
                    "evidence_selection": {
                        "sdk_projection": "terminal_history_catalog_v2"
                    },
                },
            )
            requests.append(
                {
                    "request_id": name,
                    "job_id": "same-job",
                    "job_limits": {"model_requests": 22},
                    "domain_policy_ref": str(domain),
                    "reserved_bundle_id": name + "-proposal",
                    "output_ref": str(root / "output" / name),
                    "scratch_ref": str(root / "scratch" / name),
                    "permit_ref": str(root / "permits" / name),
                    "preflight_ref": str(root / "preflights" / name),
                }
            )
            for kind in ("prepare", "model", "feedback"):
                payload = {"request_id": name}
                if i == 2 and kind == "prepare":
                    payload["parent_request_id"] = "research-one"
                operations.append(
                    {
                        "operation_id": name + "-" + kind,
                        "kind": (
                            "research_prepare"
                            if i == 1
                            else "evidence_followup_prepare"
                        )
                        if kind == "prepare"
                        else "research"
                        if kind == "model"
                        else "research_feedback",
                        "dynamic": kind == "model",
                        "payload": payload,
                    }
                )
            nodes["prepare" + str(i)] = {
                "operation_id": name + "-prepare",
                "transitions": [{"next": "model" + str(i)}],
            }
            nodes["model" + str(i)] = {
                "operation_id": name + "-model",
                "transitions": [{"next": "feedback" + str(i)}],
            }
            nodes["feedback" + str(i)] = {
                "operation_id": name + "-feedback",
                "transitions": [
                    {
                        "when": {
                            "path": ["next_action"],
                            "equals": "prepare_evidence_followup",
                        },
                        "next": "prepare2" if i == 1 else "supplement-limit",
                    },
                    {"next": "deferred"},
                ],
            }
        nodes.update(
            {
                "supplement-limit": {"terminal": "supplement-limit"},
                "deferred": {"terminal": "deferred"},
            }
        )
        core = {
            "scope_authorization_ref": str(scope),
            "protocol_ref": str(protocol),
            "source_manifest_ref": str(source),
            "repository_root": str(repo),
            "phase_ref": "old-phase",
            "base_bundle_id": "base",
            "research_store_ref": str(root / "store"),
            "operations": operations,
            "research_requests": requests,
            "continuation": {"start": "prepare1", "nodes": nodes},
        }
        config = {
            "inputs_ref": str(inputs),
            "hub_configuration_ref": "hub",
            "python": sys.executable,
            "script_ref": str(repo / "scripts/run_midterm_host.py"),
            "poll_seconds": 3,
            "node": "node",
            "autoloop_root": "autoloop",
            "autoloop_home": "home",
        }
        previous = SimpleNamespace(
            core_config=core,
            configuration_ref=root / "old-host.json",
            config=config,
            ledger=SimpleNamespace(root=root, authority=lambda: {"scope_id": "scope"}),
            coordinator=lambda: SimpleNamespace(
                status=lambda: {
                    "status": "completed",
                    "terminal": "research_preparation_technical_stop",
                }
            ),
        )
        before = {
            Path(r["domain_policy_ref"]): Path(r["domain_policy_ref"]).read_bytes()
            for r in requests
        }
        options = {} if limit is None else {"research_request_limit": limit}
        with patch(
            "rsi_vla.midterm_host.MidtermHost",
            side_effect=[previous, SimpleNamespace(coordinator=lambda: None)],
        ):
            result = prepare_research_storage_recovery(
                previous.configuration_ref, "next", "repair", **options
            )
        registered = json.loads(
            (root / "operator/next/continuation-configuration.json").read_text()
        )
        for p, blob in before.items():
            self.assertEqual(p.read_bytes(), blob)
        self.assertEqual(
            [r["job_id"] for r in registered["research_requests"]],
            ["same-job"] * len(registered["research_requests"]),
        )
        self.assertTrue(
            all(
                r["job_limits"] == {"model_requests": 22}
                for r in registered["research_requests"]
            )
        )
        self.assertEqual(registered["phase_ref"], "old-phase")
        self.assertFalse(result["actual_dispatch"])
        return registered, result

    def test_default_keeps_both_requests_and_original_followup_route(self):
        core, result = self.prepare()
        self.assertEqual(len(core["research_requests"]), 2)
        self.assertEqual(len(core["operations"]), 6)
        self.assertEqual(
            core["continuation"]["nodes"]["feedback1"]["transitions"][0]["next"],
            "prepare2",
        )
        self.assertEqual(result["research_request_limit"], 2)
        self.assertTrue(
            core["research_storage_recovery"]["evidence_followup_dispatch_authorized"]
        )

    def test_one_call_keeps_first_feedback_and_archival_but_no_second_model_registration(
        self,
    ):
        core, result = self.prepare(1)
        self.assertEqual(len(core["research_requests"]), 1)
        self.assertEqual(
            [r["kind"] for r in core["operations"]],
            ["research_prepare", "research", "research_feedback"],
        )
        self.assertEqual(
            core["continuation"]["nodes"]["feedback1"]["transitions"][0]["next"],
            "supplement-limit",
        )
        self.assertNotIn("prepare2", core["continuation"]["nodes"])
        self.assertNotIn("model2", core["continuation"]["nodes"])
        self.assertNotIn("feedback2", core["continuation"]["nodes"])
        self.assertEqual(len(core["research_storage_recovery"]["request_mapping"]), 1)
        self.assertFalse(
            core["research_storage_recovery"]["evidence_followup_dispatch_authorized"]
        )
        self.assertEqual(
            result["registered_research_requests"], ["research-one-repair"]
        )
        self.assertEqual(
            core["research_requests"][0]["request_id"], "research-one-repair"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
