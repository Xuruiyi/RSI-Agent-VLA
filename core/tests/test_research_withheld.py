"""Pure file fixtures: completed transport withheld by the whole-request storage stop."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from rsi_vla.storage import atomic_json
from rsi_vla.research_supervision import close_ended_research
from rsi_vla.midterm_host import MidtermHost, MidtermContinuationBackend
from test_research_storage import EndedResearchTests


class CompletedResponseBudgetTests(EndedResearchTests):
    def response_facts(self):
        identity = json.loads((self.output / "request_gate_live.json").read_text())[
            "identity"
        ]
        atomic_json(
            self.output / "request_gate_live.json",
            {"identity": identity, "requests_forwarded": 1},
        )
        events = [
            {
                "identity": identity,
                "kind": "request_admitted",
                "request_number": 1,
                "model": "gpt-6.1-sol",
                "reasoning_effort": "high",
            },
            {
                "identity": identity,
                "kind": "response_terminal",
                "request_number": 1,
                "backend_model": "gpt-6.1-sol",
                "usage": {
                    "input_tokens": 155568,
                    "output_tokens": 2323,
                    "input_tokens_details": {"cached_tokens": 0},
                },
            },
        ]
        (self.output / "model_requests.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in events)
        )
        atomic_json(
            self.output / "request_gate.json",
            {
                "identity": identity,
                "requests_forwarded": 1,
                "cleanup_complete": True,
                "active_requests_at_close": 0,
                "unfinished_request_numbers": [],
            },
        )
        atomic_json(
            self.output / "supervisor.json",
            {
                "identity": identity,
                "status": "completed",
                "exit_code": 0,
                "cleanup_confirmed": True,
                "remaining_process_ids": [],
            },
        )
        atomic_json(
            self.output / "sdk-result.json",
            {
                "status": "completed",
                "final_response": json.dumps(
                    {
                        "schema": "rsi-bounded-research-response-v2",
                        "request_id": "r",
                        "decision": "defer",
                        "evidence_requests": [
                            {"result_id": "source", "event_ids": ["event"]}
                        ],
                    }
                ),
            },
        )
        atomic_json(
            self.job / "runtime_context.json",
            {"limits": {"storage_bytes": 16 * 1024**2}},
        )
        phase = json.loads(self.phase.read_text())
        phase.update(
            limits={"model_requests": 10},
            used={"model_requests": 0},
            remaining={"model_requests": 10},
        )
        atomic_json(self.phase, phase)
        self.host.core_config["phase_ref"] = str(self.phase)

    def test_completed_response_is_preserved_withheld_and_known_usage_charged(self):
        self.response_facts()
        originals = {p: p.read_bytes() for p in self.output.glob("*.json")}
        with (
            patch(
                "rsi_vla.research_supervision.research_actor_evidence",
                return_value={"established": True},
            ),
            patch(
                "rsi_vla.midterm_host.inspect_owned_runtime_processes",
                return_value={"live_processes": [], "unreadable_owned_processes": []},
            ),
        ):
            receipt = close_ended_research(self.host, "r-model", self.report)
            self.assertEqual(
                close_ended_research(self.host, "r-model", self.report), receipt
            )
        value = receipt["value"]
        self.assertEqual(value["response_application"], "withheld_budget_stop")
        self.assertEqual(
            value["technical_terminal"], "research_budget_stop_response_withheld"
        )
        self.assertEqual(
            value["response_tokens"],
            {"input_tokens": 155568, "output_tokens": 2323, "cached_input_tokens": 0},
        )
        self.assertEqual(self.ledger.read()["used"]["model_requests"], 1)
        self.assertEqual(
            json.loads((self.store / "jobs/job.json").read_text())["used"][
                "model_requests"
            ],
            1,
        )
        self.assertFalse((self.output / "validated-response.json").exists())
        self.assertFalse((self.store / "llm_executions/r.json").exists())
        self.assertFalse((self.store / "research_evidence_supplements/r.json").exists())
        for p, body in originals.items():
            self.assertEqual(p.read_bytes(), body)
        backend = object.__new__(MidtermContinuationBackend)
        self.assertEqual(
            backend.observe({"receipts": {"r-model": receipt}}, {}),
            {"terminal": "research_budget_stop_response_withheld"},
        )

    def test_native_cleanup_or_nonbudget_stop_cannot_claim_withheld_budget_end(self):
        self.response_facts()
        before = self.ledger.path.read_bytes()
        atomic_json(
            self.output / "request_gate.json",
            {
                "identity": {},
                "requests_forwarded": 1,
                "cleanup_complete": False,
                "active_requests_at_close": 1,
            },
        )
        with (
            patch(
                "rsi_vla.research_supervision.research_actor_evidence",
                return_value={"established": True},
            ),
            patch(
                "rsi_vla.midterm_host.inspect_owned_runtime_processes",
                return_value={"live_processes": [], "unreadable_owned_processes": []},
            ),
        ):
            with self.assertRaisesRegex(PermissionError, "native closure"):
                close_ended_research(self.host, "r-model", self.report)
        self.assertEqual(self.ledger.path.read_bytes(), before)

    def test_phase_and_nested_job_credit_only_their_own_actual_native_charges(self):
        costs = {
            "whole_job_wall_seconds": 96,
            "artifact_bytes": 15171427,
            "model_requests": 1,
        }
        self.host.core_config.update(
            research_request_envelope="complete_v2", phase_ref=str(self.phase)
        )
        self.host.config["operations"] = {
            "r-model": {
                "budget": {"storage_bytes": 20 * 1024**2, "model_requests": 1},
                "output_ref": str(self.output),
            }
        }
        atomic_json(self.output / "execution-costs.json", {"costs": costs})
        report = {
            **self.report,
            "wall_seconds": 99,
            "business_storage_delta_bytes": 15934548,
            "storage_bytes": 17925950,
        }
        for phase_paid, job_paid in ((False, False), (False, True), (True, True)):
            with self.subTest(phase_paid=phase_paid, job_paid=job_paid):
                limits = {
                    "wall_seconds": 500,
                    "storage_bytes": 100000000,
                    "model_requests": 10,
                    "gpu_seconds": 100,
                }
                used = dict.fromkeys(limits, 0)
                if phase_paid:
                    used.update(wall_seconds=96, storage_bytes=15171427)
                phase = {
                    "limits": limits,
                    "used": used,
                    "remaining": {k: v - used[k] for k, v in limits.items()},
                    "unresolved_costs": [],
                    "research_charges": [{"request_id": "r"}] if phase_paid else [],
                }
                atomic_json(self.phase, phase)
                job = json.loads((self.store / "jobs/job.json").read_text())
                job["charges"] = []
                job["used"] = dict.fromkeys(job["limits"], 0)
                if job_paid:
                    job["charges"] = [{"reason": "llm r", "costs": costs}]
                    job["used"].update(costs)
                atomic_json(self.store / "jobs/job.json", job)
                MidtermHost.charge_phase(self.host, "r-model", report)
                phase = json.loads(self.phase.read_text())
                job = json.loads((self.store / "jobs/job.json").read_text())
                self.assertEqual(phase["used"]["wall_seconds"], 99)
                self.assertEqual(phase["used"]["storage_bytes"], 15934548)
                self.assertEqual(job["used"]["whole_job_wall_seconds"], 99)
                self.assertEqual(job["used"]["artifact_bytes"], 15934548)
                before_phase = self.phase.read_bytes()
                before_job = (self.store / "jobs/job.json").read_bytes()
                MidtermHost.charge_phase(self.host, "r-model", report)
                self.assertEqual(self.phase.read_bytes(), before_phase)
                self.assertEqual(
                    (self.store / "jobs/job.json").read_bytes(), before_job
                )


class BuilderProjectionTests(unittest.TestCase):
    def test_new_catalog_is_registered_in_new_domain_before_freeze_without_resetting_job(
        self,
    ):
        import tempfile, sys
        from types import SimpleNamespace
        from rsi_vla.research_supervision import prepare_research_storage_recovery
        import rsi_vla.midterm_host

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
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
            domain = root / "old-domain.json"
            atomic_json(
                domain,
                {
                    "purpose": "drawer diagnosis",
                    "evidence_selection": {
                        "sdk_projection": "terminal_history_v1",
                        "media_refs": "selected_events",
                    },
                },
            )
            old = domain.read_bytes()
            source = root / "old-source/manifest.json"
            atomic_json(
                source,
                {
                    "version": "5a-assets-offline",
                    "files": ["core/src/rsi_vla/research_transport.py"],
                },
            )
            inputs = root / "old-inputs.json"
            atomic_json(inputs, {"authorization": {"binding": {}, "operation_ids": []}})
            ops = [
                {
                    "operation_id": "research-one-" + kind,
                    "kind": "research_prepare" if kind == "prepare" else "research",
                    "dynamic": kind != "prepare",
                    "payload": {"request_id": "research-one"},
                }
                for kind in ("prepare", "model")
            ]
            core = {
                "scope_authorization_ref": str(scope),
                "protocol_ref": str(protocol),
                "source_manifest_ref": str(source),
                "repository_root": str(repo),
                "phase_ref": "original-phase",
                "base_bundle_id": "base",
                "research_store_ref": str(root / "store"),
                "operations": ops,
                "research_requests": [
                    {
                        "request_id": "research-one",
                        "job_id": "original-job",
                        "job_limits": {"model_requests": 22},
                        "domain_policy_ref": str(domain),
                        "reserved_bundle_id": "research-one-proposal",
                        "output_ref": str(root / "output/research-one"),
                        "scratch_ref": str(root / "scratch/research-one"),
                        "permit_ref": "old-permit",
                        "preflight_ref": "old-preflight",
                    }
                ],
                "continuation": {
                    "start": "prepare",
                    "nodes": {
                        "prepare": {
                            "operation_id": ops[0]["operation_id"],
                            "transitions": [{"next": "model"}],
                        },
                        "model": {
                            "operation_id": ops[1]["operation_id"],
                            "transitions": [{"next": "done"}],
                        },
                        "done": {"terminal": "done"},
                    },
                },
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
                ledger=SimpleNamespace(
                    root=root, authority=lambda: {"scope_id": "scope"}
                ),
                coordinator=lambda: SimpleNamespace(
                    status=lambda: {
                        "status": "completed",
                        "terminal": "research_budget_stop_response_withheld",
                    }
                ),
            )
            with patch(
                "rsi_vla.midterm_host.MidtermHost",
                side_effect=[previous, SimpleNamespace(coordinator=lambda: None)],
            ):
                prepare_research_storage_recovery(
                    previous.configuration_ref,
                    "catalog-v2",
                    "catalog-v2",
                    research_sdk_projection="terminal_history_catalog_v2",
                )
            registered = json.loads(
                (
                    root / "operator/catalog-v2/continuation-configuration.json"
                ).read_text()
            )
            request = registered["research_requests"][0]
            self.assertEqual(request["job_id"], "original-job")
            self.assertEqual(request["job_limits"], {"model_requests": 22})
            self.assertEqual(registered["phase_ref"], "original-phase")
            self.assertEqual(
                json.loads(Path(request["domain_policy_ref"]).read_text())[
                    "evidence_selection"
                ]["sdk_projection"],
                "terminal_history_catalog_v2",
            )
            self.assertEqual(domain.read_bytes(), old)
            self.assertEqual(
                registered["research_storage_recovery"]["research_sdk_projection"],
                "terminal_history_catalog_v2",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
