"""A real transport may finish with an inadmissible source; close without replay."""

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rsi_vla.storage import atomic_json, ResearchStore
from rsi_vla.midterm_host import MidtermLedger, MidtermContinuationBackend
from rsi_vla.research_tools import opportunity_source_eligibility
from rsi_vla.research_supervision import (
    research_policy_rejection_evidence,
    close_policy_rejected_research,
)


class PolicyRejectTests(unittest.TestCase):
    def setUp(self):
        from test_midterm_host import LedgerTests

        LedgerTests.setUp(self)
        self.scope["outer_limits"].update(model_requests=10, research_requests=5)
        atomic_json(self.scope_ref, self.scope)
        self.ledger.path.unlink()
        self.ledger = MidtermLedger(self.scope_ref)
        self.op = {
            "operation_id": "r-model",
            "kind": "research",
            "dynamic": True,
            "payload": {"request_id": "r"},
            "stage": "development",
        }
        self.output = self.root / "output"
        self.job = self.root / "host/jobs/r-model"
        self.store = ResearchStore(self.root / "store")
        proto = self.root / "protocol.json"
        atomic_json(proto, {"protocol_id": "p"})
        source = self.root / "frozen/manifest.json"
        atomic_json(source, {"files": []})
        policy_source = source.parent / "source/core/src/rsi_vla/research_tools.py"
        policy_source.parent.mkdir(parents=True)
        policy_source.write_text(
            '"technical diagnosis requires referenced invalid/unknown execution evidence"\n'
        )
        request = {
            "request_id": "r",
            "job_id": "job",
            "base_bundle_id": "base",
            "result_ids": ["success", "failure"],
            "output_ref": str(self.output),
            "permit_ref": str(self.root / "permit.json"),
        }
        self.core = {
            "operations": [self.op],
            "research_requests": [request],
            "base_bundle_id": "base",
            "protocol_ref": str(proto),
            "phase_ref": str(self.phase),
            "source_manifest_ref": str(source),
            "research_store_ref": str(self.store.root),
        }
        self.budget = {
            "model_requests": 1,
            "research_requests": 1,
            "storage_bytes": 16000,
            "current_disk_bytes": 16000,
        }
        self.host = SimpleNamespace(
            root=self.root / "host",
            core_config=self.core,
            ledger=self.ledger,
            config={
                "scope_id": self.scope["scope_id"],
                "operations": {"r-model": {"budget": self.budget, "gpu_devices": []}},
            },
        )
        identity = {
            "run_id": "r",
            "episode_id": "research-only",
            "bundle_id": "base",
            "protocol_id": "p",
        }
        atomic_json(
            self.output / "runtime_context.json",
            {**identity, "authorization_ref": request["permit_ref"]},
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
            self.output / "sdk-result.json",
            {
                "status": "completed",
                "final_response": json.dumps(
                    {
                        "request_id": "r",
                        "source": "technical_diagnostic",
                        "decision": "defer",
                        "result_ids": ["success", "failure"],
                    }
                ),
            },
        )
        events = [
            {
                "kind": "request_admitted",
                "identity": identity,
                "request_number": 1,
                "model": "gpt-6.1-sol",
                "reasoning_effort": "high",
            },
            {
                "kind": "response_terminal",
                "identity": identity,
                "request_number": 1,
                "backend_model": "gpt-6.1-sol",
                "usage": {
                    "input_tokens": 50,
                    "output_tokens": 10,
                    "input_tokens_details": {"cached_tokens": 0},
                },
            },
        ]
        (self.output / "model_requests.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in events)
        )
        self.store.put(
            "llm_executions",
            "r",
            {
                "request_id": "r",
                "job_id": "job",
                "base_bundle_id": "base",
                "costs": {
                    "model_requests": 1,
                    "input_tokens": 50,
                    "output_tokens": 10,
                    "cached_input_tokens": 0,
                },
                "transport_facts": {
                    "effective_model": "gpt-6.1-sol",
                    "effective_reasoning_effort": "high",
                },
            },
        )
        self.store.put(
            "research_evidence_supplements",
            "r",
            {
                "request_id": "r",
                "job_id": "job",
                "base_bundle_id": "base",
                "reads": [{"events": ["original exact facts"]}],
            },
        )
        self.error_ref = (
            self.store.root / "role_receipts/r-model-execute-revision-00001.json"
        )
        atomic_json(
            self.error_ref,
            {
                "role": "research",
                "request": {
                    "configuration": self.core,
                    "task": {"operation": self.op, "action": "execute"},
                },
                "error": "ValueError: technical diagnosis requires referenced invalid/unknown execution evidence",
            },
        )
        self.report = {
            "identity": {"run_id": "r-model"},
            "status": "completed",
            "exit_code": 1,
            "stop_reason": None,
            "cleanup_confirmed": True,
            "remaining_process_ids": [],
            "storage_bytes": 1000,
            "wall_seconds": 10,
        }
        atomic_json(
            self.job / "runtime_context.json",
            {"limits": {"storage_bytes": 16000, "wall_seconds": 180}},
        )
        atomic_json(
            self.job / "submission.json",
            {"operation": self.op, "owner_uid": os.getuid()},
        )
        atomic_json(
            self.job / "process.json",
            {
                "pid": 999999999,
                "owner_uid": os.getuid(),
                "start_ticks": 1,
                "boot_id": "fixture",
            },
        )
        atomic_json(self.job / "supervisor.json", self.report)
        atomic_json(
            self.job / "completion.json",
            {"status": "unknown", "original_role_error": str(self.error_ref)},
        )
        self.ledger.reserve(
            "r-model",
            budget=self.budget,
            phase_ref=self.phase,
            gpu_devices=[],
            receipt_ref=self.job / "completion.json",
        )
        self.ledger.mark_submitted("r-model")
        self.ledger.settle(
            "r-model",
            actual={
                "model_requests": 1,
                "research_requests": 1,
                "storage_bytes": 1000,
                "current_disk_bytes": 1000,
            },
            cleanup_confirmed=True,
        )
        self.results = [
            {
                "result_id": name,
                "outcome": name,
                "data_refs": [name + ".json"],
                "assignments": [
                    {
                        "outcome": name,
                        "validity": "valid",
                        "evidence_ref": name + ".json",
                    }
                ],
            }
            for name in ("success", "failure")
        ]
        self.tools = SimpleNamespace(query_development=lambda ids: self.results)

    def classify(self):
        with patch(
            "rsi_vla.task_completion.TaskCompletionBackend",
            return_value=SimpleNamespace(
                tools=lambda rid: (self.tools, self.core["research_requests"][0])
            ),
        ):
            return research_policy_rejection_evidence(self.host, "r-model", self.report)

    def close(self):
        with (
            patch(
                "rsi_vla.task_completion.TaskCompletionBackend",
                return_value=SimpleNamespace(
                    tools=lambda rid: (self.tools, self.core["research_requests"][0])
                ),
            ),
            patch(
                "rsi_vla.research_supervision.research_actor_evidence",
                return_value={"established": True},
            ),
            patch(
                "rsi_vla.midterm_host.inspect_owned_runtime_processes",
                return_value={"live_processes": [], "unreadable_owned_processes": []},
            ),
            patch(
                "rsi_vla.midterm_host.capture_closure_sources",
                return_value=self.root / "control/manifest.json",
            ),
        ):
            return close_policy_rejected_research(self.host, "r-model", self.report)

    def test_typed_rejection_preserves_response_reads_error_and_all_settled_costs(self):
        paths = [
            self.output / "sdk-result.json",
            self.error_ref,
            self.store.path("research_evidence_supplements", "r"),
            self.store.path("llm_executions", "r"),
            self.ledger.path,
            self.job / "completion.json",
        ]
        before = {p: p.read_bytes() for p in paths}
        value = self.classify()
        self.assertEqual(value["source_original"], "technical_diagnostic")
        self.assertFalse(value["eligibility"]["eligible"])
        receipt = self.close()
        self.assertEqual(self.close(), receipt)
        for p, blob in before.items():
            self.assertEqual(p.read_bytes(), blob)
        self.assertFalse(self.store.path("opportunities", "r-opportunity").exists())
        self.assertFalse(self.store.path("research_decisions", "r-defer").exists())
        self.assertFalse((self.output / "validated-response.json").exists())
        backend = object.__new__(MidtermContinuationBackend)
        backend.host = self.host
        self.assertEqual(backend.reconcile(self.op, {})["value"], receipt["value"])
        self.assertEqual(
            backend.observe({"receipts": {"r-model": receipt}}, {}),
            {"terminal": "research_policy_rejected"},
        )
        self.assertNotIn("research_technical_stop", receipt["value"])
        self.assertEqual(receipt["value"]["saved_evidence_read_groups"], 1)

    def test_eligible_technical_evidence_or_different_role_error_cannot_be_reclassified(
        self,
    ):
        self.results[0]["assignments"][0].update(validity="invalid", outcome="unknown")
        self.assertIsNone(self.classify())
        self.results[0]["assignments"][0].update(validity="valid", outcome="success")
        error = json.loads(self.error_ref.read_text())
        error["error"] = "RuntimeError: infrastructure fault"
        atomic_json(self.error_ref, error)
        self.assertIsNone(self.classify())

    def test_outer_budget_stop_or_created_opportunity_is_not_policy_rejection(self):
        self.report.update(status="interrupted", stop_reason="storage_bytes")
        self.assertIsNone(self.classify())
        self.report.update(status="completed", stop_reason=None)
        self.store.put(
            "opportunities", "r-opportunity", {"actual_prior_artifact": "preserved"}
        )
        self.assertIsNone(self.classify())

    def test_pure_source_helper_keeps_existing_three_gates(self):
        self.assertTrue(
            opportunity_source_eligibility("normal", self.results)["eligible"]
        )
        self.assertTrue(
            opportunity_source_eligibility("failure", self.results)["eligible"]
        )
        self.assertFalse(
            opportunity_source_eligibility("technical_diagnostic", self.results)[
                "eligible"
            ]
        )
        self.assertFalse(
            opportunity_source_eligibility("normal", self.results[1:])["eligible"]
        )
        self.assertFalse(
            opportunity_source_eligibility("failure", self.results[:1])["eligible"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
