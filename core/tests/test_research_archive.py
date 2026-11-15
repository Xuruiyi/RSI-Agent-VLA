"""Lossless archive consumers and ended one-admission/no-response closure, CPU only."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsi_vla.storage import ResearchStore, atomic_json
from rsi_vla.research_supervision import close_ended_research
from rsi_vla.midterm_host import MidtermContinuationBackend
from test_research_storage import EndedResearchTests


class ArchiveTests(unittest.TestCase):
    def test_raw_and_compressed_records_get_all_conflict_preserve_exact_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ResearchStore(
                root,
                llm_archive_format="gzip_json_v1",
                archive_record_kinds=["llm_requests", "bounded_requests"],
            )
            value = {
                "schema": "rsi-lossless-json-v1",
                "facts": [
                    {
                        "x": "完整事实" * 500,
                        "missing": None,
                        "false": False,
                        "array": [1, 2, 3],
                    }
                    for _ in range(4)
                ],
            }
            for kind in ("llm_requests", "bounded_requests"):
                ResearchStore(root).put(kind, "old", value)
                before = store.path(kind, "old").read_bytes()
                store.put(kind, "new", value)
                self.assertEqual(
                    json.loads(store.path(kind, "new").read_text())["schema"],
                    "rsi-lossless-record-v1",
                )
                self.assertEqual(ResearchStore(root).get(kind, "new"), value)
                self.assertEqual(store.all(kind), [value, value])
                self.assertEqual(store.path(kind, "old").read_bytes(), before)
                self.assertLess(
                    store.path(kind, "new").stat().st_size,
                    len(json.dumps(value).encode()),
                )
                store.put(kind, "new", value)
                with self.assertRaises(ValueError):
                    store.put(kind, "new", {"changed": True})
                damaged = json.loads(store.path(kind, "new").read_text())
                damaged["key"] = "foreign"
                atomic_json(store.path(kind, "new"), damaged)
                with self.assertRaisesRegex(ValueError, "binding"):
                    store.get(kind, "new")


class ForwardedClosureTests(EndedResearchTests):
    def test_one_admission_no_response_closes_unknown_once_and_never_creates_response(
        self,
    ):
        identity = json.loads((self.output / "runtime_context.json").read_text())
        identity = {
            k: identity[k] for k in ("run_id", "episode_id", "bundle_id", "protocol_id")
        }
        atomic_json(
            self.output / "request_gate_live.json",
            {"identity": identity, "requests_forwarded": 1},
        )
        event = {
            "identity": identity,
            "kind": "request_admitted",
            "request_number": 1,
            "model": "gpt-6.1-sol",
            "reasoning_effort": "high",
        }
        (self.output / "model_requests.jsonl").write_text(json.dumps(event) + "\n")
        phase = json.loads(self.phase.read_text())
        phase.update(
            limits={"model_requests": 10},
            used={"model_requests": 0},
            remaining={"model_requests": 10},
        )
        atomic_json(self.phase, phase)
        self.host.core_config["phase_ref"] = str(self.phase)
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
            result = close_ended_research(self.host, "r-model", self.report)
            self.assertEqual(
                close_ended_research(self.host, "r-model", self.report), result
            )
        self.assertEqual(
            result["value"]["technical_terminal"], "research_forwarded_response_unknown"
        )
        self.assertEqual(result["value"]["response_tokens"], "unknown")
        self.assertEqual(self.ledger.read()["used"]["model_requests"], 1)
        self.assertEqual(self.ledger.read()["used"]["research_requests"], 1)
        self.assertEqual(
            json.loads((self.store / "jobs/job.json").read_text())["used"][
                "model_requests"
            ],
            1,
        )
        self.assertEqual(
            json.loads(self.phase.read_text())["used"]["model_requests"], 1
        )
        self.assertFalse((self.output / "sdk-result.json").exists())
        self.assertFalse((self.store / "llm_executions/r.json").exists())
        backend = object.__new__(MidtermContinuationBackend)
        self.assertEqual(
            backend.observe({"receipts": {"r-model": result}}, {}),
            {"terminal": "research_forwarded_response_unknown"},
        )
        for p, body in self.original.items():
            self.assertEqual(p.read_bytes(), body)

    def test_complete_envelope_books_one_request_and_retains_prior_overhead(self):
        from rsi_vla.research_supervision import bill_research_overhead
        self.host.core_config['research_request_envelope'] = 'complete_v2'
        self.report['business_storage_delta_bytes'] = self.report['storage_bytes']
        atomic_json(self.job/'supervisor.json', self.report)
        self.original[self.job/'supervisor.json'] = (self.job/'supervisor.json').read_bytes()
        self.host.charge_phase = lambda operation_id, report: bill_research_overhead(self.host, self.op, {'wall_seconds':report['wall_seconds'], 'storage_bytes':report['business_storage_delta_bytes']})
        self.test_one_admission_no_response_closes_unknown_once_and_never_creates_response()
        state = json.loads((self.store/'jobs/job.json').read_text())
        self.assertEqual(state['used']['whole_job_wall_seconds'],self.report['wall_seconds'])
        self.assertEqual(state['used']['artifact_bytes'],self.report['storage_bytes'])

    def test_forwarded_count_without_corresponding_admission_is_not_closed(self):
        identity = json.loads((self.output / "request_gate_live.json").read_text())[
            "identity"
        ]
        atomic_json(
            self.output / "request_gate_live.json",
            {"identity": identity, "requests_forwarded": 1},
        )
        before = self.ledger.path.read_bytes()
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
            with self.assertRaisesRegex(PermissionError, "admission count"):
                close_ended_research(self.host, "r-model", self.report)
        self.assertEqual(self.ledger.path.read_bytes(), before)


class ActualArchiveConsumerTests(unittest.TestCase):
    from test_batch4 import RunnerTest as _Original

    result = _Original.result
    prepared = _Original.prepared
    test_actual_adapter_apply_and_reconcile_reads_compressed_archives = _Original.test_one_dispatch_reconcile_no_repeat_charge_and_no_applicable_no_dispatch

    def setUp(self):
        self._Original.setUp(self)
        self.store.llm_archive_format = "gzip_json_v1"
        self.store.archive_record_kinds = ("llm_requests", "bounded_requests")


if __name__ == "__main__":
    unittest.main(verbosity=2)
