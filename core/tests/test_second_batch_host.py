"""Receipt directories and public-role consumers use the same restarted Host."""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import test_f04_role_boundaries as role_fixture
import test_host_receipt_inputs as host_fixture
from rsi_vla.host_inputs import materialize_inputs
from rsi_vla.protocol10_roles import role_configuration, run_role
from rsi_vla.storage import atomic_json


def consume_directory(config):
    """Read real late-bound bytes inside the actual namespace before backend use."""
    role_fixture.projection_probe(config)
    op = next(row for row in config["operations"] if row["operation_id"] == "consume")
    directory = Path(op["payload"]["model_path"])
    assert json.loads((directory / "model.json").read_text()) == {"synthetic": True}
    assert not directory.joinpath("escape").exists()
    atomic_json(
        Path(config["research_store_ref"]) / "consumed-directory.json",
        {"model_path": str(directory), "source": "real-role-read"},
    )
    return {}


class SecondBatchHostTests(unittest.TestCase):
    setUp = host_fixture.HostReceiptInputsTests.setUp
    host = host_fixture.HostReceiptInputsTests.host
    end_job = host_fixture.HostReceiptInputsTests.end_job
    produce = host_fixture.HostReceiptInputsTests.produce

    def directory_producer(self):
        producer = self.config["operations"]["produce"]
        directory = Path(producer["result_ref"]).parent / "checkpoint"
        producer["command"] = [
            sys.executable,
            "-c",
            (
                "import json,os; from pathlib import Path; "
                "p=json.loads(Path(os.environ['RSI_CPU_PERMIT_REF']).read_text()); "
                f"d=Path({str(directory)!r}); d.mkdir(); (d/'model.json').write_text(json.dumps({{'synthetic':True}})); "
                "Path(p['result_ref']).write_text(json.dumps(dict(schema='rsi-cpu-operation-result-v1',"
                "scope_id=p['scope_id'],operation_id=p['operation_id'],version=p['version'],status='completed',"
                "model_path=str(d))))"
            ),
        ]
        producer["exports"]["theta"]["field"] = "model_path"
        atomic_json(self.host_ref, self.config)
        return directory

    def test_directory_and_rejections_preserve_frozen_inputs(self):
        directory = self.directory_producer()
        host, completion = self.produce()
        path, value = materialize_inputs(host, self.operations[1])
        self.assertEqual(value["references"]["theta"], str(directory))
        original = path.read_bytes()
        self.assertEqual(materialize_inputs(self.host(), self.operations[1])[0], path)
        self.assertEqual(path.read_bytes(), original)
        host.config["operations"]["consume"]["late_refs"]["theta"]["version"] = "wrong"
        with self.assertRaises(PermissionError):
            materialize_inputs(host, self.operations[1])
        host = self.host()
        report_ref = completion.parent / "supervisor.json"
        report = json.loads(report_ref.read_text())
        atomic_json(report_ref, {**report, "cleanup_confirmed": False})
        with self.assertRaises(PermissionError):
            materialize_inputs(host, self.operations[1])
        atomic_json(report_ref, report)
        native = Path(self.config["continuation_root"]) / "receipts/produce.json"
        receipt = json.loads(native.read_text())
        atomic_json(
            native,
            {
                **receipt,
                "value": {**receipt["value"], "model_path": str(self.operator)},
            },
        )
        with self.assertRaises(PermissionError):
            materialize_inputs(host, self.operations[1])
        self.assertEqual(path.read_bytes(), original)

    @patch.dict(
        "os.environ",
        {"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")
         + ":" + str(Path(__file__).resolve().parent)},
    )
    def test_same_host_public_configuration_real_isolated_successor_and_restart(self):
        self.hidden = self.root / "hidden"
        self.hidden.mkdir()
        secret = self.hidden / "labels.json"
        secret.write_text("F04-HIDDEN-LABEL")
        fixture = SimpleNamespace(
            root=self.root,
            hidden=self.hidden,
            secret=secret,
            config={
                "isolation_task_root": str(self.root),
                "offline_fixture": True,
                "role_isolation": {"hidden_paths": [str(self.hidden)]},
            },
        )
        projected, consumer = (
            role_fixture.F04RoleBoundaryTests.projection_configuration(fixture)
        )
        consumer["operation_id"] = "consume"
        self.operations[1].update(consumer)
        self.scope["scope_id"] = "f04-scope"
        self.config["scope_id"] = "f04-scope"
        atomic_json(self.scope_ref, self.scope)
        self.core.update(projected)
        self.core["scope_authorization_ref"] = str(self.scope_ref)
        self.core["operations"] = self.operations
        self.core["offline_boundary_factory"] = "test_second_batch_host:consume_directory"
        atomic_json(self.core_ref, self.core)
        public = {
            key: value
            for key, value in self.core.items()
            if key
            not in {
                "private_configuration_note",
                "public_protocol_ref",
                "public_frozen_protocol_ref",
                "public_configuration_ref",
                "private_configuration_ref",
            }
        }
        public.update(
            protocol_ref=projected["public_protocol_ref"],
            frozen_protocol_ref=projected["public_frozen_protocol_ref"],
        )
        atomic_json(projected["public_configuration_ref"], public)
        atomic_json(projected["private_configuration_ref"], self.core)
        self.config["operations"]["consume"].pop("command")
        self.config["operations"]["consume"]["late_refs"]["theta"]["target"] = [
            "payload",
            "model_path",
        ]
        directory = self.directory_producer()
        host, _ = self.produce()
        role_configuration(host.core_config, "research")
        from rsi_vla.midterm_host import read

        self.assertEqual(host.advance()["status"], "running")
        host, _completion = self.end_job("consume")
        native = Path(self.config["continuation_root"]) / "receipts/consume.json"
        original_native = native.read_bytes()
        value = read(native)["value"]
        self.assertEqual(value, {"registered": True})
        restarted = self.host()
        self.assertEqual(restarted.advance()["status"], "completed")
        self.assertEqual(read(native)["value"], value)
        self.assertEqual(native.read_bytes(), original_native)
        proof = read(Path(projected["research_store_ref"]) / "consumed-directory.json")
        self.assertEqual(proof["model_path"], str(directory))
        receipts = list(
            (self.root / "state/research/role_receipts").glob(
                "consume-execute-revision-*.json"
            )
        )
        self.assertEqual(len(receipts), 1)
        frozen = read(host.root / "jobs/consume/inputs.json")
        self.assertEqual(frozen["operation"], self.operations[1])
        self.assertNotIn("model_path", public["operations"][1]["payload"])
        original = frozen.copy()
        with self.assertRaisesRegex(PermissionError, "controls"):
            run_role(
                host.core_config,
                {"action": "execute", "operation": self.operations[1], "inputs": {}},
                resolved_operation={**self.operations[1], "dynamic": True},
            )
        self.assertEqual(read(host.root / "jobs/consume/inputs.json"), original)
        public["operations"][1]["payload"]["model_path"] = str(directory)
        atomic_json(projected["public_configuration_ref"], public)
        with self.assertRaisesRegex(ValueError, "role control"):
            role_configuration(host.core_config, "research")


if __name__ == "__main__":
    unittest.main()
