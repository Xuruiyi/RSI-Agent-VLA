# Copyright 2026 RSI-Agent-VLA Authors.
"""Real Autoloop events/timers must finish late-bound CPU jobs once across restart."""

import json
import os
import shutil
import signal
import socket
import sqlite3
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import test_host_receipt_inputs as receipt_fixture
from rsi_vla.midterm_host import build_autoloop_rules, process_matches, read
from rsi_vla.storage import atomic_json

ROOT = Path(__file__).resolve().parents[2]


class HostAutoloopReceiptTests(unittest.TestCase):
    setUp = receipt_fixture.HostReceiptInputsTests.setUp

    def test_real_hub_timer_restart_finishes_two_cpu_receipts_without_manual_advance(
        self,
    ):
        node = os.environ.get("RSI_TEST_NODE") or shutil.which("node")
        if not node:
            self.skipTest("Node 24 is required; set RSI_TEST_NODE")
        version = subprocess.run(
            [node, "--version"], capture_output=True, text=True, timeout=3, check=True
        )
        if int(version.stdout.strip().removeprefix("v").split(".", 1)[0]) < 24:
            self.skipTest("Node 24 is required; set RSI_TEST_NODE")
        node = str(Path(node).resolve())
        deadline = time.monotonic() + 45
        evidence = Path(tempfile.mkdtemp(prefix="host-autoloop-receipts-"))
        self.hub_home = self.root / "autoloop/home"
        self.hub_home.mkdir(parents=True)
        processes = []
        streams = []
        summary = {
            "status": "failed",
            "evidence_root": str(evidence),
            "manual_advances": 0,
        }
        try:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            url = f"http://127.0.0.1:{port}"
            hub_ref = self.operator / "hub.json"
            self.config.update(
                configuration_ref=str(self.host_ref),
                hub_configuration_ref=str(hub_ref),
                transport_root=str(self.root / "autoloop/transport"),
                poll_seconds=0.5,
            )
            # Keep the original operation permit and extend only its actual CPU
            # duration so a real timer fires before its native return.
            role_producer = getattr(self, "isolated_rule_producer", False)
            if not role_producer:
                command = self.config["operations"]["produce"]["command"]
                command[-1] = command[-1].replace("time.sleep(.15)", "time.sleep(4)")
            atomic_json(self.host_ref, self.config)
            rules = build_autoloop_rules(self.config)
            self.assertEqual(rules["channels"], {})
            atomic_json(self.hub_home / "rules.json", rules)
            environment = {
                **os.environ,
                "AGENT_AUTOLOOP_HOME": str(self.hub_home),
                "PYTHONDONTWRITEBYTECODE": "1",
                "NO_PROXY": "127.0.0.1,localhost",
                "no_proxy": "127.0.0.1,localhost",
            }
            runtime = (ROOT / "agent-autoloop/host/runtime.ts").as_uri()
            program = (
                f"import {{addKey,db}} from {json.dumps(runtime)};"
                f"const name={json.dumps(self.scope['scope_id'])};"
                "console.log(JSON.stringify({worker:{name,key:addKey('worker',name)},"
                "source:{name,key:addKey('source',name)},owner:addKey('owner','owner')}));db.close();"
            )
            initialized = subprocess.run(
                [
                    node,
                    "--disable-warning=ExperimentalWarning",
                    "--input-type=module",
                    "-e",
                    program,
                ],
                env=environment,
                capture_output=True,
                text=True,
                timeout=8,
                check=True,
            )
            credentials = json.loads(initialized.stdout)
            hub_config = {
                **credentials,
                "url": url,
                "hub": {"listen": f"127.0.0.1:{port}", "publicUrl": url},
                "channels": {},
            }
            atomic_json(self.hub_home / "config.json", hub_config)
            (self.hub_home / "config.json").chmod(0o600)
            atomic_json(
                hub_ref,
                {
                    "url": url,
                    "source_key": credentials["source"]["key"],
                    "owner": credentials["owner"],
                },
            )
            hub_ref.chmod(0o600)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

            def request(path, event=None):
                key = (
                    credentials["source"]["key"]
                    if event is not None
                    else credentials["owner"]
                )
                query = urllib.request.Request(
                    url + path,
                    data=json.dumps(event).encode() if event is not None else None,
                    headers={
                        "Authorization": "Bearer " + key,
                        "Content-Type": "application/json",
                    },
                )
                with opener.open(query, timeout=2) as response:
                    return json.load(response)

            def start(argv, name):
                stream = (evidence / name).open("wb")
                streams.append(stream)
                process = subprocess.Popen(
                    argv,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                processes.append(process)
                return process

            host_argv = [
                self.config["python"],
                self.config["script_ref"],
                "serve",
                "--configuration",
                str(self.host_ref),
            ]
            host = start(host_argv, "host-first.log")
            hub = start(
                [
                    node,
                    "--disable-warning=ExperimentalWarning",
                    str(ROOT / "agent-autoloop/bin/agent-autoloop.ts"),
                    "up",
                ],
                "autoloop.log",
            )
            view = None
            while time.monotonic() < deadline:
                self.assertIsNone(
                    host.poll(), (evidence / "host-first.log").read_text()
                )
                self.assertIsNone(hub.poll(), (evidence / "autoloop.log").read_text())
                try:
                    view = request("/v1/state")
                    break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.05)
            self.assertIsNotNone(view, "real hub did not start")
            source = {
                "id": "one-total-start",
                "type": "midterm.start",
                "subject": self.scope["scope_id"],
                "summary": "temporary CPU receipt-chain acceptance",
            }
            initial = request("/v1/events", source)
            duplicate = request("/v1/events", source)
            self.assertFalse(initial["duplicate"])
            self.assertTrue(duplicate["duplicate"])
            host_root = self.root / "operator/midterm-host"
            supervisor = host_root / "jobs/produce/process.json"
            responses = self.root / "autoloop/transport/responses"
            while time.monotonic() < deadline:
                if (
                    supervisor.exists()
                    and (self.counter.exists() or role_producer)
                    and list(responses.glob("*.json"))
                ):
                    break
                time.sleep(0.05)
            self.assertTrue(supervisor.exists(), "source wake did not submit producer")
            original_supervisor = read(supervisor)
            self.assertTrue(
                process_matches(supervisor), "producer ended before restart injection"
            )
            host.terminate()
            self.assertEqual(host.wait(timeout=4), 0)
            self.assertTrue(
                process_matches(supervisor),
                "Host stop killed its independent supervisor",
            )
            host = start(host_argv, "host-restarted.log")
            duplicate_after_restart = request("/v1/events", source)
            self.assertTrue(duplicate_after_restart["duplicate"])
            progress_ref = (
                Path(self.config["continuation_root"]) / "coordinator/progress.json"
            )
            state = {}
            timer_runs = []
            while time.monotonic() < deadline:
                self.assertIsNone(
                    host.poll(), (evidence / "host-restarted.log").read_text()
                )
                self.assertIsNone(hub.poll(), (evidence / "autoloop.log").read_text())
                state = read(progress_ref) if progress_ref.exists() else {}
                view = request("/v1/state")
                timer_runs = [
                    run
                    for run in view["runs"]
                    if run["state"] in {"done", "nothing"}
                    and any(
                        event["type"] == "timer.due" for event in run.get("events", [])
                    )
                ]
                published = all(
                    (host_root / "jobs" / name / "event-published.json").exists()
                    for name in ("produce", "consume")
                )
                if state.get("status") == "completed" and published and timer_runs:
                    break
                errors = [
                    read(path)
                    for path in responses.glob("*.json")
                    if read(path).get("status") == "blocked"
                ]
                self.assertFalse(errors, f"Host blocked a real worker wake: {errors}")
                time.sleep(0.1)
            self.assertEqual(
                state.get("status"),
                "completed",
                f"real Autoloop did not finish: {state}",
            )
            self.assertTrue(
                timer_runs, "no real completed timer-triggered worker round"
            )
            self.assertEqual(self.counter.read_text(),
                             "consume\n" if role_producer else "produce\nconsume\n")
            self.assertEqual(read(supervisor), original_supervisor)
            completion = {}
            for name in ("produce", "consume"):
                completion[name] = read(host_root / "jobs" / name / "completion.json")
                self.assertEqual(completion[name]["status"], "completed")
                self.assertTrue(
                    read(host_root / "jobs" / name / "supervisor.json")[
                        "cleanup_confirmed"
                    ]
                )
                self.assertTrue(
                    (host_root / "jobs" / name / "event-published.json").exists()
                )
            with sqlite3.connect(
                f"file:{self.hub_home / 'hub.db'}?mode=ro", uri=True
            ) as database:
                source_count = database.execute(
                    "SELECT COUNT(*) FROM events WHERE key = ?", (source["id"],)
                ).fetchone()[0]
                completion_events = database.execute(
                    "SELECT COUNT(*) FROM events WHERE type = 'midterm.completed'"
                ).fetchone()[0]
                timer_events = database.execute(
                    "SELECT COUNT(*) FROM events WHERE type = 'timer.due'"
                ).fetchone()[0]
                failed_runs = database.execute(
                    "SELECT COUNT(*) FROM runs WHERE attempts > 0 OR error IS NOT NULL"
                ).fetchone()[0]
            self.assertEqual(source_count, 1)
            self.assertEqual(completion_events, 2)
            self.assertGreater(timer_events, 0)
            self.assertEqual(failed_runs, 0)
            ledger = read(self.root / "accounting.json")
            self.assertEqual(
                {name: row["status"] for name, row in ledger["reservations"].items()},
                {"produce": "settled", "consume": "settled"},
            )
            self.assertEqual(ledger["used"]["model_requests"], 0)
            self.assertEqual(ledger["used"]["gpu_seconds"], 0)
            summary.update(
                status="passed",
                node_version=version.stdout.strip(),
                host_restarts=1,
                source_events=source_count,
                duplicate_deliveries=2,
                completion_events=completion_events,
                timer_events=timer_events,
                completed_timer_rounds=len(timer_runs),
                execution_order=self.counter.read_text().splitlines(),
                costs=ledger["used"],
            )
        except BaseException as error:
            summary.update(error_type=type(error).__name__, error=str(error))
            raise
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            # Autoloop deliberately detaches workers. Reap only actors carrying
            # this test's home marker or exact temporary-domain path.
            remaining = self._owned_processes()
            for pid in remaining:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            until = time.monotonic() + 3
            while self._owned_processes() and time.monotonic() < until:
                time.sleep(0.05)
            for pid in self._owned_processes():
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            summary["remaining_owned_process_ids"] = self._owned_processes()
            for stream in streams:
                stream.close()
            atomic_json(evidence / "summary.json", summary)
            print(f"Host/Autoloop evidence: {evidence}", flush=True)
            self.assertEqual(
                summary["remaining_owned_process_ids"],
                [],
                "owned processes survived cleanup",
            )

    def _owned_processes(self):
        found = []
        marker = f"AGENT_AUTOLOOP_HOME={self.hub_home}".encode()
        for directory in Path("/proc").iterdir():
            if not directory.name.isdigit() or int(directory.name) == os.getpid():
                continue
            try:
                stat = (directory / "stat").read_text().rsplit(")", 1)[1].split()
                if stat[0] == "Z":
                    continue
                command = (directory / "cmdline").read_bytes()
                environment = (directory / "environ").read_bytes().split(b"\0")
                if str(self.root).encode() in command or marker in environment:
                    found.append(int(directory.name))
            except (OSError, ProcessLookupError):
                continue
        return found


if __name__ == "__main__":
    unittest.main(verbosity=2)
