"""CPU fixtures verify contracts and process supervision, never VLA training."""
from copy import deepcopy
import json
import io
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest

from rsi_vla.adapters.pi05_training import (ACTION_CONTRACT, build_recipe,
    deployment_handoff, inspect_checkpoint, training_bundle_proposal, validate_dataset_manifest,
    _fetch_bounded, select_demonstration_files, frozen_vlm_key,
    acquire_dataset_metadata, acquire_demonstration_files)
from rsi_vla.contracts import RunBundle
from rsi_vla.training import (TrainingBudget, dynamic_lock,
    artifact_size, prepare_ray_temp_path, require_training_permit, supervise_process)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.budget = TrainingBudget(3, 3, 1, 100000, 3)

    def dataset(self):
        root = self.root / "dataset"
        (root / "meta").mkdir(parents=True)
        (root / "data").mkdir()
        info = {"fps": 20, "total_episodes": 1, "features": {
            "state": {"shape": [8]}, "actions": {"shape": [7]},
            "image": {"dtype": "image"}, "wrist_image": {"dtype": "image"}}}
        (root / "meta/info.json").write_text(json.dumps(info))
        (root / "meta/episodes.jsonl").write_text('{"episode_index":0,"tasks":["fixture"]}\n')
        (root / "meta/tasks.jsonl").write_text('{"task_index":0,"task":"fixture"}\n')
        (root / "data/fixture.parquet").write_bytes(b"fixture_not_read_as_training_data")
        return {"source_kind": "public_demonstration", "provenance_ref": "fixture-source",
                "revision": "fixture-v1", "dataset_root": str(root),
                "action_contract": deepcopy(ACTION_CONTRACT),
                "episodes": [{"episode_index": 0, "task": "fixture", "source_ref": "fixture-0"}]}

    def recipe(self):
        return build_recipe(dataset_root=self.root / "dataset", model_path=self.root / "model",
                            norm_stats_path=self.root / "norm", output_root=self.root,
                            run_id="pilot", budget=self.budget)

    def test_recipe_matches_pi05_geometry_and_native_sft(self):
        cfg = self.recipe()
        model = cfg["actor"]["model"]
        self.assertTrue(model["pi05"])
        self.assertEqual(model["openpi"]["task"], "sft")
        self.assertEqual(model["openpi"]["config_name"], "pi05_libero")
        self.assertEqual(model["openpi"]["action_horizon"], 10)
        self.assertEqual(model["openpi"]["action_chunk"], 5)
        self.assertEqual(model["action_dim"], 7)
        self.assertFalse(model["openpi"]["discrete_state_input"])
        self.assertEqual(cfg["runner"]["max_steps"], 3)
        self.assertTrue(cfg["actor"]["fsdp_config"]["save_full_model_weights"])
        self.assertEqual(cfg["rsi_training"]["scope"], "action_expert_only")
        self.assertFalse(cfg["rsi_training"]["full_tuning"])
        self.assertTrue(cfg["actor"]["optim"]["use_fp32_master_params"])
        self.assertEqual(cfg["actor"]["optim"]["lr_warmup_steps"], 0)

    def test_freeze_scope_keeps_native_action_expert(self):
        for key in ("img.stem.weight", "llm.embedder.embedding.weight", "llm.layers.0.mlps.0.w_gating",
                    "llm.layers.1.attn.q_proj.0.weight", "llm.final_norms.0.scale"):
            self.assertTrue(frozen_vlm_key(key))
        for key in ("action_out_proj.weight", "llm.layers.0.attn.q_proj.1.weight",
                    "llm.layers.0.pre_attention_norms.1.ada_modulation.weight"):
            self.assertFalse(frozen_vlm_key(key))

    def test_download_payload_cap_rejects_before_read(self):
        class Response(io.BytesIO):
            headers = {"Content-Length": "20"}
        response = Response(b"x" * 20)
        with self.assertRaisesRegex(ValueError, "Content-Length"):
            _fetch_bounded("https://fixture", remaining=10, opener=lambda *a, **k: response)
        self.assertEqual(response.tell() if not response.closed else 0, 0)

    def test_download_exact_pinned_size_and_ledger(self):
        class Response(io.BytesIO):
            headers = {"Content-Length": "4"}
        seen = []
        self.assertEqual(_fetch_bounded("https://fixture", remaining=4, expected_size=4,
            opener=lambda *a, **k: Response(b"demo"), on_bytes=lambda u, n: seen.append(n)), b"demo")
        self.assertEqual(sum(seen), 4)

    def test_select_only_first_two_task_demonstrations(self):
        task = "pick up the salad dressing and place it in the basket"
        info = {"features": {"state": {"shape": [8]}, "actions": {"shape": [7]},
                              "image": {"dtype": "image"}, "wrist_image": {"dtype": "image"}},
                "chunks_size": 1000, "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"}
        episodes = [{"episode_index": i, "tasks": [task], "length": 10} for i in (3, 1, 2)]
        sizes = {f"data/chunk-000/episode_{i:06d}.parquet": 100 for i in (1, 2, 3)}
        plan = select_demonstration_files(info, json.dumps({"task_index": 7, "task": task}),
                                          "\n".join(map(json.dumps, episodes)), sizes)
        self.assertEqual([e["episode_index"] for e in plan["episodes"]], [1, 2])
        self.assertEqual(plan["download_bytes"], 200)
        sizes[plan["files"][0]] = 512 * 1024**2
        with self.assertRaisesRegex(ValueError, "512 MiB"):
            select_demonstration_files(info, json.dumps({"task_index": 7, "task": task}),
                                      "\n".join(map(json.dumps, episodes)), sizes)

    def test_metadata_then_two_files_require_separate_byte_capped_permits(self):
        task = "pick up the salad dressing and place it in the basket"
        sha = "c" * 40
        info = {"codebase_version": "v2.0", "features": {"state": {"shape": [8]},
                "actions": {"shape": [7]}, "image": {"dtype": "image"}, "wrist_image": {"dtype": "image"}},
                "chunks_size": 1000, "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"}
        content = {"meta/info.json": json.dumps(info).encode(),
                   "meta/tasks.jsonl": json.dumps({"task_index": 0, "task": task}).encode(),
                   "meta/episodes.jsonl": "\n".join(json.dumps({"episode_index": i, "tasks": [task], "length": 2}) for i in range(2)).encode(),
                   "meta/stats.json": b"{}",
                   "data/chunk-000/episode_000000.parquet": b"a" * 100,
                   "data/chunk-000/episode_000001.parquet": b"b" * 100}
        api = json.dumps({"sha": sha, "siblings": [{"rfilename": k, "size": len(v)} for k, v in content.items()]}).encode()
        urls = []
        class Response(io.BytesIO):
            def __init__(self, data):
                super().__init__(data)
                self.headers = {"Content-Length": str(len(data))}
        def opener(request, **kwargs):
            urls.append(request.full_url)
            return Response(api if "/api/" in request.full_url else content[request.full_url.split(f"/{sha}/")[1]])
        metadata = self.root / "metadata"
        record = acquire_dataset_metadata(metadata, {"kind": "pi05-dataset-metadata", "authorized": True,
                                                      "max_bytes": 8 * 1024**2}, opener=opener)
        self.assertEqual(record["revision"], sha)
        self.assertFalse(any("/data/" in url for url in urls))
        self.assertLessEqual(record["downloaded_metadata_bytes"], 8 * 1024**2)
        with self.assertRaises(PermissionError):
            acquire_demonstration_files(metadata, self.root / "wrong-permit", {"kind": "pi05-dataset-metadata", "authorized": True}, opener=opener)
        permit = {"kind": "pi05-dataset-demonstrations", "authorized": True, "revision": sha,
                  "max_bytes": 200, "evaluation_isolation_ref": "fixture-only"}
        data = acquire_demonstration_files(metadata, self.root / "demos", permit, opener=opener)
        self.assertEqual(data["downloaded_demonstration_bytes"], 200)
        self.assertEqual(len([url for url in urls if "/data/" in url]), 2)

    def test_pilot_refuses_unmeasured_scale(self):
        with self.assertRaises(ValueError):
            build_recipe(dataset_root="d", model_path="m", norm_stats_path="n",
                         output_root=self.root, run_id="p", budget=TrainingBudget(5, 10, 2, 100, 4))

    def test_manifest_requires_all_consumed_episode_provenance(self):
        manifest = self.dataset()
        self.assertEqual(validate_dataset_manifest(manifest)["episodes"], 1)
        manifest["episodes"][0]["episode_index"] = 2
        with self.assertRaisesRegex(ValueError, "all episodes"):
            validate_dataset_manifest(manifest)

    def test_success_states_are_not_demonstrations(self):
        manifest = self.dataset()
        manifest["source_kind"] = "successful_states_transcript"
        with self.assertRaisesRegex(ValueError, "demonstration"):
            validate_dataset_manifest(manifest)

    def test_seven_dimensions_do_not_make_ur5_libero(self):
        manifest = self.dataset()
        manifest["action_contract"]["backend"] = "ur5e"
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            validate_dataset_manifest(manifest)

    def test_manifest_task_cannot_override_native_metadata(self):
        manifest = self.dataset()
        manifest["episodes"][0]["task"] = "different instruction"
        with self.assertRaisesRegex(ValueError, "task differs"):
            validate_dataset_manifest(manifest)

    def test_norm_is_required_for_checkpoint_handoff(self):
        cp = self.root / "checkpoint/actor/model_state_dict"
        cp.mkdir(parents=True)
        (cp / "full_weights.pt").write_bytes(b"fixture-layout-only")
        norm = self.root / "norm_stats.json"
        norm.write_text(json.dumps({"norm_stats": {key: {field: [0] * 32
            for field in ("mean", "std", "q01", "q99")} for key in ("state", "actions")}}))
        handoff = deployment_handoff(self.root / "checkpoint", norm, python_executable=sys.executable)
        self.assertEqual(handoff["actual_model_load"], "not_run")
        self.assertIn("--norm-stats-path", handoff["load_command"])
        self.assertIn("actor/model_state_dict/full_weights.pt", handoff["selected_weights"])
        norm.write_text('{"norm_stats":{"state":{"mean":[]}}}')
        with self.assertRaises((ValueError, KeyError)):
            inspect_checkpoint(self.root / "checkpoint", norm)

    def test_training_permit_is_separate_from_simulation(self):
        permit = {"kind": "simulation", "authorized": True}
        with self.assertRaises(PermissionError):
            require_training_permit(permit, run_id="p", budget=self.budget,
                                    dataset_root="d", model_path="m")

    def test_training_bundle_preserves_non_model_state(self):
        base = RunBundle("old", "theta", "processor", "phi", "rho", "psi", "memory",
                         "protocol", "backend", "code")
        proposed = training_bundle_proposal(base, bundle_id="new", theta_ref="trained-theta",
                                            processor_ref="fixed-norm", diff_ref="actual-diff")
        for key in ("phi_ref", "rho_ref", "psi_ref", "memory_ref", "protocol_id", "backend_ref", "code_ref"):
            self.assertEqual(getattr(proposed, key), getattr(base, key))
        self.assertEqual(proposed.parent_bundle_id, "old")
        self.assertEqual(proposed.theta_ref, "trained-theta")

    def test_permit_binds_source_identity_and_budget(self):
        permit = {"kind": "pi05-training-pilot", "authorized": True, "run_id": "p",
                  "dataset_root": "d", "model_path": "m", "budget": self.budget.__dict__}
        require_training_permit(permit, run_id="p", budget=self.budget,
                                dataset_root="d", model_path="m")
        with self.assertRaises(PermissionError):
            require_training_permit(permit, run_id="p", budget=self.budget,
                                    dataset_root="other", model_path="m")
        with self.assertRaises(PermissionError):
            require_training_permit(permit, run_id="p", budget=TrainingBudget(4, 3, 1, 100000, 3),
                                    dataset_root="d", model_path="m")

    def test_shared_dynamic_lock_is_nonblocking(self):
        lock = self.root / "dynamic.lock"
        with dynamic_lock(lock):
            with self.assertRaises(RuntimeError):
                with dynamic_lock(lock):
                    self.fail("second holder entered")
        with dynamic_lock(lock):
            pass

    def test_cpu_process_records_failure(self):
        result = supervise_process([sys.executable, "-c", "raise SystemExit(7)"],
                                    budget=self.budget, artifact_root=self.root / "failure")
        self.assertEqual(result["returncode"], 7)
        self.assertIsNone(result["stop_reason"])

    def test_device_reservation_envelope_stops_cpu_fixture(self):
        budget = TrainingBudget(3, 0.08, 1, 100000, 1)
        result = supervise_process([sys.executable, "-c", "import time; time.sleep(5)"],
                                    budget=budget, artifact_root=self.root / "timeout", poll_seconds=0.01)
        self.assertEqual(result["stop_reason"], "wall_or_device_reservation_limit")
        self.assertLess(result["wall_seconds"], 2)
        self.assertEqual(result["gpu_device_seconds_envelope"], result["wall_seconds"])

    def test_invalid_budget_rejected(self):
        for value in (0, -1, float("nan"), True):
            with self.assertRaises(ValueError):
                TrainingBudget(value, 3, 1, 100, 1)

    def test_short_ray_alias_survives_native_path_helpers_and_counts_artifacts(self):
        from types import SimpleNamespace, MethodType
        from unittest.mock import patch
        import ray
        from ray._common.utils import get_default_ray_temp_dir
        from ray._private.node import Node
        from ray._private.utils import validate_socket_filepath
        artifact = Path(os.environ.get("RSI_RAY_PATH_EVIDENCE", self.root / ("long-" + "x" * 100)))
        paths = prepare_ray_temp_path(artifact)
        session = "session_2026-11-10_23-59-59_999999_4194304"
        old_socket = str(artifact.resolve() / "ray/ray" / session / "sockets/plasma_store")
        with self.assertRaises(OSError):
            validate_socket_filepath(old_socket)
        node = SimpleNamespace(_ray_params=SimpleNamespace(temp_dir=None, runtime_env_dir_name="runtime_resources"),
                               _session_name=session, head=True)
        node._make_inc_temp = MethodType(Node._make_inc_temp, node)
        self.assertFalse(ray.is_initialized())
        with patch.dict(os.environ, {"RAY_TMPDIR": paths["ray_tmpdir"]}):
            self.assertEqual(get_default_ray_temp_dir(), paths["ray_temp_alias"])
            Node._init_temp(node, None)
            socket_path = Node._prepare_socket_file(node, None, "plasma_store")
        self.assertFalse(ray.is_initialized())
        self.assertTrue(socket_path.startswith(paths["ray_temp_alias"] + "/"))
        self.assertLessEqual(len(socket_path.encode()), 107)
        self.assertGreater(len(str(Path(socket_path).resolve()).encode()), 107)
        before = artifact_size(artifact)
        probe = Path(node._logs_dir) / "path-probe.txt"
        probe.write_bytes(b"probe")
        self.assertEqual(artifact_size(artifact) - before, 5)
        self.assertTrue(probe.resolve().is_relative_to(artifact.resolve()))
        (artifact / "path-check.json").write_text(json.dumps({
            **paths, "old_socket_path": old_socket, "old_socket_bytes": len(old_socket.encode()),
            "native_socket_path": socket_path, "native_socket_bytes": len(socket_path.encode()),
            "actual_log_path": str(probe.resolve()), "counted_probe_bytes": 5,
            "native_helper_kept_alias": True, "ray_services_started": False}, indent=2) + "\n")

    def test_detached_marked_child_is_reaped_within_cleanup_budget(self):
        root = self.root / "detached"
        root.mkdir()
        child = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"
        launcher = ("import subprocess,sys,time; from pathlib import Path; "
                    "p=subprocess.Popen([sys.executable,'-c',sys.argv[2]],start_new_session=True,"
                    "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
                    "Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(30)")
        result = supervise_process([sys.executable, "-c", launcher, str(root / "detached.pid"), child],
            budget=TrainingBudget(1.5, 1.5, 1, 100000, 1), artifact_root=root, poll_seconds=0.01)
        pid = int((root / "detached.pid").read_text())
        self.assertEqual(result["stop_reason"], "wall_or_device_reservation_limit")
        self.assertTrue(result["cleanup_confirmed"])
        self.assertEqual(result["remaining_owned_pids"], [])
        self.assertIn(pid, [entry["pid"] for entry in result["owned_pid_identities"]])
        self.assertLess(result["wall_seconds"], 1.5)
        self.assertTrue(result["resource_limits_satisfied"])

    def test_unmarked_unrelated_child_is_not_killed(self):
        unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"],
                                      start_new_session=True, env=dict(os.environ))
        try:
            result = supervise_process([sys.executable, "-c", "pass"],
                budget=TrainingBudget(1, 1, 1, 100000, 1), artifact_root=self.root / "isolated")
            self.assertTrue(result["cleanup_confirmed"])
            self.assertIsNone(unrelated.poll())
            self.assertNotIn(unrelated.pid, [entry["pid"] for entry in result["owned_pid_identities"]])
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=2)

    def test_native_cpu_fixture_subset_and_loader(self):
        import importlib.util
        if importlib.util.find_spec("lerobot") is None:
            self.skipTest("optional installed native LeRobot stack unavailable")
        import os
        import shutil
        import numpy as np
        os.environ["HF_DATASETS_CACHE"] = str(self.root / "datasets-cache")
        os.environ["JAX_PLATFORMS"] = "cpu"
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from rsi_vla.adapters.pi05_training import materialize_demonstration_subset, read_native_batch
        model = Path("/home/xzh/RSI-Agent-VLA/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT")
        norm = model / "physical-intelligence/libero/norm_stats.json"
        if not norm.is_file():
            self.skipTest("local native processor fixture assets unavailable")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        source = self.root / "source-fixture"
        features = {"state": {"dtype": "float32", "shape": (8,), "names": None},
                    "actions": {"dtype": "float32", "shape": (7,), "names": None},
                    "image": {"dtype": "image", "shape": (32, 32, 3), "names": ["height", "width", "channels"]},
                    "wrist_image": {"dtype": "image", "shape": (32, 32, 3), "names": ["height", "width", "channels"]}}
        task = "pick up the salad dressing and place it in the basket"
        writer = LeRobotDataset.create("fixture/libero", fps=20, root=source,
                                       features=features, use_videos=False, robot_type="fixture")
        for episode in range(2):
            for frame in range(2):
                writer.add_frame({"state": np.zeros(8, dtype=np.float32),
                                  "actions": np.full(7, frame * 0.1, dtype=np.float32),
                                  "image": np.full((32, 32, 3), episode * 30, dtype=np.uint8),
                                  "wrist_image": np.zeros((32, 32, 3), dtype=np.uint8)}, task=task)
            writer.save_episode()
        info = json.loads((source / "meta/info.json").read_text())
        sizes = {str(p.relative_to(source)): p.stat().st_size for p in (source / "data").rglob("*.parquet")}
        plan = select_demonstration_files(info, (source / "meta/tasks.jsonl").read_text(),
                                          (source / "meta/episodes.jsonl").read_text(), sizes)
        record = {"repo_id": "fixture/libero", "revision": "a" * 40, "selection": plan,
                  "file_sizes": sizes, "evaluation_isolation_ref": "CPU fixture, not project demonstrations"}
        (source / "source.json").write_text(json.dumps(record))
        downloaded = self.root / "downloaded-fixture"
        shutil.copytree(source / "data", downloaded / "data")
        (downloaded / "source.json").write_text(json.dumps(record))
        manifest = materialize_demonstration_subset(source, downloaded, self.root / "subset-fixture")
        self.assertEqual([e["source_episode_index"] for e in manifest["episodes"]], [0, 1])
        recipe = build_recipe(dataset_root=manifest["dataset_root"], model_path=model,
                              norm_stats_path=norm, output_root=self.root, run_id="fixture", budget=self.budget)
        batch = read_native_batch(recipe, manifest)
        self.assertEqual(batch["action_shape"], [1, 10, 32])
        self.assertEqual(batch["state_shape"], [1, 32])
        evidence = os.environ.get("RSI_TRAINING_FIXTURE_ARTIFACTS")
        if evidence:
            target = Path(evidence)
            shutil.copytree(self.root, target)
            (target / "fixture-check.json").write_text(json.dumps({
                "fixture_only": True, "batch": batch,
                "source_episode_indices": [0, 1], "source_frames": 4,
                "real_project_demonstrations": False}, indent=2) + "\n")


if __name__ == "__main__":
    unittest.main()
