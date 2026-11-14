"""Task inheritance uses the existing CLI release selector and permit binding."""

import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import test_initial_deployment as fixture
from rsi_vla.qualification_import import deployment_authorization
from rsi_vla.runtime import prepare_runtime
from rsi_vla.storage import atomic_json
from rsi_vla.task_completion import TaskCompletionBackend


class DeploymentTaskDispatchTests(unittest.TestCase):
    def setUp(self):
        fixture.InitialDeploymentTests.setUp(self)
        lock = patch("rsi_vla.runtime.DYNAMIC_LOCK", self.root / "fixture-only.lock")
        lock.start()
        self.addCleanup(lock.stop)
        self.slot = dict(
            run_id="inherit",
            bundle_id="deployment-initial",
            task_key="position_swap_salad",
            seed=1,
            split="deployment",
            cuda_device=2,
            output_ref=str(self.root / "runtime/inherit"),
            scratch_ref=str(self.root / "scratch"),
            permit_ref=str(self.operator / "inherit-permit.json"),
        )
        self.operation = dict(
            operation_id="inherit-runtime",
            kind="runtime",
            payload={"run_id": "inherit"},
        )
        self.backend = TaskCompletionBackend.__new__(TaskCompletionBackend)
        self.backend.config = dict(
            repository_root=str(fixture.ROOT),
            artifacts_root=str(self.artifacts),
            protocol_ref=str(self.protocol_ref),
            release_root=self.domains["release"],
            runtime_slots=[self.slot],
            operations=[self.operation],
        )
        # Admission has its own actual consumer tests. This fixture isolates the
        # routing while retaining the real selector, qualification and job binding.
        self.backend.validate_dynamic = Mock(return_value={})

    def valid_permit(self, active):
        plan = prepare_runtime(
            fixture.ROOT,
            self.artifacts,
            protocol_path=self.protocol_ref,
            run_id="inherit",
            bundle_id="deployment-initial",
            task_key="position_swap_salad",
            seed=1,
            split="deployment",
            output_dir=self.slot["output_ref"],
            cuda_device=2,
            deployment_binding=active,
            deployment_release_root=self.domains["release"],
            storage_write_roots=[self.slot["scratch_ref"]],
        )
        context = plan["context"]
        permit = {
            key: context[key]
            for key in ("run_id", "bundle_id", "protocol_id", "gpu_devices")
        }
        permit.update(
            authorized=True,
            task_key="position_swap_salad",
            seed=1,
            simulation_episodes=1,
            limits=context["limits"],
            storage_write_roots=context["storage_write_roots"],
            deployment_authorization_ref=str(self.auth_ref),
            release_binding=active,
        )
        atomic_json(self.slot["permit_ref"], permit)
        return permit

    def test_inheritance_cli_selects_actual_active_without_an_explicit_bundle(self):
        _, active = fixture.InitialDeploymentTests.publish(self)
        self.valid_permit(active)
        captured = {}

        def fixture_executor(plan):
            captured.update(plan)
            atomic_json(
                Path(self.slot["output_ref"]) / "runtime_context.json", plan["context"]
            )
            return {"status": "completed", "exit_code": 0}

        with (
            patch("rsi_vla.runtime.execute_runtime", side_effect=fixture_executor),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            returned = self.backend.execute(self.operation, {})
        self.backend.validate_dynamic.assert_called_once_with(
            "runtime", {"run_id": "inherit"}
        )
        argv = returned["selection_argv"]
        self.assertNotIn("--bundle", argv)
        self.assertNotIn("--deployment-authorization", argv)
        self.assertIn("--execute", argv)
        self.assertEqual(
            argv[argv.index("--deployment-release") + 1], self.domains["release"]
        )
        self.assertEqual(argv[argv.index("--cuda-device") + 1], "2")
        self.assertEqual(returned["selection"]["mode"], "deployment")
        self.assertEqual(returned["release_binding"], active)
        self.assertEqual(captured["context"]["bundle_id"], active["bundle_id"])
        self.assertEqual(returned["runtime_exit_code"], 0)

    def test_real_selector_refuses_unqualified_active_before_execution(self):
        self.manager.bootstrap("deployment-initial", scope="simulation")
        with patch("rsi_vla.runtime.execute_runtime") as execute:
            with self.assertRaisesRegex(ValueError, "initial qualification"):
                self.backend.execute(self.operation, {})
            execute.assert_not_called()
        self.assertFalse(Path(self.slot["output_ref"]).exists())

    def test_real_selector_rejects_permit_binding_different_from_active(self):
        _, active = fixture.InitialDeploymentTests.publish(self)
        permit = self.valid_permit(active)
        permit["release_binding"] = {**active, "revision": active["revision"] + 1}
        atomic_json(self.slot["permit_ref"], permit)
        with patch("rsi_vla.runtime.execute_runtime") as execute:
            with self.assertRaisesRegex(ValueError, "job permit lacks matching"):
                self.backend.execute(self.operation, {})
            execute.assert_not_called()
        self.assertFalse(Path(self.slot["output_ref"]).exists())

    def test_existing_deployment_authorization_routes_candidate_upgrade(self):
        ref = self.operator / "upgrade-fixture.json"
        atomic_json(ref, {"version": "candidate-upgrade-registration-v1"})
        with patch(
            "rsi_vla.upgrade_bridge.upgrade_authorization",
            return_value={"fixture": True},
        ) as route:
            self.assertEqual(
                deployment_authorization(
                    ref, bundle_id="candidate", protocol_id="target", role="runtime"
                ),
                {"fixture": True},
            )
        route.assert_called_once_with(
            ref, role="runtime", bundle_id="candidate", protocol_id="target"
        )


if __name__ == "__main__":
    unittest.main()
