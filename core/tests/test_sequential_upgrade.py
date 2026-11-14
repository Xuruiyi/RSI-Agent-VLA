"""A published candidate can be the independently qualified current next round."""

import contextlib
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import test_upgrade_bridge as fixture
from rsi_vla.contracts import Candidate, RunBundle
from rsi_vla.evaluation import Gate
from rsi_vla.runtime import execution_binding
from rsi_vla.storage import atomic_json
from rsi_vla.upgrade_bridge import (
    BEHAVIOR_FIELDS,
    import_candidate_evidence,
    isolation_paths,
    prepare_candidate_deployment,
    qualify_upgrade,
)


def read(path):
    return json.loads(Path(path).read_text())


def binding(source, target, path):
    def value(ref):
        result = read(ref)
        for part in path:
            result = result[part]
        return result

    return dict(
        source_ref=str(source),
        target_ref=str(target),
        source_path=path,
        target_path=path,
        source_value=value(source),
        target_value=value(target),
        reason="Exact evaluated profile value retained.",
    )


class NextRound:
    def __init__(self, case, *, formal=True, full_profile=False):
        f = self.f = fixture.UpgradeBridgeTests()
        f.setUp()
        case.addCleanup(f.doCleanups)
        if full_profile:
            profile = read(f.profile_ref)
            profile.update(
                model=dict(checkpoint="fixture", normalization="norm"),
                task_inputs={},
                planner_observation={},
                execution_version="fixture",
                vla_segment={},
                observations={},
                control=dict(action_dim=7, model_action_dim=7),
                collection={},
            )
            atomic_json(f.profile_ref, profile)
            for path, identity in [
                (f.base_raw, "comparison-base"),
                (f.raw, "comparison-trial"),
            ]:
                context = read(path.parent / "runtime_context.json")
                context["execution_binding"] = execution_binding(
                    f.research, f.research.get("bundles", identity), profile
                )
                atomic_json(path.parent / "runtime_context.json", context)
        self.first = f.publish()
        self.first_qualification = deepcopy(
            f.manager.records.get("qualifications", "new-qualified")
        )
        self.first_publication = deepcopy(
            f.manager.records.get("upgrade_publications", "upgrade")
        )
        self.protocol = deepcopy(f.source_protocol)
        self.protocol.update(
            protocol_id="comparison-next",
            anchor_bundle_id="next-base",
            dynamic_authorized=True,
            active_bundle_id=self.first["bundle_id"],
        )
        self.protocol["task_sets"]["release"].update(
            seeds=[0, 1], benchmark_id="benchmark"
        )
        self.protocol["task_sets"]["development"] = dict(
            task_keys=["controlled"], seeds=[2]
        )
        self.protocol["runtime_config"] = {}
        self.protocol["budget"] = dict(
            per_attempt_proposal=dict(full_job_wall_seconds=900)
        )
        self.source_ref = f.operator / "next-comparison.json"
        atomic_json(self.source_ref, self.protocol)
        self.base = replace(
            RunBundle(**f.research.get("bundles", self.first["bundle_id"])),
            bundle_id="next-base",
            protocol_id="comparison-next",
        )
        f.research.save_bundle(self.base)
        profile = read(self.base.backend_ref)
        profile["relation_routes"] = [
            dict(route_id="learned-route", requested_worker="pi0_pick")
        ]
        profile["relation_skill_starts"] = dict(
            controlled=dict(route_id="learned-route", tool="pi0_pick")
        )
        self.profile_ref = f.research.path("profiles", "next-trial")
        atomic_json(self.profile_ref, profile)
        rho = {
            **f.research.get("rho", self.base.rho_ref),
            "schema": "rsi-relation-feedback-v1",
            "learned": True,
            "parent_rho_ref": self.base.rho_ref,
            "feedback": ["synthetic-feedback"],
            "protocol_id": "comparison-next",
        }
        f.research.put("rho", "next-rho", rho)
        self.trial = replace(
            self.base,
            bundle_id="next-trial",
            parent_bundle_id=self.base.bundle_id,
            rho_ref="next-rho",
            backend_ref=str(self.profile_ref),
        )
        f.research.save_bundle(self.trial)
        f.research.put(
            "development_results",
            "next-dev",
            dict(
                bundle_id=self.trial.bundle_id,
                role="development",
                intervention=dict(screening_checks=dict(verified=True)),
            ),
        )
        self.candidate = Candidate(
            "next-candidate",
            self.base.bundle_id,
            self.trial.bundle_id,
            self.protocol["protocol_id"],
            "benchmark",
            dict(rho_ref=self.trial.rho_ref, backend_ref=self.trial.backend_ref),
            [str(f.research.path("rho", "next-rho"))],
            ["next-dev"],
            {},
            {},
            "offline",
        )
        f.research.save_candidate(self.candidate)
        self.ref = f.operator / "next-registration.json"
        self.target_ref = f.operator / "next-deployment.json"
        self.target = deepcopy(self.protocol)
        self.target.update(
            protocol_id="next-deployment",
            anchor_bundle_id=self.first["bundle_id"],
            comparison_only=False,
            deployment_derivation=dict(
                kind="candidate_upgrade", authorization_ref=str(self.ref)
            ),
            candidate_upgrade_derivation=dict(
                registration_ref=str(self.ref), source_protocol_id="comparison-next"
            ),
        )
        atomic_json(self.target_ref, self.target)
        self.profile_snapshot_ref = f.operator / "next-profile-snapshot.json"
        target_profile = {
            **profile,
            "profile_id": "next-profile",
            "dynamic_authorized": True,
            "release_authorized": False,
            "deployment_authorized": True,
            "authorization_reference": str(self.ref),
            "changed_configuration": "Candidate deployment identity; evaluated behavior preserved",
        }
        atomic_json(self.profile_snapshot_ref, target_profile)
        self.registration = deepcopy(f.registration)
        self.registration.update(
            registration_id="next-frozen",
            registration_ref=str(self.ref),
            registered_at=(
                datetime.now(timezone.utc) - timedelta(seconds=1)
            ).isoformat(),
            active_snapshot=self.first,
            source_protocol_ref=str(self.source_ref),
            source_protocol_snapshot=self.protocol,
            target_protocol_ref=str(self.target_ref),
            target_protocol_snapshot=self.target,
            source_candidate_id="next-candidate",
            source_baseline_bundle_id="next-base",
            source_candidate_bundle_id="next-trial",
            target_candidate_id="next-deployed-candidate",
            target_bundle_id="next-deployed",
            target_profile_id="next-profile",
            source_result_ids=dict(
                baseline="next-baseline",
                candidate="next-candidate-matrix",
                anchor="next-baseline",
            ),
            source_decision_ref=str(f.operator / "next-source-gate.json"),
        )
        self.registration["rollback"] = dict(
            allowed_ancestors=[self.first["bundle_id"]], triggers=["binding_invalid"]
        )
        self.registration["behavior_review"] = {
            key: [binding(self.profile_ref, self.profile_snapshot_ref, ["planner"])]
            for key in BEHAVIOR_FIELDS
        }
        for field in ["relation_routes", "relation_skill_starts"]:
            self.registration["behavior_review"][field] = [
                binding(self.profile_ref, self.profile_snapshot_ref, [field])
            ]
        atomic_json(self.ref, self.registration)
        self.base_raw = Path(f.domains["runtime"]) / "next-baseline/native.json"
        self.raw = Path(f.domains["runtime"]) / "next-candidate/native.json"
        self.results = {}
        for name, old, bundle, raw in [
            ("next-baseline", f.baseline, self.base, self.base_raw),
            ("next-candidate-matrix", f.candidate, self.trial, self.raw),
        ]:
            rows = [
                {**row, "seed": seed, "evidence_ref": str(raw)}
                for row in old.assignments
                for seed in (0, 1)
            ]
            if bundle == self.base:
                next(
                    row for row in rows if row["task_key"] == "old" and row["seed"] == 1
                )["outcome"] = "failure"
            result = replace(
                old,
                result_id=name,
                bundle_id=bundle.bundle_id,
                protocol_id="comparison-next",
                candidate_id="next-candidate" if bundle == self.trial else None,
                assignments=rows,
                assigned_count=6,
                metrics={
                    **old.metrics,
                    "success_old": 0.5 if bundle == self.base else 1.0,
                },
                data_refs=[str(raw)],
            )
            self.results[name] = result
            if formal:
                atomic_json(raw, dict(native=True))
                atomic_json(
                    raw.parent / "runtime_context.json",
                    dict(
                        bundle_id=bundle.bundle_id,
                        protocol_id=bundle.protocol_id,
                        execution_binding=execution_binding(
                            f.research, asdict(bundle), read(bundle.backend_ref)
                        ),
                    ),
                )
                f.evaluation.save_result(result)
        f.evaluation.put("protocols", "comparison-next", self.protocol)
        if formal:
            gate = Gate(self.protocol).evaluate(
                "next-source-gate",
                self.candidate,
                current_bundle_id=self.base.bundle_id,
                candidate_result=self.results["next-candidate-matrix"],
                baseline_result=self.results["next-baseline"],
                anchor_result=self.results["next-baseline"],
            )
            case.assertEqual(gate.verdict, "accept")
            atomic_json(self.registration["source_decision_ref"], asdict(gate))

    def prepare_import(self):
        prepare_candidate_deployment(self.f.research, registration_ref=self.ref)
        return import_candidate_evidence(
            self.f.evaluation, import_id="next-independent", registration_ref=self.ref
        )


class SequentialUpgradeTests(unittest.TestCase):
    def setUp(self):
        self.n = NextRound(self)

    def test_second_upgrade_preserves_prior_qualification_and_accepts_valid_old_failure(
        self,
    ):
        n = self.n
        f = n.f
        self.assertNotIn("qualification_ref", n.first)
        imported = n.prepare_import()
        self.assertEqual(
            imported["current_active_applicability"]["qualification_snapshot"],
            n.first_qualification,
        )
        self.assertFalse(imported["current_active_applicability"]["reinitialized"])
        qualify_upgrade(
            f.manager,
            qualification_id="next-qualified",
            decision_id="next-upgrade",
            import_id="next-independent",
            registration_ref=str(n.ref),
        )
        active = f.manager.activate(
            "next-upgrade", safe_point="task_boundary", queues_empty=True
        )
        self.assertEqual(active["revision"], n.first["revision"] + 1)
        self.assertEqual(active["bundle_id"], "next-deployed")
        self.assertEqual(
            f.manager.records.get("qualifications", "new-qualified"),
            n.first_qualification,
        )
        self.assertEqual(
            f.manager.records.get("upgrade_publications", "upgrade"),
            n.first_publication,
        )
        self.assertEqual(
            f.evaluation.get("results", "next-baseline"),
            asdict(n.results["next-baseline"]),
        )
        from rsi_vla import qualification_import as trust

        trust.require_role.reset_mock()
        self.assertEqual(
            f.manager.deployment_bundle(scope="offline", safe_point="task_boundary")[1],
            active,
        )
        self.assertTrue(trust.require_role.call_args_list)
        self.assertTrue(
            all(call.args[1] == "runtime" for call in trust.require_role.call_args_list)
        )
        trust.require_role.reset_mock()
        reverted = f.manager.rollback(
            n.first["bundle_id"],
            safe_point="task_boundary",
            queues_empty=True,
            reason="binding_invalid",
        )
        self.assertEqual(reverted["bundle_id"], n.first["bundle_id"])
        self.assertTrue(
            all(call.args[1] == "release" for call in trust.require_role.call_args_list)
        )

    def test_prior_publication_drift_does_not_establish_second_upgrade(self):
        n = self.n
        f = n.f
        damaged = {**n.first_publication, "decision_id": "unrelated"}
        atomic_json(f.manager.records.path("upgrade_publications", "upgrade"), damaged)
        with self.assertRaisesRegex(ValueError, "ancestry mismatch"):
            n.prepare_import()
        self.assertEqual(f.manager.active(), n.first)
        self.assertFalse(
            f.evaluation.path("candidate_evidence_imports", "next-independent").exists()
        )

    def test_relation_mapping_distinguishes_deployment_from_old_current_profile(self):
        n = self.n
        active = n.f.research.get("bundles", n.first["bundle_id"])
        self.assertNotIn("relation_routes", read(active["backend_ref"]))
        n.registration["behavior_review"]["planner_input"].append(
            binding(n.profile_ref, active["backend_ref"], ["planner"])
        )
        atomic_json(n.ref, n.registration)
        imported = n.prepare_import()
        self.assertEqual(
            imported["profile_snapshot"]["relation_routes"],
            read(n.profile_ref)["relation_routes"],
        )

    def test_new_profile_fields_need_explicit_equal_mapping(self):
        n = self.n
        del n.registration["behavior_review"]["relation_routes"]
        atomic_json(n.ref, n.registration)
        with self.assertRaisesRegex(ValueError, "relation profile mapping missing"):
            n.prepare_import()
        self.assertEqual(n.f.manager.active(), n.first)

    def test_new_profile_fields_cannot_change_during_deployment_projection(self):
        n = self.n
        profile = read(n.profile_snapshot_ref)
        profile["relation_skill_starts"]["controlled"]["tool"] = "vla_execute"
        atomic_json(n.profile_snapshot_ref, profile)
        with self.assertRaisesRegex(ValueError, "changes evaluated relation profile"):
            n.prepare_import()

    def test_isolation_includes_original_published_upgrade_sources(self):
        n = self.n
        f = n.f
        paths = set(
            isolation_paths(
                {},
                dict(action="candidate_evidence_import", registration_ref=str(n.ref)),
            )
        )
        self.assertIn(f.manager.records.path("qualifications", "new-qualified"), paths)
        self.assertIn(f.manager.records.path("upgrade_publications", "upgrade"), paths)
        self.assertIn(f.manager.records.path("upgrade_decisions", "upgrade"), paths)
        self.assertIn(f.ref, paths)
        current = f.research.get("bundles", n.first["bundle_id"])
        self.assertIn(f.research.path("bundles", n.first["bundle_id"]), paths)
        self.assertIn(Path(current["backend_ref"]), paths)
        self.assertIn(f.research.path("rho", current["rho_ref"]), paths)
        self.assertIn(
            f.evaluation.path("candidate_evidence_imports", "independent-v1"), paths
        )


class UpgradeHelperV3Tests(unittest.TestCase):
    def setUp(self):
        self.n = NextRound(self, formal=False, full_profile=True)
        root = (
            fixture.ROOT
            / "artifacts/5a-completion-20261113-v1/operator/helpers/build_upgrade_registration_v3.py"
        )
        spec = importlib.util.spec_from_file_location("upgrade_helper_v3_test", root)
        self.helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.helper)
        self.helper.ROOT = self.n.f.root

    def test_actual_current_provenance_does_not_force_scope_origin_revision(self):
        n = self.n
        f = n.f
        active, qualification, registration = self.helper.current_provenance(
            f.research,
            f.evaluation,
            f.manager.records.root,
            expected={key: f.active[key] for key in ["bundle_id", "revision"]},
        )
        self.assertEqual(active, n.first)
        self.assertEqual(qualification, n.first_qualification)
        self.assertEqual(registration, f.registration)

    def test_rho_preview_and_materialize_use_actual_roots_authority_and_profile(self):
        n = self.n
        f = n.f
        authority_ref = f.operator / "scope-v2.json"
        atomic_json(
            authority_ref,
            dict(
                authorized=True,
                dynamic_authorized=True,
                scope_id="fixture-upgrade",
                expires_at=f.registration["expires_at"],
                allowed_actions=["qualification and normal upgrade after accept"],
                expected_active={
                    key: f.active[key] for key in ["bundle_id", "revision"]
                },
            ),
        )
        live = f.root / "fixture-consumer.py"
        live.write_text("source for the next round\n")
        frozen = f.operator / "source/fixture-consumer.py"
        frozen.parent.mkdir(exist_ok=True)
        frozen.write_bytes(live.read_bytes())
        manifest = f.operator / "next-source-manifest.json"
        atomic_json(manifest, dict(files=["fixture-consumer.py"]))
        config = dict(
            schema="rsi-task-completion-v1",
            scope_authorization_ref=str(authority_ref),
            protocol_ref=str(n.source_ref),
            research_store_ref=str(f.research.root),
            evaluation_store_ref=str(f.evaluation.root),
            release_root=str(f.manager.records.root),
            base_bundle_id=n.base.bundle_id,
            source_manifest_ref=str(manifest),
        )
        config_ref = f.operator / "next-configuration.json"
        atomic_json(config_ref, config)
        acceptance = f.operator / "next-acceptance.json"
        reports = {}
        for role, domain in f.domains.items():
            probe = Path(domain) / "fixture-probe.json"
            atomic_json(
                probe,
                dict(
                    role=role,
                    scope="fixture-upgrade",
                    allowed_write=True,
                    denied_writes=True,
                ),
            )
            reports[role] = dict(exit_code=0, report_ref=str(probe))
        atomic_json(
            acceptance,
            dict(scope="fixture-upgrade", role_domains=f.domains, roles=reports),
        )
        temp = tempfile.TemporaryDirectory(prefix="upgrade-helper-v3-", dir="/tmp")
        self.addCleanup(temp.cleanup)
        argv = [
            "--task-root",
            str(f.task),
            "--configuration",
            str(config_ref),
            "--candidate-id",
            n.candidate.candidate_id,
            "--bundle-id",
            n.trial.bundle_id,
            "--operator",
            "rho_candidate",
            "--target-candidate-id",
            "v3-candidate",
            "--target-bundle-id",
            "v3-deployment",
            "--target-profile-id",
            "v3-profile",
            "--target-protocol-id",
            "v3-protocol",
            "--baseline-result-id",
            "future-baseline",
            "--candidate-result-id",
            "future-candidate",
            "--source-decision-ref",
            str(f.operator / "future-gate.json"),
            "--isolation-acceptance-ref",
            str(acceptance),
            "--inheritance-task-key",
            "controlled",
            "--preview-root",
            temp.name,
        ]
        registration_ref = f.operator / "v3-deployment-registration.json"
        before = deepcopy(f.manager.active())
        with contextlib.redirect_stdout(io.StringIO()):
            self.helper.main(argv)
        self.assertFalse(registration_ref.exists())
        plan = read(Path(temp.name) / "v3-deployment-materialization-plan.json")
        reg = plan["writes"][str(registration_ref)]
        self.assertEqual(reg["active_snapshot"], n.first)
        self.assertEqual(reg["approval_reference"], str(authority_ref))
        self.assertIn("relation_routes", reg["behavior_review"])
        self.assertIn("relation_skill_starts", reg["behavior_review"])
        self.assertIn("rho", reg["behavior_review"])
        with contextlib.redirect_stdout(io.StringIO()):
            self.helper.main(argv + ["--materialize"])
        self.assertTrue(registration_ref.exists())
        self.assertEqual(f.manager.active(), before)
        self.assertEqual(
            f.manager.records.get("qualifications", "new-qualified"),
            n.first_qualification,
        )
        self.assertEqual(read(config_ref), config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
