"""Protocol10 admission is offline and keeps the frozen experiment denominator."""

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from rsi_vla.contracts import ExperimentResult, RunBundle
from rsi_vla.improvement_protocol import (
    ACTIVE,
    BASELINE,
    DIMENSIONS,
    TASKS,
    admit_resources,
    baseline_readiness,
    check_stage,
    derive_baseline,
    load_frozen_protocol,
    prepare_baseline_attempt,
    resource_envelope,
    validate_attempts,
)
from rsi_vla.storage import Records, ResearchStore

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/pilot_5a_protocol10.json"
FROZEN = CONFIG


class ImprovementProtocolTest(unittest.TestCase):
    def setUp(self):
        self.p = load_frozen_protocol(CONFIG, FROZEN)

    def phase(self, cap=None):
        cap = cap or resource_envelope(self.p)["gross"]
        return {
            "limits": dict(cap),
            "used": dict.fromkeys(DIMENSIONS, 0),
            "remaining": dict(cap),
            "unresolved_costs": [],
        }

    def baseline(self, failed=()):
        rows = [
            {
                "task_key": t,
                "seed": s,
                "outcome": "failure" if (t, s) in failed else "success",
                "validity": "valid",
                "evidence_ref": f"isolated/{t}/{s}.json",
                "outcome_source": "native_benchmark",
            }
            for t in TASKS
            for s in (0, 1)
        ]
        values = {
            t: sum(r["outcome"] == "success" for r in rows if r["task_key"] == t) / 2
            for t in TASKS
        }
        metrics = {
            "success_base": values["base_salad"],
            "success_controlled": values["position_swap_salad"],
            "success_old": (values["old_milk"] + values["old_drawer"]) / 2,
        }
        return ExperimentResult(
            result_id="isolated-baseline",
            bundle_id=BASELINE,
            protocol_id=self.p["protocol_id"],
            benchmark_id=self.p["benchmark_id"],
            role="release",
            evidence_level="simulation",
            intervention={},
            data_refs=["isolated-fixture"],
            criteria_ref="fixture-native",
            metrics=metrics,
            hard_constraints=dict.fromkeys(self.p["hard_constraints"], True),
            costs={},
            assigned_count=8,
            assignments=rows,
        )

    def attempt(
        self,
        run="first",
        side="baseline",
        task="base_salad",
        seed=0,
        outcome="failure",
        validity="valid",
        order=1,
    ):
        return {
            "run_id": run,
            "output_ref": "isolated/" + run,
            "protocol_id": self.p["protocol_id"],
            "versions": self.p["versions"],
            "started_order": order,
            "side": side,
            "task_key": task,
            "seed": seed,
            "bundle_id": BASELINE,
            "outcome": outcome,
            "validity": validity,
            "failure_class": "technical" if validity == "invalid" else "ability",
            "cost_closed": True,
            "cleanup_complete": True,
            "costs": dict(zip(DIMENSIONS, (1, 1, 2, 30, 40, 40, 100))),
        }

    def test_frozen_copy_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            changed = copy.deepcopy(self.p)
            changed["task_sets"]["release"]["seeds"] = [0, 3]
            path = Path(td) / "changed.json"
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "frozen"):
                load_frozen_protocol(path, FROZEN)
            with self.assertRaisesRegex(ValueError, "denominator"):
                load_frozen_protocol(path, path)

    def test_exact_full_envelope_and_no_scratch_double_reserve(self):
        actual = resource_envelope(self.p)
        self.assertEqual(
            actual["runtime"],
            dict(zip(DIMENSIONS, (22, 440, 1320, 44000, 19800, 19800, 23622320128))),
        )
        self.assertEqual(
            actual["gross"],
            dict(zip(DIMENSIONS, (22, 441, 1320, 44000, 21180, 19800, 24763170816))),
        )
        used = resource_envelope(
            self.p,
            offline_used_seconds=101,
            offline_used_bytes=123,
            runtime_starts_used=8,
            research_completed=True,
        )
        self.assertEqual(used["remaining"]["simulation_episodes"], 14)
        self.assertEqual(used["remaining"]["wall_seconds"], 21180 - 101 - 7200 - 180)
        self.assertEqual(
            used["remaining"]["storage_bytes"],
            24763170816 - 123 - 8 * 1073741824 - 67108864,
        )

    def test_each_hard_dimension_blocks_and_phase_is_unchanged(self):
        for key in DIMENSIONS:
            with self.subTest(key=key):
                phase = self.phase()
                phase["limits"][key] -= 1
                phase["remaining"][key] -= 1
                before = copy.deepcopy(phase)
                report = admit_resources(
                    self.p, phase, offline_used_seconds=0, offline_used_bytes=0
                )
                self.assertFalse(report["resource_fit"])
                self.assertEqual(report["additional_required"][key], 1)
                self.assertEqual(phase, before)

    def test_unknown_cost_and_false_remaining_are_rejected(self):
        for mutate in (
            lambda p: p.update(unresolved_costs=["unknown"]),
            lambda p: p["used"].update(tool_calls=None),
            lambda p: p["remaining"].update(wall_seconds=1),
        ):
            phase = self.phase()
            mutate(phase)
            with self.assertRaises(ValueError):
                admit_resources(
                    self.p, phase, offline_used_seconds=0, offline_used_bytes=0
                )

    def test_full_baseline_ceiling_stops_candidate_work(self):
        result = self.baseline()
        report = baseline_readiness(self.p, result)
        self.assertEqual(report["J"], 1)
        self.assertEqual(report["action"], "freeze_stop_ceiling")
        for stage in ("research", "development", "candidate", "inheritance"):
            stage_report = check_stage(
                self.p,
                self.phase(),
                stage=stage,
                baseline=result,
                offline_used_seconds=0,
                offline_used_bytes=0,
            )
            self.assertFalse(stage_report["preparation_ready"])
            self.assertFalse(stage_report["dispatch"])

    def test_headroom_keeps_original_criterion(self):
        result = self.baseline(failed=(("position_swap_salad", 0),))
        report = baseline_readiness(self.p, result)
        self.assertTrue(report["candidate_generation_allowed"])
        self.assertAlmostEqual(report["J"], 0.825)
        self.assertEqual(self.p["min_gain"], 0.02)
        self.assertFalse(
            check_stage(
                self.p,
                self.phase(),
                stage="research",
                baseline=result,
                offline_used_seconds=0,
                offline_used_bytes=0,
            )["dynamic_admitted"]
        )

    def test_baseline_identity_denominator_and_old_capability(self):
        result = self.baseline(failed=(("old_milk", 0), ("old_milk", 1)))
        self.assertEqual(
            baseline_readiness(self.p, result)["action"],
            "stop_baseline_capability_unestablished",
        )
        with self.assertRaises(ValueError):
            baseline_readiness(
                self.p, replace(result, bundle_id="historical-release08")
            )
        with self.assertRaises(ValueError):
            baseline_readiness(
                self.p, replace(result, assignments=result.assignments[:-1])
            )

    def test_missing_slots_stay_in_sixteen_row_matrix(self):
        report = validate_attempts(self.p, [self.attempt()])
        self.assertEqual(len(report["matrix"]), 16)
        self.assertEqual(
            sum(r["selected_run_id"] is None for r in report["matrix"]), 15
        )
        self.assertEqual(report["formal_denominator_per_side"], 8)
        self.assertEqual(report["matrix"][0]["outcome"], "failure")

    def test_valid_failure_cannot_be_replaced(self):
        first = self.attempt()
        retry = self.attempt(run="retry", outcome="success", order=2)
        retry.update(recovery_of_run_id="first", recovery_cause_resolved=True)
        with self.assertRaisesRegex(ValueError, "cannot replace"):
            validate_attempts(self.p, [first, retry])

    def test_closed_technical_recovery_retains_all_costs(self):
        first = self.attempt(outcome="unknown", validity="invalid")
        retry = self.attempt(run="retry", outcome="success", order=2)
        retry.update(recovery_of_run_id="first", recovery_cause_resolved=True)
        report = validate_attempts(self.p, [first, retry])
        self.assertEqual(report["matrix"][0]["selected_run_id"], "retry")
        self.assertEqual(report["all_attempt_costs"]["simulation_episodes"], 2)
        self.assertEqual(report["recoveries"]["technical_recovery"], 1)
        for missing in ("cleanup_complete", "cost_closed"):
            bad = copy.deepcopy(first)
            bad[missing] = False
            with self.assertRaises(ValueError):
                validate_attempts(self.p, [bad, retry])

    def test_round_recovery_cap_and_per_slot_cap(self):
        rows = []
        for n, task in enumerate(TASKS[:3]):
            first = self.attempt(
                run=f"a{n}",
                task=task,
                outcome="unknown",
                validity="invalid",
                order=n * 2 + 1,
            )
            retry = self.attempt(run=f"b{n}", task=task, order=n * 2 + 2)
            retry.update(
                recovery_of_run_id=first["run_id"], recovery_cause_resolved=True
            )
            rows += [first, retry]
        with self.assertRaisesRegex(ValueError, "round recovery cap"):
            validate_attempts(self.p, rows)
        rows = rows[:2]
        rows[1]["validity"] = "invalid"
        rows[1]["failure_class"] = "technical"
        third = self.attempt(run="third", order=3)
        third.update(recovery_of_run_id="b0", recovery_cause_resolved=True)
        with self.assertRaises(ValueError):
            validate_attempts(self.p, rows + [third])

    def test_attempt_id_versions_bundle_splits_and_limits(self):
        for mutate in (
            lambda r: r.update(seed=3),
            lambda r: r.update(bundle_id="other"),
            lambda r: r["versions"].update(evaluation="unfrozen"),
            lambda r: r["costs"].update(model_requests=21),
        ):
            row = copy.deepcopy(self.attempt())
            mutate(row)
            with self.assertRaises(ValueError):
                validate_attempts(self.p, [row])
        rows = [
            self.attempt(
                run=f"d{i}", side="development", task=TASKS[i], seed=2, order=i + 1
            )
            for i in range(3)
        ]
        with self.assertRaisesRegex(ValueError, "side start cap"):
            validate_attempts(self.p, rows)

    def test_isolated_source_derivation_without_qualification_import(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = ResearchStore(root / "source-research")
            release = root / "source-release"
            release.mkdir()
            (release / "active.json").write_text(
                json.dumps({"bundle_id": ACTIVE, "activation": "qualified_initial"})
            )
            new_profile = json.loads(
                (ROOT / "config/run_profile_5a_protocol10.json").read_text()
            )
            old_profile = {
                k: v for k, v in new_profile.items() if k != "protocol10_derivation"
            }
            old_profile.update(
                profile_id=ACTIVE, dynamic_authorized=True, deployment_authorized=True
            )
            source.put("profiles", ACTIVE, old_profile)
            profile_ref = str(source.path("profiles", ACTIVE))
            for kind in ("psi", "memory", "rho"):
                source.put(
                    kind, "fixture-" + kind, {"kind": kind, "legal_information": []}
                )
            bundle = RunBundle(
                ACTIVE,
                profile_ref + "#/model",
                profile_ref + "#/model/normalization",
                profile_ref + "#/control",
                "fixture-rho",
                "fixture-psi",
                "fixture-memory",
                "rsi-libero-pro-initial-release08-deployment09",
                profile_ref,
                "fixture-code",
            )
            source.save_bundle(bundle)
            active_before = (release / "active.json").read_bytes()
            destination = root / "destination-research"
            args = {
                "source_research": source.root,
                "source_release": release,
                "destination": destination,
                "profile_path": ROOT / "config/run_profile_5a_protocol10.json",
            }
            first = derive_baseline(self.p, **args)
            self.assertEqual(first, derive_baseline(self.p, **args))
            derived = json.loads(
                (destination / "bundles" / (BASELINE + ".json")).read_text()
            )
            self.assertEqual(derived["parent_bundle_id"], ACTIVE)
            self.assertEqual(derived["protocol_id"], self.p["protocol_id"])
            for kind in ("psi", "memory", "rho"):
                self.assertEqual(
                    source.get(kind, "fixture-" + kind),
                    Records(destination).get(kind, "fixture-" + kind),
                )
            self.assertFalse((destination / "results").exists())
            self.assertFalse((destination / "qualifications").exists())
            new_profile["control"]["action_chunk"] = 1
            changed_path = root / "changed.json"
            changed_path.write_text(json.dumps(new_profile))
            with self.assertRaisesRegex(ValueError, "execution inputs"):
                derive_baseline(self.p, **{**args, "profile_path": changed_path})
            self.assertEqual(active_before, (release / "active.json").read_bytes())

    def test_stage_reuses_consumed_starts_without_erasing_cost(self):
        row = self.attempt()
        phase = self.phase()
        for k, v in row["costs"].items():
            phase["used"][k] += v
            phase["remaining"][k] -= v
        report = check_stage(
            self.p,
            phase,
            stage="baseline",
            attempts=[row],
            offline_used_seconds=0,
            offline_used_bytes=0,
        )
        self.assertEqual(report["envelope"]["remaining"]["simulation_episodes"], 21)
        self.assertEqual(report["attempts"]["all_attempt_costs"], row["costs"])
        with self.assertRaisesRegex(ValueError, "retain all"):
            check_stage(
                self.p,
                self.phase(),
                stage="baseline",
                attempts=[row],
                offline_used_seconds=0,
                offline_used_bytes=0,
            )

    def hook_fixture(self, root):
        source = ResearchStore(root / "source")
        profile = {
            "dynamic_authorized": False,
            "planner": {
                "backend": "codex",
                "model": "gpt-6.1-sol",
                "reasoning_effort": "medium",
            },
            "research_llm": {"model": "gpt-6.1-sol", "reasoning_effort": "high"},
            "model": {"checkpoint": "fixture-weights", "normalization": "fixture-norm"},
            "control": {
                "libero_type": "pro",
                "action_chunk": 5,
                "max_episode_steps": 2000,
            },
            "token_policy": "observed_only",
        }
        source.put("profiles", BASELINE, profile)
        profile_ref = str(source.path("profiles", BASELINE))
        for kind in ("psi", "rho"):
            source.put(kind, "fixture-" + kind, {"type": "fixed"})
        source.put(
            "memory",
            "fixture-memory",
            {
                "legal_information": ["public-fixture"],
                "upstream": {
                    "profile": "hf",
                    "version": "fixture",
                    "repo": "fixture/repo",
                    "revision": "fixture-revision",
                },
            },
        )
        bundle = RunBundle(
            BASELINE,
            profile_ref + "#/model",
            profile_ref + "#/model/normalization",
            profile_ref + "#/control",
            "fixture-rho",
            "fixture-psi",
            "fixture-memory",
            self.p["protocol_id"],
            profile_ref,
            "fixture-execution",
            parent_bundle_id=ACTIVE,
        )
        source.save_bundle(bundle)
        source.put(
            "protocol_derivations",
            BASELINE,
            {"parent_bundle_id": ACTIVE, "protocol_id": self.p["protocol_id"]},
        )
        authority_path = root / "authority.json"
        authority = {
            "authorized": True,
            "scope": self.p["scope_id"],
            "protocol_id": self.p["protocol_id"],
            "frozen_protocol_ref": str(FROZEN.resolve()),
            "allowed_runtime_splits": ["release"],
        }
        authority_path.write_text(json.dumps(authority))
        output, scratch = root / "output", root / "scratch"
        permit_path = root / "permit.json"
        permit = {
            "authorized": True,
            "scope": self.p["scope_id"],
            "run_id": "fixture-baseline-a1",
            "bundle_id": BASELINE,
            "protocol_id": self.p["protocol_id"],
            "task_key": "base_salad",
            "seed": 0,
            "simulation_episodes": 1,
            "gpu_devices": [0],
            "output_dir": str(output),
            "storage_write_roots": [str(output), str(scratch)],
            "administrative_authorization_ref": str(authority_path),
            "limits": {
                "wall_seconds": 900,
                "planner_seconds": 600,
                "tool_calls": 60,
                "environment_steps": 2000,
                "storage_bytes": 1073741824,
                "gpu_seconds": 900,
                "model_requests": 20,
            },
        }
        permit_path.write_text(json.dumps(permit))
        return {
            "root": ROOT,
            "source_research": source.root,
            "preparation_root": root / "prepared",
            "protocol_path": CONFIG,
            "frozen_path": FROZEN,
            "phase": self.phase(),
            "administrative_authorization": authority_path,
            "permit_path": permit_path,
            "run_id": "fixture-baseline-a1",
            "task_key": "base_salad",
            "seed": 0,
            "output_dir": output,
            "scratch_dir": scratch,
            "offline_used_seconds": 0,
            "offline_used_bytes": 0,
        }

    def test_prepare_hook_real_runtime_cpu_plan_without_assets_or_collection(self):
        with tempfile.TemporaryDirectory() as td:
            args = self.hook_fixture(Path(td))
            source_profile = (
                Path(args["source_research"]) / "profiles" / (BASELINE + ".json")
            )
            before = source_profile.read_bytes()
            receipt = prepare_baseline_attempt(**args)
            self.assertEqual(receipt["plan"]["blockers"], [])
            context = receipt["plan"]["context"]
            self.assertEqual(
                context["storage_write_roots"],
                [str(args["output_dir"]), str(args["scratch_dir"])],
            )
            self.assertEqual(context["bundle_id"], BASELINE)
            self.assertEqual(context["protocol_id"], self.p["protocol_id"])
            self.assertEqual(context["limits"]["storage_bytes"], 1073741824)
            self.assertTrue(context["dynamic_authorized"])
            self.assertFalse(receipt["execute"])
            self.assertFalse(args["output_dir"].exists())
            self.assertFalse(args["scratch_dir"].exists())
            self.assertEqual(before, source_profile.read_bytes())
            projected = json.loads(Path(receipt["runtime_projection_ref"]).read_text())
            self.assertEqual(projected["tasks"], self.p["tasks"])
            self.assertEqual(projected["metrics"], self.p["metrics"])
            self.assertEqual(
                projected["budget"]["runtime_starts"],
                self.p["budget"]["runtime_starts"],
            )
            self.assertEqual(
                projected["runtime_projection"]["version"],
                "protocol10-to-existing-runtime-v1",
            )

    def test_prepare_hook_fake_records_actual_consumer_arguments(self):
        with tempfile.TemporaryDirectory() as td:
            args = self.hook_fixture(Path(td))
            with patch(
                "rsi_vla.runtime.prepare_runtime",
                return_value={"blockers": ["fixture asset absent"]},
            ) as prepare:
                result = prepare_baseline_attempt(**args)
            prepare.assert_called_once()
            positional, named = prepare.call_args
            self.assertEqual(positional, (ROOT, args["preparation_root"] / "state"))
            self.assertEqual(named["bundle_id"], BASELINE)
            self.assertEqual(named["split"], "release")
            self.assertEqual(named["authorization_path"], args["permit_path"])
            self.assertEqual(named["storage_write_roots"], [args["scratch_dir"]])
            self.assertFalse(result["ready_for_separately_authorized_dispatch"])
            self.assertFalse(result["dispatch"])

    def test_prepare_hook_rejects_missing_authority_budget_and_permit_before_call(self):
        for reason in ("authority", "budget", "permit", "seed", "reused-root"):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as td:
                args = self.hook_fixture(Path(td))
                if reason == "authority":
                    authority = json.loads(
                        args["administrative_authorization"].read_text()
                    )
                    authority["authorized"] = False
                    args["administrative_authorization"].write_text(
                        json.dumps(authority)
                    )
                elif reason == "budget":
                    args["phase"]["limits"]["simulation_episodes"] = 1
                    args["phase"]["remaining"]["simulation_episodes"] = 1
                elif reason == "permit":
                    permit = json.loads(args["permit_path"].read_text())
                    permit["storage_write_roots"] = [str(args["output_dir"])]
                    args["permit_path"].write_text(json.dumps(permit))
                elif reason == "seed":
                    args["seed"] = 3
                else:
                    args["scratch_dir"].mkdir()
                with patch("rsi_vla.runtime.prepare_runtime") as prepare:
                    with self.assertRaises(ValueError):
                        prepare_baseline_attempt(**args)
                    prepare.assert_not_called()
                self.assertFalse(args["preparation_root"].exists())


if __name__ == "__main__":
    unittest.main()
