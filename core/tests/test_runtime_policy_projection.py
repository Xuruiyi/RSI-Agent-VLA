"""Optional Skill and L1 policy inputs reach frozen Runtime contexts."""

import json
import unittest
from dataclasses import replace
from pathlib import Path

import test_runtime_plan
from rsi_vla.contracts import RunBundle
from rsi_vla.storage import ResearchStore, atomic_json


class RuntimePolicyProjectionTests(unittest.TestCase):
    setUp = test_runtime_plan.RuntimePlanTests.setUp
    plan = test_runtime_plan.RuntimePlanTests.plan

    def profile(self):
        store = ResearchStore(self.artifacts / "research")
        base = RunBundle(**store.get("bundles", "initial-medium"))
        profile = json.loads(Path(base.backend_ref).read_text())
        profile["skill_contracts"] = {
            "schema": "libero-skill-contracts-v1",
            "skills": {
                "vla_execute": {
                    "skill_id": "input-compatible-vla",
                    "authorized": True,
                    "version_signature": {
                        name: getattr(base, name)
                        for name in (
                            "theta_ref",
                            "processor_ref",
                            "phi_ref",
                            "code_ref",
                        )
                    },
                    "action_prefix_steps": 5,
                    "require_verified": False,
                    "validation": {
                        "status": "unverified",
                        "evidence_refs": [],
                        "task_scope": [],
                    },
                    "preconditions": [],
                    "completion": [],
                    "failure_conditions": [],
                    "requires": [],
                }
            },
        }
        profile["l1_media_policy"] = {
            "schema": "rsi-l1-media-policy-v1",
            "optional_artifacts": ["agentview_world_high.npz", "wrist_world_high.npz"],
        }
        ref = self.artifacts / "policy-profile.json"
        atomic_json(ref, profile)
        child = replace(base, bundle_id="policy-child", backend_ref=str(ref))
        store.save_bundle(child)
        return ref, profile, child

    def test_new_policy_context_inherits_refs_and_old_context_keeps_original_inputs(
        self,
    ):
        old = self.plan()["context"]
        ref, profile, child = self.profile()
        context = self.plan(bundle_id=child.bundle_id)["context"]
        self.assertEqual(context["skill_contracts"], profile["skill_contracts"])
        self.assertEqual(context["l1_media_policy"]["rho_ref"], child.rho_ref)
        self.assertEqual(
            context["l1_media_policy"]["max_persisted_bytes"],
            context["limits"]["storage_bytes"],
        )
        for field in ("theta_ref", "processor_ref", "phi_ref", "code_ref"):
            self.assertEqual(context[field], getattr(child, field))
        profile["skill_contracts"]["schema"] = "changed"
        atomic_json(ref, profile)
        self.assertEqual(
            context["skill_contracts"]["schema"], "libero-skill-contracts-v1"
        )
        self.assertNotIn("skill_contracts", old)
        self.assertNotIn("l1_media_policy", old)

    def test_media_policy_rejects_wrong_version_unknown_schema_and_over_budget(self):
        ref, profile, child = self.profile()
        for changed in (
            {"rho_ref": "other"},
            {"schema": "unsupported"},
            {"max_persisted_bytes": True},
            {"max_persisted_bytes": 10**15},
        ):
            with self.subTest(changed=changed):
                atomic_json(
                    ref,
                    {
                        **profile,
                        "l1_media_policy": {**profile["l1_media_policy"], **changed},
                    },
                )
                with self.assertRaises(ValueError):
                    self.plan(bundle_id=child.bundle_id)


if __name__ == "__main__":
    unittest.main()
