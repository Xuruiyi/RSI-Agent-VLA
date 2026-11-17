"""Research selection and trusted queries consume existing candidate lineage."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import test_core
from rsi_vla.candidate_archive import (
    applicability,
    compose_candidates,
    inherit_parent,
    register_context,
)
from rsi_vla.contracts import Candidate, RunBundle
from rsi_vla.evaluation import EvaluationStore
from rsi_vla.release import ReleaseManager
from rsi_vla.research import ResearchJob
from rsi_vla.research_tools import ResearchToolkit, ResearchTools
from rsi_vla.storage import ResearchStore, initialize


class CandidateArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="f09-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = ResearchStore(self.root / "research")
        self.base = RunBundle(
            "base",
            "theta",
            "processor",
            "phi",
            "rho",
            "psi",
            "memory",
            "protocol",
            "backend",
            "code",
        )
        initialize(
            self.store,
            self.base,
            psi={"kind": "fixed"},
            memory={"provenance": "synthetic", "legal_information": ["development"]},
        )
        self.protocol = {
            "protocol_id": "protocol",
            "benchmark_id": "bench",
            "scope": "offline",
        }
        self.job = ResearchJob(self.store, "job", "base", {"cpu_seconds": 20})
        self.tools = ResearchTools(
            self.store, self.job, self.protocol, domain_policy={}
        )
        self.a = self.branch("a", "phi_ref")
        self.b = self.branch("b", "psi_ref")

    def branch(self, key, component, *, value=None, cost=3):
        artifact = self.root / (key + ".json")
        artifact.write_text('{"synthetic":true}')
        ref = value or (str(artifact) if component != "psi_ref" else "psi-" + key)
        if component == "psi_ref":
            self.store.put("psi", ref, {"program_ref": str(artifact)})
        bundle = replace(
            self.base,
            bundle_id="bundle-" + key,
            parent_bundle_id="base",
            **{component: ref},
        )
        self.store.save_bundle(bundle)
        job = ResearchJob(self.store, "job-" + key, "base", {"cpu_seconds": 20})
        job.consume({"cpu_seconds": cost}, "failed attempt")
        job.consume({"cpu_seconds": 1}, "following valid attempt")
        candidate = Candidate(
            key,
            "base",
            bundle.bundle_id,
            "protocol",
            "bench",
            {component: ref},
            [str(artifact)],
            [],
            {"cpu_seconds": cost + 1},
            {},
        )
        self.store.save_candidate(candidate)
        self.store.put(
            "research_experiments",
            "experiment-" + key,
            {
                "experiment_id": "experiment-" + key,
                "job_id": "job-" + key,
                "bundle_id": bundle.bundle_id,
            },
        )
        register_context(
            self.store,
            key,
            experiment_ids=["experiment-" + key],
            hypothesis_refs=["hypothesis-" + key],
            selection_reason="bounded development branch",
            dependencies={
                "versions": {"theta_ref": "theta", "processor_ref": "processor"},
                "task_scope": ["task"],
            },
            origin={"kind": "program", "ref": "test-fixture"},
        )
        return candidate

    def test_historical_parent_selection_is_consumed_by_new_job_without_rebinding_old_job(
        self,
    ):
        toolkit = ResearchToolkit(self.tools)
        self.assertEqual(len(toolkit.call("query_candidates", {})), 2)
        selected = toolkit.call(
            "select_parent",
            {
                "selection_id": "selection",
                "candidate_id": "a",
                "reason": "reuse a retained failed branch",
            },
        )
        new_job = ResearchJob(
            self.store, "new-job", selected["selected_bundle_id"], {"cpu_seconds": 2}
        )
        inherited = inherit_parent(
            self.store, "selection", new_job, self.protocol, domain_policy={}
        )
        self.assertEqual(inherited.base.bundle_id, "bundle-a")
        self.assertEqual(inherited.base.phi_ref, self.a.changes["phi_ref"])
        self.assertEqual(self.tools.base.bundle_id, "base")
        self.assertEqual(ResearchJob(self.store, "job").state["bundle_id"], "base")
        self.assertFalse((self.root / "release/active.json").exists())
        self.assertEqual(len(self.tools.query_candidates()), 2)
        with self.assertRaisesRegex(ValueError, "bound exhausted"):
            self.tools.select_parent(
                "another", "b", reason="not within the selection allocation"
            )
        Path(self.a.artifact_refs[0]).unlink()
        with self.assertRaisesRegex(ValueError, "retained"):
            self.tools.select_parent(
                "selection", "a", reason="reuse a retained failed branch"
            )
        self.assertEqual(
            self.store.candidate_archive("a")["candidate"]["candidate_id"], "a"
        )

    def test_complete_composition_costs_and_version_dependencies_are_consumed_by_queries(
        self,
    ):
        result = self.tools.compose_candidates(
            "combined",
            bundle_id="combined-bundle",
            source_candidate_ids=["a", "b"],
            selection_reason="test joint interfaces",
            dependencies={
                "versions": {"theta_ref": "theta", "processor_ref": "processor"},
                "task_scope": ["task"],
            },
        )
        bundle = result["bundle"]
        self.assertEqual(bundle["phi_ref"], self.a.changes["phi_ref"])
        self.assertEqual(bundle["psi_ref"], self.b.changes["psi_ref"])
        self.assertEqual(bundle["rho_ref"], "rho")
        self.assertEqual(result["candidate"]["costs"], {"cpu_seconds": 8})
        evaluations = EvaluationStore(self.root / "evaluation")
        release = ReleaseManager(self.root / "release", self.store, evaluations)
        trace = release.candidate_archive("combined", task_scope="task")
        self.assertEqual(trace["applicability"]["status"], "applicable")
        self.assertEqual(len(trace["costs"]["job_ledgers"]), 2)
        self.assertEqual(
            sum(len(j["charges"]) for j in trace["costs"]["job_ledgers"]), 4
        )
        self.assertEqual(
            evaluations.candidate_archive(self.store, "combined")["bundle"], bundle
        )
        self.assertEqual(trace["evaluation_status"], "unconfirmed")
        self.assertFalse(trace["qualification_inherited"])
        changed = {**bundle, "theta_ref": "new-theta"}
        applicability_result = release.candidate_archive(
            "combined", target_bundle=changed, task_scope="task"
        )["applicability"]
        self.assertEqual(applicability_result["status"], "unconfirmed")
        self.assertIn("skill_start", applicability_result["checks"])
        self.assertEqual(
            release.candidate_archive("combined", task_scope="other")["applicability"][
                "status"
            ],
            "inapplicable",
        )
        self.assertEqual(
            applicability({}, bundle, invalidated=True)["status"], "invalid"
        )

    def test_composition_preserves_each_sources_version_validation(self):
        # A phi validated only with the original psi is not silently certified
        # when a second candidate changes that dependency.
        context = self.store.get("candidate_context", "a")
        context["dependencies"]["versions"]["psi_ref"] = "psi"
        from rsi_vla.storage import atomic_json

        atomic_json(self.store.path("candidate_context", "a"), context)
        self.tools.compose_candidates(
            "dependent",
            bundle_id="dependent-bundle",
            source_candidate_ids=["a", "b"],
            selection_reason="joint check required",
            dependencies={"versions": {"theta_ref": "theta"}},
        )
        trace = self.store.candidate_archive("dependent", task_scope="task")
        self.assertEqual(trace["applicability"]["status"], "unconfirmed")
        self.assertIn("research_program", trace["applicability"]["checks"])
        self.assertEqual(self.store.get("candidate_context", "a"), context)

    def test_legacy_cost_lineage_is_preserved_without_claiming_complete_total(self):
        artifact = self.root / "legacy.json"
        artifact.write_text("{}")
        target = replace(
            self.base,
            bundle_id="legacy-bundle",
            parent_bundle_id="base",
            phi_ref=str(artifact),
        )
        self.store.save_bundle(target)
        self.store.save_candidate(
            Candidate(
                "legacy",
                "base",
                target.bundle_id,
                "protocol",
                "bench",
                {"phi_ref": str(artifact)},
                [str(artifact)],
                [],
                {"cpu_seconds": 7},
                {},
            )
        )
        composed = self.tools.compose_candidates(
            "part-known",
            bundle_id="part-known-bundle",
            source_candidate_ids=["legacy", "b"],
            selection_reason="preserve old unknown accounting owners",
        )
        self.assertFalse(composed["candidate"]["compatibility"]["costs_complete"])
        trace = self.store.candidate_archive("part-known")
        self.assertIn(
            "composition_source_cost_lineage_incomplete", trace["costs"]["unknown"]
        )
        self.assertEqual(
            composed["diff"]["cost_snapshots"][0]["costs"],
            {"cpu_seconds": 7},
        )

    def test_conflict_stale_base_and_duplicate_contribution(self):
        self.branch("conflict", "phi_ref")
        with self.assertRaisesRegex(ValueError, "conflicting"):
            compose_candidates(
                self.store,
                "bad",
                base_bundle_id="base",
                bundle_id="bad-bundle",
                source_candidate_ids=["a", "conflict"],
                selection_reason="bad",
            )
        self.assertFalse(self.store.path("bundles", "bad-bundle").exists())
        with self.assertRaisesRegex(ValueError, "stale"):
            compose_candidates(
                self.store,
                "stale",
                base_bundle_id="bundle-a",
                bundle_id="stale-bundle",
                source_candidate_ids=["a", "b"],
                selection_reason="wrong base",
            )
        self.branch("same", "phi_ref", value=self.a.changes["phi_ref"])
        same = self.tools.compose_candidates(
            "dedup",
            bundle_id="dedup-bundle",
            source_candidate_ids=["a", "same"],
            selection_reason="same patch is one contribution",
        )
        self.assertEqual(len(same["diff"]["contributions"]), 1)
        self.assertEqual(len(same["diff"]["cost_snapshots"]), 2)
        self.assertEqual(same["candidate"]["costs"], {"cpu_seconds": 8})


class ExistingTrustedQueryTests(unittest.TestCase):
    setUp = test_core.CoreTest.setUp
    result = test_core.CoreTest.result

    def test_existing_release_evaluation_remains_independent_of_archive_provenance(
        self,
    ):
        for result in (
            self.result(),
            self.result("baseline", "initial", 0.5),
            self.result("anchor", "initial", 0.5),
        ):
            self.evals.save_result(result)
        decision = self.release.evaluate(
            "decision",
            "candidate",
            candidate_result="candidate-result",
            baseline_result="baseline",
            anchor_result="anchor",
        )
        trace = self.release.candidate_archive("candidate")
        self.assertEqual(trace["decisions"][0]["verdict"], decision.verdict)
        self.assertEqual(trace["evaluation_status"], "evaluated")
        self.assertEqual(trace["costs"]["unknown"], ["job_cost_lineage_unavailable"])
        self.assertEqual(self.release.active()["bundle_id"], "initial")
        self.assertFalse(trace["qualification_inherited"])


if __name__ == "__main__":
    unittest.main()
