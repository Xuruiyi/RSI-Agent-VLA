"""Reuse the existing Release Host graph for synthetic theta artifact publication."""

import os
from dataclasses import replace
from pathlib import Path

from rsi_vla.contracts import Candidate
from rsi_vla.midterm_host import MidtermHost, read
from rsi_vla.storage import atomic_json
from rsi_vla.theta_b import training_theta_version
from test_release_host import ReleaseHostTests


class ThetaArtifactHost(ReleaseHostTests):
    matrix_evaluation_kind = "theta_artifact_engineering"

    def configure_fixture(self, protocol, profile):
        protocol.update(
            protocol_id="theta-artifact-engineering-v1",
            benchmark_id="theta-artifact-engineering-release",
            theta_artifact_engineering="rsi-theta-artifact-engineering-v1",
            hard_constraints=["no_device_safety", "artifact_interface"],
        )
        protocol["task_sets"]["release"]["benchmark_id"] = protocol["benchmark_id"]
        return protocol, profile

    def prepare_candidate(self, base):
        self.source = Candidate(**read(self.source_candidate_ref))
        self.version = training_theta_version(
            training_result_ref=self.training_result_ref,
            load_result_ref=self.load_result_ref,
            scope_id=self.training_scope_id,
        )
        assert self.source.changes["theta_ref"] == self.version["theta_ref"]
        assert self.source.changes["processor_ref"] == self.version["processor_ref"]
        target = replace(
            base,
            bundle_id="changed",
            theta_ref=self.version["theta_ref"],
            processor_ref=self.version["processor_ref"],
            parent_bundle_id=base.bundle_id,
            diff_ref=self.source.changes.get(
                "diff_ref", self.version["training_result_ref"]
            ),
        )
        self.store.save_bundle(target)
        fields = {
            "theta_ref",
            "processor_ref",
            "phi_ref",
            "rho_ref",
            "psi_ref",
            "memory_ref",
            "backend_ref",
            "code_ref",
            "diff_ref",
        }
        changes = {
            key: getattr(target, key)
            for key in fields
            if getattr(target, key) != getattr(base, key)
        }
        self.store.save_candidate(
            Candidate(
                "candidate",
                base.bundle_id,
                target.bundle_id,
                self.protocol["protocol_id"],
                self.protocol["benchmark_id"],
                changes,
                list(
                    dict.fromkeys(
                        [*self.source.artifact_refs, str(self.source_candidate_ref)]
                    )
                ),
                [],
                dict(self.source.costs),
                dict(self.source.compatibility),
                "offline",
            )
        )

    def problem_document(self, protocol, tasks):
        groups = {row["task_key"]: row["group"] for row in protocol["tasks"]}
        return {
            "schema": "rsi-theta-artifact-problem-v1",
            "information_scope": "release",
            "training_result_ref": str(self.training_result_ref),
            "load_result_ref": str(self.load_result_ref),
            "training_scope_id": self.training_scope_id,
            "source_candidate_ref": str(self.source_candidate_ref),
            "tasks": [
                {
                    "task_key": row["task_key"],
                    "seed": row["seed"],
                    "criterion": "traced_training_handoff"
                    if groups[row["task_key"]] == "controlled"
                    else "parsed_interface",
                }
                for row in tasks
            ],
        }


def consume_training_publication(
    *, training_result_ref, load_result_ref, scope_id, candidate_ref
):
    """Execute the existing independent Host/Release chain, with no model service."""
    flow = ThetaArtifactHost()
    flow.training_result_ref = Path(training_result_ref).resolve(strict=True)
    flow.load_result_ref = Path(load_result_ref).resolve(strict=True)
    flow.source_candidate_ref = Path(candidate_ref).resolve(strict=True)
    flow.training_scope_id = scope_id
    try:
        flow.setUp()
        # The source graph may live under /tmp, which each role replaces with
        # private scratch. Grant these completed synthetic inputs read-only in
        # all Host roles; their write domains remain the existing publication domains.
        source_root = Path(
            os.path.commonpath(
                [
                    str(flow.training_result_ref),
                    str(flow.load_result_ref),
                    str(flow.source_candidate_ref),
                ]
            )
        )
        core = read(flow.core_ref)
        core["role_isolation"]["read_grants"] = {
            role: [str(source_root)] for role in flow.domains
        }
        atomic_json(flow.core_ref, core)
        flow.test_automatic_matrix_qualification_activation_runtime_binding_and_rollback()
        host = MidtermHost(flow.host_ref)
        receipts = host.coordinator().status()["receipts"]
        independent = flow.evals.get("results", "candidate-result")
        published = flow.store.get("candidates", "candidate")
        inherited = receipts["inherit"]["value"]["plan"]
        assert (
            inherited["context"]["execution_binding"]["bundle"]["theta_ref"]
            == flow.version["theta_ref"]
        )
        assert (
            inherited["context"]["execution_binding"]["bundle"]["processor_ref"]
            == flow.version["processor_ref"]
        )
        assert inherited["selection"]["mode"] == "deployment"
        assert published["costs"] == flow.source.costs
        assert str(flow.source_candidate_ref) in published["artifact_refs"]
        assert independent["intervention"]["real_model_compatibility"] == "unmeasured"
        assert independent["intervention"]["physical_safety"] == "unmeasured"
        assert independent["intervention"]["learning_gain"] == "unmeasured"
        closure = independent["intervention"]["candidate_cost_closure"]
        assert closure["candidate_costs_as_serialized"] == flow.source.costs
        assert closure["pending_cost_operations"] == []
        assert (
            closure["closed_serialization_operations"]
            == flow.source.compatibility["pending_serialization_cost_operations"]
        )
        assert (
            independent["intervention"]["candidate_acquisition_costs_included"] is True
        )
        assert (
            independent["costs"]["whole_job_wall_seconds"]
            >= closure["all_attempt_costs"]["whole_job_wall_seconds"]
        )
        assert (
            closure["all_attempt_costs"]["whole_job_wall_seconds"]
            >= flow.source.costs["whole_job_wall_seconds"]
        )

        assert set(independent["hard_constraints"]) == {
            "no_device_safety",
            "artifact_interface",
        }
        assert all(
            row["outcome_source"] == "synthetic_task"
            for row in independent["assignments"]
        )
        qualification = flow.manager.qualification(
            "changed", scope="offline", protocol_id=flow.protocol["protocol_id"]
        )
        assert qualification["kind"] == "evaluated_candidate"
        assert len(flow.manager.records.all("activations")) == 1
        assert len(flow.manager.records.all("rollbacks")) == 1
        return {
            "schema": "rsi-theta-artifact-publication-integration-v1",
            "independent_evaluation": "synthetic_theta_artifact_engineering",
            "qualification": "evaluated_candidate",
            "activation": "passed",
            "new_job_binding": "effective_runtime_selector",
            "rollback": "passed",
            "source_candidate_ref": str(flow.source_candidate_ref),
            "source_candidate_costs": published["costs"],
            "closed_candidate_costs": closure["all_attempt_costs"],
            "closed_serialization_operations": closure[
                "closed_serialization_operations"
            ],
            "real_model_compatibility": "unmeasured",
            "physical_safety": "unmeasured",
            "learning_gain": "unmeasured",
            "model_requests": host.ledger.read()["used"]["model_requests"],
            "gpu_seconds": host.ledger.read()["used"]["gpu_seconds"],
        }
    finally:
        flow.doCleanups()
