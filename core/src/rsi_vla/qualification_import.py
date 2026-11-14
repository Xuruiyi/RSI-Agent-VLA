"""Import an approved historical initial result without relabeling its run.

The operator approval is a protected input. Evaluation and release writers must
run inside the accepted OS role boundary; a JSON assertion alone grants no role.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path

from .contracts import ExperimentResult


BEHAVIOR_CHECKS = ("model", "processor", "phi", "legal_memory", "planner_input",
                   "actions", "collection", "execution_sources")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _readonly(path):
    return bool(os.statvfs(path).f_flag & os.ST_RDONLY)


def require_role(authorization, role):
    """Check the current mount boundary, including domain replacement parents."""
    domains = {name: Path(value).resolve(strict=True)
               for name, value in authorization["role_domains"].items()}
    required = {"operator", "research", "evaluation", "release", "runtime"}
    if set(domains) != required or role not in required or not _readonly("/"):
        raise ValueError("protected OS role boundary missing")
    if len(set(domains.values())) != len(domains):
        raise ValueError("OS role domains must be distinct")
    for name, path in domains.items():
        if any(other != path and path.is_relative_to(other) for other in domains.values()):
            raise ValueError("OS role domains must not overlap")
        if _readonly(path) != (name != role):
            raise ValueError("OS role write boundary mismatch: " + name)
        if any(not _readonly(parent) for parent in path.parents):
            raise ValueError("OS role domain parent permits replacement: " + name)
    protected = authorization.get("protected_paths", [])
    if not protected or any(not _readonly(Path(path).resolve(strict=True)) for path in protected):
        raise ValueError("source/interpreter/evidence protection missing")
    acceptance = _read(authorization["isolation_acceptance_ref"])
    if (acceptance.get("scope") != authorization["scope"]
            or acceptance.get("role_domains") != authorization["role_domains"]):
        raise ValueError("OS acceptance scope/domain mismatch")
    for name in required:
        probe = acceptance.get("roles", {}).get(name, {})
        if probe.get("exit_code") != 0 or not probe.get("report_ref"):
            raise ValueError("OS role acceptance incomplete: " + name)
        report = _read(probe["report_ref"])
        if (report.get("role") != name or report.get("scope") != authorization["scope"]
                or report.get("allowed_write") is not True
                or report.get("denied_writes") is not True):
            raise ValueError("OS role probe failed: " + name)


def deployment_authorization(reference, *, bundle_id=None, protocol_id=None, role=None):
    """Read the separate, time-bounded permission consumed by publication/runtime."""
    value = _read(reference)
    if value.get("version") == "candidate-upgrade-registration-v1":
        from .upgrade_bridge import upgrade_authorization
        if role is None:
            raise ValueError("candidate upgrade authority requires an isolated consumer role")
        return upgrade_authorization(reference, role=role, bundle_id=bundle_id, protocol_id=protocol_id)
    if (value.get("version") != "initial-deployment-v1" or value.get("authorized") is not True
            or value.get("publication_authorized") is not True
            or value.get("inheritance_authorized") is not True or not value.get("scope")):
        raise ValueError("explicit initial deployment authorization missing")
    expiry = datetime.fromisoformat(value["expires_at"].replace("Z", "+00:00"))
    if expiry.tzinfo is None or expiry <= datetime.now(timezone.utc):
        raise ValueError("deployment authorization expired or lacks timezone")
    if ((bundle_id is not None and value.get("bundle_id") != bundle_id)
            or (protocol_id is not None and value.get("protocol_id") != protocol_id)):
        raise ValueError("deployment authorization target mismatch")
    if role is not None:
        require_role(value, role)
        if (not Path(reference).resolve(strict=True).is_relative_to(
                Path(value['role_domains']['operator']).resolve())
                or not _readonly(Path(reference).resolve(strict=True))):
            raise ValueError("deployment authorization is writable by consumer")
    return value


def validate_compatibility(compatibility, *, source_bundle, source_protocol,
                           source_profile, bundle, protocol, profile):
    """Bind the operator's behavior review to the exact source/target snapshots."""
    expected = dict(source_bundle=source_bundle, source_protocol=source_protocol,
                    source_profile=source_profile, bundle=bundle, protocol=protocol, profile=profile)
    if (compatibility.get("version") != "initial-deployment-compatibility-v1"
            or any(compatibility.get(key) != value for key, value in expected.items())
            or any(compatibility.get("checks", {}).get(key) is not True for key in BEHAVIOR_CHECKS)
            or compatibility.get("historical_os_isolation_claimed") is not False
            or not compatibility.get("source_manifest_ref")
            or not compatibility.get("target_manifest_ref")):
        raise ValueError("deployment behavior compatibility is not established")


def import_initial_evidence(evaluation_store, *, import_id, approval_ref, authorization_ref):
    """Evaluation identity independently checks and imports operator-approved facts.

    Approval carries source_bundle/source_protocol/source_profile/source_result
    snapshots, plus original evidence references. Historical identities survive.
    """
    from .release import inspect_initial_result

    authorization = deployment_authorization(authorization_ref, role="evaluation")
    if evaluation_store.root.resolve() != Path(authorization["role_domains"]["evaluation"]).resolve():
        raise ValueError("evidence import store is outside evaluation role domain")
    approval_path = Path(approval_ref).resolve(strict=True)
    if (not approval_path.is_relative_to(Path(authorization["role_domains"]["operator"]).resolve())
            or not _readonly(approval_path)):
        raise ValueError("evidence approval must be protected operator input")
    approval = _read(approval_path)
    source_bundle, source_protocol = approval["source_bundle"], approval["source_protocol"]
    result = ExperimentResult(**approval["source_result"])
    if (approval.get("scope") != authorization["scope"] or approval.get("approved") is not True
            or approval.get("historical_os_isolation_claimed") is not False
            or source_bundle["bundle_id"] != authorization["source_bundle_id"]
            or source_protocol["protocol_id"] != authorization["source_protocol_id"]
            or result.result_id != authorization["source_result_id"]
            or not approval.get("evidence_refs")
            or any(approval.get("checks", {}).get(key) is not True for key in BEHAVIOR_CHECKS)):
        raise ValueError("operator evidence approval identity/scope mismatch")
    check = inspect_initial_result(source_protocol, source_bundle, result)
    if check["passed"] is not True:
        raise ValueError("imported initial evidence failed: " + check["reason"])
    value = dict(import_id=import_id, approval_ref=str(approval_path), approval_snapshot=approval,
                 authorization_ref=str(Path(authorization_ref).resolve()),
                 authorization_snapshot=authorization, initial_check=check,
                 historical_result_relabelled=False, historical_os_isolation_claimed=False)
    evaluation_store.put("protocols", source_protocol["protocol_id"], source_protocol)
    evaluation_store.put("results", result.result_id, approval["source_result"])
    evaluation_store.put("initial_evidence_imports", import_id, value)
    return value


def renewal_authorization(reference, manager, *, scope, role):
    """Validate fresh permission to reuse an unchanged, already effective version.

    This permission never initializes a version or imports a new qualification.
    The original authorization is checked as immutable historical evidence;
    only this separate permission supplies current time and scope authority.
    """
    value = _read(reference)
    if (value.get('version') != 'effective-version-renewal-v1'
            or value.get('authorized') is not True or value.get('scope') != scope
            or value.get('publication_authorized') is not False):
        raise ValueError('explicit effective-version renewal authorization missing')
    expiry = datetime.fromisoformat(value['expires_at'].replace('Z', '+00:00'))
    if expiry.tzinfo is None or expiry <= datetime.now(timezone.utc):
        raise ValueError('renewal authorization expired or lacks timezone')
    require_role(value, role)
    path = Path(reference).resolve(strict=True)
    if (not path.is_relative_to(Path(value['role_domains']['operator']).resolve())
            or not _readonly(path)):
        raise ValueError('renewal authorization must be protected operator input')
    active = manager.active()
    if (active.get('activation') == 'bootstrap' or active != value.get('active_snapshot')
            or str(manager.records.root.resolve()) != value.get('source_release_root')
            or str(manager.research.root.resolve()) != value.get('source_research_root')
            or str(manager.evaluations.root.resolve()) != value.get('source_evaluation_root')):
        raise ValueError('renewal effective source mismatch')
    qualification = manager.records.get('qualifications', active['qualification_ref'])
    if qualification != value.get('qualification_snapshot') or not qualification.get('evidence_import_id'):
        raise ValueError('renewal qualification source mismatch')
    if scope == qualification['authorization_snapshot']['scope']:
        raise ValueError('renewal requires a new task scope')
    for source in (manager.records.root, manager.research.root, manager.evaluations.root):
        if not _readonly(source):
            raise ValueError('renewal source evidence must be read-only')
    if not value.get('tasks') or not value.get('task_authorization_ref'):
        raise ValueError('renewal task scope missing')
    task_path = Path(value['task_authorization_ref']).resolve(strict=True)
    task_authority = _read(task_path)
    if (not task_path.is_relative_to(Path(value['role_domains']['operator']).resolve())
            or not _readonly(task_path)
            or task_authority.get('scope') != scope
            or task_authority.get('reuse_effective_version_authorized') is not True):
        raise ValueError('renewal task authorization source/scope mismatch')
    return value, qualification


def _historical_authorization_snapshot(reference, expected):
    """Read immutable qualification provenance; this grants no current authority."""
    value = _read(reference)
    if value != expected:
        raise ValueError('historical deployment authorization snapshot changed')
    return value
