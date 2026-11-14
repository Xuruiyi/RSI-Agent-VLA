---
name: verify-change
description: Select and run existing RPent checks for a code, documentation, dependency, or robot integration change, and report validation evidence before handoff or a pull request.
---

# Verify a change

Read [CONTRIBUTING.md](../../../CONTRIBUTING.md),
[tests/README.md](../../../tests/README.md), and the affected CI workflow. Use
their current commands and prerequisites; reuse existing runners and fixtures.

## Select checks

- Identify changed behavior and its callers before selecting tests. Use focused
  cases during development, then the required repository checks before a PR.
- Core Python changes use `tests/unit_tests/rpent/`; robot adapters use
  `tests/unit_tests/robots/`. Unit tests run offline on CPU. Add or extend tests
  for observable new behavior or a concrete regression, rather than mirroring
  implementation details or creating a parallel test framework.
- Documentation changes use [docs-check](../docs-check/SKILL.md). Dependency,
  optional-import, CLI, or robot-discovery changes also need the relevant
  installation and startup checks in the affected environment.
- Env/VLA or runtime changes may need the real component and policy-chain
  tests documented in the test guide. Check GPU, assets, checkpoint, and
  platform prerequisites before launching them.

## Choose scope and environment

Use the changed behavior and its callers to select a focused set. A shared
contract change can justify broader integration checks; the full pre-PR checks
below remain required. Do not run a whole suite solely because a small edit
followed an already successful run. Recheck the affected behavior after that edit.

Reuse known environment evidence while it remains valid. Loopback/socket tests
need an environment that permits local sockets; role-isolation tests need the
supported namespace mechanism and a source/interpreter path visible inside it.
A `/tmp` source checkout can be hidden when an isolation test mounts scratch at
`/tmp`. Run such tests from their intended checkout. An unavailable prerequisite
is not a product regression; repair that prerequisite or report the precise gap.
Do not skip required checks or relax assertions to obtain a pass.

Record commands, outcomes and material gaps once in the existing owning record.
Ordinary checking does not require another manifest, proof file, review agent or
reconstruction of unrelated historical costs. Finish when the requested change
and its applicable checks are complete.

## Run existing commands

Use an isolated environment as described in CONTRIBUTING. Before opening a PR,
run from the repository root:

```bash
pre-commit run --all-files
pytest tests/unit_tests -v
```

Inspect any files changed by hooks and rerun affected checks. The
[unit-test workflow](../../../.github/workflows/unittest.yml) defines the full
Python matrix and additional Flywheel dependencies; one local environment does
not cover that matrix. Do not install every robot extra for an unrelated change.

For the supported GPU suites, follow the environment setup in the test guide
and call [the existing runner](../../../tests/e2e_tests/run_gpu_suite.sh):

```bash
bash tests/e2e_tests/run_gpu_suite.sh \
  <libero-pro|robocasa|robotwin> \
  /path/to/new-output-dir \
  /path/to/new-venv-root
```

Choose the actual target and fresh paths. Inspect the runner's logs, JUnit
results, and cleanup evidence. Real-robot diagnostics are separate opt-in
procedures with operator prerequisites; follow their documented scope rather
than treating them as ordinary automated checks.

## Report evidence

Record the revision or working diff, exact commands, relevant environment,
results, artifact locations, and checks not run with reasons. Distinguish a
failed check from unavailable prerequisites and from an untested path.

An import or health check establishes less than a real component request.
A bounded policy-chain check verifies action execution, recorded `finish`,
state/observation artifacts, and owned-process cleanup; it does not require
task success. Report native success separately when evaluating a task or
benchmark. Never infer it from the planner's prose or a clean process exit.
