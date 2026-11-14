# Working on RPent

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for development conventions, including
module ownership and interface contracts, and for testing and pull requests.
Use the implementation and its callers to verify behavior; report disagreements
with documentation instead of assuming either is correct.

## Agent and model defaults

- Do not use sub-agents by default. When sub-agents are enabled, default to
  GPT-6.1-Sol (`gpt-6.1-sol`) with `high` reasoning effort.
- When configuring or running project agent calls, including planners, default
  to GPT-6.1-Sol (`gpt-6.1-sol`) with `medium` reasoning effort.
- RSI Research requests use GPT-6.1-Sol (`gpt-6.1-sol`) with `high`
  reasoning effort; Planner requests keep `medium`. Historical medium Research
  records remain valid for offline parsing, not new requests.
- Explicit task-specific model and reasoning settings take precedence over
  these defaults.

## Project context

- RPent is the primary workspace for ongoing project implementation. For
  research carried over from ReinFlow, consult its `项目文档/研究基调与方向规划.md`
  for goals and decisions and `项目文档/90_待办与执行状态.md` for research status
  and TODOs. Keep those records authoritative until explicitly migrated.
- Record repository responsibilities and interfaces as implemented; changing
  the workspace or documentation does not establish a code migration.
- Keep goals, decisions, evidence, and next steps in their owning project
  documents, with entry points here. Update them when handing work between
  sessions; avoid duplicate status records.
- When drafting RSI-Agent-VLA execution prompts, consult the
  [task-prompt reference](../项目文档/04_实施路线与待决事项.md#51-任务提示词编写参考)
  for bounded autonomy, recovery, and completion criteria. Apply the current
  task's explicit authorization to all execution.
  Prefer the linked [task-wide authorization guidance](../项目文档/04_实施路线与待决事项.md#阶段总授权与主代理自主调度):
  user-approved hard totals with soft phase allocations, applicable permits
  issued within that authority, and bounded recovery for actual faulted jobs.
  Use the linked [unexpected-problem guidance](../项目文档/04_实施路线与待决事项.md#目标范围内的异常处置与受控续程)
  for autonomous diagnosis, reversible repair, and preauthorized recovery.
  Ask again for material scope, hard-limit, expiry, or acceptance changes. This
  guidance grants no new dynamic authority and does not reopen closed campaigns.

## Repository map

- `rpent/planner/`, `rpent/session/`, and `rpent/prompt/`: planner adapters,
  execution sessions, and shared prompt construction.
- `rpent/tools/` and `rpent/memory/`: tool execution, observations, and memory.
- `rpent/robots/`: robot discovery, descriptors, shared runtime, and Env/VLA
  components. `robots/<robot>/` owns each concrete integration.
- `rpent/cli/`, `rpent/dashboard/`, and `rpent/evaluation/`: entry points,
  interactive execution, and run results.
- `tests/`: offline tests and integration checks; see [tests/README.md](tests/README.md).
- `docs/source-en/` and `docs/source-zh/`: paired user and developer documentation.

## Development principles

- Confirm the working branch and relevant upstream state before editing. Read
  the closest current implementation and both sides of changed interfaces.
  Preserve unrelated working-tree changes.
- Reuse existing configuration, registration, RPC, runtime, and artifact
  helpers. Prefer RLinf's supported Env/VLA implementations with thin RPent
  adapters.
- Base compatibility paths and new abstractions on actual consumers or
  supported data.
- Validate external inputs at the boundary that owns the operation. Keep
  internal control flow direct, avoid repeated checks of established state,
  and preserve useful errors. Make resource ownership and cleanup explicit.
- Keep optional simulator/model imports at their use sites so basic imports,
  robot discovery, and CLI help work without every robot extra installed.
- Follow neighboring code for naming and structure, use the project logger,
  and follow CONTRIBUTING's type annotation and docstring conventions.
- Write prose about current behavior and non-obvious constraints. Keep review
  exchanges and implementation history in PR discussions or migration notes.
  Translate meaning into natural Chinese and preserve technical identifiers.
- Keep prompts and tool descriptions focused on the model's task. Remove
  repetition without dropping required behavior or operational constraints.
- Test observable behavior and concrete regressions, reusing existing cases
  and fixtures where possible. Avoid tests that merely restate implementation
  details. Never weaken failing tests to make a change pass.

## Execution and evidence

- Confirm the interpreter, import paths, and relevant environment before
  environment-dependent work. Keep machine-specific paths and credentials in
  local configuration.
- Choose available, compatible GPUs for each job or service according to current
  memory and load. GPU numbers are runtime choices, not frozen experiment
  conditions; examples and historical device IDs impose no device restriction.
  Keep resource budgets and admission rules, and record the selected devices
  consistently in configuration, permits, and runtime evidence.
- Start with bounded offline checks. Package changes, training, GPU evaluation,
  batch rendering, downloads, and hardware access require explicit authorization
  covering the action. Reuse existing authorization within its scope.
- Coordinate file ownership across active sessions; use separate worktrees
  when edits would overlap. Preserve other sessions' changes.
- Select checks for the affected behavior and follow CONTRIBUTING's required
  checks before a PR. Reuse valid results; repeat checks only after relevant
  changes or new evidence. Use known socket, namespace and interpreter
  prerequisites to choose the test environment before running the checks.
- Keep each check tied to a concrete decision. Routine reversible repairs
  proceed within existing authorization; final publication conditions apply
  at publication. Limit failures to their supported impact and continue
  independent work. Do not require a new checklist or report for this judgment.
- Finish once the requested deliverable and relevant required checks are
  complete. Add a further audit, source copy or historical reconstruction only
  when it can change execution, budget, recovery or the stated conclusion.
- Distinguish static inspection, offline tests, simulation, and real-robot
  evidence. Report research results with their sources and conditions; a smoke
  test or historical result does not establish current task or baseline success.

## Task skills

Read the relevant skill when the task calls for it; ordinary edits do not
require loading every skill. Reuse instructions already read in this session
unless they change or the task needs a different source.

| Task | Skill |
| --- | --- |
| Explain and review a PR, or review an explicitly selected local diff | [review-pr](.agents/skills/review-pr/SKILL.md) |
| Check documentation, comments, translations, or user/model-facing prose | [docs-check](.agents/skills/docs-check/SKILL.md) |
| Select and run checks for a change, or prepare validation evidence | [verify-change](.agents/skills/verify-change/SKILL.md) |
| Add a robot integration or extend its Env/VLA/tool/runtime wiring | [add-robot](.agents/skills/add-robot/SKILL.md) |

Reuse repository tools and the commands documented in CONTRIBUTING and the
test guide. Report what ran, its result, and what remains unverified; keep
integration evidence distinct from benchmark task success.

Skill sources live in `.agents/skills/`. `.claude/skills` links to that whole
directory, and `CLAUDE.md` links to this file. Edit the canonical files so both
clients share the same instructions. Keep each skill self-contained in its
`SKILL.md` and link to existing repository resources.
