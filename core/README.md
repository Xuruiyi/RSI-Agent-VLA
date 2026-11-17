# RSI-VLA 研究 core

`core/` 是聚合目录维护的研究与执行协调代码，要求 Python ≥3.10。基础合约、存储与离线状态转换使用标准库，入口为 `PYTHONPATH=core/src`；实际 Runtime、Research SDK 和原生数据 batch 使用各自匹配的安装依赖。实现入口见下文，运行状态、历史暂停与接续边界统一见[项目文档04 §1.22](../项目文档/04_实施路线与执行状态.md#122-f07f08-隔离补修与-f10f14-第三批2026-11-17)，检查证据见 [OFFLINE_CHECKS.md](OFFLINE_CHECKS.md)。

在聚合目录执行：

```bash
PYTHONPATH=core/src RPent/.venv/bin/python -m unittest discover -s core/tests -v
PYTHONPATH=core/src RPent/.venv/bin/python -m rsi_vla check-protocol config/pilot_v9.json
PYTHONPATH=core/src RPent/.venv/bin/python -m rsi_vla init \
  --artifacts /tmp/rsi-medium \
  --profile config/run_profile_v9_resultfix.json --protocol config/pilot_v9.json \
  --bundle initial-medium-v9-resultfix
```

`init` 默认使用协议的 `anchor_bundle_id`（v9 为 `initial-medium-v9-bounded-vla`，可显式 `--bundle`），保存不可按同一 ID 覆盖的运行 profile、ψ、合法记忆来源和 ρA/B/C 快照。RunBundle 引用保存后的 profile，不依赖以后对原配置的修改；模型大文件仍按 profile 里的 revision 和路径引用，不复制权重。上游记忆登记为外部先验。重复导入相同事实幂等；同一事件 ID 或回合序号提交不同事实会报冲突。

v9 profile 的聚合源码标识为 `aggregate-core-v0.5.0-bounded-vla`；包元数据仍为 `pyproject.toml` 的 0.4.0，旧 profile 未指定源码标识时继续使用 `aggregate-core-v0.4.0`。当前代码包括既有合约/存储/证据/研究/Gate模块，以及 `runtime.py`、`evaluator.py`、`training.py`、`research_tools.py`、`rule_updates.py` 和对应 adapters/tests。聚合目录现由独立 Git 仓库维护，RPent 仍有自身 Git 边界；研究归档需保存实际未提交差异和运行产物，版本标签不能代替实际源码绑定。

历史 low 日志应导入单独的 v1 store，不绑定 `initial-medium`；历史命令和证据见 OFFLINE_CHECKS.md §1—4。

`--result /path/to/result.json` 可接入 LIBERO 新 finalizer 的 `schema_version=1.0` 独立结果。adapter 检查任务身份、原生来源、结果内部语义以及与末步原生标记的一致性。原 runtime 未记录 run/bundle ID 时，把调用者绑定作为导入元数据并显式登记缺失；这不能充当正式发布的完整版本关联。旧 `environment_success`、Planner `finish`、`pi0_pick.success` 和 `terminated` 原样保留为不同来源；没有独立结果的旧轨迹结果为 `unknown`。媒体引用只包含实际存在的逐步文件。adapter 不导入 RPent，也不打开服务或设备。

`import-rpent` 可显式传 `--action-events` 与 `--protocol-id` 导入本轮动作 JSONL；适配器逐行核对身份和连续序号。

各模块的消费边界：

| 模块 | 行为 |
| --- | --- |
| `contracts.py` | RunBundle、ExecutionEvent、EvidenceView、ExperimentResult、Candidate、GateDecision |
| `storage.py` | 单 worker JSON 持久化、同 ID 冲突检查、候选实际修改对象核对 |
| `evidence.py` | L0 全保留，仅选择 L1 媒体；关系提交去重和失效历史 |
| `cooperation.py` | 固定 B 只选择授权现有 Worker；固定 C 按证据、权限和原预算选择检查/有界实验/暂缓 |
| `research.py` | 作业绑定起点，分项成本、预算终止、恢复不重置 |
| `evaluation.py` | 固定协议、独立发布结果、全指派矩阵重算、三态 Gate |
| `release.py` | 受信写入端，安全点激活、重读有效版、返回曾激活版本 |
| `runtime.py` / RPent `execution_arms.py` | 冻结 D/S/R 执行程序或绑定动态 A，沿既有 Toolkit/VLA 与结果链运行 |
| `research_transport.py` / `bounded_runner.py` | 保存业务请求、生成实际 SDK 输入并消费有界 Research 回执 |
| `relation_learning.py` / `relation_consumption.py` | 关系反馈、有限探索及实际消费者的独立核验 |
| `task_continuation.py` / `task_completion.py` | 从角色回执选择预登记后继，复用 Coordinator 与角色隔离 |
| `adapters/pi05_training.py` / `theta_b.py` | 原生 CPU 数据准备、训练交接与冻结 θ×B 评价计划 |
| `effective_psi.py` | 冻结实际研究程序、提示、ρC 与检索规则，事实记忆单独保存 |

`TaskState.update` 只接受已入账且同回合事件、合法观察/独立判据和完成规则引用。调用者负责执行该具体判据；工具结束或启发式标签本身不能作为完成提交。当前关系失效或未知不会删除原事件及提交历史。未知状态由 `ResearchJob.unknown_action` 按既有观察/恢复权限和剩余额度返回 `observe`、`recover` 或 `stop`。

固定 ρA 的执行入口是 `EventLedger.view`；ρB 是 `fixed_route(rho_snapshot, requested_worker, allowed_workers)`，未授权则返回 `defer`；ρC 是 `fixed_research_step(rho_snapshot, job, evidence_available=..., legal_repairs=..., inspected=..., experiment_permitted=...)`，只有合法修复集合非空、已检查证据且预算和权限允许时才返回 `bounded_experiment`。`rho_snapshot` 从该 RunBundle 的唯一 `rho_ref` 读取，配置类型或规则变成未实现策略时显式报错。固定快照保留原行为；显式反馈快照通过现有调用链消费有界经验表，ρA选择媒体引用、ρB选择注册兼容技能、ρC参与有预算/权限约束的研究准入。实际受控证据和未发布状态见04 §1.17；策略本身不授予额外权限。

`ResearchStore` 只能写开发结果和候选；`EvaluationStore` 写独立结果与冻结协议；`ReleaseManager` 从可信评价区读取结果并自行计算 GateDecision，研究者不能通过其 API 直接发布。部署时应让研究进程仅可写 research 区，独立 evaluator/release 进程拥有各自目录写权限。本实现是单 worker 的本地状态机，Python 对象分层本身不构成操作系统 ACL。

正式评价需要 candidate/current/固定 anchor 三份 `ExperimentResult`，逐份匹配 bundle、协议、benchmark、`role=release` 和实际证据层级；候选结果还必须匹配 candidate ID。开发结果不能升级为独立结果。对于有任务矩阵的协议，每条 `assignments` 包含：

```json
{
  "task_key": "base_salad", "seed": 3,
  "outcome": "success", "validity": "valid",
  "evidence_ref": "independent-episode-result.json",
  "outcome_source": "native_benchmark"
}
```

consumer 要求完整且不重复的预定 task/seed 集合，从每个槽的原生结果重算分母、unknown/invalid 和组内任务宏平均指标，并与传入指标核对。`verified_budget_exhaustion` 仅能支持有终局证据的失败。每项关键旧任务的 candidate 成功数至少 1，并不得低于 current 和固定 anchor。安全、兼容、关键推进和完整成本账本等具体验证由独立 evaluator 提供对应硬约束的 `true/false/null`；这里不把调用者填的布尔值当作已实施的仿真或安全检查。已知且身份匹配的硬违反优先拒绝，缺失/冲突/未授权未决，两者均不改变有效版。

`config/pilot_v9.json` 是独立 JSON consumer 的运行协议。其 `dynamic_authorized=false` 时不允许仿真发布；离线测试用独立 `scope=offline` 的小型 fixture 演示接受、继承和回滚。Gate 通过与激活分开，激活还要求相同 scope、基准未过期、安全点和队列已清空；恢复只能回到该发布历史中实际激活过的版本。新作业绑定新版，已启动研究作业仍引用原起点。

`ResearchJob.run` 只封装已知成本的同步有界离线动作，提前登记成本、额度不足不调度，耗尽后停止后续动作。它不抢占外部进程，也不负责 GPU/LLM/训练的实时监督；实际后端必须先实现对应资源的进程内监督和授权。生成数据放 `--artifacts` 指定的外部目录，不写入代码树。

离线检查验证状态转换与接口；实际 VLA、Research、训练、部署和 H1—H4 分别消费各自运行证据。具体批次的已验证范围和缺口见[项目文档04](../项目文档/04_实施路线与执行状态.md)。

历史 v9 静态开发计划使用 `run_profile_v9_resultfix.json / pilot_v9.json`，当时 Planner 和 Research LLM 固定 `gpt-6.1-sol / medium`；新 Research 请求使用 `gpt-6.1-sol / high`，Planner 保持 medium。v1 和历史 low 证据保持原样。下面只生成计划，不调用模型或启动服务：

```bash
scripts/run_libero.sh --artifacts /tmp/rsi-medium \
  --protocol config/pilot_v9.json --bundle initial-medium-v9-resultfix \
  --task-key position_swap_salad --seed 1 --split development \
  --run development-input-v1 --output /tmp/rsi-medium/run-01
```

先用上述 `init` 命令创建同一 artifacts 根。输出包括完整 argv、run/bundle/protocol 身份、预算和实际阻塞；`--execute` 需要对应授权与受支持的请求预算，默认配置不会启动。当前部署默认让 Runtime、Research 与训练共用 `/tmp/rsi-agent-vla-dynamic.lock` 的单作业额度；这是现行串行实现，后续隔离并发需另接设备/写域互斥与计量。实际外部监督入口为 `rsi_vla.adapters.runtime_supervisor`，不是离线 `ResearchJob`。

`rsi_vla.evaluator` 从独立产出的结果、动作 JSONL、成本与 supervisor 文件读取全部预指派槽；缺槽为 unknown，身份冲突为 invalid，low 结果不能进入 medium 基线。安全、兼容、关键推进和预算等未取得独立证据时保持 null；单纯收齐文件不会自动获得发布资格。受保护 evaluator/release 的操作系统写权限部署仍需在正式发布前建立；本地独立文件消费和不同角色 API 不构成 OS ACL。

2026-11-10 用户已授权动态推进，v3保留Codex，将token改为记录项；本地HTTP接点在转发前限制每回合20个实际模型请求，完整墙钟/工具/总环境步仍有硬停止。v1/v2及历史快照不改写。实际进展以04最新运行记录为准。

默认静态 consumer 使用 v7 工具60协议；本轮新输入/Runtime 的单回合计划显式使用 v9。v1—v8、历史 low 和5A证据保留。`scripts/run_research_review.py` 将合法开发视图交给一次受外部监督的真实 Codex Research LLM 请求，实际 model/medium 来自本地转发证据，token 为记录项。`--reconcile` 只校验已有响应，不重新调用模型。有限规则 v3 的 `planner_finish_reserve_calls=1` 只给 Planner 预算规划提示，实际上限从当次 context 读取，finish 仍经过原工具计数；不改变 θ、动作 chunk 或原生成功判据。

Runtime 在首次工具/环境步/turn 预算停止时保存同身份 `budget_stop.json`，允许已准入操作收尾，阻止后续模型转发。原始 SDK 状态与明确预算原因分开记录；完整原生负反馈和清理可形成预算失败，其它中断仍为 unknown。后续运行在 `code_snapshot/` 留存启动前的实际 Runtime 源码，不把未提交代码的版本标签当作完整源码证明。历史缺少明确 stop cause 的第三槽采用独立补件，原始结果和停止传播缺口均保留，详见04。


`libero-startup-v1` 在任务启动时传入操作边界、技能入口和三份匹配主题的原始上游记忆，完整指南/记忆索引按需查询；缺少当前任务专用记忆时直接登记缺失。`back_project_batch` 逐点调用原 `back_project`，保留顺序和逐点错误，不执行机器人动作。每次工具反馈的 `runtime_budget` 读取计费 owner 的快照，不再次扣费，缺失值为 null。

Launcher 在进程与传输清理后独立保存 `independent_result.json` 和 `cost_breakdown.json`，保留原 `result.json`；取消、超时、传输失败和缺证保持 unknown。分项计时注明包含/重叠，cached input 是 input 子集；不完整 usage 只有观测下界。新回合验收须消费独立结果和原始服务证据，不能只看 Planner finish 或退出码。

**历史状态（2026-11-12，第一批准备时）：** 当时唯一待运行计划为 [development_plan.json](../artifacts/offline-20261112-vla-result/development_plan.json)，源码/输入绑定见同目录 `source_manifest.json`、`task_input.json`。当时暂停且无该批动态授权；计划中的 permit 是未签发提案，未安排第二回合。公共执行基础优化不计为 RSI 自主改进。检查及限制见 [OFFLINE_CHECKS.md §8](OFFLINE_CHECKS.md#8-2026-11-12-vla-技术错误向最终结果传播修复)。


**历史状态（2026-11-12，第二批执行前）：** 当时唯一待运行版本为 v9 的结果传播修复：`config/pilot_v9.json` / `config/run_profile_v9_resultfix.json`，bundle `initial-medium-v9-resultfix`，代码 `aggregate-core-v0.5.1-vla-result-validity`，输入 `libero-startup-v2-bounded-vla`。v8 配置及历史快照保留其身份。`vla_execute` 在明确的 1—24 次预测额度内复用五步 chunk 连续执行；Planner 保留目标、监督和恢复职责。该批静态计划及当时暂停/许可缺口见项目文档04 §1.13；当前执行状态统一见04 §1.17。检查入口见 `OFFLINE_CHECKS.md §8`。

执行技术错误由 Toolkit 已存执行段和绑定回合的 action events 进入 finalizer；launcher 再独立检查推理失败、动作 RPC 错误或不确定反馈及技术停止事件。原生事实保存在原始结果和派生结果的 `native_task_result`，未解决故障的可评价结论保持 unknown/technical_invalid。acceptance 默认读取 `independent_result.json`，显式原始结果也须检查事件；evaluator 将已有原始路径切换到同目录的 launcher 收尾。轨迹导入保留原始独立证据，技术无效结论保持 unknown。尚未定义执行故障恢复判据，正常退出、后续工具成功或 Planner 叙述均不构成恢复；已有有证据的预算派生收尾继续保留。


第三批离线接线状态见04 §1.15，验证见OFFLINE_CHECKS §10。上述计划和暂停描述按各自历史时点阅读；当前授权、停止和阻塞统一见[04 §1.17](../项目文档/04_实施路线与执行状态.md#117-5a-数据经验有限往返与资格准备2026-11-13)，任务政策见[04 §5.4](../项目文档/04_实施路线与执行状态.md#54-任务政策与历史比较的作用域)。

正式研究导入（先用同一profile/protocol/bundle初始化新的版本化store）：

```bash
PYTHONPATH=core/src RPent/.venv/bin/python -m rsi_vla import-development \
  --artifacts artifacts/offline-20261112-batch3/state --protocol config/pilot_v9.json \
  --run-dir artifacts/development-v9-resultfix-position-swap-salad-s1-01 \
  --result-id development-v9-resultfix-s1-origin-v2
```

`development.import_development` 复用原事件、EvidenceView和ExperimentResult，仅接受冻结开发任务/seed和匹配身份。同store同run等值导入幂等，冲突报错；语义变化另用版本store。原事件、旧派生和历史成本保留，不重复charge。现行批次研究请求携带所选合法result的完整L0视图，可批量输入多个result。新协议可显式采用文末的 `selected_l0` 输入政策，权威L0继续完整留存。新旧视图出处见真实run下 `research-derivation-v2.json`。

`scripts/prepare_bounded_research.py --help` 是新有限研究输入/响应准备入口，无模型或实验调度；它要求已有预算job及人的domain policy，只接受已有实际消费者的规则算子（chunk前缀、native finish guard、finish reserve及显式获准的handoff hint），实际产物仍由ResearchTools/rule_updates生成。synthetic响应只作offline工程检查。旧 `run_research_review.py`/v1响应保留；不得运行其历史execute入口恢复队列。

普通部署使用 `scripts/run_libero.sh --deployment-release <trusted-release-root> ...`，任务边界读取ReleaseManager有效版；实验/评价/复现继续用显式 `--bundle`，二者互斥。未合格simulation bootstrap会拒绝部署选择。选中的模型/processor/规则/合法记忆进入现有Runtime参数，已绑定作业不重新选版。发布端仍须独立权限部署；本地目录分开不构成ACL。

`config/run_profile_eval_draft.json` 只准备四任务startup输入适配，未改变现行Gate或矩阵。评价建议和5A/5B阻塞统一见04及06，不据此启动任何动态作业。

## protocol10 本地阶段协调入口

GPU 按当前显存、负载和模型/环境兼容性自主选择，不预绑定 GPU0、GPU1 或固定卡号集合。每次启动可选择当时可用的兼容设备，设备编号不是冻结的实验条件。现有入口仍使用显式设备参数；启动前按实际选择准备一致的 `cuda_device`、permit `gpu_devices` 和运行参数，记录真实设备。旧配置中的卡号不是必须等待该卡的理由，预算、显存准入和单卡串行要求继续按适用协议执行。 对既有 campaign，用 `scripts/revise_protocol10_gpu.py --configuration <现有配置> --root <operator/protocol10> --revision-id <唯一修订身份> --devices <run到物理GPU编号的JSON>` 同步尚未提交的配置、具体permit和pending。修订在共享动态锁及协调锁内保留旧字节与操作身份；已启动、已有派发请求、完成或记账的作业拒绝切卡。首次派发前的必要来源补正可加 `--refresh-source-manifest`，旧来源副本由operator保留；之后将回执中的binding同步到admission，再先reconcile、后经guard首次执行。

`protocol10_coordinator.Coordinator` 保存控制进度，`protocol10_backend.Protocol10Backend` 将固定阶段接到既有消费者：

```text
baseline Runtime → 费用结算 → 单尝试评价 → 8槽起点评价/天花板判断
→ 唯一 bounded Research → 候选和 seed2 开发槽登记
→ development Runtime/结算/开发消费 → Candidate 提交
→ candidate Runtime/结算/单尝试评价 → 8槽候选评价
→ 独立 Gate → comparison_only 资格交接
```

Runtime 使用 `prepare_runtime/execute_runtime`；Research 使用 `bounded_runner` 和 `Protocol10ResearchAdapter`；开发使用 `research_cycle`；Candidate 使用 `ResearchTools.submit_candidate`；矩阵与 Gate 使用 `evaluate_matrix`、`apply_comparison_attempt_evidence` 和 `Gate`。开发循环通过 `auto_submit=False` 保留各槽 finding，最后一次提交带齐所有开发结果。protocol10 的完整起点 J=1 时结束该成功率正增益分支，Research 和候选启动数均为 0；新任务的其它研究目标须使用独立协议与评价计划。

从聚合目录使用已有解释器；以下是入口用法，不是运行授权：

```bash
RPent/.venv/bin/python scripts/run_protocol10.py status \
  --root /path/to/task-root/operator/protocol10 --configuration /path/to/new-registration.json
RPent/.venv/bin/python scripts/run_protocol10.py advance \
  --root /path/to/task-root/operator/protocol10 --configuration /path/to/new-registration.json \
  --inputs /path/to/current-admission.json
RPent/.venv/bin/python scripts/run_protocol10.py reconcile \
  --root /path/to/task-root/operator/protocol10 --configuration /path/to/new-registration.json \
  --inputs /path/to/current-admission.json
RPent/.venv/bin/python scripts/run_protocol10.py event \
  --root /path/to/task-root/operator/protocol10 --configuration /path/to/new-registration.json \
  --event /path/to/control-event.json
```

注册配置由具备该 scope 授权的调用方准备，入口不签发 permit。`protocol_ref` 与 `frozen_protocol_ref` 指向等值冻结协议；`runtime_protocol_ref` 是现有 Runtime 字段投影，`profile_ref` 和 `artifacts_root` 指向本 scope 的合法派生版本；`research.store_ref` 必须为 `artifacts_root/research`。生产侧 Runtime/tools 协议必须等于 `runtime_projection` 的确定输出，其行政授权显式允许 `release` 与所需的 `development`，冻结原协议不变。`runtime_slots` 登记完整 baseline/candidate 各 8 槽、可供 Research 选择的 seed2 开发身份及最多 2 个技术恢复身份，每条包含 `side/task_key/seed/bundle_id/run_id/output_ref/scratch_ref/permit_ref`；恢复条另有 `recovery_of`。登记可选身份不算实际启动，也不由 CLI 选择开发槽。

`research` 包含 `store_ref/job_id/request_id/domain_policy_ref/output_ref/scratch_ref/permit_ref/preflight_ref/cost_limit`，可指定 `tools_protocol_ref`。请求、ResearchJob、合法历史开发证据、领域策略、源码引用和具体 permit 必须先由既有准备消费者登记；Research 响应新增 `development_slots`，候选响应选 1—2 个不同 task/seed2 并保留顺序，`defer` 选空列表。独立比较资料不进入 Research 输入。已拒绝 hint 不复活，改变 chunk timing 不能沿用本协议的新起点；没有合法干预时结束分支。

`--inputs` 提供 `authorization` 和 `resources`；生产 CLI 用真实 UTC 覆盖文件中的 `now`，协调器 API 保留显式时间供受控测试使用。每次真实派发前还重新读取行政授权与具体 permit 的期限，不能用旧 admission 时间延长授权。动态授权包含 `dynamic_authorized=true`、与注册完全相同的 `binding`、`expires_at` 和本次 `operation_ids`；资源包含 `available/required/reserve` 逐维数值以及未决成本、资源阻塞。protocol10 作为本次完整比较批次，其冻结协议完整剩余包络还由权威阶段账重新准入；继承和恢复预留已在包络内，不能再重复相加。调用方通过 `offline_used_seconds/offline_used_bytes` 明确提供离线已用墙钟和字节。资源、授权、绑定不满足时保存等待条件并结束本轮，唤醒后重新核对。

事件 JSON 仅接受 `event_id/type/operation_id`，可加 `reference`；campaign 级信号的 `operation_id` 为 `null`。支持 `receipt_available/resource_changed/timer/authorization_expired/pause/resume/cancel/terminate`。同 ID 同值去重，冲突拒绝。事件引用只触发核对，不能指定命令、预算或权限；迟到的操作信号不会推进当前另一操作。`resume` 是显式控制操作，计时器不会解除暂停；cancel/terminate 后不重开 campaign。

每次 `advance` 最多派发一个操作，执行前保存 intent。`reconcile` 核对已提交操作并闭合证据，不派发新的 Runtime/Research；已有 Runtime 完成回执时评价修复只读原输出。控制目录中的 `progress.json` 保存阶段、操作、待处理身份、版本/授权绑定、回执引用和等待原因；`backend/registration` 保存不可变注册，业务账、Research findings、独立 evaluation 和正式 active 各由原消费者维护。状态摘要不充当预算或有效版权威。

生产配置的 `isolation_task_root` 划分 `operator`、`runtime`、`state/research` 和 `state/evaluation`。CLI 的宿主 broker 启动固定角色入口，每个进程由 `run_isolated_role.py` 的独立 mount namespace 限制为自身写域。operator 保存协调进度并请求结算，宿主 phase owner 是唯一阶段账写入方；Runtime 写模型传输和原始运行产物，Research 写请求消费、finding 与 Candidate，evaluation 只读原始事实并写独立评价。跨域交付核对操作、请求与角色回执，重复消费保留原执行及费用。技术失败的 owner/进程/请求清理闭合独立于技术有效性；非零退出仍保留 invalid，符合原恢复条件后才考虑恢复。

Runtime 和 Research 共享宿主网络及同一动态锁 inode；Research 不挂载 GPU。两者的 SDK 会话与缓存写入各自 scratch，`auth.json/config.toml` 通过只读凭据来源链接使用。permit 的 `storage_write_roots` 精确列出 output 与对应 scratch；Research 派发既遵守冻结的 180 秒上限，也保留既有 transport 更窄的存储上限。所有动态许可还要匹配注册的 `phase_ref/source_manifest_ref`、行政授权、操作身份和逐维额度。

具体生产配置为 `artifacts/continuation-20261113-protocol10/batch2/operator/configuration.json`。行政授权、适用 permit、实际来源冻结、阶段资源与当前 GPU 条件须同时满足；本轮状态见04 §1.17。准备派生 store 与推进命令如下：

```bash
# 从 /home/xzh/RSI-Agent-VLA 执行；数值从当时的离线实耗核准，不沿用旧余额。
PYTHONDONTWRITEBYTECODE=1 RPent/.venv/bin/python scripts/prepare_protocol10_registration.py \
  --configuration artifacts/continuation-20261113-protocol10/batch2/operator/configuration.json \
  --administrative-authorization artifacts/continuation-20261113-protocol10/batch2/operator/administrative-authorization.json \
  --source-research artifacts/continuation-20261113-protocol10/protocol-design/state/research \
  --offline-used-seconds <current-measured-seconds> --offline-used-bytes <current-measured-bytes>
PYTHONDONTWRITEBYTECODE=1 RPent/.venv/bin/python scripts/run_protocol10.py advance \
  --root artifacts/continuation-20261113-protocol10/batch2/operator/protocol10 \
  --configuration artifacts/continuation-20261113-protocol10/batch2/operator/configuration.json \
  --inputs artifacts/continuation-20261113-protocol10/batch2/operator/admission-inputs.json
```

准备入口不签发许可或派发请求。`admission-inputs.json` 必须在启动前绑定准备后的真实 registration、明确批准的操作身份、当前阶段资源及一次计算的强制预留；资源事实和阻塞原因按当前权威证据更新。Research 如需继续，先经现有工具登记合法开发证据、可执行算子、唯一 transport 和完整 input 绑定；无合法干预则零请求结束分支。开发 permit 的 experiment 身份取自唯一实际响应，在首个开发观测前固定；不提前虚构实验或候选。

推进锁串行化本 campaign 的 advance/reconcile；共享动态锁仍由 Runtime/Research 原消费者持有。status 原子读取当前状态，event 经独立锁持久保存控制信号，二者可在长 advance 期间并发执行。每个 CLI 使用独立 broker socket，只清理自身连接。运行中 pause 阻止后续派发，当前原生操作仍须闭合费用与清理；本入口不抢占执行中的工具。unknown 保留原身份，不能凭超时认定 GPU、请求或写域释放；须核对原 owner、supervisor、费用和清理。恢复只消费已预登记、故障已解决、费用与清理已闭合且有适用许可的技术恢复槽，有效 failure 不补跑。缺失费用保持未知，取得同身份真实新观测后仅追加缺维差额，保留原观测并拒绝修改已知金额。

离线端到端测试显式注入 fake Runtime/Research 进程与边界报告，使用全新临时账和 `scope=offline`；生产 CLI 拒绝 CPU factory；隔离 CPU fixture 仅在显式 `--cpu-boundary-factory` 下验证进程边界。comparison_only 的 accept 只产生比较交接，`release_eligible=false`，不会调用发布、起点初始化或 active 写入接口。当前工程/动态状态统一见 [04 §1.17](../项目文档/04_实施路线与执行状态.md#117-5a-数据经验有限往返与资格准备2026-11-13)。


### Autoloop 本地外部入口

Autoloop 使用本地源码和局部 Node24，通过既有 command adapter 调用 `scripts/run_protocol10_autoloop.py`。服务只监听127.0.0.1，独占 home、scratch 和 ingress/requests；responses、阶段账、active 和源码只读，私有 `/dev` 没有 GPU。独立 host 消费固定 RPC，再调用现有 core CLI 与隔离角色。外部事件只给登记身份和核验引用，文字不决定命令、目录、权限、预算、模型或发布对象；重复事件与 trigger 幂等，内容冲突拒绝。

历史 protocol10 专用实例的准备与启停命令保存在 `artifacts/continuation-20261113-protocol10/batch3/verification/A/production-templates/START.md`。以下命令保留其原部署入口；实际启停须符合该实例的当前授权及停止状态。新任务由宿主物化自己的配置和 permit，不复制历史 permit 启动：

```bash
bash artifacts/continuation-20261113-protocol10/batch3/verification/A/production-templates/start-host.sh &
host_pid=$!
bash artifacts/continuation-20261113-protocol10/batch3/verification/A/production-templates/start-service.sh &
service_pid=$!
# 等待当前 worker 和 host 请求闭合后停止；不清空已有数据和回执。
kill -TERM "$service_pid" "$host_pid"
wait "$service_pid"
wait "$host_pid"
```

当前生产模板部署模式为 `allow_advance=false`；这是本部署选择。其它获准部署可在 host 持续核对身份、预算、期限和人工暂停的条件下启用已有受限推进接口。当前模式下，本地入口提供状态、固定信号、等待和原操作核对；正式推进仍由 host 的受限 core CLI 消费已签发授权。本人回答只能选择卡片上的暂停或继续核验，host 会核对具体 card/about/by/choice；普通 timer/资源/回执事件不会解除人工暂停。正常等待、任务有效 failure、Gate reject、额度不足及提交未明均作为正常轮次回执保存，不映射成 Autoloop failed 来重新实验。通知不决定业务是否重做；未知提交先查原 core/角色回执。

### 独立候选部署资格桥接

`upgrade_bridge.py` 通过受保护的 `candidate-upgrade-registration-v1` 登记，把新的比较证据导入独立评价版本。原 protocol10、run/result/Gate 与 `comparison_only` 保持原样；比较起点 `initial-deployment09-protocol10` 和真实 active `initial-medium-release08-deployment09` 各保留身份。登记在正式结果前冻结九类行为绑定、逐文件源码差异、评分/硬约束、原 active/revision、新部署协议/候选身份、期限、写域与回滚条件。

原比较协议使用实际授权投影；桥接从其 frozen/administrative 引用精确重算，拒绝改变固定分母、标准或预算。research 角色只物化确定的部署候选身份与 profile 元数据，新候选以真实 active 为 parent；evaluation 只读原始事实，重算原比较 accept 和新部署 Gate，并独立检查当前 active 在新协议下的适用资格。每个原始运行保存实际 Candidate、bundle、backend profile、规则与引用快照；独立导入和资格建立前逐槽核对，评价后漂移须拒绝。release 核对精确 active/revision、导入、授权期限、安全点和清空队列，在共享动态锁中调用正常激活路径。 已验收发布保存不可变回执；故障回滚核对该回执、实际激活祖先、当前revision、未过期授权及登记触发条件，并独立验证目标完好。`release_root` 必须是原有效版的真实既有拥有域；不会在新空根 bootstrap，也不复制旧资格改 protocol。

生产 broker 固定 action 顺序为 `candidate_deployment_prepare`（research）、`candidate_evidence_import`（evaluation）、`candidate_upgrade_qualify` 与 `candidate_upgrade_activate`（release）。各自携带登记的 `registration_ref` 与稳定 identity；导入再指定 import_id，资格指定 qualification_id/decision_id，激活指定 safe_point/queues_empty。缺失候选、证据或适用映射，比较 reject/inconclusive，以及直接 comparison accept 均不能发布。

合法激活后的继承通过新部署协议及 `--deployment-release` 选择真实 active，不能用显式 `--bundle` 代替。该登记由目标协议内 `deployment_derivation.authorization_ref` 消费；不要把它传给专用于旧版续用授权的 `--deployment-authorization`。继承固定 position_swap_salad/seed2，任务有效 failure 不重跑。只有登记的安全/兼容性条件触发，才可回到本轮历史中真实激活过的原有效祖先；回到 deployment09 后恢复其原 store/protocol 与适用续用授权。


## 新任务的分段准入与探索准备

`task_policy.py` 为新协议提供显式选择，缺省继续使用原消费者行为；`protocol10` 及其 Runtime 投影拒绝附加新政策。新任务使用新的 protocol/scope 身份，保存自己的预算与准备输入。这里的准备接口不签发 permit，也不启动服务、模型、仿真或硬件。

协议中的 `task_policy` 支持：

- `schema: "rsi-task-policy-v1"`。
- `resource_admission: "staged"`：`admit_resources` 接受 `next_required/cleanup/reserve` 三份互不重叠的向量。只检查下一必要阶段、该阶段之外的清理和明确的强制预留；条件分支到达后再准入。所有向量使用 phase 的硬资源维度，已消费成本不再预留，检查不预扣费用。缺省 `whole_comparison` 保留完整剩余包络。
- `research_branches`：每项包含 `objective/criterion_ref`；`evaluation_ref` 可在独立评价阶段补齐，缺少该引用不阻断研究准备。现有四任务评分消费者支持 `success_gain/efficiency/robustness/new_capability/repair` 的入口判断；J=1 关闭 success_gain，能力不足时只开放 repair 或 new_capability。完整匹配事实和硬约束仍先核对。入口开放只允许准备研究；效率、鲁棒性和新能力的独立评价及发布规则要在对应协议中实现。
- `exploration`：登记 `hypothesis/criterion_ref/stop_condition/operators/assignments/cost_limit`，其中 criterion 为本协议 ID，assignments 必须属于 development。domain policy 选择 `research_mode: "exploratory"` 并列出 `rule_operators` 后，允许无匹配触发证据的假设采集。`result_ids` 可为空，preflight 和实验计划标为 `registered_hypothesis`；实际开发仍须匹配任务 permit、原 job 额度和所登记的槽。chunk timing 改动仍需新的匹配基线；handoff 仍需该对象的显式批准。候选提交仍消费真实开发 finding。新任务及技术诊断只需一条完整可检验解释，有实际竞争解释时再补充；历史普通请求和 protocol10 继续要求两条，诊断证据仍不能直接作为行为试验前提。
- `research_per_request`：可登记每请求 `wall_seconds/storage_bytes` 上限。真实 transport 仍是一请求一身份一 permit；多轮研究在原 job 总额内使用不同请求及适用 permit，不自动重试旧请求。未登记时保留 300 秒/10 MiB 的既有单请求提案，protocol10 保留自己的更窄额度。
- `gpu_admission`：登记 `required_memory_mib/safety_margin_mib/minimum_compute_capability`，可用 `compatible_names` 收窄硬件兼容性。`select_gpu` 及 `run_release_comparison.gpu_admission(policy=...)` 按新鲜空闲显存、兼容性和负载选择；旧调用仍保留 12288 MiB 政策。
- `resource_wait`：登记 `max_wait_seconds/check_interval_seconds`。等待耗尽返回 `waiting_resources` 检查点和累计等待状态，不关闭任务。恢复需要资源、仍适用的授权及非人工暂停/取消状态；等待计量由调用方记回原离线池。原 release08/protocol10 的等待额度保持原约定。
- `evidence_input`：`mode: "selected_l0"`，显式填写 `reason` 与覆盖本次所有 result 的 `event_ids: {result_id: [event_id, ...]}`。输入记录来源、总事件数、选中事件及遗漏范围，原 EvidenceView 不变。`ResearchTools.read_development_events(result_id, event_ids=None)` 默认返回事件索引；指定 event_ids 时只读合法开发结果绑定的原始事件，拒绝 release/sealed 和未知事件。新请求可通过下文 `evidence_requests` 由宿主消费补证，SDK 单次调用保持无工具。
- `research_input`：`mode: "readable_v1"`，登记 `max_read_requests/max_events_per_read`，由 `readable_input` 将原始事实、选择、遗漏索引、缺失和媒体引用呈现为可读 JSON。未启用时保留既有 transport 语义。

从聚合目录准备下一阶段：

```bash
RPent/.venv/bin/python scripts/prepare_task_stage.py \
  --protocol /path/to/new-protocol.json --phase /path/to/phase.json \
  --request /path/to/stage-request.json --output /path/to/stage-preparation.json
```

stage request 提供 `repository_root/artifacts/run_id/bundle_id/task_key/seed/split/output_dir/cuda_device/cleanup/reserve`。脚本调用真正的 `prepare_runtime`，从其 limits 派生下一回合向量（Planner timeout 包含在 wall 内），保留 Runtime blockers。可另传 `gpu_cards/waited_seconds/offline_remaining_seconds`，按政策选择设备或保存资源等待检查点。硬件快照只供准备；派发前仍由真实 host 核对新鲜资源、授权、预算和暂停状态。

通用准备接口可在不同请求和阶段间复用；新任务的结果驱动后继由下文 `TaskContinuationBackend` 消费，protocol10 保留固定链及历史终态。隔离并行和新领域算子按实际消费者与计量条件启用，具体后续项见04 §5.4。


检查和交付按[04 §5.1](../项目文档/04_实施路线与执行状态.md#51-任务提示词编写参考)选择范围。新任务政策按需启用，普通修复无需先配置研究、评价和资源等待字段。准备入口与真实SDK写出的response schema按当前任务生成相同的解释数量要求，历史响应继续按其保存的请求消费。监督优化只减少消失进程组及清理循环中的冗余扫描，仍检查存活成员、原始硬期限和清理结果。

### completion11 的有限文本提示

新任务可在显式领域政策和 `rsi-task-policy-v1` 下允许 `planner_task_hint`。真实Research提交的非空文本最长1200字符，由phi v5与其他既有规则共同保存；Runtime将其追加到Planner启动输入并在run_start记录实际文本。它不改变工具、动作chunk、资源上限或原生成功定义。缺省无hint时，两版启动输入与原实现逐字一致。旧protocol10不展示或接受这个新算子。

实际新任务由受限角色消费者 `task_completion.py` 接通多请求、开发、Candidate及独立矩阵/Gate；具体预算和恢复仍由已登记scope、permit及host准入核验。升级继续经过独立 `upgrade_bridge`，开发或trial不等于正式Candidate、比较accept或发布。当前权威断点见04 §1.17；旧protocol10保持终止。


## 新任务的实际执行与研究消费者

本节说明接口与证据去向。各臂运行、Research、费用和清理的当前状态统一见[项目文档04](../项目文档/04_实施路线与执行状态.md#framework15近期框架优化交付2026-11-15)。

### D/S/R/A 整回合执行

[execution_arms.py](../RPent/robots/libero/execution_arms.py) 的 `libero-execution-arm-v1` 规格由 `validate_execution_arm` 检查，`runtime.prepare_runtime` 将其冻结到 Runtime context。D/S/R 经现有 Flash robot hook 调用 `run_execution_arm`，直接沿 `Toolkit.execute_tool("vla_execute", ...)`、原生 VLA processor、动作 RPC、journal 和 finalizer 执行；A 继续使用 Codex Planner。这里的 Flash hook 消费冻结程序，不启动在线 Planner 或历史 Flash 计划。

| 臂 | 实际消费者与信息 |
| --- | --- |
| D | 每个 chunk 使用原始任务指令和最新原生图像/状态；禁止阶段指令替换和ρB路由 |
| S | 按冻结列表执行技能指令、参数、公开状态完成条件和阶段额度；顺序不在线修改 |
| R | 读取公开 EEF/夹爪状态，按冻结的 `on_complete/on_timeout/on_unknown` 选择分支或 `stop` |
| A | 现有动态 Agent 选择子目标、监督、恢复和停止；Planner 消费合法运行输入 |

S/R 的条件由 `_observe` 实际读取，独立评价私有真值不进入执行程序。四臂继续绑定模型、processor、动作前缀、物理预算、原生结果、技术有效性与成本；程序用尽而缺少原生终态时保持 unknown。D 的原始指令、零模型请求和原生 success 已有[独立消费验收](../artifacts/framework15-20261115/state/evaluation/audits/framework15-D-base-s0-recovery1-zero-request-v2.json)；各臂的完整尝试记录见04。

### 可读证据、SDK 输入与宿主补证

[research_transport.py](src/rsi_vla/research_transport.py) 的 `readable_input` 与 `model_prompt` 供 `BoundedResearchAdapter.prepare_request` 和 [bounded_runner.child](src/rsi_vla/bounded_runner.py) 实际消费。`bounded_requests` 保存业务内容，`bounded_transports` 保存可逆传输；`sdk-input.json` 保存实际传给 `thread.turn` 的 prompt、模型、推理配置、开发者指令和响应 schema。返回后的 `sdk-input-consumption.json` 关联请求与实际 HTTP 计量证据；HTTP 序列化和模型实际关注内容仍为unknown。

当前 Research 为 text-only：媒体路径、文字说明和图像字节分别记录，`image_bytes_sent=false`，视觉内容未知。ρA 的媒体引用选择进入模型输入；视觉诊断质量和实际媒体存储节省须有各自证据。

启用 `readable_v1` 后，响应可在 `defer` 中提交 `evidence_requests: [{result_id, event_ids, reason}]`。`BoundedResearchAdapter.apply_response` 经 `ResearchTools.consume_evidence_requests` 读取精确合法开发事件，保存 `research_evidence_supplements`；`prepare_evidence_followup` 合并原选择与补证事件，生成新请求身份及父请求链接。后续派发仍核对新 permit、原 job 剩余额度和宿主准入。

### 上下文ρC与有效ψ

[relation_learning.py](src/rsi_vla/relation_learning.py) 的 `contextual_research_policy` 创建显式新版本ρC，按任务、算子、证据类别、可执行版本和数据条件分开上下文。`ResearchTools.research_relation_admission` 保存选择、理由及有限探索预留；`record_contextual_research_feedback` 关联实际回执，分开技术有效、筛选通过、任务收益、旧能力与成本。`learn_relations` 不把技术无效计成任务筛选负奖励，冻结继承时保留已用探索额度；技术、数据、预算及权限准入继续先行。旧 `rsi-relation-feedback-v1` 快照与筛选目标按原语义读取。

ρA 继续优化记录事件覆盖与引用数量。`relation_consumption.evaluate_relation_consumption` 分别报告引用消费、实际 SDK prompt 字节、媒体写入和诊断质量的已知与未知。[历史回放](../artifacts/framework15-20261115/state/research/relation_replays/framework15-historical-v1.json)提供合法旧快照的决策依据，未执行分支收益保持未知。

[effective_psi.py](src/rsi_vla/effective_psi.py) 的 `freeze_effective_psi/consume_effective_psi` 冻结实际程序、提示、ρC、检索、依赖、来源和开发成本，事实写入独立 `fact_memory_snapshots`。新请求通过 `domain_policy.effective_psi_ref` 绑定快照；没有实际策略变化就不创建后期ψ。[初始物化回执](../artifacts/framework15-20261115/state/research/policy_materialization/framework15-initial-complete.json)绑定独立 `framework15-research-base`，保留四臂 baseline。预留任务族和机制相对本轮开发划分，外部先验与历史暴露单列。

### 从回执选择后继

[task_continuation.py](src/rsi_vla/task_continuation.py) 的 `TaskContinuationBackend` 在 `rsi-task-completion-v1` 配置内消费有限 `continuation` 图，按原角色回执 `value` 的字段等值条件选择后继。操作继续交给 [task_completion.py](src/rsi_vla/task_completion.py)、既有 `Coordinator` 和隔离 `run_role`；pending、operation、receipt 与 reconcile 保持原提交关联。补证准备、Research反馈、有限试验、暂缓与结束是登记操作或终态，分支条件不提供新命令、permit 或额度。

宿主先物化新 scope、注册配置、准入输入与适用 permit，再从聚合目录调用：

```bash
RPent/.venv/bin/python scripts/run_task_continuation.py status \
  --root /path/to/new-task/operator/continuation \
  --configuration /path/to/new-task/continuation-registration.json
RPent/.venv/bin/python scripts/run_task_continuation.py reconcile \
  --root /path/to/new-task/operator/continuation \
  --configuration /path/to/new-task/continuation-registration.json \
  --inputs /path/to/new-task/admission.json
RPent/.venv/bin/python scripts/run_task_continuation.py advance \
  --root /path/to/new-task/operator/continuation \
  --configuration /path/to/new-task/continuation-registration.json \
  --inputs /path/to/new-task/admission.json
```

`status` 返回检查点，`reconcile` 核对原操作，`advance` 在当前准入允许时派发后继。`event --event <signal.json>` 提交登记控制事件；`drive --inputs <admission.json>` 连续推进并在有限等待政策内重读资源输入，人工暂停、终止和提交未知仍返回检查点。Autoloop 使用 [autoloop_bridge.py](src/rsi_vla/autoloop_bridge.py) 的固定 `core_consumer="task_continuation"` 适配同一 CLI，不另建任务平台。历史任务及 permit 保留其原停止、期限和权限。

### 原生 CPU 数据准备与θ×B交接

[adapters/pi05_training.py](src/rsi_vla/adapters/pi05_training.py) 的 `prepare_cpu_demonstrations` 贯通固定来源示范、LeRobot 子集、匹配 RLinf/OpenPI 配方和原生 CPU dataloader；入口为 [prepare_pi05_training.py](../scripts/prepare_pi05_training.py) 的 `--cpu-pipeline`。许可、来源、动作语义、processor、归一化、评价隔离和产物计量随作业保存。[准备回执](../artifacts/framework15-20261115/training-preparation/readiness.json)确认真实 batch 已通过，数据仍为 `not_admitted_interface_batch_only`、训练步数为0；自主轨迹的 `autonomous_execution/not_admitted` 保持原来源。实际训练须另有示范准入与训练授权，checkpoint、原生加载和独立能力评价分别交接。

[theta_b.py](src/rsi_vla/theta_b.py) 的 `freeze_theta_b/prepare_theta_b_cell/evaluate_theta_b_cell` 与 [prepare_theta_b.py](../scripts/prepare_theta_b.py) 冻结 B0/B1 内容、θ及processor；通过现有 `prepare_runtime` 准备单元，显式记录适用性unknown及请求/实际B的fallback。θ不存在时返回 `blocked_theta_unavailable`。[接口检查](../artifacts/framework15-20261115/training-preparation/theta-b-interface-check.json)只证明冻结和交接；未来θ更新后的适配收益仍需独立实验，生产准入继续走原评价/Gate。

## F01—F04：冻结、Host CPU 接续与读写边界

历史 RunBundle 字段及序列化保持兼容。`version_binding.bundle_references` 按 execution/research/facts/evaluation 组织原引用，同一ρ记录不复制成两套权威。`prepare_runtime` 保存实际 profile、θ/φ描述、processor、既有 norm JSON 内容和策略/事实记录；`execute_runtime` 启动前重新核对，漂移时拒绝原作业；RPent `RuntimeJournal` 在实际读入规则时再次核对冻结φ，已启动journal保留原规则。实际模型路径、归一化路径、配置和动作由既有 loader receipt/`runtime_acceptance` 核对。新 Research 请求保存 `version_binding` 与有效ψ；`bounded_runner.validate_permit` 在转发前重核，旧请求读取不要求新增字段，新ψ不改旧请求。训练、θ×B、评价/资格继续消费既有 permit 和实际加载证据；F11/F13 新生产者及新发布接线仍归后续包。

Host 保持 Coordinator（意图/pending/后继）、各业务 Store（研究/评价/发布）和 Host（准入/进程/监督/计量）的原权威。所有已登记操作复用同一 `scripts/run_midterm_host.py serve/worker` 入口；未知提交先对账，人工暂停不被资源/定时事件解除。CPU 业务失败/defer/reject/inconclusive 是原生结果，进程崩溃、超时或清理未知保持技术状态，不转换成可重提的成功。

### 显式 CPU model0 作业

新 scope 除 `authorized/status/owner_uid/期限/写域/outer_limits` 外声明 `cpu_authorized=true` 和准确的 `cpu_operation_ids`。模型/GPU上限可为0，`dynamic_authorized=false` 不代表 CPU 已获准；CPU permit 在 Host 派发、实际 supervisor 及固定业务消费者处核验。`mandatory_reserve` 可在scope或登记 Host template 中声明，准入将其计入一次，实际消费/占用释放/累计费用分开。清理不会退款，缺失成本保留unknown。

`build_configuration` 支持如下登记 operation，默认调用固定 [run_cpu_operation.py](../scripts/run_cpu_operation.py)，不接受模型返回任意模块或命令：

```json
{
  "operation_id": "freeze-b0", "kind": "host_command", "stage": "engineering",
  "dynamic": false, "cpu_purpose": "baseline_freeze", "version": "b0-v1",
  "payload": {"manifest_ref": "/task/operator/arm-bindings.json"}
}
```

生成后按实际任务在已登记 Host 配置中设置 `budget`、`supervisor_limits` 和完整 `storage_write_roots`；它们须在scope/phase剩余和期限内。默认180秒/64MiB只是准备值，不能覆盖硬额度。运行中的配置身份不可改写。现有六类数据/教师/训练/load fixed families 原消费者及计量保留。

| `cpu_purpose` | 固定消费者；所需 payload | 写角色 |
| --- | --- | --- |
| `analysis` | `exact_weighted_paired`；manifest_ref 的 differences | operator/evaluation |
| `baseline_freeze` | `freeze_midterm_b0`；manifest_ref 的四臂真实 binding | operator |
| `data_review` | `audit_demonstration_payload`；manifest_ref | evaluation |
| `result_import` | `EvaluationStore.save_result`；manifest_ref、evaluation_store_ref | evaluation |
| `scientific_evaluation` | `compute_admission`；registration_ref、可选 contrast | evaluation |
| `qualification` | `ReleaseManager.qualify_initial`；release_root、research_store_ref、evaluation_store_ref、qualification_id、result_id、可选 evidence_import_id | release |
| `successor_prepare` | `inheritance_plan`；registration_ref（保留其角色/授权要求） | runtime |

所有CPU命令都隔离网络与设备，`cpu_role`只允许已登记可信角色；资格创建仍执行原独立判据，固定入口不提供activate。业务输出为 `artifact.json` 与绑定scope/operation/version/终态的 `rsi-cpu-operation-result-v1`。工程probe也使用此permit；新部署scope需明确包括 `<scope>-engineering-probe`，准备入口不扩写原授权。

### Receipt-bound late refs

Host template（或由配置工厂复制的operation）登记生产者 exports 和后继 late_refs：

```json
{
  "exports": {"trained": {"field": "theta_ref", "version": "theta-v1", "type": "path"}},
  "late_refs": {"theta": {"operation_id": "train", "export": "trained", "version": "theta-v1"}}
}
```

exports 属于前驱template，late_refs属于后继template；两者各有对应 `version`。Host 核对原operation、scope、监督终态/清理、结算actual、版本和允许字段，再从原生回执取值，写入后继 `jobs/<id>/inputs.json`。path 必须实际存在于scope内前驱登记的写域；任意JSON路径、命令、许可或设备字段不能成为导出。相同身份内容不等时拒绝覆盖；原回执漂移在消费点拒绝。

CPU 固定命令通过 `RSI_OPERATION_INPUT_REF` 读取独立输入，通过 `RSI_CPU_PERMIT_REF` 读取原permit。可选 `target: ["payload", "manifest_ref"]` 或 `["permit_template", "model_path"]` 仅填列明数据字段，不能覆盖不同的原值，也不能改argv/授权。现有role消费者收到匹配的已物化payload；六类fixed job继续消费原family permit，原生trainer/loader的`handoff_ref`可作为版本化文件导出。训练/加载与发布生产者使用同一晚绑定契约，实际接口见下文F10—F14；通用绑定器与具体完成度分别验收。

### 隐藏材料和候选边界

`configuration.role_isolation.hidden_paths` 登记私有协议、标签和未公开元评价路径；`read_grants[role]` 只暴露审核过的公开材料。`run_operator/run_role` 均消费此策略，隐藏其它角色scratch、评价共享回执及其它operator请求。评价/release为可信消费者，Runtime/Research只拿合法输入；清空继承环境后按实际运行依赖提供白名单，原SDK认证仍只读链接，不复制到研究产物。

含隐藏协议的新任务配置使用 `public_protocol_ref/public_frozen_protocol_ref`；可另提供 `public_configuration_ref/private_configuration_ref` 去除私有顶层配置。没有独立公开配置时须显式 `configuration_is_public=true`。Host核对公开协议身份、原字段与授权；tasks/task_sets仅可取原文子集，预算/策略不漂移。真实子进程已用公开task_sets准备Runtime并消费TaskCompletionBackend，私有原件保持不可读。legacy protocol10按原契约读取，不自动重写历史协议。

`run_role(config, {"action": "code_candidate_execute", "execution_id": "check"})` 仅消费 `code_candidate_executions` 中固定的 execution_id/source_ref/input_refs/read_grants/timeout_seconds。候选收到固定 `--output/--input`，在empty-root、独立网络/PID、私有 `/dev`、清空环境中执行；每execution有独立scratch，只能写其output，Host在候选域外写回执。没有GPU、设备、broker、SDK凭据或发布写域；启动标记无返回时拒绝重提。候选生成、补丁选择与反馈沿下文F05—F09接口接入。

真实namespace检查须在支持bwrap/user namespace的宿主环境执行；受限开发sandbox的NETLINK_ROUTE EPERM不作为产品回归。具体检查及进程边界见 [OFFLINE_CHECKS §34](OFFLINE_CHECKS.md#34-f00f04-第一批工程验证2026-11-17)，当前状态见 [04 §1.20](../项目文档/04_实施路线与执行状态.md#120-f00f04-第一批工程交付2026-11-17)。本批不激活正式发布，不解除历史暂停。

<a id="f05f09执行契约证据出口与研究候选"></a>

## F05—F09：执行契约、证据出口与研究候选

实际接口映射见[项目文档01 §3.4](../项目文档/01_系统设计与框架复用.md#34-f05f09-实际消费者2026-11-17)，当前状态见[04 §1.21](../项目文档/04_实施路线与执行状态.md#121-f03f04-补修与-f05f09-第二批2026-11-17)，检查见[OFFLINE_CHECKS §35](OFFLINE_CHECKS.md#35-f03f04-补修与-f05f09-第二批工程验证2026-11-17)。LIBERO字段与维护示例见[RPent开发说明](../RPent/docs/RSI_FRAMEWORK.md)。

`prepare_runtime`把profile的可选`skill_contracts`、`l1_media_policy`与实际θ/processor/φ/ρ/代码引用冻结到context。Skill由Planner启动输入、ρB选择后的Toolkit准入及结果入口共同消费。条件只读当前公开policy图像和EEF/夹爪状态；工具结束、公开子目标判据和原生任务结果分别记录。五步前缀不变。动作反馈未知需要对账；新观察可使旧交接关系失效。私有评价文件与独立物体判据不交给运行决策。

L0保持完整，`EvidenceView.selection`另列引用选择、概率、零覆盖和断流。`EnvState.save`在明确启用L1策略时先选择，再按实际编码临时文件检查可选媒体占用。只允许开发范围的冻结Flash D/S/R且没有关系控制时筛选不参与控制的高分辨率world派生物。控制、终局、审计与评价媒体保留。`review_media_persistence`核对真实文件和回执；引用数减少不等于字节节约。持久占用、临时占用、累计下界分列，采样间隙下的精确累计保持unknown。评价记忆使用运行绑定的冻结副本，拒绝合并/同步/索引写入；导出区分自主、公共示范、教师与独立评价来源，成功不自动形成训练准入。

`ResearchTools.discover_research_opportunity/consume_research_plan/record_plan_feedback`把四类机会接到可信operation模板和原回执。提案只有capability/输入依赖/成本/预测/停止与产物选择，不能提供命令或许可。`install_research_plan`生成新任务配置的有界后继图，并核验/沿用原完成回执；原pending、未知效果和人工暂停不被覆盖。随后由既有`build_configuration`和`MidtermHost`派发；每个新operation仍须经过scope与permit核验。text-only缺少必要视觉内容时暂缓；未授权训练只有提案，没有派发。

`prepare_code_candidate/execute_code_candidate`只允许一个`select_action(context)`研究规则插件。独立副本、源码差异、AST/接口检查后，沿现有无设备namespace执行。程序输出仅为`inspect_evidence/bounded_experiment/defer`，不能修改治理或正式评分。技术有效可形成ψ Candidate；收益仍未测。原执行回执保存wall和实际StorageMeter观测，Job保留失败尝试与元数据费用，重复读取不再次执行/计费。新Research Job绑定新的ψ程序；`TaskCompletionBackend.research_prepare`实际执行它并据结果分流，`bounded_experiment`仍须通过原preflight和许可。

| 更新对象 | 输入与有限动作 | 更新规则与反馈目标 | 冻结、停止与消费者 |
| --- | --- | --- | --- |
| φ | 已登记开发证据；一种调用提示/阶段或既有规则修订 | `write_rule_trial`每次选择一个实际干预；技术、任务结果和成本分开 | 不变动作时序；新Bundle/规则文件由RuntimeJournal和Planner消费，旧作业保持快照 |
| ρA | 合法L0及媒体引用；all_available/one_per_event | 有界经验均值，目标为事件覆盖减引用成本；字节和视觉质量单列 | `learn_relations/freeze_relations`后接引用与真实L1出口，受保护证据不受选择影响 |
| ρB | 独立起态/交接反馈与执行版本；兼容的已注册Skill | 版本匹配的经验均值；未知不生成成功/失败标签 | 新Runtime真实路由消费；不兼容回退/暂缓，不绕过Skill准入 |
| ρC | 任务/算子/证据/执行版本/数据条件；检查、有限探索或暂缓 | contextual经验统计，技术有效与筛选分别更新；任务收益单列 | 探索预留随冻结继承；预算、技术和数据准入先行，实际Research准入消费 |
| ψ | 冻结起点与受限规则源码；三个合法研究动作 | 一种程序修订，经语法/接口与有界试验筛选；技术有效不等于研究收益 | 生成Candidate与ψ程序，后继Job实际执行；源码漂移、未知或超额拒绝，旧Job不改 |

`ResearchToolkit`提供`query_candidates/select_parent`；操作物化与代码执行仍属可信调用。`candidate_archive`沿已有Candidate/base Bundle/changes解析假设/实验/来源、全部Job账、失败分支、依赖与评价状态。历史父节点有保留产物和有限选择上限；`inherit_parent`只接纳已经独立准入、绑定所选Bundle的新Job。组合生成新的完整Bundle、真实diff和未评价Candidate；贡献去重不删除任何尝试成本，来源资格不能拼接。版本变更生成受影响核验项，显式失效、范围不适用和未确认分别返回。`EvaluationStore.candidate_archive`和`ReleaseManager.candidate_archive`可以追溯同一完整组合；正式新评价/发布能力仍归F12/F13。

<a id="f10f14数据原生更新独立评价发布与元考试"></a>

## F10—F14：数据、原生更新、独立评价、发布与元考试

第三批授权、完成度和停止边界见[04 §1.22](../项目文档/04_实施路线与执行状态.md#122-f07f08-隔离补修与-f10f14-第三批2026-11-17)，工程检查见[OFFLINE_CHECKS §36](OFFLINE_CHECKS.md#36-f07f08-隔离补修与-f10f14-第三批工程验证2026-11-17)。以下接口复用原 Host、Coordinator、Store 和 Release 单写入口。

| 固定 operation kind | 实际消费者 | 写角色 |
| --- | --- | --- |
| `data_review` | `collection.review_training_dataset`，保存明确准入、拒绝或未决 | evaluation |
| `training_prepare` | `prepare_admitted_training`，消费审核记录并准备原生输入 | runtime |
| `training_output_parse` | `finalize_training_job`，核对原启动/监督与保存结果 | runtime |
| `training_bundle_prepare` | `register_training_bundle`，从训练/加载原回执建立 Bundle/Candidate | research |
| `goal_materialization` | `trusted_goal.materialize_goal`，固定外部组合规则与独立任务物化 | evaluation |
| `materialized_task_prepare` | RPent `runtime_binding`，核对任务与材料身份 | runtime |
| `evaluation_feedback` | `EvaluationStore.public_feedback`，按协议返回固定公开投影 | evaluation |
| `release_*` | `release_flow` → 原矩阵/Gate/Release/Runtime 消费者 | 各既有可信角色 |
| `meta_exam_research/meta_exam_score/meta_policy_qualify/meta_policy_activate` | 受限 ψ → 实际开发候选 → 独立考试 → Release → 新 Research Job | research/evaluation/release |

`framework_operations.py` 只分派这些固定消费者，操作仍受原 scope、phase、注册 operation、permit、监督和结算约束。CPU 非动态角色必须在 `cpu_operation_ids` 中，并使用现有 `generic-cpu-model0` grant；未授权模型、GPU、设备和训练额度保持为0。`build_configuration` 复制角色操作的 `exports/late_refs/storage_write_roots`，后继只从已闭合原回执晚绑定审核、配方、训练/加载和任务材料引用。

### Research 程序执行与输入隔离

`research_prepare` 和 `meta_exam_research` 在 Research namespace 内只向专属 Unix socket 提交有界执行身份。Host 重读绑定 Job、ψ 源快照、布尔上下文、预算和固定 runner，再创建 candidate scratch、启动无设备程序并写 operator 原始回执。Research 消费该外部回执的结果快照；不取得 operator 或其他角色 scratch 写权限。已有启动标记没有返回时保持未知并要求原操作对账，不能重发。原 Job 版本和费用不随新 ψ 激活重置。

Host `inputs.json` 保留完整前驱来源；CPU 角色只读取 `public-inputs.json` 中的固定身份、公开 references 和版本。Research/Runtime 看不到 Host 原始日志、Coordinator 回执、元评分和私有资格记录；精确 completion/supervisor 文件只读提供成本消费者。公开回归反馈仅在协议 `feedback_policy.release=aggregate` 时投影；sealed/meta feedback 固定 withheld。`ResearchTools.query_public_feedback` 只消费独立 evaluator 的公开目录。

### 数据与原生训练/加载

`review_training_dataset` 每次重读实际载荷。传入 `payload_audit_ref` 时，缓存须与这次独立结果全内容一致；快照和自述成功不替代审核。Host 的 `training_prepare` 拒绝 Research 可写域中的准入 JSON，只接受 operator／evaluation 受保护域的回执；随后原消费者继续核对 scope、protocol、审核者、完整数据身份和质量／隔离证据。第四批补修及真实最小数据准入见[OFFLINE_CHECKS §37](OFFLINE_CHECKS.md#37-第四批分范围验收与文档准备2026-11-17)。

公共、授权示范、教师、自主和评价材料保留来源及版本。采集阶段已有 LeRobot 子集只是未准入暂存物；独立审核核对 episode/frame、时钟、动作/相机契约、执行掩码、文件身份和外部质量/隔离记录。缺少必要证据为未决，不支持的监督来源保留后拒绝训练用途。训练准备复用这一份已审核冻结子集，物化 processor/norm、transforms 和 recipe，并进入原生 dataloader；不是审核后另选一份数据。

`native_train` 调用安装 RLinf 的 `SFTRunner` 和原 actor worker；resource pilot 与 learning 配方分开。源码核验使用 `verify_installed_sources` 的 RLinf `2348ece3df113874671478be1360ae3115b346e3`、`rpent-openpi` `a560f4dd8205b8423ecd4c8a0fabb5f54140b8a0`。本地仓库与实际安装来源分别核对，不改二者源码。

`finalize_training_job` 只从原启动标记、监督回执和完整保存目录解析 θ、processor、norm、配方及源码。半保存、路径漂移、超预算和 processor 错配不导出有效 θ。`native_verify_load` 进入当前 `Pi05VLAFacade` 的 init/predict 路径；加载身份、处理器、动作解码与任务收益分别记录。新 Candidate 从实际训练/加载回执及原 Host 成本账建立，资格等待独立评价；当前 Candidate 序列化操作尚未结算的费用单列；后续独立资格消费者重读原Host回执派生闭合，保留原Candidate历史数值。

`prepare_theta_b_matrix` 要求完整且不重复的 θ×冻结 B 矩阵。不存在 θ1 时保持阻塞；适用性和显式 fallback 保留原 cell 身份，不能将 θ0 改标为 θ1。

### 评价、目标、发布与元考试

`objective_evaluation` 支持登记的成功率、效率、鲁棒性、新能力与关键旧能力保持，重算完整指派矩阵并保留 unknown/invalid；效率使用预定结果/安全条件与全部尝试成本，配对统计只消费注册的配对。评价材料、正式规则和评分由可信消费者拥有。

`trusted_goal` 对固定场景/对象/谓词内的合法新 conjunction 生成 BDDL、指令、匹配初态与新协议。公开 `materialization.json` 只包含任务绑定；完整协议和证据保存在 evaluation 私有 `protocol.json/evaluation-materialization.json`。RPent 使用 `--suite materialized --task 0 --materialized-task <task.json>`，把 BDDL/初态/回执/原生判据身份带入实际 Env daemon 与独立结果。薄适配复用 RLinf reset、动作和成功判据；真实初始化与物理可达性在 P/V 验证。

`ReleaseManager.register_safety` 为发布域登记 operator/runtime 的独立观察来源。资格、active 期望版、服务实际加载版和新任务绑定分别核对；实际加载证据在新 Runtime namespace 中只读。激活记录与 active 切换之间崩溃沿原 intent 对账，重复提交不再激活。未知动作、费用或清理阻断受影响发布；reject/inconclusive 保持当前版，回滚要求已有激活历史。

`meta_exam.build_operations` 将冻结 ψ、合法同事实问题、开发候选、独立评分和策略资格编成原 Host 图。支持 ψ0、ψ1、固定 ψ0 加匹配晚期事实、终点与预定里程碑；每个问题/重复/策略的 Job 保留累计预算。隐藏题目与评分不进入策略修订。单终点不输出曲线或到目标成本；缺失、失败、无更新和超预算按登记规则保留分母和费用。新 Research Job 可通过 `research_release_root/research_policy_scope` 绑定合格有效 ψ；已存在 Job 保持原绑定。

这些接口的离线证据使用隔离合成域与外部重组件边界替身。真实梯度学习、真实模型加载兼容、机器人能力、H3 与正式发布收益分别需要后续 P/V 证据；本批不自动启动 F15 或 P/V。

离线 `theta_artifact_evaluation` 仅在显式合成产物协议中核验文件/回执与发布绑定，沿同一Release/Runtime入口演示θ侧资格、激活和回滚；不把这些任务的接受升级为真实模型兼容或学习收益。


## 第四批：仿真范围验收与准备

当前状态、硬资源上限和单作业准入统一见[04 §1.23](../项目文档/04_实施路线与执行状态.md#123-第四批仿真验收与准备2026-11-17)。`G_F_sim` 覆盖 F00—F14 仿真适用项、F16_sim 和 C01—C07/C09/C10；证据成立后可进入适用且获授权的 P_sim。C08 未验收、延期；全局 G_F 继续等待 F15、C08、真机 F16 及共享版本兼容。

前三批证据分别保留在 OFFLINE_CHECKS §34—36，第四批适用检查见[§37](OFFLINE_CHECKS.md#37-第四批分范围验收与文档准备2026-11-17)。P03/P05 草案见[实验设计 §9](../项目文档/03_实验设计与验收协议.md#9-第四批-p03p05-草案2026-11-17)，未冻结正式比较。实际模型、仿真初始化、原生梯度更新/保存/加载与能力收益按各自真实回执验收；合成离线产物不建立这些结论。
