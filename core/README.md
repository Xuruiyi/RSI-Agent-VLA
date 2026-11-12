# RSI-VLA 离线研究 core

`core/` 是聚合目录维护的自研代码，采用 Python 标准库，要求 Python ≥3.10，无需安装包。入口使用 `PYTHONPATH=core/src`。它提供可重读的研究与版本单位、证据视图、预算账本和独立 Gate 状态机；固定 ρ 和初始 ψ 是基线表示，尚无真实训练或正式发布的动态验收。新增 Runtime 证据/监督、有限研究候选及 π0.5 训练适配的本轮检查见 OFFLINE_CHECKS.md；当前状态、暂停指令和阻塞统一见项目文档 04。

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

v9 profile 的聚合源码标识为 `aggregate-core-v0.5.0-bounded-vla`；包元数据仍为 `pyproject.toml` 的 0.4.0，旧 profile 未指定源码标识时继续使用 `aggregate-core-v0.4.0`。当前代码包括既有合约/存储/证据/研究/Gate模块，以及 `runtime.py`、`evaluator.py`、`training.py`、`research_tools.py`、`rule_updates.py` 和对应 adapters/tests。聚合目录没有 Git 版本，正式研究归档应同时保留这批实际源码/差异和运行产物；版本标签本身不构成 Git 可复现证明。

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

`TaskState.update` 只接受已入账且同回合事件、合法观察/独立判据和完成规则引用。调用者负责执行该具体判据；工具结束或启发式标签本身不能作为完成提交。当前关系失效或未知不会删除原事件及提交历史。未知状态由 `ResearchJob.unknown_action` 按既有观察/恢复权限和剩余额度返回 `observe`、`recover` 或 `stop`。

固定 ρA 的执行入口是 `EventLedger.view`；ρB 是 `fixed_route(rho_snapshot, requested_worker, allowed_workers)`，未授权则返回 `defer`；ρC 是 `fixed_research_step(rho_snapshot, job, evidence_available=..., legal_repairs=..., inspected=..., experiment_permitted=...)`，只有合法修复集合非空、已检查证据且预算和权限允许时才返回 `bounded_experiment`。`rho_snapshot` 从该 RunBundle 的唯一 `rho_ref` 读取，配置类型或规则变成未实现策略时显式报错。这些离线基线不会修改 RPent 的实时路由，也不会自行启动实验。学习与运行接入仍按实际反馈和后续授权实现。

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

测试覆盖的证据层级是离线状态转换。真实 VLA 基线、动作全覆盖、实际候选修复/训练、三类关系学习、仿真部署和 H1—H4 仍需各自运行证据。

本轮静态开发计划使用 `run_profile_v9_resultfix.json / pilot_v9.json`，Planner 和 Research LLM 显式固定 `gpt-6.1-sol / medium`。v1 和历史 low 证据保持原样。下面只生成计划，不调用模型或启动服务：

```bash
scripts/run_libero.sh --artifacts /tmp/rsi-medium \
  --protocol config/pilot_v9.json --bundle initial-medium-v9-resultfix \
  --task-key position_swap_salad --seed 1 --split development \
  --run development-input-v1 --output /tmp/rsi-medium/run-01
```

先用上述 `init` 命令创建同一 artifacts 根。输出包括完整 argv、run/bundle/protocol 身份、预算和实际阻塞；`--execute` 需要对应授权与受支持的请求预算，默认配置不会启动。动态 Runtime 与训练共用 `/tmp/rsi-agent-vla-dynamic.lock` 的单作业额度。实际外部监督入口为 `rsi_vla.adapters.runtime_supervisor`，不是离线 `ResearchJob`。

`rsi_vla.evaluator` 从独立产出的结果、动作 JSONL、成本与 supervisor 文件读取全部预指派槽；缺槽为 unknown，身份冲突为 invalid，low 结果不能进入 medium 基线。安全、兼容、关键推进和预算等未取得独立证据时保持 null；单纯收齐文件不会自动获得发布资格。受保护 evaluator/release 的操作系统写权限部署仍需在正式发布前建立；本地独立文件消费和不同角色 API 不构成 OS ACL。

2026-11-10 用户已授权动态推进，v3保留Codex，将token改为记录项；本地HTTP接点在转发前限制每回合20个实际模型请求，完整墙钟/工具/总环境步仍有硬停止。v1/v2及历史快照不改写。实际进展以04最新运行记录为准。

默认静态 consumer 使用 v7 工具60协议；本轮新输入/Runtime 的单回合计划显式使用 v9。v1—v8、历史 low 和5A证据保留。`scripts/run_research_review.py` 将合法开发视图交给一次受外部监督的真实 Codex Research LLM 请求，实际 model/medium 来自本地转发证据，token 为记录项。`--reconcile` 只校验已有响应，不重新调用模型。有限规则 v3 的 `planner_finish_reserve_calls=1` 只给 Planner 预算规划提示，实际上限从当次 context 读取，finish 仍经过原工具计数；不改变 θ、动作 chunk 或原生成功判据。

Runtime 在首次工具/环境步/turn 预算停止时保存同身份 `budget_stop.json`，允许已准入操作收尾，阻止后续模型转发。原始 SDK 状态与明确预算原因分开记录；完整原生负反馈和清理可形成预算失败，其它中断仍为 unknown。后续运行在 `code_snapshot/` 留存启动前的实际 Runtime 源码，不把未提交代码的版本标签当作完整源码证明。历史缺少明确 stop cause 的第三槽采用独立补件，原始结果和停止传播缺口均保留，详见04。


`libero-startup-v1` 在任务启动时传入操作边界、技能入口和三份匹配主题的原始上游记忆，完整指南/记忆索引按需查询；缺少当前任务专用记忆时直接登记缺失。`back_project_batch` 逐点调用原 `back_project`，保留顺序和逐点错误，不执行机器人动作。每次工具反馈的 `runtime_budget` 读取计费 owner 的快照，不再次扣费，缺失值为 null。

Launcher 在进程与传输清理后独立保存 `independent_result.json` 和 `cost_breakdown.json`，保留原 `result.json`；取消、超时、传输失败和缺证保持 unknown。分项计时注明包含/重叠，cached input 是 input 子集；不完整 usage 只有观测下界。新回合验收须消费独立结果和原始服务证据，不能只看 Planner finish 或退出码。

唯一待运行计划为 [development_plan.json](../artifacts/offline-20261112-vla-result/development_plan.json)，源码/输入绑定见同目录 `source_manifest.json`、`task_input.json`。当前仍暂停，无动态授权；计划中的 permit 是未签发提案，未安排第二回合。公共执行基础优化不计为 RSI 自主改进。检查及限制见 [OFFLINE_CHECKS.md §8](OFFLINE_CHECKS.md#8-2026-11-12-vla-技术错误向最终结果传播修复)。


当前唯一待运行版本为 v9 的结果传播修复：`config/pilot_v9.json` / `config/run_profile_v9_resultfix.json`，bundle `initial-medium-v9-resultfix`，代码 `aggregate-core-v0.5.1-vla-result-validity`，输入 `libero-startup-v2-bounded-vla`。v8 配置及历史快照保留其身份。`vla_execute` 在明确的 1—24 次预测额度内复用五步 chunk 连续执行；Planner 保留目标、监督和恢复职责。静态计划及暂停/许可缺口只见项目文档04 §1.13，当前不得运行动态入口。检查入口见 `OFFLINE_CHECKS.md §8`。

执行技术错误由 Toolkit 已存执行段和绑定回合的 action events 进入 finalizer；launcher 再独立检查推理失败、动作 RPC 错误或不确定反馈及技术停止事件。原生事实保存在原始结果和派生结果的 `native_task_result`，未解决故障的可评价结论保持 unknown/technical_invalid。acceptance 默认读取 `independent_result.json`，显式原始结果也须检查事件；evaluator 将已有原始路径切换到同目录的 launcher 收尾。轨迹导入保留原始独立证据，技术无效结论保持 unknown。尚未定义执行故障恢复判据，正常退出、后续工具成功或 Planner 叙述均不构成恢复；已有有证据的预算派生收尾继续保留。


第三批离线接线状态见04 §1.15，验证见OFFLINE_CHECKS §10。旧“唯一待运行”段落及第一批源码清单为已执行第二批的历史，不是当前运行许可；动态工作继续暂停。

正式研究导入（先用同一profile/protocol/bundle初始化新的版本化store）：

```bash
PYTHONPATH=core/src RPent/.venv/bin/python -m rsi_vla import-development \
  --artifacts artifacts/offline-20261112-batch3/state --protocol config/pilot_v9.json \
  --run-dir artifacts/development-v9-resultfix-position-swap-salad-s1-01 \
  --result-id development-v9-resultfix-s1-origin-v2
```

`development.import_development` 复用原事件、EvidenceView和ExperimentResult，仅接受冻结开发任务/seed和匹配身份。同store同run等值导入幂等，冲突报错；语义变化另用版本store。原事件、旧派生和历史成本保留，不重复charge。研究请求携带完整L0视图，可批量输入多个result。新旧视图出处见真实run下 `research-derivation-v2.json`。

`scripts/prepare_bounded_research.py --help` 是新有限研究输入/响应准备入口，无模型或实验调度；它要求已有预算job及人的domain policy，允许chunk前缀、native finish guard、finish reserve三种既有规则算子，实际产物仍由ResearchTools/rule_updates生成。synthetic响应只作offline工程检查。旧 `run_research_review.py`/v1响应保留；不得运行其历史execute入口恢复队列。

普通部署使用 `scripts/run_libero.sh --deployment-release <trusted-release-root> ...`，任务边界读取ReleaseManager有效版；实验/评价/复现继续用显式 `--bundle`，二者互斥。未合格simulation bootstrap会拒绝部署选择。选中的模型/processor/规则/合法记忆进入现有Runtime参数，已绑定作业不重新选版。发布端仍须独立权限部署；本地目录分开不构成ACL。

`config/run_profile_eval_draft.json` 只准备四任务startup输入适配，未改变现行Gate或矩阵。评价建议和5A/5B阻塞统一见04及06，不据此启动任何动态作业。
