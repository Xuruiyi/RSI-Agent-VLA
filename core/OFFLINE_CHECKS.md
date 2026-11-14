# 本轮离线交付检查

**GPU 选择说明：** 后续执行按 [04 的当前 GPU 选择规则](../项目文档/04_实施路线与待决事项.md) 自主选择可用设备。本文历史卡号、固定卡号方案和当时未换卡的记录仅用于说明已有证据，不约束后续选卡；原始产物与实际设备记录保留。

§1—4 是 2026-11-09 原离线交付记录；本轮增量见 §5。日期：2026-11-09。对应 04 §1.3 第 1—3 项。全部是静态/离线证据，无新模型请求、下载、安装、训练、GPU、仿真或设备操作。

## 1. 实现与证据位置

- 聚合目录自有：`core/src/rsi_vla/`、`core/tests/test_core.py`、`core/README.md`。
- 配置：`config/run_profile_v1.json`、`config/runtime_observed_v1.json`、`config/pilot_v1.json`；脚本 `scripts/{run_libero.sh,check_runtime.py,check_pilot.py}`。
- RPent：`robots/libero/{result.py,robot_spec.py,toolkit.py}`、`rpent/cli/{main.py,dashboard.py}`、结果/CLI/Dashboard 测试及中英文 LIBERO 页面。
- 先导协议：`项目文档/06_首批任务先导协议.md`；唯一推进状态见 `04` §1.3—1.4。
- 本地证据根：`artifacts/offline-20261109/`。RPent 原始日志 `RPent/logs/smoke-20261109-222602/` 保持原样。

RPent 起点 `main / c41c0e6c3dc44e3347070ac1361bc2c288901873`，集成前 `git status --short` 为空；补丁归档 `artifacts/offline-20261109/runtime/runtime.patch`。聚合目录不是 Git 仓库，没有初始化 Git 或提交。子终端分别在 `/tmp/rsi-runtime-work`、`/tmp/rsi-core-work`、`/tmp/rsi-protocol-work` 隔离工作，未继续委派；模型为 GPT-6.1-Sol、high。主会话无法通过当前工具切换模型，未声称已完成主模型切换。

## 2. 当前通过的检查

| 检查 | 结果 | 证据 |
|---|---|---|
| 主终端集成后的 core 行为测试 | 29 tests，OK，0.179s | `core-tests.log` |
| RPent LIBERO 全离线单元测试 | 132 passed，exit 0 | `runtime/libero-unit-tests.log` |
| 受影响 CLI/Dashboard 合约 | 82 passed，5 deselected，exit 0 | `runtime/cli-dashboard-tests.log` |
| 新结果定向检查（与上两组重叠） | 29 passed，66 deselected，exit 0 | `runtime/focused-result-tests.log` |
| RPent Ruff check / format check | passed，exit 0 | `runtime/ruff-check.log`、`runtime/ruff-format.log` |
| 任务/BDDL/init、划分、协议与预算 | PASS，exit 0，标准库静态解析 | `pilot-check.log` |
| 匹配安装核对 | PASS，exit 0；版本、安装来源、导入发现位置、模型 preset 未变 | `runtime-check.json` |
| `bash -n scripts/run_libero.sh` | exit 0 | 主终端本轮输出 |
| 初始快照与真实文件接口 | init 成功；raw 9 events→unknown；derived 9 events→success、身份缺口显式保留 | `init.log`、`import-raw.log`、`import-derived.log` |
| Python AST、JSON、文档本地链接/标题/空白；RPent diff --check | PASS | `delivery-check.log`、主终端输出 |

core 测试覆盖未知不冒充成功、工具/子目标/整任务分开、事件重放幂等与冲突、完成提交去重、当前关系失效保留历史、L0 不受媒体选择隐藏、版本/初始快照重新读取、候选真实差异与协议/基准关联、全槽评分重算、旧任务保持、预算终止及恢复不重置、已知违规拒绝、拒绝/未决不发布、安全点激活与回滚，以及固定 B/C 规则的授权/证据/预算分支。固定规则不算关系学习。

初始化实际产物位于 `artifacts/offline-20261109/core-state/research/`：`bundles/initial.json`、`profiles/initial.json`、`psi/psi-initial.json`、`memory/memory-initial.json`、`rho/rho-initial.json`。这是离线起点，不是已验收运行锚点。RunBundle 引用保存后的 profile 副本；大模型与外部记忆仍按固定 revision 引用。

历史恢复结果为 `runtime/recovered-smoke/result.json`；依据 `runtime/native-semantics.log` 中匹配安装的 LIBERO `_check_success()`/RLinf truncation 源码及原始 step 7 标记，将 benchmark_success 恢复为 true。抓取启发式单列，释放/落稳等过程标签仍 null。原始 transcript 缺独立结果，默认导入仍 unknown；显式导入恢复结果保留 `run_id_not_recorded_by_runtime`、`bundle_id_not_recorded_by_runtime`，不充当正式发布证据。恢复脚本与stdout见 `runtime/recover_smoke.py`、`runtime/historical-recovery.log`。

## 3. 可复用命令

在聚合目录运行；使用既有 `RPent/.venv/bin/python`（3.11.13），系统默认 Python 3.8 不适用本 core：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core/src RPent/.venv/bin/python -m unittest discover -s core/tests -v
RPent/.venv/bin/python scripts/check_runtime.py --expected config/runtime_observed_v1.json
RPent/.venv/bin/python scripts/check_pilot.py
bash -n scripts/run_libero.sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core/src RPent/.venv/bin/python -m rsi_vla check-protocol config/pilot_v1.json
```

原 RPent 检查在隔离目录 `/tmp/rsi-runtime-work` 执行，`PYTHONPATH=/tmp/rsi-runtime-work`，解释器为目标 RPent 的 `.venv/bin/python`。对应命令如下；主终端确认补丁适用、集成后 `git diff --check` 通过，复用该离线测试证据，没有重复跑整套测试。

```bash
python -m pytest tests/unit_tests/robots/libero -q
python -m pytest tests/unit_tests/rpent/cli/test_main_contracts.py tests/unit_tests/rpent/dashboard/test_session_contracts.py -k 'not full_cli_exploration_finalizes_memory_without_starting_gpu_runtime' -q
python -m pytest tests/unit_tests/robots/libero/test_result_contracts.py tests/unit_tests/rpent/cli/test_main_contracts.py tests/unit_tests/rpent/dashboard/test_session_contracts.py -k 'native_result or pick_heuristic or full_cli_calls_robot_result or startup_failure or dashboard_exploration' -q
```

前两条使用 shell `timeout 60s`；Ruff 对上述 5 个实现文件与 3 个测试文件执行 `check --preview` 和 `format --check`。原日志中的隔离路径用于定位实际执行源码，不是另一个安装链。

## 4. 未运行、限制与下一步

Sphinx 未安装（`runtime/docs-build-prerequisite.log`），没有安装依赖，中英文页面未构建。一次较宽的 CLI 运行进入已有 interactive API exploration 场景后被停止，exit 130；最终选择器明确排除该组，不计作全 CLI 通过。pytest 报现有 `timeout` 配置项未识别警告；上述有界检查另有 shell timeout，未安装插件。

当前无动态授权。06 §6 和 `pilot_v1.json.next_step_proposal` 已给出可审查的单回合方案：已见 swap/task 2/seed 0、initial、900 秒完整作业、600 秒 Planner、2000 环境步、1M 输入/8k 输出等边界；执行前还需实际后端成本监督和具体 GPU/LLM/仿真授权，不自动扩展到 16 回合。启动脚本的模型/记忆显式固定及新增结果出口未做新动态验收。正式 evaluator、受保护发布进程、动作级 L0 全覆盖、完整身份关联、真实锚点与训练往返仍待完成；本地状态机不是进程权限隔离或后端实时抢占器。

本轮不宣称自主更新、关系学习、H3、正式独立发布或完整系统验收通过。

## 5. 2026-11-10 第4、5A、5B项的离线集成

代码、模型配置、阻塞和待授权范围统一见 04 §1.5。本节只记录检查证据，不代表动态验收。证据根 `artifacts/offline-20261110/`；A/B/C 隔离目录分别为 `/tmp/rsi-20261110/{a,b,c}`，RPent Git worktree 为 `/tmp/rsi-20261110-runtime`，主集成在 `/tmp/rsi-20261110/main`。旧 low 日志、v1 配置与 2026-11-09 归档原样保留。

| 检查 | 实际结果 | 日志 |
|---|---|---|
| 编码主/子代理实际模型配置 | 4份 turn_context 均为 gpt-6.1-sol/high；未来运行 medium 与历史 low 分开 | `model-settings.json` |
| 主终端最终共同接口与监督检查 | 38 tests，OK，1.062s；含 C 的16项最终路径/协议交叉检查及8项含cleanup的CPU子进程监督 | `integration-tests.log` |
| A RPent LIBERO/Codex fake/CLI/Dashboard定向组 | 274 passed；1个loopback socket在沙箱EPERM失败；5 deselected。仅失败项在允许loopback环境补跑1 passed，不重复整组 | `a-runtime-tests.log`、`loopback-test.log` |
| A core接口检查（先前证据） | 39 tests，OK；既有29项与动作导入/旧版监督检查。监督修改后的8项以主终端集成日志为准 | `a-core-tests.log` |
| B 数据/配方/模型交接与训练监督 | 14 tests，OK，0.202s；AST通过；实际安装配置校验与worker cloudpickle往返通过，没有Cluster/模型/worker实例 | `b-prior-verification.log` |
| B 实际安装链 prepare | exit0；实际 norm_stats、pi05_libero处理链和812 tensors header读取；未训练、未加载模型、无合格数据，未实际读batch | `b-training-prepare.log`、`training-prepared/preparation.json` |
| 后续集成补证 | 物理GPU编号 CLI/env一致性离线测试1项OK；目标目录实际CLI初始化/计划/拒绝准入/JSONL导入均exit0，合成缺失结果保持unknown | `gpu-binding-test.log`、`medium-init.log`、`medium-smoke-plan.json`、`runtime-admission.log`、`cli-journal-import.log` |
| 最终交付静态检查 | 38文件传输与源一致，历史v1未变，Python AST/JSON/空白、文档本地路径和标题、RPent diff --check通过 | `delivery-static.log` |
| v2 任务/初态/评分/预算静态consumer | PASS；medium、initial-medium、开发/发布/封存指派及零授权均核对 | `pilot-v2-check.log` |

主终端集成检查命令（仅CPU/离线）：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core/src:core/tests \
  RSI_RPENT_ROOT=/tmp/rsi-20261110-runtime timeout 30s RPent/.venv/bin/python -m unittest \
  test_runtime_plan test_evaluator test_research_tools test_runtime_supervisor \
  test_core.CoreTest.test_cli_init_consumes_immutable_profile_snapshot -v
```

命令在主隔离目录执行时，解释器使用目标项目 RPent/.venv 的绝对路径。RPent原组命令和解释器已写在 `a-runtime-tests.log`；loopback补跑只选 `test_codex_contracts.py::test_http_mcp_readiness_ignores_environment_proxy`。pytest 的既有 `Unknown config option: timeout` 警告保留，另有shell timeout；未安装插件。未跑被排除的 interactive exploration 场景、GPU/仿真/真实LLM、下载、装包或硬件操作。

实际外部监督可停止进程组并确认清理；GPU时间是占用包络，实际util未知；存储是采样阈值，记录超额，不称硬配额。动作RPC反馈不等于控制器实际应用或测得运动。setup/reset内部动作及完整成本仍有缺口，Gate相应证据保持未决。Research LLM只有显式medium请求准备/响应校验，未取得实际运行生效证据。5A的真实候选、5B的真实训练—实际加载—任务评价均未执行。

最终静态归档检查首跑误用系统 Python 3.8，解析已有 RPent 3.11 语法失败；改用项目指定 Python 3.11.13 后通过。未因此重跑已通过的行为检查。


## 6. 2026-11-11 离线执行基础改造

当前状态/唯一后续回合见04 §1.11；本节只维护检查证据。项目保持暂停，未发真实模型请求，未启动仿真/GPU/训练/下载/安装/设备或历史重试。聚合目录不是Git；RPent保持既有未提交成果。待集成补丁先在临时副本 `patch --dry-run -p1` 全部适用，再按逐文件原内容一致性集成；没有整份覆盖目录或重复应用历史补丁。

环境：目标 `RPent/.venv/bin/python`，Python3.11.13，实际 `rpent` 导入目标源码；所有检查启用 `PYTHONDONTWRITEBYTECODE=1`。标准库AST首轮误用了系统Python3.8，不支持既有3.11语法；换成指定解释器后通过。临时副本的pytest初轮缺少父conftest，补齐并明确root/confcutdir后所缺5项通过，目标目录定向组另行通过。新增CLI fixture最初mock错了局部导入的位置，改为mock `rpent.memory.loading.prepare_run_memory`。Ruff缓存默认位置不可写，最终使用 `--no-cache`。这些工具/fixture问题不计作系统能力失败。

| 检查 | 本轮已检查的结果/范围 | 日志（本目录相对项目根） |
|---|---|---|
| core定向集成 | 70 tests，OK；plan/较低permit、evaluator、research、CPU进程监督；后续收尾/usage变化以最终定向项为准 | `artifacts/offline-20261111/core-checks.log` |
| RPent预算/几何/结果/服务证据 | 51 passed，1 skipped；skip是未指定原生RLinf reset源码的可选静态检查，不是动态作业 | `artifacts/offline-20261111/rpent-checks.log` |
| RequestGate真实loopback假上游 | 5 tests，OK；127.0.0.1客户端/假上游、trust_env=False，确认不转发外部；上限后/运行停止后请求不转发、不重复计数；传输错误无重试 | `artifacts/offline-20261111/request-gate-final.log` |
| 自动收尾与usage最终检查 | 3 tests，OK，含8种终态子场景：成功、任务失败、预算、取消、超时、传输、缺失/错身份；无finish/无backend结果时仍保存独立unknown/成本；cached缺失不补0，部分usage仅下界 | `artifacts/offline-20261111/closure-final.log` |
| CLI精简视图与原finalizer | 新增compact消费者1 passed；原CLI/finalizer等14项已通过，新fixture导入位置修复后单独补跑 | `artifacts/offline-20261111/cli-fixture-initial.log`、`cli-compact-final.log` |
| 协议实际静态consumer | v7和v8均PASS：任务/初态、划分、模型配置、评分、授权形状、预算 | `artifacts/offline-20261111/pilot-v7.log`、`pilot-v8.log` |
| 历史关联与静态计划 | 见静态交叉记录：v6前三槽和5A身份/原预算；新输入只含公共上游原文，缺失可追溯；模型/processor及其余预算保持，唯一静态计划没有运行目录 | `artifacts/offline-20261111/historical_link_check.json` |
| 格式/语法/差异 | 最终Ruff check/format、Python3.11 AST、JSON、bash语法、本地文档链接/编号及增量空白、RPent git diff --check，见日志 | `artifacts/offline-20261111/static-checks.log`、`ruff-check.log` |

核心命令：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core/src:core/tests timeout 40s RPent/.venv/bin/python -m unittest test_runtime_closure test_runtime_plan test_evaluator test_research_tools test_runtime_supervisor -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core/src timeout 25s RPent/.venv/bin/python -m unittest discover -s core/tests -p test_request_gate.py -v
PYTHONDONTWRITEBYTECODE=1 RPent/.venv/bin/python scripts/check_pilot.py --protocol config/pilot_v7.json
PYTHONDONTWRITEBYTECODE=1 RPent/.venv/bin/python scripts/check_pilot.py --protocol config/pilot_v8.json
```

RPent定向pytest及最终命令记录在同目录 `verification_commands.json`；`timeout`是shell外层有界检查，pytest既有 `Unknown config option: timeout` 警告保留，未安装插件。Sphinx构建器未安装，未装包；两种语言的新增段落做源码/本地引用检查，未声称构建通过。没有触发真实模型、仿真、GPU或设备验收。

5A只复用已有响应/audit记账：研究请求1+试验14，未知token单列；正式Gate缺三份独立发布结果而inconclusive，无激活。`reconcile_5a.py`是本轮离线记账脚本，记录与当前版保留事实见04；不是一次新研究调用。公共执行基础修改属于人工离线工程工作。辅助模型加载的独立耗时及真实模型理解/任务能力/实际节约仍待证据。


## 7. 2026-11-11 bounded-vla-v1 离线执行粒度改造

状态见04 §1.12。证据根 `artifacts/offline-20261111-vla-v9/`；Python 为目标 RPent `.venv/bin/python` 3.11.13，`rpent` 导入实际目标源码。仅 CPU/fake/历史观测和 loopback MCP/假上游，无外部模型、仿真、GPU或设备。复用 §6 的未受影响监督、服务动作映射及安装链证据，不重跑无关完整套件。

| 检查 | 已检查结果与适用范围 | 日志 |
|---|---|---|
| 有界 VLA 基础 | 首组17通过；启动输入拼接问题修复后该1项单独通过。覆盖指令隔离、最新观测、1/24和非法边界、成功/截断/取消/推理错误/含setup步数预算/时间预算 | `rsi-v9-unit-initial.log`、`verification_commands.json` 的启动输入补跑记录 |
| 实际 Toolkit/MCP 及增量 | 修正fixture后4项通过：真实MCP tools/list/call、参数必填、新旧工具并存、两张/三张历史图像字节序列化、RPC失败不确定步数、chunk间取消、WAM隔离；其中MCP合为1测试，分支分开断言 | `rsi-v9-integration-final.log` 中4通过；同组标定项后续单独修复通过 |
| 标定及旧注册表 | 2项通过：历史全局/腕部1024图与同相机世界网格的四位小数定位输出一致；更新注册表列出新工具及上一轮back_project_batch | `rsi-v9-calibration-compat-final.log` |
| 既有兼容路径 | 首组67通过；唯一注册表期望修正后如上一行通过，合计68项覆盖CLI启动/旧观测/几何/结果/预算/原生终止等 | `rsi-v9-compatibility.log` |
| 成功停止最终项 | 11项通过（选择器与以上部分重叠）：实际假Codex stream停止、native成功收口、预算兼容、本地成功准入拒绝与无关技术错误区分 | `rsi-v9-success-final.log` |
| RequestGate | 1项通过；实际127.0.0.1假上游，原生成功后0转发、0新增预算/取消标记 | `rsi-v9-gate.log` |
| 共享chunk消费者 | 首组33项通过；WAM标量反馈计数兼容和原证据测试的末事件假设修正后，3个失败项单独通过，合计36项 | `rsi-v9-chunk-consumers.log`、`rsi-v9-chunk-compat-final.log` |
| core自动收尾/静态计划 | 初组12项通过；后续成本缺失及新增源码快照变化以最终定向日志为准 | `rsi-v9-core-initial.log`、`core-final.log` |
| 语法、格式、文档、版本计划 | 最终AST/JSON、本地路径、配对RST内容、Ruff check/format、git diff --check、v9协议consumer及冻结预算/模型/矩阵对比 | `static-final.log`、`pilot-v9.log` |

所有检查使用 `PYTHONDONTWRITEBYTECODE=1`、shell timeout 和 pytest `-p no:cacheprovider`；既有 pytest 未识别 timeout 配置警告保留。实际命令和退出状态见 `verification_commands.json`。初轮启动输入多余加号、MCP fixture模板变量/历史状态锁缺失、旧工具集合漏列，以及定位测试未考虑原四位小数输出，均已按实际失败原因修复；没有修改历史观测或改变定位输出精度。成功的检查只在受影响变化需要时补跑，数量不重复相加。

Sphinx 仍缺已登记构建器，未安装，RST仅做源码和配对内容检查。真实模型理解/任务完成/节约效果/中途时限抢占仍需获准单回合验证。原生 chunk 内发送与终止处理不变：连续工具仅在 chunk 边界再预测；已准入 chunk 的服务预算仍逐步生效，RPC断开时仅记不确定范围，以服务owner为准。辅助模型加载独立耗时仍未知。没有发布、训练、能力提升或RSI自主学习结论。


## 8. 2026-11-12 VLA 技术错误向最终结果传播修复

本批状态及唯一暂停计划见04 §1.13。证据根 `artifacts/offline-20261112-vla-result/`；使用 RPent `.venv/bin/python` 3.11.13，实际导入本工作区源码。仅 CPU、mock/fixture 和历史记录读取；新增反例不启动服务或转发外部请求。聚合目录非 Git；RPent 目标源码起初无已有 diff，原有 staged `.idea`、AGENTS.md 与其它终端文件保留。源码前件、增量及新源码见 `source-before/`、`incremental.patch`、`source/`。

| 检查 | 结果与范围 | 日志 |
|---|---|---|
| RPent 初始有界 VLA / 原生结果 / 预算兼容 | 44 passed，1 deselected；排除已验过且未受本次影响的 MCP 图像网络序列化项。finalizer/Toolkit 后续变化以下两行为准 | `bounded-compatibility.log` |
| 最终原生 finalizer / 预算 / 跨项目链 | 33 passed，包括未启用 RuntimeJournal 的已存错误传播；后续 acceptance 分离执行健康与收尾可用性，以最终11项为准 | `result-chain-final.log` |
| 最终跨项目端到端验收 | 11 passed：推理失败无新动作正常finish、故障后预算收尾、部分RPC/不确定动作、后续原生成功仍未解决、旧工具推理错误、正常失败、成功自动收尾、参数纠正后成功、未完成、预算和取消。实际 Toolkit→事件→finalizer→launcher 独立结果→acceptance/evaluator/轨迹导入；显式原始结果路径也不能绕过故障 | `chain-acceptance-final.log` |
| 共享 chunk / Toolkit / 运行证据消费者 | 37 passed；未改 processor、动作时序及旧工具参数语义 | `consumers-final.log` |
| core 受影响离线检查 | 51 tests OK：runtime closure/静态计划/evaluator/预算派生/core；轨迹导入后续改动另补2项通过 | `core-final.log`、`import-final.log` |
| 历史预算恢复语义 | 实际读取 v6 position_swap_salad-s1 的原记录，派生 failure / verified_budget_exhaustion；原始文件不写入，既有3次晚转发缺口仍保留 | `historical-budget.log` |
| 静态源码、格式、计划与协议关联 | Python3.11 AST、JSON、Ruff check --preview / format --check、git diff --check、本地引用和计划冻结项核对 | `static-final.log`、`ruff-final.log`、`plan-check.log` |

命令与退出状态记录在 `verification_commands.json`。使用 `PYTHONDONTWRITEBYTECODE=1`、shell timeout、pytest `-p no:cacheprovider`。首轮跨文件 fixture 导入不符合项目 importlib 模式，修正为完整 tests 包名后通过；格式采用项目配置，未用 preview formatter 改写无关源码。pytest 仍有既有 Unknown config option: timeout 警告，shell timeout 提供检查时限，未安装插件。

复用 §7 仍适用的 MCP/图像、原生成功准入、RequestGate、标定及共享服务动作映射证据，§6 的安装链/进程监督/usage证据与既有阶段预算快照；它们不证明本次真实模型或任务能力。不重跑无关完整套件，未发 PR，未运行全仓 pre-commit/全部单元测试或 CI Python矩阵。Sphinx 构建器缺失的既有证据继续适用，未安装；中英文 RST 只做配对内容与本地引用检查。模型、仿真、GPU、训练、设备与动态清理均未做本轮验收。

执行故障恢复判据仍缺失，故障不会因后续工具成功或正常退出而自动恢复。原生任务事实单列，不把技术无效记录混入正常任务 failure；合法预算收尾及普通参数纠正保留。协议仍为 v9，仅 profile/bundle/code 和源码快照更新；唯一计划未获动态许可，无第二回合。

## 9. 2026-11-12 第二批唯一开发回合

状态、结果解释及第三批输入统一见04 §1.14。唯一动态run为 `development-v9-resultfix-position-swap-salad-s1-01`，bundle=`initial-medium-v9-resultfix`，任务 `position_swap_salad/seed1`；只执行一次 `execute_runtime`，没有Research模型或追加回合。第一批45份源码/配置/文档快照与启动前实际文件逐字节一致，复用 §8 仍有效的离线错误传播证据，不重复运行成功测试。

| 检查/证据 | 实际结果与边界 | 入口 |
|---|---|---|
| 执行前身份/额度/许可 | 源码、计划核心字段、提示内容匹配；历史4回合消耗核对一致；当前输入按context重绑定run/bundle；一作业许可受原阶段剩余额度约束 | `artifacts/offline-20261112-vla-result/second-batch-preflight.json`、`second-batch-permit.json` |
| 真实仿真执行 | success/valid/completed，native_success；4 HTTP模型请求、3 RPent工具、36预测、180 policy+30 setup步、161.179秒全墙钟，原result不覆盖 | `artifacts/development-v9-resultfix-position-swap-salad-s1-01/independent_result.json`、`cost_breakdown.json` |
| 独立现有Runtime消费者 | `inspect_attempt` 19/19 true，模型/processor/动作映射/请求/usage/清理和预算均通过；成功回合不验证技术失败的动态分支 | 同run `independent-audit.json`、`supervisor.json` |
| 冻结段额度符合性 | 实际24/12，第二段不同于计划24；schema范围内但固定24条件未完整验收。单列偏差，不改原结果或协议 | 同run `offline-diagnostics.json.plan_adherence` |
| 逐请求/动作/观测/停止 | 4份usage完整；成功后0转发/0拒绝，末chunk内原生成功后1步；35个预测边界先有前chunk反馈，精确逐推理观测ID缺失 | 同run `offline-diagnostics.json`、原始client/service事件及SDK stream |
| 研究输入临时离线消费 | 轨迹237事件/末结果success；临时store中 `ResearchTools.query_development`接受1个合法开发assignment，研究charge为空、正式store未写。高层vla_execute来源误标tool，raw事件保留；现有两个finish机会读器均返回空 | 同run `offline-diagnostics.json.research_ingestion_offline_check` |
| 阶段记账 | 仅新增本run消耗；保留旧limits/used及原campaign/prior引用，没有重置或恢复其它队列 | `artifacts/offline-20261111/phase_budget_snapshot.json`；同run `phase_budget_before.json` |

目标解释器Python3.11.13；RPent导入工作区，RLinf/OpenPI/LIBERO消费原 `.venv` 匹配安装链。宿主GPU0 RTX4090D，单卡、全局串行。实际驱动脚本、只读分析脚本与命令分别保存在本run `launch_driver.py`、`analysis_driver.py`、`handoff_evidence_driver.py`、`verification_commands.json`。`launch_driver.py` 是一次性执行记录，不是重试入口：许可/输出已存在即拒绝再次启动。

只更新04、此证据索引及本回合产物/原阶段账本；源码未改，没有PR、全仓pre-commit/全套单测或CI矩阵。文档检查为本地引用/数值/JSON一致性，不调用Sphinx或安装构建器。原生结果和技术有效性通过，固定24条件偏差及研究接口缺口保留；未建立正式能力比较、发布锚点或自主更新结论。已再次停止。

最终本地检查：本run所有JSON文件（排除以 `.json` 结尾的媒体目录）、留存驱动脚本AST、新增04/索引本地链接及空白检查通过，RPent `git diff --check`通过。阶段limits与旧used加单次charge一致，最终run逻辑字节31,580,845与charge相等、低于1GiB；未新增消耗或重复运行已通过的Runtime验收。


## 10. 2026-11-12 第三批离线补正与接线

状态与历史解释见04 §1.15，评价草案见06末节。证据根 `artifacts/offline-20261112-batch3/`；Python为RPent `.venv/bin/python` 3.11.13，`rpent`解析至本工作区。聚合目录非Git；RPent仍在main/c57b134，本地origin/main领先2/落后2，没有联网更新上游。进入任务时已有的RPent暂存/未暂存/未跟踪改动保留，本轮没有修改RPent源码、提交或推送。

| 检查 | 结果与实际范围 | 日志/产物 |
|---|---|---|
| 首组接口定向集 | 62 tests OK：新有限响应/发布读取，以及原runtime plan/import/research兼容 | `checks.log` |
| 模型/processor/规则/记忆真实参数消费者与四任务输入 | 2 tests OK：隔离模拟accept后选版进入现有prepare_runtime；四任务草案输入选取及工具60/请求20，旧显式bundle仍可准备；无模型加载 | `runtime-consumption.log` |
| 正式真实回合离线消费 | 237事件、2个高层VLA、完整L0请求；等值导入幂等，合成defer成功，研究charge=[] | `formal-replay.log`、`formal-consumption.json`、`state/research/` |
| 增加来源/导入边界后的定向集 | 67 tests OK：包括unknown/invalid保留、同run换result拒绝、发布seed拒绝、合成产物/Candidate/暂缓及原发布状态转换 | `final-checks.log` |
| 旧研究接口兼容补验 | 45 tests OK；仅带正式view的请求增加evidence_views，无view的旧v1请求格式保持 | `v1-compat.log` |
| 部署用途最终补验 | 20 tests OK：有限响应、正式导入、参数消费者、原runtime plan/import；deployment须有效版绑定、不计release，旧作业绑定及回滚/过期决定拒绝 | `deployment-mode-final.log` |
| 最终静态检查 | 8份Python AST、JSON、本地Markdown路径、v9原profile/协议逐字节与第一批快照一致；Ruff E9/F、3个新文件format --check、CLI --help、bash -n、原pilot consumer和RPent git diff --check通过 | `static-final.json` |

以上测试集合有重叠，不相加为独立证据数。最后两组只因对应接口变更补验，未重跑前两批动态回合或无关全套。初轮广泛Ruff风格探查输出保留在 `ruff.log`，包括新文件初始格式/import和聚合core沿用风格；新文件已格式化，最终只对本轮Python做错误类lint及新文件格式检查，没有以重排全部旧core来消除历史风格差异。`format-final.log`保留格式工具既有DOC preview提示，不是语法失败。

命令、退出状态与适用关系见 `verification_commands.json`；最终源码列表/副本为 `source-manifest.json / source/`，已有目标的前件及增量为 `source-before/ / incremental.patch`，未生成哈希。正式导入版本为 `rpent-development-v2-action-origin`，真实run新增 `research-derivation-v2.json` 只索引新派生。旧offline-diagnostics及plan_adherence=false保留，04补充max_chunks表述歧义；没有把解释追认为事前注册或验收每段固定24。阶段账本不重复charge，真实active未写。

复用第一批错误传播、第二批真实执行与独立结果，以及此前未受影响的RequestGate/进程监督/MCP图像/安装链证据。本轮没有模型请求、外部转发、本地假服务、仿真、GPU、训练、下载、安装或设备访问；没有实际启动研究实验。模拟Gate接受/继承/回滚和合成研究响应只支持接口，不证明真实发布、继承、能力提升或自主研究。OS权限隔离、合格current/anchor、独立评价矩阵和真实训练往返仍缺。未改RST/Sphinx输入，不需重建Sphinx；未发PR，未跑全仓pre-commit/全部单测/CI矩阵。全部完成后停止，动态工作继续暂停。


## 11. 2026-11-12 第三批补件与第四批准备

当前状态、具体输入/命令/预算与用户待决见04 §1.16；评价依赖和评分草案修订见06末节。证据根 `artifacts/offline-20261112-batch4-preparation/`。复用前三批动态及未受影响离线证据，不重做第三批/重跑前两批。Python为RPent `.venv/bin/python` 3.11.13；聚合目录非Git，RPent main/c57b134，本地origin/main领先2/落后2，没有联网更新上游。原用户改动保留；RPent本轮只在tools.py增加观测绑定事件、在原bounded单测文件增加一个CPU断言，其它原改动未覆盖。

| 检查 | 结果与证据范围 | 产物 |
|---|---|---|
| 新补件首组 | 6 tests OK：技术诊断/非法数据、独立推进与Gate三态、JSON全类型无损、进程内传输替身、新许可/源码匹配/幂等重解析 | `checks-initial.log` |
| 完整矩阵与evaluator兼容 | 13 tests OK：补充完整矩阵True/False/None正式消费者及原evaluator缺槽/身份/结果语义；未启动任何矩阵任务 | `checks-final.log` |
| 推进证据收口补验 | 2 tests OK：指令额度、实际适用项、调用闭合、观测绑定、源引用可用性及Gate分支；信息不足保持None | `progress-final.log` |
| 无证据与本地解析修复 | 3 tests OK：空引用成功记录拒绝、无触发不dispatch、JSON解析错误后离线修复、Research/阶段charge均不重复、旧permit/源码变化拒绝 | `diagnostic-reparse-final.log` |
| 实际RPent观测事件生产者 | 3 passed、22 deselected：新inference绑定最新observation_update、原指令/观测/chunk连续性、推理故障保持技术无效；仅FakeEnv/FakeVLA，无网络或设备 | `rpent-focused.log` |
| 原始真实回合推进检查 | 第二批实际tool/segment/command/源码引用可定位；逐推理观测ID历史缺失，critical_progress=None；holding/坐标/完成提交按实际路径不适用，未新增原生成功条件 | `historical-critical-progress.json` |
| 同一回合表示与覆盖 | 原917431字节/字符；原压紧546216字节/字符；新211040字节/字符；actual prompt211358字节/字符。精确解码等于原JSON，237事件完整，token未测 | `transport-measurement.json`、`real-request-lossless.json` |
| 真实规则产物/Runtime预检 | 三算子均不可试验：chunk前缀被时序保护拒绝，两个finish缺当前触发证据；原上限未变，无Research调用 | `operator-preflight.json`、`preparation.log`、`dry-run-final.json` |
| 执行上限收口 | 仅补验受墙钟/存储响应准入及人允许对象筛选影响的RunnerTest；不重跑其它成功项 | `runner-final.log` |
| 最终静态检查 | Python AST、JSON、Markdown本地路径；Ruff E9/F、新模块格式、RPent两个受影响文件项目preview lint/format、git diff --check、CLI --help；原profile/协议/阶段账本不变、输出未启动、job charges=[] | `static-final.json` |

命令与退出状态见 `verification_commands.json`。上述测试集合重叠，不相加。RPent pytest仍有既有Unknown config option: timeout警告，使用shell timeout，不安装插件。初次格式检查只发现新增测试长行，已格式化；core初次lint的三个unused import已移除。后续只补受影响的观测绑定/完整矩阵/本地解析/上限项，未跑全仓pre-commit/全部单测/CI或动态作业。未修改RST/Sphinx输入，无须Sphinx构建。

完整L0保留在第三批不可变证据store；新的bounded传输只做可逆去重，实际新SDK路径使用紧凑表示，不是旁路摘要。没有适用于Research模型的现成本地tokenizer，未用VLA tokenizer或字符数冒充token。没有评估真实模型对编码的理解效果或节省收益；64k输入/4k输出参数仍为observed_only参考。技术诊断记录不变成正常任务失败；原native事实/unknown/故障和缺失保留。关键推进报告不足时保持None，不能凭原生成功或技术audit补True。

后续准备仅包含新的 `development-plan.json / development-request.json / development-permit-proposal-not-issued.json`、source-manifest及source副本；permit.authorized=false，当前全部算子不可试验，dry-run dispatch=false。旧research-plan等早期本轮草案及第三批派生原样保留、不恢复调度。A最多Research1+任务1（任务20请求/工具60），B为4起点+4候选且current/anchor共用，继承任务另1次。逐任务工具上限总600超过原阶段剩余517，资源缺口明确登记 `resource-plan.json`，不扩大预算或承诺全部收齐。阶段账本没有本轮新charge，真实active未写。

源码前件/增量/终件见 `source-before/ / incremental.patch / source/`，新source-manifest版本为batch4-preparation，无哈希；旧清单/许可不授权新代码。脚本和索引只记录离线工程证据。本轮未调用外部模型、启动假网络服务、仿真/GPU/训练/下载/安装/设备或实际候选激活，未恢复旧队列、提交/推送或委派。权限隔离、匹配发布矩阵、真实继承、5A完成及5B真实模型往返均未验收。动态工作继续暂停。

## 12. 2026-11-12 5A 数据与往返准备

当前唯一状态/预算/命令见04 §1.17，未生效协议见06末节。证据根 `artifacts/offline-20261112-5a-assets/`。复用前三批和§11仍适用的真实回合、故障/关键推进/codec/监督证据；没有真实Research、仿真/GPU/训练、安装下载、设备或本地网络服务。聚合根实际为main/c4d7fd9b、origin本地同位；RPent独立main/c57b134、origin本地领先2/落后2；最终静态检查纠正了旧摘要的非Git假设，未初始化仓库。用户改动和旧原件/派生保留，未提交/推送/委派。

| 检查 | 结果及适用范围 | 日志/产物 |
|---|---|---|
| 采集→实际export→重读 | 2 passed；25步有界机制用2步fixture覆盖批次边界/尾部/mask，实际VLA消费者记录指令/input/proposal/send/feedback；LeRobot真实导出重读10合成步，自主数据示范导出拒绝 | `collection-check.log`、`collection-current-export.log`、`collection-fixture/` |
| 新manifest与原Flywheel兼容 | 新manifest反例通过；原legacy source/success/export/dtype在修复Path序列化后5项通过，新增owner关节/速度无额外RPC1项通过；总6 passed/4 deselected | `manifest-final.log`、`collection-compat-final.log`（原manifest反例从RPent移至所属core，独立RPent不依赖聚合core） |
| 精确原始状态 | 1 passed：原始float64状态在input/raw_states完整保存，float32导出合约保留；命令与结果单列 | `raw-state-check.json` |
| 原生None反馈 | 1 passed/4 deselected：终止值None不变成False，pending及缺失保存 | `missing-feedback-final.log` |
| 初始动作合约 | 1 passed：formal reset复用action_specs缓存，保存真实frame/映射描述；0控制步的进程内fixture；原命令记录在`verification_commands.json` | `formal-contract-check.json` |
| 原生CPU数据consumer | 真实installed `build_openpi_sft_dataloader`读通合成诊断export，actions[1,10,32]、state[1,32]、native images[1,3,224,224]；无模型加载/训练；不构成真实示范准入 | `native-batch.log`、`native-diagnostic-batch.json` |
| 经验/trial/Candidate/stop与资格 | 首组6 OK，后续仅补受费用和查询变化影响分支：真实有限规则差异、跨版本合法来源、技术中断native事实保留、Candidate/stop及不重复执行 | `core-final.log`、`accounting-cycle.log`中通过项、`experience-feedback-final.log` |
| 计费与重解析 | 最终Runner1 OK，cycle账本相等断言通过；新增证据存储/输出尾部单列增量，重复有效解析收费账本完全不变，LLM与task费用各只一次；请求达到cap后只可记已发生存储，不重开作业 | `accounting-final.log`、`experience-feedback-final.log` |
| 独立evaluator采集消费者 | 1 OK，四任务fixture覆盖完整/不完整/缺失采集；逐数组validator实际接入，native outcome不变、reference_coverage三态 | `evaluator-final.log`；数组validator生产者见采集日志 |
| 资格/任务边界 | 最终5 OK：合格/未合格起点、accept/reject/未决、范围/过期/回滚；另1 OK：启动前过期绑定不创建输出、版本切换复用全局动态锁；均临时store | `qualification-final.log`、`safe-boundary-final.log` |
| 新prepare正路径 | 隔离配置/临时store，批准字段仅是fixture；实际生成合法新请求，only hint，旧job/许可不迁移、charges=[]，dispatch=false | `positive-prepare-final.log`、`positive-prepare-check.json` |
| 真实历史经验/输入 | 同一第二批237事件，完整cost/audit缺失与技术事实可查询；codec精确roundtrip。v5传输227079字节/字符、实际prompt237869；上一版prompt211358，原917431/压紧546216；token未测 | `historical-experience.json`、`transport-measurement.json`、`actual-prompt.txt` |
| 实际起点绑定 | 2 OK：实际request含完整base RunBundle和人工来源；许可/源码/原始响应重解析检查通过 | `base-binding-final.log` |
| 未生效预检与dry-run | hint有产物/消费者/历史交接机会，scope与permit未批准；其它3算子仍不适用；三个CLI默认dispatch=false，无实际输出 | `operator-preflight.json`、`dry-runs.json` |
| 静态收口 | AST/JSON、RPent项目preview lint/format、core错误类lint/新模块format、git diff --check、CLI help、本地路径/配对RST及原profile/protocol/phase字节不变 | `static-final.json`、`verification_commands.json` |

检查集合有重叠，不相加。早期日志保留失败原因：系统Python不支持项目类型语法，随后统一使用RPent Python3.11；导出检查默认cache只读，改为证据根显式cache；旧Path配置需要JSON化；单槽fixture不符合现有三组指标分母，改四任务fixture；新增存储记账暴露cap后记成本及defer剩余额度快照重解析问题，均已按实际消费者修复。没有弱化原断言；费用断言区分模型/任务与新存储，同时增加完整charges相等检查。

新派生版本及源码/前件/增量见 `source-manifest.json / source/ / source-before/ / incremental.patch`。源码清单无哈希。旧请求和派生留在原store；本轮v1—v5审查输入亦按新ID保留。所有真实permit proposal=false，真实active/资格及权威阶段账本未写。合成fixture复制后的原临时路径通过 `fixture-location-map.json`关联保留副本，不改原manifest，不冒充真实数据。

静态首轮将未修改的源码快照一并纳入lint，发现既有runtime_supervisor.py未使用elapsed变量；收敛为本轮改动文件检查，没有修改该模块；最终静态收口再次包含RPent lint/format并通过，未重跑未受影响的行为套件。

新增paired Flywheel RST记录分批采集与诊断用途；Sphinx executable在目标venv缺失，复用既有构建依赖未就绪证据，只做配对源码/本地链接检查，不安装。pytest有既有Unknown config option: timeout警告，检查用shell timeout。未发PR，未跑全仓pre-commit/全部单测/CI或未受影响成功套件。

三个检查点均为离线接线/格式/替身证据；实际Research理解、新数据采集质量/代价、candidate任务行为、匹配矩阵、可信OS权限隔离、真实激活/继承、训练数据准入和5B真实训练仍未动态验收。未完成的关系学习/H3准备不另加为首次5A门槛。动态工作继续暂停。


## 13. 2026-11-12 冻结收尾与GPU0准入停止

证据根：[bounded-20261112-5a-pilot-01](../artifacts/bounded-20261112-5a-pilot-01/)。新协议/profile/policy为 `*_5a_frozen_20261112_pilot01.json`，历史草案原件保留；状态唯一入口为04 §1.17，实际方案为06本轮冻结方案。

| 定向检查 | 结果与证据 |
|---|---|
| 旧八槽pilot_v9与新四槽静态一致性 | `scripts/check_pilot.py --protocol config/pilot_v9.json`及新协议各通过；task×seed数量、逐任务初态映射和分区隔离、资产/动作、预算/成功/治理检查保留 |
| 错误槽数/初态重叠 | 两个新协议反例均拒绝，见negative-protocol-checks.json |
| 共同profile和唯一hint差异 | 实际隔离产物核对hint false→true，其它bundle槽冻结；Runtime拒绝本轮release split，见common-profile-hint-check.json；非真实trial |
| 采集、筛选、费用和停止消费者 | 5项定向unittest通过，包含CollectionEvaluatorTest、两个ExperienceAndCycleTest及RunnerTest两项；新增必要L0反例后仅重跑CollectionEvaluatorTest，1项通过；不相加为6项独立用例 |
| 新prepare与合法历史输入 | 新store只有一个新job、charges=[]，不迁移旧job/队列/permit；完整237 L0与原正式store相同，原bundle/protocol保留；actual-prompt.txt为237778 UTF-8字节/237778字符，不是token数 |
| 新源码、完整input、许可绑定 | supported `5a-assets-offline`清单合约，当前源码精确字节副本/清单，无哈希；合法permit仅在内存校验，落盘authorized=false提案拒绝，实际签发数0 |
| 无trial不启动 | run_development_trial默认dry-run返回等待真实Research trial，两个输出目录不存在，无dispatch；见round-stop.json |
| 原阶段余额与有效消费 | 校验campaign/增量运行及余额算术，无未决成本，完整本轮上限可容纳；阶段账本字节未改，无重复charge，见phase-admission-check.json |
| GPU0准入 | 沙箱驱动不可见后仅宿主只读检查：PID3182707占用GPU0/5226MiB；触发用户明确停止条件，未换卡/抢占/重试，见gpu-admission-stop.json |

精确命令与输出保存在verification-records.json；纯离线验收程序为accept_frozen_pre_stop.py（记录当时的prepare后/pre-stop条件，不用于恢复作业）。该程序早期临时fixture在复制psi/rho/memory前保存bundle失败，调整依赖顺序后通过，没有动态请求或费用。最小检查复用§12未变化的采集/原生格式/Runtime/资格证据，没有重跑全套检查。

新增代码范围：check_pilot.py按实际数量及本轮窄授权验证；runtime.py消费允许split并绑定两项采集要求；research_cycle.py显式记录资源核算筛选通过；test_5a_assets.py区分可选数组与必要L0。源前件、新源码和本轮增量保留在证据根。本轮未改RPent源码、暂存区或已有其它成果。

本轮实际Research/Planner外部请求0、Runtime启动尝试0、工具/VLA预测/环境步/未知动作/回合0；完整动态墙钟和GPU包络0，input/cached/output0（无请求）、动态作业输出0字节。离线artifact占用另列final-static-check.json，不冒充运行成本。没有原生结果、技术有效性、critical_progress或采集完整性的本轮动态结论。无可执行permit、无trial/Candidate；prepared job已stopped，进程及输出无需动态清理，其它GPU作业未动。

未执行Research实际输入消费、hint动态消费、真实采集/结果回流、发布/资格/激活/继承、训练/加载专项、下载/安装、真机或提交/推送；训练数据仍not_admitted。源码/输入和历史合法开发事实可复核、再准备；不能作为新能力收益、发布或训练证据。当前动态工作继续暂停。


## 14. 2026-11-12 GPU阈值修订与真实有限Research/开发停止

证据根：[bounded-20261112-5a-pilot-02](../artifacts/bounded-20261112-5a-pilot-02/)，round-summary.json为程序事实汇总；唯一状态在04 §1.17，方案在06末节。

- 复用§13有效检查：进入pilot02时，原74文件逐字节一致，未重跑未变化全套；新增协议身份和profile引用静态检查通过，新源码快照77文件，见prepare-checks.json。
- 新prepare、实际prompt/full237 L0与历史正式store、唯一hint预检、phase/job完整余额、源码及两个单作业permit全部核对，见research-preflight.json、development-preflight.json、两个dry-run记录。实际prompt237778字节/237778字符；token使用真实HTTP usage另记。
- GPU准入改为其它作业合计≤12288MiB，实测5226MiB；仍GPU0一张、不抢占/换卡。用户明确批准OpenAI具体目的地址后才实际调度；前两次自动审批CreateProcess拒绝未执行。
- Research真实1请求、gpt-6.1-sol/medium，decision=prepare_rule_trial，唯一hint false→true，原响应保留；完整墙钟36.776347395032644秒，input94307/cached0/output891，cleanup确认。
- Runtime真实启动1次，开发position_swap_salad/seed1；采集初始化native action spec查询失败，worker无env_call。原生unknown、technical_invalid；hint加载true但未交付Planner，关键推进null、完整采集false、资源核算false。程序stop/Candidate=null，无重启、第二候选或额外模型请求。
- 已测开发完整墙钟/GPU包络各116.492241486907秒；模型加载88.17775344848633秒、setup30步、policy0步、unknown0步、VLA预测0、Planner请求0。tool_calls缺客户端记录，保留unknown并在phase/job登记未决；不把已记0当实际总数。
- 累计请求1、回合1、环境30步、完整墙钟153.26858888193965秒、GPU包络116.492241486907秒；已记存储1687269字节。Research phase原基数比job多计execution-costs.json的315字节，且output growth已覆盖它，已离线修正一次；原始执行产物未改。bounded_runner统一phase/job原始测量基数，新增RunnerTest一项通过，见post-run-accounting-test.json、storage-correction.json。
- 源码归属分开保留：source-manifest.json/source为真实调度前77文件，不被最终文档或记账代码覆盖；post-run-source保存收尾记账修正及文档，post-run-source-before保存其前件。该修正不是RSI候选，也没有再次执行Runtime验证。
- 清理证据：两个supervisor均cleanup_confirmed/remaining_process_ids=[]；本轮env/VLA/SAM3 PID已不存在，其它两项GPU作业仍在。没有真实发布矩阵、资格、激活/继承/回滚、5B、真机、下载/安装或提交/推送。

未解决：安装worker与native action-spec查询接口不匹配；客户端tool_calls未决；完整采集、实际hint交付、原生任务结果及关键推进未动态验收。保存的Research原响应、setup原始journal、错误、缺失manifest、finding和已测成本用于离线诊断，训练仍not_admitted，不能作有效开发成功或收益证据。阶段余额与未决成本见phase-after.json，动态工作继续暂停；本轮job已stopped，未恢复历史队列或旧许可。


## 15. 2026-11-12 后续Research high、账号来源与启动故障定位

证据根：[offline-20261112-research-high-diagnosis](../artifacts/offline-20261112-research-high-diagnosis/)。本轮是配置修改和离线诊断，未执行Research、Planner、Runtime、仿真/GPU或模型加载，没有改认证文件或阶段账。

- 后续Research请求工厂默认gpt-6.1-sol/high，new prepare及dispatch拒绝medium；SDK effort、permit与gate context同步high。旧medium响应允许离线解析。Planner medium及其审计/机会依据保持；原冻结配置/permit/运行原件保留，新暂停profile/protocol只改Research effort和必要新身份/准入元数据。
- 11项定向unittest通过：默认high请求/合法证据、历史medium解析且禁止新medium、observed-only请求/成本/可信transport、结构化提案、RunnerTest四项（含实际SDK effort离线替身、medium输入拒绝及单次费用）、经验/hint入口。命令与结果索引见verification.json；无真实模型调用。
- check_pilot.py对pilot_v9、pilot02和新high协议分别通过；新high/旧medium配置匹配其原profile，原资产、分区、动作/预算、成功和治理检查保留。旧pilot02 stage spec在新prepare入口被high一致性检查阻止，dispatch=false，无store/job改写。
- 模型、processor、control、Planner、memory、observations、collection、vla_segment及task_inputs与pilot02逐字段一致，见configuration-checks.json。RPent AGENTS同步Research high例外，Planner默认不变。
- 只读login status显示Logged in using ChatGPT；Planner/Research继承同一CODEX_HOME，未设置独立API key或endpoint环境变量。脱敏账号来源见authentication-source.json，不保存凭据、完整email/account ID或auth.json副本；无法从历史运行确定账户切换时间。官方认证说明：https://learn.chatgpt.com/docs/auth 。
- 根因：env.get_action_spec要求worker.env_call，而本机RLinf0.4.0的ReconfigureSubprocEnvWorker及基类无该方法，有get_env_attr/getattr管道。用生产方法AST和匹配安装worker接口的替身离线重现同一AttributeError，见root-cause.json；不声称完成真实接口修复或采集验收。
- 原离线采集合约用例直接给fake env.action_specs赋值，绕过RPC/真实worker；原HEAD已有不兼容调用，正式采集reset首次惰性读取时暴露。故障接口本轮未改；真实修复/完整采集与任务结果仍待后续工作。

只运行受本轮修改影响的检查；未重复完整套件、动态失败回合或模型诊断请求。源码前件、当前源码和增量无哈希保存；历史成本、unknown/technical_invalid及tool_calls未决保持。动态工作继续暂停。


## 16. 2026-11-12 原生动作合约修复与repair03真实开发

证据根：[bounded-20261112-5a-repair-03](../artifacts/bounded-20261112-5a-repair-03/)。状态唯一入口为04 §1.17，具体许可/方案见06末节。

- 复用add-robot的薄适配：新增robots/libero/action_spec.py在原生worker factory构造后提取实际合约，沿支持的get_env_attr读取；不编辑RLinf/robosuite、不替换控制循环。固定task/controller元数据跨普通reset保留，不新增任务reconfigure支持。Runtime源码快照纳入新helper。
- 14项pytest定向检查通过：两种原生controller布局、facade RPC/getattr、缺失合约拒绝、任务/seed选取、初始化失败成本关闭、正式采集缓存合约保存；原task-selection fixture改为真实class形状以匹配生产构造契约，原身份/边界断言全部保留。preview lint、format及core错误lint通过。
- actual installed worker.get_env_attr与robosuite loader函数AST在离线接口夹具中实际执行，原生Panda配置先按实际loader展开arms/right，再走真实factory及facade RPC；未启动worker进程/仿真/GPU。见installed-worker-contract-check.json。早期夹具直接读取未展开JSON失败，按真实loader修正后通过，没有修改控制规则。
- 已核准旧pilot02启动前工具成本0，依据冻结代码顺序、完整事件及零请求/清理证据；旧原始结果/日志不变，阶段硬成本未决项解除，没有重复收费。
- 新common profile/协议/源码84文件、原Research trial provenance、新job/experiment/permit及完整余额检查通过。旧pilot_v9、pilot02与新repair03协议静态检查通过；无新Research请求、旧job/permit迁移或模型诊断请求。
- 新Runtime实际启动1次，修复的native action-spec链真实消费通过；position_swap_salad/seed1在20次VLA预测/100策略步原生成功，总130步含setup30、unknown0。规则加载、critical_progress、collection与硬资源筛选通过，hint交付条件未触发。
- 独立完整validate_episode检查一份episode/100策略转移通过，原始状态/两路图像/输入/proposal/发送反馈与native action spec保存，无pending/unflushed/errors。见independent-collection-check.json；训练仍not_admitted。
- 第6次Planner请求已准入后，native-success controller关闭SDK，代理下游BrokenPipe；仅前5请求有terminal usage。保留6次成本和observed input75467/cached67584/output420下界，不补第6为0，不重新请求。独立结果unknown/technical_invalid、complete_usage=false，程序stop/Candidate=null，见transport-closure-diagnosis.json。
- 新消耗6模型/2工具/1回合/130步，完整launcher墙钟/GPU包络各173.34670307300985秒；后续独立CPU数组审计另计入phase/job墙钟，不重复GPU/回合。新存储44806238字节（含派生）。原Research和失败尝试费用均保留。
- supervisor退出0、cleanup_confirmed/remaining_process_ids=[]，宿主核对三个本轮daemon PID已消失，其它GPU作业未改。没有第二次任务、发布矩阵/资格/激活/继承、训练、真机或安装/提交/推送。

verification-records.json保存命令与结果；source/source-manifest是调度前冻结源码，source-before保留修改前件，最终文档另存post-run-source。阶段账及新job停止状态均保存。新的待办是原生成功关闭时的在途模型流/usage处理和实际hint交付覆盖；本轮不改评价标准或人为升级结果。动态工作继续暂停。


## 17. 2026-11-12 原生成功后的模型准入/流收尾离线修复

证据根：[offline-20261112-native-stream-closure](../artifacts/offline-20261112-native-stream-closure/)。本轮无新模型请求、仿真/GPU、运行许可、训练或阶段账消费；真实repair03保持原判。

| 检查 | 结果 |
|---|---|
| RequestGate loopback | 12项通过；早期准入关闭/本地等待，断管前usage记录，有界drain/严格原墙钟内取消，上游故障仍错误，错误effort/端点不转发，计数和隐私日志保留 |
| 独立runtime closure | 4项通过；预期native关闭与真实故障区分，身份/时序/路径匹配，缺usage为None，原始结果不改 |
| budget closure | 3项既有定向回归通过；错误/未决不能转成普通失败或成功 |
| 费用重解析幂等 | 1项通过；模型/任务及派生费用不重复charge，无额外执行 |
| Runtime原生反馈边界 | 5项通过；成功反馈在媒体前关准入，truncation不冒充成功，早期标记不关闭Recorder任务，完整成功才走原SDK正常收口 |
| 完整结果链 | 12项通过；生产Toolkit/原生finalizer/独立closure/audit/evaluator/import，新增本地下游关闭正路径、真实上游错误、usage缺失及代理清理未完成反例 |
| 真实repair03只读回查 | audit保持未通过/outcome unknown；原生success、完整采集、8份所选原始文件及阶段账与既有phase-after逐字节不变；无导入、模型调用或记账 |

合计37个不同用例，分批有重叠，重跑次数不相加。初始完整结果链夹具错误地将两条初始JSONL当单JSON读取，修为逐行保留全部事实与原断言后通过；未删事件或放宽断言。post-scope检查复用未变化结果，未跑全仓测试/CI/Sphinx（无Sphinx源修改），未安装依赖。

实现范围：RequestGate可取消上游I/O、有界等待/清理和先记录后转发；RuntimeJournal/LiberoPrimitives将新请求禁入与完成落盘分开；runtime_closure/runtime_acceptance共用严格的transport_issues并消费cleanup_complete。新增事件/标记只用于请求准入，不替代原生成功定义。真正transport_error、request_incomplete、错身份/时序及缺usage保持失败/未决，既有Gate和评分不变。

源码前件、当前副本、增量和精确命令/结果在证据根。repair03-readonly-recheck.json确认旧原件、旧费用、Candidate=null及动态暂停不变；原第6usage不能从新代码追补。真实新版本SDK/仿真验证仍待后续单作业授权，未自动恢复队列或使用旧permit。


## 18. 2026-11-12 repair04新冻结与GPU准入停止

证据根：[bounded-20261112-5a-repair-04](../artifacts/bounded-20261112-5a-repair-04/)。唯一状态在04 §1.17，授权范围与停止规则在06末节。

| 检查 | 结果 |
|---|---|
| Git/环境接手 | 聚合main/c4d7fd9b与本地origin同位；独立RPent main/c57b134相对本地origin领先2/落后2；保留原工作树和staged成果，无fetch/提交/推送；既有Python3.11.13解释器 |
| 既有离线证据复用 | §17全部12源码/测试/文档副本逐字节匹配；§16动作合约helper、env/client/flywheel/toolkit匹配；复用14项/37项及安装接口证据，不重跑成功检查 |
| 新协议/冻结 | pilot_v9、pilot02、新repair04协议检查通过；89文件逐字节冻结，覆盖实际生产者/消费者，无哈希和源码清单校验改动；共同工程修复及唯一hint差异核对通过 |
| 原Research与输入 | pilot02 validated-response、trial、rules、diff精确副本及medium provenance保留，新experiment绑定；相同模型/processor/记忆/输入/采集/控制；Research新增0 |
| 单次预算/阶段余额 | 20请求/60工具/900秒/2000步/900GPU秒/1GiB及Planner600秒/20turns、恢复1/reset0、启动1/重试0全部绑定方案；无硬成本未决，余额充足，未改变历史账 |
| GPU准入 | 两条只读nvidia-smi宿主查询；物理GPU0其它作业21826MiB>12288MiB，admitted=false；未等待/重查/抢占/换卡 |
| 具体许可与dry-run | GPU检查断言在Runtime/许可签发前停止；仅development-permit-not-issued.json（authorized=false），既有development入口dry-run退出0，其它静态项就绪，缺未签发许可和GPU为阻塞 |
| 程序停止 | 既有ResearchTools.defer记录确定性准入停止，新job stopped；未伪造Runtime audit/import/finding或Candidate；模型/Runtime/预测/环境/动态成本均0 |
| 发布资源算术 | 当前9回合；8槽原包络可覆盖，8槽加继承的540工具比515余额多25；若后续开发使用1回合则只剩8；无Candidate槽清单或可执行发布许可 |

没有运行--execute、网络/模型冒烟、基线、全仓测试/CI/Sphinx、模型专项、训练、设备、依赖安装下载或子代理。全部动态流收尾分支未覆盖，不能用本次准入停止建立技术验收通过；repair03原件与第6usage缺失不改。未启动任务无逐请求终态/usage，采集输出未创建，训练仍not_admitted。阶段账无新增charge、保持逐字节不变；本轮冻结/离线文件逻辑字节另测于final-checks.json，不混入未启动的动态任务成本。

源码/协议与绑定在source-manifest.json、input-source-binding.json；工程修复相对repair03的精确差异在common-engineering-delta.json/patch，未改生产代码。final-checks.json记录文档路径/语义、JSON、冻结副本及旧原件/阶段账检查；最终文档另存post-stop-source，调度前冻结副本不覆盖。无本轮动态进程或在途请求，清理结果为空；动态工作继续暂停。


## 19. 2026-11-12 repair05唯一开发技术有效、关键推进审计停止

证据根：[bounded-20261112-5a-repair-05](../artifacts/bounded-20261112-5a-repair-05/)。唯一状态在04 §1.17，协议与GPU范围变更见06末节。用户允许GPU0/1任选一张，原其它作业显存阈值和全部限额保持。

| 检查 | 结果 |
|---|---|
| Git/环境与冻结 | 保留聚合根和独立RPent已有工作树；local refs状态保持、Python3.11.13；91文件精确副本，实际生产源码与repair04相同，无哈希或生产代码修改 |
| 证据复用与输入 | 复用§16/§17有效定向证据；原pilot02响应/trial/rules/diff逐字节相同，medium来源保留，Research新增0；共同修复、唯一hint差异及模型/processor/记忆/输入/采集/控制匹配 |
| 新协议/准入 | 新协议静态检查及既有development入口dry-run通过；实际trial/完整rules/diff/源码/输入/采集/输出和完整限额/阶段余额由新permit绑定；GPU0其它作业5226MiB准入，GPU1为21782MiB，选择一张GPU0 |
| 唯一真实Runtime | 启动1次，position_swap_salad/seed1/development；原生success、outcome success、valid/completed，独立audit20项全true，prediction29/policy145/setup30/total175，unknown0，无中途reset |
| 原生成功时序 | feedback214→admission215约0.001655秒；完整native_completion219晚6.030909秒，在状态/媒体保存后；media_state_end为外层计时，不是另一次落盘 |
| 请求/流清理 | 9次response.completed/完整usage/HTTP200；成功后新准入0、在途0、本地拒绝0、上游转发9；无transport_error/request_incomplete/downstream_closed；cleanup_complete、unfinished=[]，代理清理0.0028793951496481895秒 |
| 动态覆盖边界 | 早期禁入与完整完成分离、正常流及最终usage/清理实际通过；成功后新请求拒绝/等待、下游断管、在途drain/超时取消和真实上游故障未动态发生，复用§17，不追加覆盖 |
| hint消费 | 已加载；首段24chunk自然用尽时SDK工具结果实际交付；后续可见说明及新子目标/max_chunks16，5chunk成功。明确hint归因/因果收益未建立 |
| critical_progress与程序筛选 | Planner用已计数read_text_file读取stop-that-shit/SKILL.md，现有progress对该实际前提缺独立审计而unknown；其它五项筛选true，程序finding→stop/Candidate=null；未改规则或人工放行 |
| 完整采集 | 一份episode/145转移；两路图像、原始精度状态、任务/子目标、29组输入/proposal/映射/native action spec；validate_episode(spec=LIBERO_SPEC)完整数组检查通过，pending=null/unflushed0/errors=[]；训练not_admitted |
| 成本与清理 | 模型9/工具4/回合1/175步；完整launcher墙钟/GPU各243.67328591598198秒；额外CPU审计2.2496391721069813秒单计，存储67401578字节；supervisor清理确认，所属三个PID已消失，代理无在途 |
| 发布准备限制 | Candidate空，不建实际候选四槽清单或可执行许可；剩8回合/511工具可覆盖8槽原包络，再加继承缺1回合/29工具；Gate和效率仅诊断保持 |

逐请求usage（cached为input子集；请求6的cached0是实际返回值，未用0补缺失）：

| 请求 | 上游终态 | input | cached input | output | 流清理 |
|---|---|---:|---:|---:|---|
| 1 | response.completed | 12512 | 3712 | 88 | HTTP200，正常返回 |
| 2 | response.completed | 13139 | 12416 | 59 | HTTP200，正常返回 |
| 3 | response.completed | 16269 | 13056 | 151 | HTTP200，正常返回 |
| 4 | response.completed | 18335 | 16384 | 72 | HTTP200，正常返回 |
| 5 | response.completed | 18435 | 18304 | 31 | HTTP200，正常返回 |
| 6 | response.completed | 18494 | 0 | 44 | HTTP200，正常返回 |
| 7 | response.completed | 18566 | 18432 | 44 | HTTP200，正常返回 |
| 8 | response.completed | 18638 | 18560 | 42 | HTTP200，正常返回 |
| 9 | response.completed | 22395 | 18432 | 125 | HTTP200，正常返回 |

合计input156783/cached119296/output656；本轮usage无缺失。旧repair03第6usage仍unknown，原生成功不清除旧技术故障。token observed_only，无token/货币硬上限承诺。成本分项加载/工具/媒体/cleanup重叠，仅完整launcher与额外独立CPU审计相加。

per-request-closure.json、native-stream-closure-check.json、hint-and-screening-check.json和critical-progress-check.json保存时序、SDK交付/后续可见规划及筛选缺口；independent-collection-check.json保存完整数组证据。phase-before/after与独立审计charge、原program cycle/finding/stop均保留；模型/回合无重复charge。最小下一步是离线审计实际read_text_file前提，不需新模型或Runtime来读取现有证据。

未重跑已通过的离线套件、网络/模型冒烟、基线、全仓检查、Sphinx、安装下载、训练、资格/激活/继承/回滚、硬件、提交/推送或子代理。最终文档另存post-run-source，不覆盖运行前冻结副本；运行原件不改。final-checks.json记录新增产物JSON、源码/输入精确副本、费用一次性核对、文档链接/语义和Git空白检查。动态工作继续暂停。


## 20. 2026-11-12 repair05实际读取审计、不可变重解析与首次release停止

唯一状态和完整费用在04 §1.17，派生release协议在06末节。证据根：[offline-20261112-repair05-audit-closure](../artifacts/offline-20261112-repair05-audit-closure/) 和 [release-20261112-repair05-seed3](../artifacts/release-20261112-repair05-seed3/)。不新增Research/诊断/总结请求、development Runtime、子代理或提交/推送。

| 检查 | 实际证据与结果 |
|---|---|
| 实际read_text_file | rpent MCP/具体Toolkit/CommonTools/普通MemoryManager权限与record_read全链核准；SDK参数12000、完整8197字符、未截断，调用ID/返回/L0终态/工具计数对应；readonly未当副作用或权限证明 |
| 缺失旧源码来源 | 8模块以运行前Git状态/已有c57b134版本/launch来源/冻结具体构造逐字节核准，明确事后来源关系；旧快照未增加当前文件 |
| 全部后续关键推进 | 指令/max_chunks24与16/states、29次最新观测绑定、proposal→五步native发送/服务执行mask、段终态、setup及前提通过；不适用项有具体消费者；其它工具和混合几何不放宽 |
| 读取/修订/既有5A定向组 | 13项unittest通过，verification-0.log；覆盖正常读取、缺证/错身份/错来源/错误路径/截断、修订Candidate与stop、旧记录及单次charge、停job不重开；原关键推进true/false/unknown两项此前通过并保留 |
| release派生 | 新定向1项通过，含拒绝改变chunk/划分；原development结果不改protocol/run身份；旧/新协议静态资产检查通过 |
| 初始资格/Runtime接口 | 改为共享纯检查的原资格3项通过；Runtime组8项通过。另1项既有错误消息断言失败，已用source-before复现相同失败：旧测试期待must explicitly use，现有实现返回model/effort does not match，错误模型实际仍拒绝；未改测试或消息 |
| 发布驱动及新增收尾 | 新4项pytest通过：GPU阈值/未知显存拒绝、无卡不签许可不启动、禁止恢复、普通有效failure完成起点四槽再由不合格停止候选、launcher后处理错误到独立消费者；同函数复用既有真实生产者CPU夹具 |
| 真实修订消费链 | --reparse actual-read-v2 → 新import/collection/finding/cycle → submit_candidate；完整六项true。164原输出/493既有非财务记录逐字节不变；第二次同CLI返回同cycle，job/phase字节不变，模型/Runtime重执行0。原stop/Candidate=null保留，新的Candidate由程序产生 |
| 费用来源 | 原模型/工具/回合/环境/墙钟/GPU/存储不重复charge，离线CPU/新派生存储单列。Candidate引用权威阶段完整历史，包含原Research和旧失败；新增发布实际消费继续加原账 |
| 共同版本/快照 | 109文件精确source及实际Runtime code_snapshot一致，无新哈希；实际RPent源码与repair05一致，AST证明Runtime仅扩展快照列表。源码覆盖通用工具、注册、权限、采集、代理、关键推进和回流/评价；执行及评价源码引用分开 |
| 新release准入 | 新release-only协议/派生Candidate/完整8槽最大包络/硬成本/源码检查与逐槽dry-run通过。首槽GPU0，卡0/1其它作业各1744MiB；新具体permit，启动1次，未恢复旧job/output |
| 实际首槽 | base_salad/seed3，24预测/120policy＋30setup/150总步、环境原生success；独立unknown/invalid，critical_progress=true，两者分开。SDK本地429保留错误导致exit1，停止其余7槽 |
| 请求/代理 | 3上游HTTP200/response.completed/usage完整，input41587/cached34048/output285。原生边界后新转发0、在途0；本地等待/拒绝各1。SDK错误为通用429、will_retry=false，没有实际SDK重试事件，非已观察到上游429。代理cleanup_complete/unfinished=[] |
| 本地成功后收尾缺口 | RequestGate正常等待完整落盘后本地拒绝，SDK不保留详细拒绝消息，Recorder现有精确消息识别未命中；保留原audit/result/stop，不以原生成功升级valid。不追加请求制造覆盖。BrokenPipe、在途drain/取消、上游故障和超时取消仍复用§17，未动态触发 |
| 采集/模型边界 | 新一份120转移完整数组复核通过，pending=null/unflushed0/errors=[]；原repair05的145转移保持。controller applied/setup轨迹/未返回RPC转移声明缺失，训练not_admitted。新起点hint=false、候选未启动；开发hint交付/可见后续规划成立，因果收益不成立 |
| 独立矩阵/Gate | 全8槽保留；起点unknown4/invalid1，候选unknown4/invalid0；初始资格纯检查失败。现行critical_preserved对关键旧能力至少1次成功要求未建立，程序Gate reject，未改成手工inconclusive/accept；零贡献分数不是候选能力为0或退化证据。没有真实资格/OS权限/激活/继承 |
| 清理/后续 | 完整墙钟/GPU各171.4808468609117秒；supervisor确认清理，宿主PID143759/143760/143761不存在，代理清理0.0036198990419507027秒。原8槽＋继承包络缺1回合/29工具；实际剩7槽再加继承仍缺1回合，工具无缺口。新动态需新授权/新证据，不恢复本轮矩阵 |

核心命令：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core/src:core/tests:RPent RPent/.venv/bin/python -m unittest test_read_text_audit test_trial_reparse test_5a_assets -v`；`python -m unittest test_release_preparation -v`；`python -m pytest core/tests/test_release_driver.py -q -p no:cacheprovider`。真实重解析和发布命令、退出码/日志分别在 actual-reparse-check.json、preflight.json、driver-stdout.log 和 driver-stderr.log；Ruff错误检查、Python AST及两层git diff --check通过。首次Ruff因聚合根只读缓存失败，改用/tmp缓存后通过，没有装包。未为形式检查重跑全仓/CI/Sphinx，未修改Sphinx输入；§16—19仍适用证据复用，不能替代本次触发失败分支的验收。


## 21. 2026-11-13 SDK 收尾、三态 Gate 与 release06 完整后续

唯一状态见04 §1.17，协议见06末节。证据根：[离线收尾](../artifacts/offline-20261113-release06-closure/)、[全新 release06](../artifacts/release-20261113-repair05-seed3-sdk06/)。使用既有 Python3.11.13，不安装依赖。主代理＋3直接gpt-6.1-sol/high编程子代理，无继续委派；各自文件/临时数据分离，真实账/许可/GPU/Runtime仅主代理操作。一次已完成子代理只读关键复核后，owner修复了预算拒绝SDK/L0关联和启动缺证的已测费用保留；歧义工具费用通过真实_phase_charge临时账进入unresolved，不传None造成算术异常。

- A：保存真实SDK generic429形状、通知error/failed/exception顺序、同步/异步inflight写入排空、超时/清理错误保留、错误身份/标记/媒体/L0/usage反例；RequestGate12项仅loopback、独立closure12子场景、原生/预算27项及后续实际drain回归通过。详见 [A验证索引](../artifacts/offline-20261113-release06-closure/verification/A/verification.json)，重叠用例未叠加。
- B：最终15工具审计＋3Gate＋10evaluator＋29core＋17batch4/assets＋3budget，共77不同离线用例通过；精确命令/复用范围在 [B索引](../artifacts/offline-20261113-release06-closure/verification/B/validation-commands.txt)。已确认违规跨缺槽保留，一般技术错误/缺配置不伪造safety/compatibility失败。
- C：7项修订→筛选Candidate/stop→release派生完整离线链通过；扩大关联检查初轮61项通过，与7项定向有重叠，不叠加。成本测试旧重复证据夹具在旧源码复现后，只修第二实验来源，原保护和全部成本断言保留，单项通过。见 [C日志](../artifacts/offline-20261113-release06-closure/verification/C/verification-3.log)。
- 主集成：发布调度、完整Toolkit→原生finalizer→独立closure/audit→evaluator、SDK诊断透传等23项＋2子场景通过；一项downstream夹具原缺SDK/launch-context，补齐证据后独立通过，不放宽审计或断言。初次pytest受宿主/home/xzh/__init__.py影响，使用--import-mode=importlib和显式rootdir消除收集错误，无依赖安装。精确命令/结果在 [集成记录](../artifacts/offline-20261113-release06-closure/verification/main/integration.json) 和 [修复后用例](../artifacts/offline-20261113-release06-closure/verification/main/downstream-fixture-final.log)。
- 真实旧repair05新revision：六筛选全true，程序Candidate；1104历史文件字节不变、原stop/Candidate=null及stopped job保留、重复job/phase字节不变且不重复charge；新增CPU9.308863秒/派生存储另计。原始与派生来源、不可变snapshot和全阶段费用范围见 [实际执行](../artifacts/offline-20261113-release06-closure/actual-reparse-check.json)。
- 新共同源码221文件逐字节/AST、原release05与新release06静态协议、源码清单、全9槽包络与dry-run、两层git diff --check及Ruff通过，见 [最终准入](../artifacts/release-20261113-repair05-seed3-sdk06/preflight.json)。未生成新源码哈希、未提交推送。
- 旧release05原Gate=reject，最终v6重评价=inconclusive；旧unknown/invalid/缺SDK终态/七槽未启动及全部费用不改。新八槽由程序独立evaluator/Gate消费：4启动、4未启动、冻结起点资格纯检查=False（停止后v5补审通过）、Gate=inconclusive，③尚未闭环。这是实际Runtime与独立评价证据，子代理审查不替代Gate或OS隔离验收。

停止后新增真实finish分支补审：5项反例＋2项既有ProgressTest通过，_control_checks AST不变；v5评价中四个起点critical_progress均true、纯资格检查通过，原始资格失败与停止保持，候选四槽仍未启动。全新finish-audit-v4 revision完整筛选产生Candidate，原件1104份/费用幂等复核通过；没有恢复动态。actual Runtime源码221文件与posthoc评价源码222文件分开保存。[v5报告](../artifacts/release-20261113-repair05-seed3-sdk06/post-evaluation-v5/report.json)。实际old_drawer槽动态复现generic429并success/valid，SDK缺1终态保留、L0闭合、状态/媒体/三个上游终态usage/清理全部核准；真实上游429/其它故障及超时取消仍以离线证据为限。

费用/usage/SDK-L0/源代码/采集/清理分别保留在发布根；原Research与失败不重置，历史repair03缺usage不补零。hint开发交付和后续规划成立，因果收益未建立，效率10%例外不启用；采集训练not_admitted。④真实资格、权限、激活、继承、回滚均未执行，动态继续暂停。未动态覆盖的故障分支以离线证据为限，不能从新成功槽推出真实上游429/断流/清理失败也已动态验证。


## 22. 2026-11-13 release07 共同冻结、新八槽及结束

当前状态唯一入口04 §1.17：实际4/8启动，Gate=inconclusive，③尚未闭环，结束后动态暂停。新release07严格scope/身份及费用时点闭合，没有新增Research或development Runtime，没有真实资格/权限部署/激活/继承/回滚/训练/依赖下载。

- 派生/费用：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core/src:core/tests:RPent RPent/.venv/bin/python -m unittest test_release_preparation test_pilot_scope -v` 首轮3方法通过；加入反向身份→scope绑定后，仅重跑 `... -m unittest test_pilot_scope -v` 的3方法，最终4个不同方法通过。覆盖新旧scope正例、错误预算/划分/身份/重试/启动数及未知scope落入16槽分支反例；源成本与快照核准、后续准入分开、实际账/job/源Candidate不改及幂等。完整命令/子场景见 [A报告](../artifacts/release-20261113-repair05-seed3-finish07/verification/A/verification.json)。
- driver：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core/src:RPent RPent/.venv/bin/python -m unittest discover -s core/tests -p test_release_driver.py -v` 8项通过，含固定任务/重复run或output在GPU准入前拒绝、关键推进unknown立即停止、有效failure继续、J=1仍完成候选、无GPU不签permit、不恢复及继承预留不扣费。[driver记录](../artifacts/release-20261113-repair05-seed3-finish07/driver-test-evidence.json)。
- 未变化的finish/SDK/L0/closure/evaluator/Gate复用保存通过证据：正常原生停止、真实generic429+SDK缺终态/L0闭合、position_swap failure/valid/finish关键推进、上游/错身份/缺标记媒体usage清理反例、缺证inconclusive与已确认违规reject、原VLA前提。源码逐字节保持，未重跑SDK修复。历史主集成记录含中间失败，须配最终downstream单项通过日志，不把中间JSON写成全绿。[覆盖映射](../artifacts/release-20261113-repair05-seed3-finish07/verification/B/admission-evidence.json)。
- `... prepare_release07.py` 实际产生Candidate与8槽新计划，派生重复同值、阶段账不重复charge；`scripts/check_pilot.py --protocol config/pilot_5a_release07.json` 通过；227冻结文件逐字节和Python AST通过，错误类Ruff、两仓git diff --check通过。完整argv/结果及最新权威预算复算见 [preflight](../artifacts/release-20261113-repair05-seed3-finish07/preflight.json)。完成的B复用一次有边界只读复核通过，没有以审查代替独立程序评价。[复核](../artifacts/release-20261113-repair05-seed3-finish07/verification/B/final-key-path-review.json)。
- 主代理实际执行上述冻结driver，每槽前只读GPU0/1显存门槛并签发独立permit，串行单卡；无中途源码修改、补槽/重试或跨版本拼接。独立evaluator计算完整8槽分母，起点资格纯检查和Gate程序消费，原判据不变；起点old_drawer有效失败使关键旧能力准入未满足，其余硬约束true；当前verdict=inconclusive，理由为candidate result incomplete/unknown/invalid；critical old capability evidence incomplete。[执行记录](../artifacts/release-20261113-repair05-seed3-finish07/execution-record.json)、[独立Gate](../artifacts/release-20261113-repair05-seed3-finish07/independent-gate.json)。
- 每请求终态/usage、SDK原错误与缺口、L0、完整媒体/采集、源码和owned进程清理分别保留，本轮动态覆盖与离线反例分列。新增费用按实际记权威账，源Candidate继承快照不重charge；完整九槽包络只作准入保护。编程协作usage与项目Runtime分列，缺失不填0、不估货币账单。[证据与费用汇总](../artifacts/release-20261113-repair05-seed3-finish07/round-summary.json)。

本轮没有PR，未运行全仓pre-commit或无关全量测试；未安装依赖。仅更新独立Markdown，Sphinx构建不适用；本轮新增链接和当前实现语义另作检查。真实上游故障/超时取消等未动态遇到分支仍以对应离线证据为限。训练not_admitted；缺证/停止原因见独立Gate与逐槽审计；未启动槽保持完整分母。当前授权不恢复、不重试；下一步先离线关闭具体缺口，再取得新共同版本/完整新矩阵的明确授权和预算，成功前语义变化另需development验证授权。④未执行。


## 23. 2026-11-13 release08 比较契约修复、全新矩阵与结论

当前状态唯一入口04 §1.17：8正式槽、8实际启动、0技术恢复、0未启动，独立Gate=reject，③已闭环，动态暂停。起点A=True，资格B=True；比较与部署资格分开，④未执行。

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core/src:core/tests:RPent RPent/.venv/bin/python -m unittest test_pilot_scope test_release_preparation -v`：9方法通过；新scope/身份/预算/划分、8正式与10启动/2恢复/单槽1恢复及未知scope回退反例、历史scope、源Candidate成本时点/幂等。[主检查](../artifacts/release-20261113-repair05-seed3-compare08/verification/main-checks.json)。
- `... python /tmp/run-release08-driver-check.py RPent/.venv/bin/python -m unittest test_release_driver.Release08DriverTest -v`：15方法全类通过；随后针对只读复核发现的确认关键推进违规遗漏，新增False→reject/None→inconclusive方法并连同2项现有选择/违规用例通过，共16个不同新增方法；历史8项复用前轮通过。包括A/B分文件、J=1完整候选、有效failure不恢复、8/10/2/1、恢复前清理/成本/GPU/permit、局部unknown继续、首个有效尝试、全部尝试费用与确认违规保留。原停止仅0启动且精确账项/清理例外可进入首次矩阵，任何已启动仍拒绝。[driver精确命令和日志](../artifacts/release-20261113-repair05-seed3-compare08/verification/driver/)。
- `... RPent/.venv/bin/python -m unittest core.tests.test_release_comparison -v`：11方法通过；`... -m unittest core.tests.test_core core.tests.test_gate_tristate core.tests.test_evaluator -v`：42方法通过。覆盖能力失败可比但资格failed、技术缺证/旧能力标签unknown或冲突不放行、候选base与每项旧能力、三态与硬违规优先、comparison accept无法资格/激活/部署。[评价证据](../artifacts/release-20261113-repair05-seed3-compare08/verification/evaluation/)。
- 185文件源码逐字节/AST、错误类Ruff、两仓git diff --check通过。唯一只读复核指出critical_progress=False遗漏，修复与定向证据由主代理核准后重新冻结，运行后未混用源码。[准入](../artifacts/release-20261113-repair05-seed3-compare08/preflight.json)、[实际源码绑定](../artifacts/release-20261113-repair05-seed3-compare08/source-bindings.json)。
- 独立评价修订v2：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=artifacts/release-20261113-repair05-seed3-compare08/evaluation-revisions/v2/source/core/src:RPent RPent/.venv/bin/python -m unittest discover -s artifacts/release-20261113-repair05-seed3-compare08/evaluation-revisions/v2/tests -p test_read_text_audit_v2.py -v`，14项通过；仅副本read_text_audit.py变化。默认参数、null/非法值、截断、路径/身份/源码、read缺终态和非read缺终态无合法local-stop反例均覆盖。[修订与检查](../artifacts/release-20261113-repair05-seed3-compare08/evaluation-revisions/v2/source-manifest.json)。主代理对四起点正式重评后，用原冻结启动函数执行剩余候选，`PYTHONDONTWRITEBYTECODE=1 RPent/.venv/bin/python -u /tmp/release08_continue_matrix.py` 的保存源码见[主代理调度](../artifacts/release-20261113-repair05-seed3-compare08/operator-continuation.py)；未恢复旧Runtime。实际SDK/L0/finish/closure生产者未改，复用§20—22有效证据；本轮实际SDK故障/缺终态以逐请求及SDK报告记录，不补造。[全部尝试](../artifacts/release-20261113-repair05-seed3-compare08/SDK-L0-and-attempt-results.json)。
- 主代理运行 `PYTHONDONTWRITEBYTECODE=1 RPent/.venv/bin/python -u scripts/run_release_comparison.py --plan artifacts/release-20261113-repair05-seed3-compare08/matrix-plan.json --execute`；逐次具体permit/只读GPU/单卡串行，程序独立evaluator/Gate统一评价，current=anchor同一完整新起点。A/B/比较Gate/部署许可分列，没有以子代理审查替代正式评价。[Gate](../artifacts/release-20261113-repair05-seed3-compare08/independent-gate.json)。
- 临时基线756257057字节越界与原0启动停止保留，用户最新精确授权后仅排除已清理份额的本地准入计数；全部费用仍记阶段账。三项阶段上限只修订一次；每请求终态usage、SDK原错误与缺终态、L0、完整媒体采集和清理分别保存，缺失不填0。编程协作usage单列，不估货币账单；离线监督墙钟去重，不重复计内部测试和Runtime等待。[费用与证据](../artifacts/release-20261113-repair05-seed3-compare08/round-summary.json)。

本轮无PR，未做全量CI或安装下载；只改Markdown，Sphinx不适用，新链接和语义另核准。未动态遇到的恢复/等待/真实上游故障/取消等继续以离线证据为限，历史原件/Gate/费用保持，hint因果收益未建立，训练not_admitted，④未执行。


## 24. 2026-11-13 全局长任务规范与④方案核准（仅文档）

全局规范覆盖所有长任务。04 §5.1逐项评估八项机制，按风险影响、操作副作用、任务规模、证据时效和用户交付类型采用；安全瞬时重试不要求每次新假设，普通任务不强制实验冻结，关键决策可及时询问并继续独立工作。全局Codex指令已同步。

release08③完整reject及历史费用保持。下一任务默认完成③工程补充与④准备；初始起点实发/继承作为用户可明确选择的追加路线，不能自动替换被拒候选升级。04 §5.2/§5.3和06明确目标、授权及实际消费者缺口。

本轮仅修改全局指令和已有Markdown，检查前后内容、链接、实际接口与预算算术、历史源码/阶段账保持和两仓diff；不重复已通过单元/SDK/Runtime检查，不运行Sphinx（未改rst/build输入）。通用续程、资格导入和OS部署仍是下一任务的实施/验收责任。本轮未新增项目模型请求或Runtime、未执行提示词里的发布动作、未改权威阶段账，无提交推送。


## 25. 2026-11-13 deployment09 工程接线、OS验收与真实继承

唯一当前状态见04 §1.17。产物根：[deployment09](../artifacts/initial-20261113-release08-deploy09/)。release08八槽保存事实经正式v2评价一致，Gate仍reject；可信起点导入、真实资格/OS边界/初始发布及真实继承分别验收，子代理报告或静态active读取没有替代动态证据。

- A在`/tmp/rsi-evaluation-resume`使用`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core/src /home/xzh/RSI-Agent-VLA/RPent/.venv/bin/python -m unittest discover -s core/tests -p 'test_read_text_audit*.py' -v`：17项通过，正式read模块与冻结v2字节相等。driver同命令以`-p test_release_driver.py`执行31项通过；之后仅针对实际receipt/version/完成槽/共享故障变化补测5项通过，存在重叠不叠加。早期隔离目录缺config两错、误替换历史评价一次失败均保留记录并修正；未将其算通过。[精确argv与计时](../artifacts/initial-20261113-release08-deploy09/verification/A/evidence.json)。
- B在`/tmp/rsi-qualification-consumer`使用`/usr/bin/time -p env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core/src:core/tests /home/xzh/RSI-Agent-VLA/RPent/.venv/bin/python -m unittest test_initial_deployment test_release_comparison test_release_preparation test_batch3.DeploymentTest test_5a_assets.QualificationTest -q`：21项通过；新增permit断言后仅运行`test_initial_deployment.InitialDeploymentTests.test_import_publish_and_real_cli_deployment_selection_preserve_source_identity`一项通过。初宽测33项中的3个既有错误在原源码精确复现：Research medium fixture两项、旧RuntimeConsumption缺初始资格一项；未弱化断言或修无关项。[完整命令](../artifacts/initial-20261113-release08-deploy09/verification/commands.json)、[B日志](../artifacts/initial-20261113-release08-deploy09/verification/B/focused-tests.log)。
- 主线程通过`python scripts/run_isolated_role.py --task-root <deployment09> --role <role> -- <python> <operator脚本>`实际执行5角色允许/拒绝探针；父链、源码/依赖/入口/证据/permit/资格/active受保护，根只读、capabilities清空，嵌套namespace remount拒绝。锁例外与scratch均显式限定；a2配置修复后只重验受影响的Runtime域和配置读取，不重复无关成功检查。主线程正式评价/导入、资格和发布均在真实角色中执行。
- 主线程实际运行`... -m rsi_vla.runtime --root /home/xzh/RSI-Agent-VLA --artifacts <deployment09>/state --protocol <deployment09>/operator/deployment-protocol.json --deployment-release <deployment09>/release --run <a1或a2> --task-key position_swap_salad --seed 1 --split deployment --output <新output> --cuda-device 0 --authorization <具体permit> --execute`，没有`--bundle`。精确argv在每次[返回回执](../artifacts/initial-20261113-release08-deploy09/operator/runtime-returned-a2.json)。a1配置锁错误为unknown/invalid；完整费用和清理核准后a2为success/valid。实际模型/processor/记忆/规则加载、原生结果、SDK/L0、270转移完整数组及host进程/请求清理分别保存。[独立结果](../artifacts/initial-20261113-release08-deploy09/state/evaluation/attempt-a2/inheritance-result.json)、[清理](../artifacts/initial-20261113-release08-deploy09/cleanup-check.json)。
- 新控制源码187文件与a2实际快照逐字节一致；137执行生产者与release08一致。执行/评价/调度和OS配置修订分别关联；不把调度变化自动当作无语义变化，不重新打旧run标签。11个实现/测试文件AST及`RPent/.venv/bin/ruff check --no-cache --select E9,F63,F7,F82 <changed-files>`通过，两个仓库`git diff --check`通过，RPent既有staged及工作树diff逐字节保持。最终Markdown检查见[交付检查](../artifacts/initial-20261113-release08-deploy09/final-checks.json)。

两次Runtime完整外层墙钟/GPU包络共260.673210秒，8请求/4工具/300步/107206392原输出字节。原launcher内部时长作为嵌套诊断保留，外层身份/命令/返回/完整计时另存，阶段按完整进程追加一次。离线测试进程、暂存保留物、源码正增长和共享scratch另列，未重复计动态等待。部分短读写操作及清理fixture未精确累计，actual=null且与保守余量区分；详见[计量边界](../artifacts/initial-20261113-release08-deploy09/accounting-supplement.json)。缺失usage不填0，不估货币账单，编程token_count单列。

无PR，未运行全仓pre-commit/CI或无关全量测试；没有安装依赖或改Sphinx输入，Sphinx不适用。真实上游429/故障/断流、超时取消、GPU等待、候选升级与回滚未动态覆盖。两次新启动已用完，完成证据/费用收尾后动态暂停。


## 26. 2026-11-13 continuation10：计量、授权续用与protocol10

当前状态统一见04 §1.17；证据根 [continuation10](../artifacts/continuation-20261113-protocol10/)。聚合根main与本地origin/main同步，独立RPent main领先2/落后2；已有staged逐字节与deployment09保存快照相等，无fetch/commit/push。只改聚合根core/scripts/config和拥有文档，未改RPent源码或既有staged。

- 主定向组：process accounting3、续用3、旧initial4、supervisor8共18不同方法通过；unittest因导入旧fixture类重复执行其中4方法，原输出22，不重复累计。首组子进程exit0后，工具外层清理/tmp/.aws报错；单列工具错误，不重跑成功检查。
- 增量6方法通过：实际计量CLI完整子进程/重复拒绝/幂等续程，新scope permit写域，既有driver评价暂停/末评价续程/未知提交拒绝/已扣费checkpoint恢复。之后授权来源责任改变，仅重验续用5方法及5B实际CLI限额1方法；新scratch重定向只重验实际CLI1方法。迟到评价文件费用修复后只验新增1方法与受影响driver2/supervisor1。以上集合有重叠。
- 新protocol10原14方法通过，新增Runtime适配3方法通过；包括真实prepare_runtime的隔离CPU dry-run（fixture关闭assets/collection，非真实资产准入）。实际CLI用现余额退出2，未写准备根。此前成功方法未重复运行。
- 新scope 5角色实际bwrap允许/拒绝探针均exit0，旧active/授权、源码、父链与其他角色域只读；无GPU或模型。授权过期和新授权合法续用由隔离回归验证，未签发正式新续用授权/permit。
- Ruff错误类、两仓git diff --check与staged保持检查通过；完整命令/墙钟/返回码见[主回执](../artifacts/continuation-20261113-protocol10/verification/main-command-receipts.json)、[协议原检查](../artifacts/continuation-20261113-protocol10/protocol-design/verification.json)、[新实际消费者](../artifacts/continuation-20261113-protocol10/protocol-design/runtime-hook-verification.json)。核心测试共44不同方法，不把重叠执行数相加。
- 5B只读本地metadata/资产与真实CPU原生配置，后者16.667945秒；未读取不合格batch。固定两个episode主体、公开许可声明与具体命令/包络见[data qualification](../artifacts/continuation-20261113-protocol10/data-qualification/)。

项目Runtime/Research新消费均0，编程token_count单列。完整命令与外层回执可恢复部分已补记；历史完整实际、bootstrap存储及采样间已删除fixture字节仍null。存储是跨明确写域的采样阈值与逻辑增长下界，记录overshoot和峰值，清理不退历史费用；不称内核硬配额。阶段只追加有据费用，不改变原limits，真实总量精度见accounting.json。

本轮未发PR，不跑无关全量CI/pre-commit/Sphinx；改的是独立Markdown与core，未改rst/build输入。尚无新仿真/GPU/真实上游故障/候选Gate/新发布继承/训练证据。完整动态包络不足且集中授权待答复，保存待续而非宣称整项实施完成。

## 27. 2026-11-13 protocol10 可恢复阶段协调第一批离线交付

当前状态由 [04 §1.17](../项目文档/04_实施路线与待决事项.md#117-5a-数据经验有限往返与资格准备2026-11-13) 维护，本节只记本批验证。工作目录 `/home/xzh/RSI-Agent-VLA`，解释器 `/home/xzh/RSI-Agent-VLA/RPent/.venv/bin/python` 3.11.13，`PYTHONPATH=core/src:core/tests`；core 与 RPent 实际导入均来自该工作区。聚合 `main/c4d7fd9b`、RPent `main/c57b1345` 的已有未提交工作保留；Autoloop 只读参考 HEAD 为 `c9ec4a187cab49627947e7e33db265ec143db3b8`，开始时工作树干净。

新增代码为 `protocol10_coordinator.py`、`protocol10_backend.py` 和 `scripts/run_protocol10.py`；现有 ResearchTools/research_cycle、Runtime 协议投影及全部尝试费用/违规汇总做必要接线。汇总 helper 从 release08 脚本移到 evaluation，历史脚本仍导入同名函数；没有把 release08 driver 扩为通用工作流。

| 检查 | 命令与结果 |
| --- | --- |
| 第一组受影响离线检查 | `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core/src:core/tests timeout 90s RPent/.venv/bin/python -m unittest test_improvement_protocol test_research_tools test_batch4 test_process_accounting test_release_comparison -q`：87 项通过，4.302s |
| Research/开发/既有 release 调用链接线后的定向检查 | `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core/src:core/tests timeout 90s RPent/.venv/bin/python -m unittest test_protocol10_research test_improvement_protocol test_5a_assets test_release_comparison test_release_driver -q`：79 项通过，4.123s；其中新 Research 合约 7 项 |
| 控制恢复 | `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core/src RPent/.venv/bin/python -m unittest discover -s core/tests -p test_protocol10_coordinator.py -v`：15 个方法在定向及下述 full core 中通过；包括显式阶段、继续条件、同 ID 冲突、并发、暂停/关闭、授权/预算及 unknown |
| 实际生产适配器的离线集成 | `PYTHONPATH=core/src RPent/.venv/bin/python -m pytest core/tests/test_protocol10_integration.py -q -p no:cacheprovider`：前 15 个方法整体通过；后续新增开发技术恢复、Research 终局/未知和实际并发方法分别定向通过，不重复跑已成功部分。共 19 个方法，全部使用新临时数据及 fake 进程边界 |
| full core | `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core/src:core/tests timeout 180s RPent/.venv/bin/python -m unittest discover -s core/tests -q`：344 项，51.241s；328 通过、12 个本机 socket 沙箱错误、4 项既有测试契约不一致，未报告全套通过。输出 `/tmp/protocol10-core-tests.log` |
| 本机 HTTP 定向补证 | `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core/src:core/tests timeout 60s RPent/.venv/bin/python -m unittest test_request_gate -q`：在允许 loopback 的环境补跑 12 项，全部通过，1.855s；无外部请求。输出 `/tmp/protocol10-loopback-tests.log` |
| 新文件 Ruff | `RPent/.venv/bin/ruff check --no-cache` 及 `ruff format --no-cache --check`，目标为上述 3 个新实现文件及 `test_protocol10_{coordinator,research,integration}.py`：通过 |
| 所改 Python 的语法、未定义名与导入检查 | `RPent/.venv/bin/ruff check --no-cache --select E4,E7,E9,F`，目标为新实现/测试、`improvement_protocol.py/research_tools.py/research_cycle.py/evaluation.py`、`run_release_comparison.py`、`test_5a_assets.py`：通过 |
| 入口/冻结/文档 | `scripts/run_protocol10.py --help`、`scripts/prepare_protocol10_attempt.py --help`：通过；config 与 continuation10 的 `protocol-design/frozen-protocol.json` 等值且模型/三版本保持；Markdown 新增链接/接口说明与实际消费者、局部空白检查通过 |

full core 的 4 项契约不一致分别是 `test_batch3.BoundedResponseTest` 中两个仍用 medium 准备新 Research 请求的用例、`test_batch3.RuntimeConsumptionTest.test_selected_bundle_changes_all_runtime_consumers_and_explicit_history` 缺独立部署资格的旧 fixture，以及 `test_runtime_plan.RuntimePlanTests.test_runtime_profile_binding_and_stale_authorization` 仍断言旧错误文案。它们触发的 high 请求要求、部署资格检查和 Planner 校验文案均为本批开始时已有实现，本批未放宽这些规则或改这些测试。full core 后只补跑受环境限制或新增/修正相关用例，不重跑已成功组。

全规则 Ruff 首次检查旧/新共 12 文件报告 85 项（含既有整文件风格项和当时新文件项）；本批新文件已修正并通过全规则，旧文件按所改行为做上述语法/导入检查，未扩大为历史代码风格清理。首次 Ruff 因默认父目录 cache 只读失败，之后统一 `--no-cache`。

集成覆盖两条完整分支：起点 J=1 仅 8 个 fake Runtime、Research=0、候选=0；非天花板为 8 起点＋1 Research＋2 开发＋8 候选，经真实 Candidate/evaluate_matrix/Gate 得到 comparison_only 交接。开发技术恢复 fixture 为 19 个 fake Runtime、3 个开发 finding/费用回执，Candidate 仅消费各槽所选的 2 个技术有效结果；原失败尝试及费用保留。还覆盖重复完成信号和并发继续、外部完成但确认丢失、评价修复零 Runtime 重放、unknown/清理未决、暂停/缺授权/过期/预算不足、有效 failure 不替换、非法及无合法 Research 响应、Gate reject/inconclusive/accept 均无 active 或资格写入。只有进程和独立边界报告被 fake；实际准备、响应合法性、费用消费者、开发循环、Candidate 和 Gate 均走生产模块。

未运行真实 Planner/Research、GPU/仿真、训练、硬件、下载、安装、通知或真实发布；未签发真实许可，未改真实阶段账、冻结 comparison_only、deployment09 资格或 active。未准备 PR，未执行 RPent 的全仓 pre-commit/unit suite；本批未改 RPent 源码。没有 Sphinx 页面或构建输入改动，未构建 Sphinx。新调度在真实隔离角色、资源竞争、上游故障和停止/清理条件下的动态证据仍待后续适用授权；离线 Gate fixture 不建立实际收益或发布资格。

新增集成方法的定向命令如下（相同解释器，均 `-p no:cacheprovider`，不调用真实后端）：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core/src RPent/.venv/bin/python -m pytest core/tests/test_protocol10_integration.py -q -p no:cacheprovider -k test_development_technical_recovery_retains_original_attempt_and_costs
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core/src RPent/.venv/bin/python -m pytest core/tests/test_protocol10_integration.py -q -p no:cacheprovider -k 'research_terminal_response_failures or closed_zero_admitted_research_startup or invalid_single_research_response'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core/src RPent/.venv/bin/python -m pytest core/tests/test_protocol10_integration.py -q -p no:cacheprovider -k test_concurrent_continue_and_duplicate_receipts_keep_one_runtime_and_charge
```

以上分别通过 1、3、1 个方法（中间包含既有非法响应测试的受影响补跑），最新新增集成共 19 个方法，与控制 15 个、Research 合约 7 个合计 41 个新增方法均通过。

第二批补正见 §28。上节四项旧测试问题已逐一修正 fixture/断言并验证，现行 high Research、独立资格和 Planner 规则保持。第一批开发技术恢复 fake 曾把技术失败同时报告为 clean_completion=true，不能用它证明真实非零退出的清理闭合；第二批已改为 clean_completion=false、owner_cleanup=true，并以实际 owner/请求回执覆盖该恢复前提。第一批原检查记录保留。

## 28. 2026-11-13 protocol10 第二批补正与生产角色验证

唯一当前状态及授权缺口见 [04 §1.17](../项目文档/04_实施路线与待决事项.md#117-5a-数据经验有限往返与资格准备2026-11-13)。本节为证据索引，证据根 `artifacts/continuation-20261113-protocol10/batch2/verification/`，source、配置和未签发 permit 在同级 `operator/`。聚合 `main/c4d7fd9b`、RPent `main/c57b1345` 及其既有工作树修改保持；解释器为 RPent/.venv/bin/python 3.11.13，实际导入为本工作区 core/src/rsi_vla 与 RPent/rpent。未提交、推送、安装、下载或运行真实训练/模型/GPU。

| 检查 | 结果及证据 |
| --- | --- |
| 真实冻结协议投影与准入 | `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core/src:core/tests timeout 60s RPent/.venv/bin/python -m unittest test_protocol10_admission -v`：9 个新方法均通过（首组8、CLI新增及Python3.10时间写法改动分别定向核准），最终也进入下述 core 检查。覆盖 observed_only Research 准备/费用消费、旧输入时间、实际行政授权到期、三类错账 permit、注册 source/output/scratch/身份以及每维上限，Research180秒。 |
| 四项既有测试契约 | `... -m unittest test_batch3 test_runtime_plan -q`：17 通过，0.282s。两个新 Research fixture 从 medium 改 high；部署 fixture 先证明无独立资格会拒绝，再 qualify_initial/establish_initial、重绑 decision 后 activate；Planner 错误断言更新为当前精确文案。约束未放宽。日志 `protocol10-batch2-legacy-tests.log`。 |
| 注册准备与动态拒绝边界 | `... -m unittest test_protocol10_registration -v`：2 通过；临时完整冻结协议/store 验证幂等、零派发/不发 permit/原账不变，拒绝未签发、过期、错账和不足包络。`registration-final.json/.log`。当前具体配置执行准备命令退出2、dispatch=false、prepared=false，明确缺已批准行政授权，未物化授权投影：`unissued-registration-boundary.json/.log`。 |
| 非零退出清理与恢复 | `... -m unittest test_runtime_cleanup.FailedRuntimeClosureTests test_protocol10_integration.Protocol10IntegrationTests.test_development_technical_recovery_retains_original_attempt_and_costs -v`：两方法通过。clean_completion=false 不阻止已闭合 owner/进程/请求和完整费用进入 invalid/technical 评价；未知 critical_progress 不伪造成功，未闭合请求及错身份仍拒绝。 |
| 真实生产 OS 角色链 | `RSI_ROLE_EVIDENCE_ROOT=/tmp/protocol10-role-evidence ... -m unittest test_protocol10_roles.RoleBusinessTests -v`：1 通过，19.435s。host broker、实际 run_protocol10_role.py 与 bwrap 挂载均真实，模型/Runtime边界替身；18 Runtime、1 Research、2开发、8+8固定槽经真实业务消费者交接。18 runtime/8 research/38 evaluation 回执中保留1次评价错误及修复版本，覆盖跨域拒绝、SDK scratch/只读凭据、丢确认、变化now后的重复消费及零重扣。顶层CLI实际进入operator后拒绝fixture。原件复制到 `isolated-cpu/`，排除所有scratch及符号链接，未复制凭据。 |
| 计量与不确定提交补证 | `... -m unittest test_artifact_accounting test_5a_assets.ExperienceAndCycleTest.test_cycle_candidate_stop_idempotent_and_independent_qualification -v`：3 通过，0.154s；raw嵌套store、非零scratch、后处理增长仅计一次及Research/development消费幂等。`... -m unittest test_runtime_cleanup.UnknownRoleLaunchTests -v`：1 通过，0.008s，动态role marker存在而receipt未返回时不再启动子进程。均晚于OS整链的相应修改，复用整链未变的角色证据。记录 `isolated-cpu/operator/validation.json` 的 supplemental_checks。 |
| 最终共享 core 回归 | `... timeout 180s RPent/.venv/bin/python /tmp/protocol10-core-check.py`：361个去重方法，359通过、2失败，runner完整50.908647秒；排除已单独验证的OS整链，新增unknown-launch方法另有上述定向证据。runner及日志已保存：`core-regression-final-pathfixed.json/.log`。首次runner因/tmp历史同名test_5a_assets遮蔽而在发现阶段报错，修正sys.path后才启动测试，原失败记录保留。 |
| 两项宿主短期限失败的范围 | 宿主独立复核仍失败：Runtime `.5s` 得 `.792518s`、detached CPU进程 `1.5s` 得 `1.668994s`；宿主/proc约1851 PID，group/ownership单次扫描 `.075394/.092147s`。用实际 `scripts/run_isolated_role.py --role runtime` 的PID namespace，在原代码/原断言/原上限下两项通过，1.796s。`isolated-timing-failures.*`、`proc-scan-unsandboxed-host.json`、`production-pid-namespace-timing.*`。保留宿主超限缺口，不声称宿主全绿；未改监督算法或训练入口。训练模块用例仅监督CPU sleep子进程，没有训练或GPU。 |
| 静态与来源 | 18个实现/新测试文件完整 Ruff及format通过；最后role/marker两文件改动后定向复核通过。既有improvement_protocol/test_batch3/test_runtime_plan的 E4/E7/E9/F通过；旧fixture同一行两语句只拆行为两行。`static-final.*`、`format-final.*`、`final-role-marker-*`、`existing-files-syntax-corrected.*`。202份实际来源精确副本、冻结协议等值和正式active保持见 `../operator/source-bindings.json`。未生成新hash。 |

OS链的临时 phase 数字属于 offline fixture：18 episodes/19 requests/18 tools/90 steps/gpu_seconds=18 是注入边界的合成值，未计入真实阶段账或解释成GPU消耗。其CPU外层时间和实际文件材料另按离线费用下界登记。第一轮OS链和后续修复验证各保留费用，已通过分支未无理由重跑。未动态遇到的上游故障等分支仍只有适用离线证据。

本批已测离线费用下界96.003951776秒、17751436字节，由主代理追加原唯一 `artifacts/offline-20261111/phase_budget_snapshot.json` 的 `continuation_scope_charges`，entry key=`protocol10-batch2-offline-v1`；没有复制迁移权威账、扩额、重置历史或清理退款。完整会话、删除fixture、部分未测命令与programming usage等保持null，不能把下界当完整实际。具体口径、逐次验证命令与资源差额见 `../accounting.json`、`../resource-admission.json`。账后原simulation/model/tool/environment/GPU used不变，正式动态0启动/0请求。

未准备PR，未改RPent实现，不运行其全仓pre-commit/unit suite或CI矩阵；Markdown与入口文档只作本地路径、协议/命令语义及JSON检查，未改变Sphinx页面或构建输入；最终13条新增本地链接、JSON、202份源码一致性、23份未签发permit、真实输出/store为空和唯一离线charge均通过，见 `final-collateral-check.json`。集中补充授权仍awaiting_user_decision，行政授权及permit未签发，当前Research领域对象/preflight为空；停止在真实动态边界，正式比较未完成。


## 29. 2026-11-13 protocol10 第三批入口、候选升级桥接与正式资源准入

唯一当前状态见 [04 §1.17](../项目文档/04_实施路线与待决事项.md#117-5a-数据经验有限往返与资格准备2026-11-13)。证据根 `artifacts/continuation-20261113-protocol10/batch3/`。本轮新增明确授权已先记录，再幂等应用精确amendment与独立离线scope；聚合main/c4d7fd9b、RPent main/c57b1345及用户既有工作树修改保留，无commit/push。解释器为RPent/.venv/bin/python 3.11.13，导入实际来自core/src/rsi_vla与RPent/rpent。

| 检查 | 证据与结果 |
| --- | --- |
| seed2 preflight与零Research闭合 | `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core/src:core/tests RPent/.venv/bin/python -m unittest test_research_preflight -v`：4通过，真实prepare_runtime消费protocol10 development seed2，保留领域、合法证据、非no-op与无派发。新fixture最初缺memory provenance及复用不可变result身份，两次失败保留后修fixture；未放宽约束。`test_protocol10_no_research`：2通过，完整8槽CPU起点后无合法算子不要求模型permit、不发Research、不重扣；preflight纳入不可变registration。原日志在verification/main/。 |
| protocol10业务消费者回归 | `... timeout 60s RPent/.venv/bin/python -m unittest test_protocol10_integration test_protocol10_admission test_protocol10_registration -v`：30中29通过，UTC CLI方法因新增read导致fixture side_effect耗尽；A移除重复读取后该1方法定向通过，29已通过方法复用，未重复整套。日志main/backend-regression.log与A逐命令回执。 |
| broker/并发控制/外部桥接 | A共22个去重方法通过：既有coordinator15、并发/稳定trigger2、command bridge2、host边界2、真实UTC CLI1。每CLI独立UUID socket，深路径通过短/proc/self/fd绑定；status/pause/ordinary event在长advance期间保持原broker、回执与暂停，不重复业务。Ruff错误类、7文件format及diff检查通过。A/HANDOFF.json、all-command-receipts.json。 |
| 真实Autoloop服务CPU链 | A/final_e2e.py最终完整10.407128411秒，真实loopback hub→command adapter→文件RPC→受限host→CLI→bwrap CPU runtime/evaluation；重复事件与重启后重复均duplicate，真实owner卡片回答核对recentCards后恢复独立settlement，Runtime execute恰1、费用结1次。service仅home/requests可写，真实阶段账/active O_WRONLY被拒，无GPU设备。先前socket路径、引用、fixture binding、提前终止与费用时点等失败均保留，不删失败费用。A/e2e-result-v6.json、service-write-boundaries.json、command-spans-v6.json。 |
| 角色整链受影响回归 | `test_protocol10_roles.RoleBusinessTests`最终1方法通过，22.235测试秒；保留18 runtime、8 research、38 evaluation角色回执与1次旧评价错误。Research模型/Runtime边界为CPU替身，8+8比较与2开发仍验证真实角色和消费者，不计正式槽、实际模型请求或GPU消费。A/protocol10-roles-lnp8k9a8/。 |
| 独立候选升级桥接 | B发布/续用/比较/核心/5A相关组合73方法通过；新增精确runtime_projection来源检查后，`test_upgrade_bridge`8方法通过。覆盖合法新资格/升级、重复激活、active/revision变化、过期、错误映射/源码差异、缺证据、reject/inconclusive、comparison accept不得直接发布、真实五角色跨域拒绝及受授权祖先回滚。新模块/测试/隔离脚本错误类与import检查通过，旧release/qualification既有import格式未扩大修改。B/command-*.log、integration-note.md。 |
| 生产冻结与实际角色 | 正式结果前冻结418份当前源码副本、执行/评价/调度及入口配置；未生成hash。真实生产五角色分别写自身域，拒绝其他四域，私有/dev无GPU；release根是真实deployment09已有根。evaluation实际核验九类行为绑定、精确源码差异、固定criteria及新部署协议，mapping_valid=true。operator/production-mapping-validation.json、batch2/operator/deployment11-isolation-acceptance.json。 |
| 生产专用外部入口 | main/production-entry/result.json：专属127.0.0.1实例经真实文件RPC→host→operator core查询生产状态，重复事件duplicate=true，Runtime/Research0，channels={}。allow_advance=false；服务与host按PID结束，保留home/ingress与可重启配置。启停：A/production-templates/START.md。 |
| 正式动态准入 | 复用scripts/run_release_comparison.py的gpu_admission阈值12288MiB及共享锁，完整298.241417232秒、11次只读探测；最后两卡他人占用13703/19244MiB，均不合格，锁空闲。真实生产advance保存首槽not_submitted，未启动任何正式Runtime/Research、GPU或模型；16固定槽全部保留，尝试/选择为空，J/Gate不存在。operator/gpu-admission.json、formal-slot-closure.json。 |
| 实际发布拒绝与清理 | 无正式Candidate时，真实evaluation导入入口因缺Candidate拒绝；真实release资格入口因缺独立import拒绝，无新资格/决定/激活/继承/回滚，active保持原bundle/revision1。所有本轮Autoloop/host及命名home标记进程核对后消失，未触碰其他GPU任务。operator/upgrade-refusal.json、cleanup.json。 |

原两项宿主短期限用例与§28的生产PID namespace通过证据继续复用；本轮未放宽计时、监督或测试断言，不为宿主全绿重跑。新增代码/测试之外的RPent既有修改保留，未准备PR；不跑RPent全仓pre-commit/unit或未改动的Sphinx构建。本轮第三批费用只进入原阶段账的独立scope，GPU等待只进入原离线池；所有失败保留，完整未知仍null。最终费用以 `batch3/operator/accounting.json` 及同一阶段账记录为准，账面为已测下界。工程完成，正式动态仍被GPU准入阻塞，实际发布未发生。


### §29 本次恢复：三项补正与第6槽技术停止

续用同一scope、原协调器、固定槽及一次amendment，未新建工程批次。新增证据根 `batch3/operator/resume-20261113/`，唯一当前状态见04 §1.17。

| 检查 | 结果与证据 |
| --- | --- |
| GPU生产选择/修订 | 12个不同CPU方法通过；真实Backend重开/prepare使用临时物理编号7核对context、参数、permit，未创建output。全物理GPU查询保留12288MiB和未知拒绝；未提交修订保留旧字节、run与operation，已派发/完成/记账拒绝切卡。三新文件Ruff check/format通过。额外误写测试名的loader错误保留，12个成功方法不重跑。engineering/gpu-*.json。 |
| 候选绑定/回滚 | 11桥接方法、62相关消费者通过；新增真实prepare快照断言定向通过。覆盖Candidate/bundle/backend_ref/profile/规则/引用在import及qualification前漂移拒绝，故障候选合法回滚、篡改import/目标损坏/错祖先/激活/revision/过期授权拒绝。首个bwrap环境限制后在授权宿主定向通过；失败、缓存限制及旧Python3.10风格项保留。engineering/release-binding-verification/。 |
| 首正式前登记 | 420份精确来源一致，旧source/manifest/发布注册保留，真实evaluation角色mapping_valid=true；既有五角色隔离复用。source-freeze-r2.json、production-mapping-r2.json。 |
| 新GPU等待与首槽 | 唯一控制/charge首探测准入，完整外层0.250588255秒，仅第三批独立池；旧298.241417232秒不改。物理GPU1/UUID d4fa...，原首operation先reconcile，再用新trigger经guard首次execute。gpu-wait-receipt.json、原revision journal。 |
| 正式结果 | Runtime6；前5槽技术有效，3success+2failure。第6槽old_milk/seed1第4个已计费请求RemoteProtocolError/502，原生结果unknown，独立评价invalid/technical；150环境步、native success=false、完整token未知。评价符合原事实，未放宽或重跑。old_drawer两槽未启动，8+8身份及全部尝试保留，完整J/Gate不存在。formal-slot-closure.json、原native/request/evaluation回执。 |
| 技术恢复边界 | 现有两预留恢复身份及未签发permit模板只指向两侧base_salad/seed0，不覆盖第6槽，恢复启动0；原协调器已按协议进入baseline_technical_stop，未重开或改绑。总额度尚有余额不代表该恢复适用或故障已解决。Research/候选/新资格/激活/继承/回滚均未触发。delivery.json。 |
| 费用/清理 | 请求41、工具20、环境步1435，墙钟/GPU包络各1656.377545秒，新增逻辑字节下界920084038。所有故障费用保留，token完整三项null，观测下界单列；编程/货币/历史实际unknown。工程/新等待/收尾归第三批独立池，旧原池不重复计。6个owner/请求闭合，共享锁空闲，无本轮Autoloop/临时host服务。accounting.json、cleanup.json。 |

未重复整套Autoloop或无关RPent CI。工程完成=true，正式动态完成=false，实际发布=false；已保存真实技术边界并停止新增动态。


### §30 新任务政策、分段准入与探索准备

2026-11-13。方案与未启用项见[04 §5.4](../项目文档/04_实施路线与待决事项.md#54-任务政策与历史比较的作用域)，接口见[README](README.md#新任务的分段准入与探索准备)。聚合分支 main/c4d7fd9b，与本地 origin/main 同位，未联网 fetch；RPent 自身 main 与本地 origin/main 为领先2/落后2。保留进入任务时全部未提交改动；本轮不改真实 phase、permit、正式结果或 active，不启动模型、GPU/仿真、训练或下载。

源码先在 `/tmp/rsi-policy-adjustment` 准备，并保留 `before/` 原字节。回写前逐文件核对原字节，仅应用本轮增量。新文件为 `task_policy.py`、`prepare_task_stage.py` 与 `test_task_policy.py`；消费者变更覆盖 `admit_resources/baseline_readiness`、Research preflight/请求/规则计划/开发槽、完整事实传输和 GPU 查询。协议10的冻结配置和生产注册未修改。

| 检查 | 证据与范围 |
| --- | --- |
| 最终定向回归 | `PYTHONPATH=core/src:core/tests RPent/.venv/bin/python -m unittest test_task_policy test_research_preflight test_improvement_protocol test_batch3 test_batch4 test_research_tools test_release_driver -v`：130项通过，`/tmp/rsi-policy-adjustment/final-regression.log`。覆盖无证据假设→实际Runtime preflight→试验产物、越开发槽拒绝、无finding不提交、原预算/请求绑定、J天花板/修复分支、完整/分段准入、L0留存/只读权限及容量型GPU选择。 |
| core 全套初检 | `PYTHONPATH=core/src:core/tests RPent/.venv/bin/python -m unittest discover -s core/tests -v`：当时408项中394项通过，14项受沙箱限制（12项本地HTTP socket、1项Unix socket、1项bwrap网络命名空间），见 `full-tests.log`；未改测试断言规避限制。后续仅补跑受影响用例，相关源码后续修改由上面的最终定向回归覆盖。 |
| 宿主补验 | `test_request_gate` 的12项本地替身用例通过，`host-tests.log`。两项真实隔离用例在临时副本因其 `/tmp` 被角色scratch覆盖而找不到解释器；回写后在原目录补验，结果追加于下。测试不访问外部模型/GPU。 |
| 静态与CLI | 三个新文件完整 Ruff check 与 format check通过；全部12个改动Python文件的 `--select E9,F63,F7,F82` 通过。更宽的 `E4,E7,E9,F` 检查报告 `runtime.py` 三处原有E701单行格式（283/410/411附近），本轮仅改其锁docstring，未重排旧代码。两个准备入口 `--help` 通过。文档是独立Markdown，核对新增本地链接/锚点及历史作用域，无Sphinx改动。 |

早期新测试的fixture字典/约束配置错误已修正并纳入最终定向检查；初轮失败日志保留。新增API只验证离线准备和状态转换，效率/鲁棒性/新能力的独立评价、模型自主工具补读、隔离并行调度与动态效果仍按04 §5.4推进。

最终回写与补验：16个目标文件在回写前全部与原字节一致，应用后与验证副本逐字节一致；聚合 `git diff --check` 通过。原项目目录的两项真实角色隔离用例通过（21.799秒），见 `/tmp/rsi-policy-adjustment/original-root-isolation.log`；连同已通过的12项HTTP用例，原14项环境阻塞全部补验通过。最后将空输入限定于探索入口、提前核对请求政策后，受影响的11项请求/响应/探索用例通过，见 `final-input-regression.log`；只重复对应变更的检查，没有重复已成功的整套。所有日志及本轮独立增量 `changes.patch` 留在同一临时证据目录。


### §31 适度检查与执行流畅性优化

2026-11-13。复核与实施原则见[04 §5.1](../项目文档/04_实施路线与待决事项.md#51-任务提示词编写参考)及§5.4末段。确认历史审计误判与暂停范围失当；不把来源留存、费用记录或跨权限边界的重复核对一概认定为多余。本轮采用局部实现与已有执行指引，未新增审批框架或检查平台。

- 新任务和技术诊断的解释数量最小为1，仍须有完整解释、预测和区别测试，拒绝空、重复及无测试内容。普通历史请求及protocol10保留2；请求保存minimum_hypotheses，CLI与实际SDK准备按保存请求生成schema。后续策略变化不改变旧请求的消费要求。诊断仍不能直接作为行为试验前提。
- 新任务research_branches保留可检验criterion，允许缺evaluation_ref时准备研究；字段返回None明确待补，独立评价、Gate和发布消费者未放宽。
- 监督对消失的进程组用signal 0直接确认；存活组仍扫描实际成员。清理先poll回收已退出子进程，每轮只作必要的成员检查，TERM/KILL、硬期限、超限及剩余进程报告保留。原0.5秒清理用例未改断言。
- RPent AGENTS与verify-change指引明确具体检查用途、既有授权、最小影响暂停、已读指引/成功证据复用、按环境选检查和满足交付后的结束条件。完整PR检查要求保留。

源码从现有工作树按目标文件准备，原字节保存在 `/tmp/rsi-defense-optimization/before/`，仅回写本轮增量；未改真实phase、permit、冻结protocol10或正式结果。测试从原项目目录执行，避免/tmp源码被角色scratch覆盖。使用已有RPent Python3.11环境，没有安装依赖。

| 检查 | 结果 |
| --- | --- |
| 受影响消费者回归 | `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core/src:core/tests RPent/.venv/bin/python -m unittest test_proportionate_research test_task_policy test_research_tools test_protocol10_research test_batch3 test_batch4 test_protocol10_integration test_protocol10_no_research test_runtime_supervisor -v`：117项通过，25.074秒。日志 `/tmp/rsi-defense-optimization/regression.log`。包含真实准备CLI、无证据探索产物、诊断分离、非法解释拒绝、p10严格数量和完整离线业务链，以及原CPU进程/孤儿清理/0.5秒期限用例。 |
| 保存请求的schema绑定补验 | 最后仅修正schema从保存请求读取解释数量，运行 `... -m unittest test_proportionate_research test_batch4.RunnerTest test_protocol10_research -v`，16项通过，2.865秒，见 `/tmp/rsi-defense-optimization/request-final.log`；未重复117项整套。 |
| 静态与文档 | 所改Python错误类Ruff通过，新测试完整Ruff及新/修改的task-policy测试format检查通过；`git diff --check`通过。文档只改Markdown，核对新增链接、接口和阶段条件。没有Sphinx构建输入改动。 |

本轮验证为离线准备、CPU监督与替身业务链；不提供模型/GPU效果或吞吐提升比例。正式动态、费用、授权及断点仍以04 §1.17为准。


## 32. 2026-11-13 completion11 新任务与有限文本提示

scope `5a-completion-20261113-v1`，新任务授权/两池账/回执均在 `artifacts/5a-completion-20261113-v1/`；唯一当前状态为04 §1.17。沿用聚合main/c4d7fd9b与RPent自身main的既有工作树，保留其它会话修改，未fetch、提交或发布源码。

本轮必要工程接通 `task_completion.py` 与原host受限角色，多请求/开发/独立矩阵和Gate保留实际消费者；每次动态核对真实期限、范围、permit、来源、写域、阶段余额及次数。实际故障槽恢复消费者核对原无效技术故障、费用/清理/原因闭合、每槽1次及全轮4次，不改旧protocol10。finish guard独立审计补齐已审源码与原生成功关系，原断言保留；v5文本算子由真实模型选择文本，经实际rule/preflight/runtime/Planner启动消费者，guard/reserve/handoff旧语义与旧协议作用域保留。

| 检查 | 结果与入口 |
| --- | --- |
| 初次角色与Runtime定向回归 | `python -m unittest test_task_completion test_finish_audit test_protocol10_roles test_protocol10_admission test_runtime_plan -v`，37项通过；新增实际恢复准入1项随后定向通过。operator/offline/focused-consumer-regression-v1与freeze-and-prepare-base-v1 |
| 文本算子跨模块检查 | `python -m unittest test_planner_task_hint test_task_completion test_runtime_plan test_protocol10_research -v`，30项通过；实际Runtime与旧协议拒绝路径覆盖。operator/offline/check-task-hint-contracts-v1 |
| RPent文本消费者 | `pytest tests/unit_tests/robots/libero/test_task_hint_consumer.py -q`，17项通过。首次因测试输出父目录未建而17个fixture未执行，修复目录后仅补跑此组，原失败与费用保留。operator/offline/check-rpent-hint-fixture-v2 |
| 共享Runtime实际调用方 | `pytest tests/unit_tests/rpent/session/test_runtime_counted_protocol.py tests/unit_tests/robots/libero/test_offline_runtime_inputs.py tests/unit_tests/robots/libero/test_budget_stop_contracts.py -q`，通过；结果见operator/offline/check-runtime-callers-v1/backend.log |
| 原启动输入兼容 | v1/v2在hint缺省或空串时完整输出与进入时源码逐字一致；新增文本只追加user_message，system/source/权限不变。原字节保存在operator/engineering-before-hint |

无PR/Sphinx页面变更，未重跑全仓PR检查或既有完整文档整理。无依赖安装、5B数据下载/batch/训练/专项模型加载或硬件。CPU检查、真实仿真和原生任务成功分别报告：首个起点开发原生failure/独立valid；Research01完整响应因探索来源约束拒绝，未生成Candidate；Research02已生成合法文本trial，其首个开发正在运行。正式收益、资格、升级及继承尚未形成。完整动态终态与后续检查继续在本条 owning evidence及04 §1.17更新，不能据本段工程通过宣布5A完成。


### §32 completion11最终证据与收尾（2026-11-14）

已完成真实Research2、开发3、正式8+8、合法Candidate、独立accept(.825-.7625=.0625)、独立升级资格和normal upgrade revision2。原生失败均保留，恢复0，当前ρ/ψ/θ不变。实际加载继承未启动，用户确认短期GPU不足后停止等待；scope执行闭合不等于5A整体完成。最终状态以04§1.17和artifacts/5a-completion-20261113-v1/delivery.json为准。

原语补审覆盖实际latest high projection、字面gripper move_to与release，NPZ/control/RPC逐项核验，不能证明目标识别或模型内坐标推导；native failure不变。独立v7结果用current指针，原raw与旧v6均保持；修复16条旧evaluator未落盘的audit引用，只允许相同Runtime身份的真实派生报告替代不存在派生路径，所有existing/raw/native引用保持。

新增最终组合77项中76通过，唯一benchmark事实变更异常分类修正后该方法补测通过，原断言不弱化；派生引用修复50项（含真实role namespace）全部通过。早期37、文本30+17、共享Runtime20、条件47等有效结果复用，不重跑全仓/未更改Sphinx页面。所有测试/准备失败及费用保留。资格current兼容保留真实旧initial资格及过期auth历史快照，使用本scope合法renewal；候选完整initial门槛照常执行，rollback同一原资格证明路径通过离线反例，未制造真实回滚。

scope实际Runtime19、project上游144（Planner medium142、Research high2）、实际候选1；源码与全部槽、原结果、独立版本、token、完整命令/GPU包络、存储下界、unknown及清理均见同目录回执。5B只读增量发现本地自主数据，训练仍not_admitted，未下载/构建5B原生训练batch/训练/专项加载。全局账和scope两池按唯一增量结算，不重补同amendment，清理不退款。


### §32 completion11 CPU接续（2026-11-14）

当前交付入口改为`artifacts/5a-completion-20261113-v1/current-delivery.json`，唯一当前状态仍为04§1.17；原delivery与上段收尾保留历史事实。checkpoint已按真实资格/发布与EvaluationStore当前版本核正，未重复补丁、比较或激活。

新增`relation_learning.py`的有界经验反馈、冻结派生替换和跨轮继承，接`EventLedger.view`、Research请求构造及`prepare_relation_step`；SC3显式v3正常机会/new_goal提案、独立evaluation价值复核、预算暂缓及槽清除均已接通。继承helper v2从真实配置/target bundle/profile派生protocol/code/collection，核对真实发布/current及源码；完整900秒/内部880秒不变。已有active、ρ/ψ/φ和正式评价未改。

本次统一计量定向检查的最终有效结果134项通过，命令/逐项状态在`continuation-cpu-20261114/validation-results.json`，原失败/回执在`operator/offline/`：SC3/旧Research/Task/独立角色组合86项中85通过，namespace fixture修复后只补1项；关系/证据/Task选择41项全通过，去重与预算投影变化后补8项通过；继承初次4通过+4个fixture setup错误，源/目标collection fixture修复及发布/来源核对变化后定向补6项、2项全通过。未放宽断言，未重复全仓或Sphinx检查。

预登记三个真实development run的元数据回放成功：A引用78→6、26→2、26→2，全部741/158/186条L0保留；A6+C2反馈跨轮继承分母不重置；B缺独立路由反馈、baseline C缺screening均保留。无媒体/NPZ/tensor读取、实际存储节省或因果收益主张。B/C受限准备消费已离线验证，新策略的真实Research/Runtime、独立评价和继承仍缺。

新实际源码快照`operator/continuation-cpu-source-v1/source-manifest.json`用于未提交继承，旧formal source-v3/evaluation-v5保持；真实current纯读prepare_configuration通过，新增slot/permit/Runtime/Research/GPU查询均0。scope/一次补额/截止不变，费用与unknown按accounting-v2及原phase账结算；5B仍仅只读提案。


### §32 relations12真实续程里程碑（2026-11-14，执行中）

用户再次要求完成余下5A，资源和授权由主代理把握。一次新鲜GPU检查确认恢复后，唯一继承实际使用`--deployment-release`闭合：binding正确/技术valid/progressTrue/native failure，430步/12请求；完整312.107127秒。原pre-Runtime隔离拒绝有独立回执和费用，3项实际namespace修复检查通过，未弱化资格、未虚计Runtime或技术恢复。有效failure不重跑。

新增ρ有限提案/真实消费门槛、B同目标技能路由与环境私有反馈、C开发准入及独立catalog目标价值测量。定向证据分列在operator/offline：ρ桥接77项通过；Runtime/Task与逐槽门槛30项通过；B投影5项、RPent路由8项通过；控制推进3新+2旧通过；首技能与旧输入19项通过；私有file边界4项通过；目标测量10项+独立角色4项+φschema1项通过，真实目录/unused orphan约定4项定点通过；独立B反馈/去重6项+真实消费3项通过。引用各receipt，不将重测加总成额外独立样本。

新协议relations12固定两技能控制臂/原四任务新8+8比较/真实Research03+goal-only04，scope Runtime总上限按最新委托33→41、其它硬维度与原截止保持，旧topup不重复。代码在新数据前冻结；首臂原生success且独立交接反馈3项True，第二臂运行中。真实ρ候选、正式比较和新领域目标尚未产生；本条是里程碑，最终结论继续更新04§1.17与checkpoint，不将准备称完成。5B权限保持只读。


### §32 当前暂停接续（2026-11-14）

用户明确暂停第三候选。原completion11真实继承已闭合（binding正确/技术valid/native failure）；relations12的独立版本化结果与Gate为accept（6/8→7/8，J=.7375→.9375，两侧hard全True），目标部署绝对success_old=1未满足，导入拒绝，current仍revision2。旧unknown/Gate、原始失败及费用保留，当前版本通过EvaluationStore.path/get读取。

新14 private drawer getter/reviewer/finite contact route已通过core43和RPent有效48项检查；原计量context污染的4个Env选择失败在清子进程环境后9项通过，不改product/断言。原失败回执保留，勿将重测加总为独立样本。真实首控制臂因人工暂停终止：159.688408秒、4请求/3完整响应、150确认环境步，native unknown，工具下界2/上界3，normal finalizer未完成。外层supervisor和实际进程/GPU审计确认清理，锁可取得；不是有效failure或技术恢复，不自动重开。第三候选/新Research均尚未产生。

暂停回执、实际费用、原始证据和接续约束见artifacts/5a-completion-20261113-v1/current-pause.json及continuation-live-20261114/candidate03-pause-handoff.md。scope-v6与原截止保持；5B只读准备、真实batch/训练未准入。后续等待用户指令，不自动继续任何动态。
