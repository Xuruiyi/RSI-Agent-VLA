# 本轮离线交付检查

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
