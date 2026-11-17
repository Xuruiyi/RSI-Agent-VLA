# RSI 执行与证据接口维护

本文说明 LIBERO 在 RSI core 启动时消费的可选接口。独立 RPent 运行不要求启用这些配置。聚合接口见[core README](../../core/README.md)，工作包当前状态见[项目文档04 §1.23](../../项目文档/04_实施路线与执行状态.md#123-第四批仿真验收与准备2026-11-17)。

## Skill 契约

聚合 profile 的 `skill_contracts` 原样冻结到 Runtime context，schema 为 `libero-skill-contracts-v1`。`skills` 的键只能是 `vla_execute`、`pi0_pick`、`pi0_doubled`。每项包含：

- `skill_id`、`authorized`、`require_verified`、`requires`；`requires` 引用同一配置中已登记的前驱 skill_id。
- `version_signature` 的 `theta_ref/processor_ref/phi_ref/code_ref`，及固定为 5 的 `action_prefix_steps`。
- `validation` 的 `status`、`evidence_refs`、`task_scope`。`verified` 需要非空证据和 `{suite, task}` 范围；`unverified` 不构成已验证声明。
- `preconditions/completion/failure_conditions`，每个条件为 `{signal, op, value}`。公开信号限于 `eef_z/gripper_opening/state_valid/main_image/wrist_image`，比较限于 `ge/le/eq`；状态/图像可用性使用布尔 `eq`。

`task_input.build_task_input`向Planner呈现可调用契约与范围；`LiberoToolkit`在ρB选择实际worker后消费同一契约。未授权/无效输入返回rejected；状态未知、版本不兼容、未验证或交接缺失返回deferred及后继建议。无显式配置时，内置契约检查现有输入和公开观察，验证状态为unverified。

技能结果和journal分别记录 `tool_status`、`subgoal_outcome`、`native_outcome` 与 `action_feedback_unknown`。子目标只表示登记的公开谓词；夹爪闭合不证明持物。关系保留建立来源和观察时点，新观察不满足或未知时使其失效。半段RPC反馈未知会阻止后续运动，直到原执行被对账；取消和预算停止由原工具/动作链传播。D/S/R/A仍沿同一VLA、动作RPC和五步前缀。私有评价材料只供独立评价器读取。

## L1 持久出口

profile 可显式启用：

```json
{
  "l1_media_policy": {
    "schema": "rsi-l1-media-policy-v1",
    "optional_artifacts": ["agentview_world_high.npz", "wrist_world_high.npz"],
    "protected_artifacts": [],
    "protected_steps": [],
    "max_persisted_bytes": 1048576
  }
}
```

core填入并核验该Bundle的 `rho_ref`，限制上限不超过Runtime存储额度。`configure_media_policy → EnvState.save`读取冻结ρA，先选择媒体，再按实际编码临时字节检查是否发布。未配置时全部保留。

高分辨率world图也可能用于 `back_project`。因此只有development中的Flash D/S/R、无 `relation_control` 时才能筛选上述可选派生物。A、交互控制、其他附件、终局/截断状态、`extras.audit_required`、显式保护步骤与评价媒体全部保留。策略不删除历史文件。

事件 `media_retention_selection/media_persistence_admission/media_materialized`记录选择机制、条件概率及实际字节。确定性选择的概率为0/1；未知的结果标注/修复选择概率保持null。持久字节、临时字节/峰值、累计下界分别报告，清理不退款；采样间隙下 `exact_cumulative_bytes` 为null。core的 `review_media_persistence`独立核对回执和真实文件。

## 记忆、来源与验证

评价scope的 `MemoryManager`把合法语料绑定到该运行的 `frozen_memory`。实际文件工具读取该副本并记录 `memory_reads.jsonl`；merge/sync/rebuild与语料写入被拒。经验合并保留修订前版本与原冲突来源。

Flywheel导出保持 `autonomous_execution/public_demonstration/teacher_data/independent_evaluation` 来源，历史缺失保留 `legacy_unspecified`。自主成功和评价轨迹不能仅凭成功标签进入示范导出；正式数据审核由 core F10 的独立消费者处理。

现有离线检查位于 `test_skill_contracts.py`、`test_skill_routing.py`、`test_execution_result_chain.py`、`test_media_persistence_contracts.py`、`test_evaluation_freeze_contracts.py` 和 `test_source_admission_contracts.py`。跨core更新与实际消费者的证据统一见[OFFLINE_CHECKS §35](../../core/OFFLINE_CHECKS.md#35-f03f04-补修与-f05f09-第二批工程验证2026-11-17)。本批外部模型/Env使用边界替身；进程、RPC、存储与隔离为真实实现。真实机器人收益和正式评价不由这些检查建立。

## 审核、原生训练与新目标

独立数据审核由 core `collection.review_training_dataset` 每次重读实际载荷后生成准入回执；可写 `payload_audit_ref` 只有与独立结果全内容一致时才可使用。Host 的 `training_prepare` 只接受 operator 或 evaluation 受保护域中的准入回执；Research 自述成功、更新快照或自行放置 admitted JSON 均不构成独立准入。训练准备只消费匹配 scope、protocol、审核者和数据字节的记录。来源保留公共、授权示范、教师、自主和评价身份；未支持的监督来源可保存，不能自动升级为训练示范。准备产出原生 dataloader、transforms 和 norm 配置，实际训练沿 RLinf `SFTRunner`，加载检查沿当前 `Pi05VLAFacade`，再从原始训练/加载回执绑定后继模型和处理器。

受信物化器可将固定场景、对象和谓词范围内的新 conjunction 写为独立 BDDL、指令、匹配初态引用和新协议。RPent 用以下参数消费该任务：

```text
--suite materialized --task 0 --seed 0 --materialized-task /path/to/task.json
```

`task.json` 必须与独立 `materialization.json` 的身份一致。`task_key`、BDDL、初态、物化回执和 `native_success=bddl_goal` 进入 `RunConfig.task_desc`、运行上下文及独立结果；改文件或混用任务身份会被拒绝。薄适配只选择任务文件，复用 RLinf 的 reset、动作、终止和原生成功判据。合成检查验证解析和消费者；真实环境初始化、匹配初态可运行性与物理可达性在 P/V 验证。

本批工程边界替换外部模型、trainer 和 dataloader 重组件，控制、进程、RPC、存储、原回执与角色隔离使用真实实现。合成模型文件与独立评分仅属于隔离工程域。接口和当前完成度见[项目文档01](../../项目文档/01_系统设计与框架复用.md)及[唯一状态记录04](../../项目文档/04_实施路线与执行状态.md)。


## 仿真验收与真实组件边界

第四批 `G_F_sim` 覆盖 F00—F14 仿真适用项、F16_sim 和 C01—C07/C09/C10；通过后仅进入适用且获授权的 P_sim。C08 未验收、延期；F15、真机 F16、P07 继续等待 Piper 接口、现场证据与共享版本兼容，全局 G_F 尚不能由仿真通过替代。检查记录见[OFFLINE_CHECKS §37](../../core/OFFLINE_CHECKS.md#37-第四批分范围验收与文档准备2026-11-17)。

真实模型请求、实际 Env 初始化、训练梯度/完整保存、原生加载与任务收益分别验收。D/S/R/A 的主问题与强简单基线、协议统计和评价信息流先按[03 §9](../../项目文档/03_实验设计与验收协议.md#9-第四批-p03p05-草案2026-11-17)准备草案；正式比较要补齐实测数值并预先注册。RPent 不因合成工程文件或 core 的一次三态结论自动取得真实组件资格。
