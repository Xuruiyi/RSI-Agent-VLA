.. _libero-data-flywheel:

采集轨迹与数据飞轮
========================

可选的数据飞轮功能会记录 LIBERO 评测中实际执行的轨迹，但不会改变规划器或动作原语。它首先保存不可变的原始轨迹；转换为训练格式是独立的后续步骤。

安装
----

环境准备和仿真资源下载请参考 :doc:`../get_started/quickstart`。在 RPent 仓库根目录安装 LIBERO-PRO 及数据飞轮导出依赖：

.. code-block:: bash

   uv pip install -e ".[libero-pro,flywheel]"

采集轨迹
--------

在普通 LIBERO 评测命令中开启采集，并指定数据根目录：

.. code-block:: bash

   rpent --robot libero \
     --suite libero_goal --task 0 --seed 0 \
     --planner codex \
     --collect-flywheel-data \
     --flywheel-root /path/to/datacollection

每次运行会在 ``/path/to/datacollection/raw/libero/<suite>/task_<id>/seed_<seed>/`` 下写入一条轨迹，其中包含策略观测、实际执行的动作、奖励、终止标记、原语调用编号和 VLA 预测的动作序列。采集功能默认关闭，仅支持 LIBERO 评测模式。

校验并导出成功轨迹
------------------

使用轨迹前可以单独校验原始数据：

.. code-block:: bash

   rpent-flywheel validate /path/to/raw/episode

将同一任务套件中某个任务下所有已完成的成功轨迹导出为 LeRobot 数据集：

.. code-block:: bash

   rpent-flywheel export-lerobot \
     --data-root /path/to/datacollection \
     --suite libero_goal \
     --task 0 \
     --dataset-id goal-task-00 \
     --output-root /path/to/lerobot

其中 ``--data-root`` 是原始数据根目录，``--output-root`` 是导出数据集的父目录。本例导出的数据集位于 ``/path/to/lerobot/goal-task-00``。

对于成功轨迹，导出从开始到首次完成任务的所有实际执行动作，包含使任务成功的那一步。其中既有脚本原语动作，也有 VLA 动作；成功后的后续动作不导出。

导出程序不会改写原始轨迹。失败轨迹会继续保留以便审计，但不会进入这份监督训练数据。

使用 RLinf 训练
---------------

将成功轨迹导出为 LeRobot 数据集后，可以在独立的 RLinf 环境中进行 Pi0.5 监督微调。RPent 负责采集和导出；模型训练、日志和权重保存由 RLinf 管理。

环境安装和训练配置请参考 `RLinf OpenPI 监督微调指南 <https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/sft_openpi.html>`_。训练数据路径应指向导出的 LeRobot 数据集目录，即包含 ``meta/info.json`` 的目录，而不是原始轨迹目录。

先准备支持 LIBERO LeRobot 数据加载的 RLinf 环境，并将 Pi0.5 LIBERO SFT 配置保存为 ``/path/to/sft-config/libero_pi05_sft.yaml``，再运行：

.. code-block:: bash

   cd /path/to/RLinf
   unset PYTHONPATH
   export PYTHONPATH="$PWD"
   export EMBODIED_PATH="$PWD/examples/sft"

   .venv/bin/python examples/sft/train_vla_sft.py \
     --config-path /path/to/sft-config \
     --config-name libero_pi05_sft \
     data.train_data_paths=/path/to/lerobot/goal-task-00 \
     actor.model.model_path=/path/to/pi05-checkpoint \
     runner.logger.log_path=/path/to/new-training-output

根据可用 GPU 设置配置中的资源分配和 batch size，并使用与所选模型和数据集相匹配的 checkpoint 及归一化设置。

受监督的有界采集
----------------

launcher 可在 ``RSI_RUNTIME_CONTEXT.collection`` 中显式绑定
``aligned-batches-v1``。``--flywheel-root`` 必须与该上下文一致，并位于
受监督作业的输出目录内。此模式每批保存最多 25 个已返回转移，分别保存
策略实际输入、动作候选及对齐日志。``collection.json`` 记录已提交批次、
待反馈发送和最近一次记录的尾部。强杀可能丢失未提交尾部或在途转移；
部分轨迹保留作诊断。策略图像保持原生策略方向，与标定后的 Planner
图像分开。名义 20 Hz 不代表实测墙钟频率。

schema-2 记录的来源为自主执行，训练准入为 ``not_admitted``。任务成功
不能自动把这些动作变成专家监督。格式检查可用
``export-lerobot --purpose diagnostic``，将全部已返回控制步（含失败轨迹）
导出为 LeRobot；``meta/rpent_flywheel.json`` 保留用途、准入和来源引用。
默认示范导出会拒绝自主 schema-2 数据。训练仍需独立、可追溯的示范
manifest 及其准入检查。
