RPent 简介
============

RPent（Recursive Physical Agent）是一个具身智能体开源框架，将规划器与感知、动作模型、程序化工具和任务记忆连接起来。规划器观察环境、选择工具，再根据执行结果决定下一步；探索模式将经验保存下来，供后续运行复用。

Pent 源自五芒星（Pentagram），五个顶点象征不同能力的组合，中心的无穷符号（∞）代表持续的观测与动作循环。

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/rpent_framework.png
   :alt: RPent 规划、感知、记忆和执行架构
   :width: 100%

RPent 遵循服务化、标准化和可组合的设计原则。模型与环境作为可复用服务运行，通过统一接口连接，规划器、工具和机器人可独立扩展。执行流程与模块职责见 :doc:`../development/architecture`。

排行榜
------

对比 LIBERO、LIBERO-PRO、RoboCasa365 Target50 和 RoboTwin C2R 上的成功率。
排名仅限图中方法及评测范围；完整结果、模型配置和来源见 :doc:`../leaderboard/index`。

.. raw:: html

   <a href="../leaderboard/index.html">
     <img class="only-light" src="../../_static/leaderboard/leaderboard-zh-light.svg" alt="RPent 排行榜" width="100%">
     <img class="only-dark" src="../../_static/leaderboard/leaderboard-zh-dark.svg" alt="RPent 排行榜" width="100%">
   </a>

选择平台
------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 平台
     - 文档内容
   * - :doc:`LIBERO <../simulators/libero>`
     - Pi0.5、SAM3、LIBERO / LIBERO-PRO 运行与复现。
   * - :doc:`RoboCasa365 <../simulators/robocasa>`
     - RLDX-1、厨房任务和 Target50 复现。
   * - :doc:`RoboTwin <../simulators/robotwin>`
     - LingBot-VLA、双臂仿真任务和 C2R 复现。
   * - :doc:`RoboDojo <../simulators/robodojo>`
     - Isaac Sim 双臂 ARX-X5 操作，使用 cuRobo 运动规划与 SAM3。
   * - :doc:`单臂 Franka <../real_world_robots/franka>`
     - 硬件准备、标定、自检和运行。
   * - :doc:`双臂 Franka <../real_world_robots/dual_franka>`
     - 双节点部署、操作与人工参与的探索。
   * - :doc:`YAM <../real_world_robots/yam>`
     - 双机部署、操作员控制、任务运行与探索。
   * - :doc:`SO-101 <../real_world_robots/so101>`
     - 内容即将推出。

在线规划器可选择 ``api``、``claude_code`` 或 ``codex``。LIBERO 支持 :doc:`Flash Mode <../guides/flash>`，单臂和双臂 Franka 支持 :ref:`经审核的 Flash 任务卡 <franka-flash>`。模型服务配置见 :doc:`../guides/configure_planner`。

从 :doc:`quickstart` 开始运行第一个任务。评测成绩和资源开销见 :doc:`../leaderboard/index`，研究背景见 :doc:`../resources/harnessvla`。

Cosmos Policy、Cosmos3 和 Fast-WAM 的部署及支持的平台组合见 :doc:`../development/add_action_model`。DreamZero 仍在规划中。
