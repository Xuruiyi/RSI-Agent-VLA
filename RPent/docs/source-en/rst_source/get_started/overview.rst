Introduction to RPent
=====================

**RPent (Recursive Physical Agent)** is an open framework for embodied agents.
It connects a planner to perception, action models, scripted tools, and task
memory. The planner observes the environment, chooses tools, and uses their
results to decide the next action. Exploration records experience for later runs.

The five points of the Pentagram represent these combined capabilities; the
central infinity symbol represents the recurring observation and action loop.

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/rpent_framework.png
   :alt: RPent planning, perception, memory, and execution architecture
   :width: 100%

RPent uses **service-oriented, standardized, and composable** components.
Models and environments run as reusable services with shared interfaces, so
planners, tools, and robots can be extended independently. See
:doc:`../development/architecture` for the execution flow and module boundaries.

Leaderboard
-----------

Compare success rates on LIBERO, LIBERO-PRO, RoboCasa365 Target50, and RoboTwin
C2R. Rankings apply to the methods and evaluation coverage shown; see
:doc:`../leaderboard/index` for detailed results, configurations, and sources.

.. raw:: html

   <a href="../leaderboard/index.html">
     <img class="only-light" src="../../_static/leaderboard/leaderboard-en-light.svg" alt="RPent Leaderboard" width="100%">
     <img class="only-dark" src="../../_static/leaderboard/leaderboard-en-dark.svg" alt="RPent Leaderboard" width="100%">
   </a>

Choose a Platform
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Platform
     - Documentation
   * - :doc:`LIBERO <../simulators/libero>`
     - Pi0.5, SAM3, and LIBERO / LIBERO-PRO runs and reproduction.
   * - :doc:`RoboCasa365 <../simulators/robocasa>`
     - RLDX-1, kitchen tasks, and Target50 reproduction.
   * - :doc:`RoboTwin <../simulators/robotwin>`
     - LingBot-VLA, dual-arm simulation tasks, and C2R reproduction.
   * - :doc:`RoboDojo <../simulators/robodojo>`
     - Dual-arm ARX-X5 manipulation in Isaac Sim, with cuRobo motion planning and SAM3.
   * - :doc:`Single-Arm Franka <../real_world_robots/franka>`
     - Hardware preparation, calibration, motion checks, and operation.
   * - :doc:`Dual-Arm Franka <../real_world_robots/dual_franka>`
     - Two-node deployment, operation, and exploration with an operator.
   * - :doc:`YAM <../real_world_robots/yam>`
     - Two-machine deployment, operator controls, task execution, and exploration.
   * - :doc:`SO-101 <../real_world_robots/so101>`
     - Coming soon.

Choose ``api``, ``claude_code``, or ``codex`` for online planning. LIBERO supports :doc:`Flash Mode <../guides/flash>`; single- and dual-arm Franka support :ref:`reviewed Flash task cards <franka-flash>`. See :doc:`../guides/configure_planner` for model-service configuration.

Start with :doc:`quickstart`. See :doc:`../leaderboard/index` for reported results and resource costs, and :doc:`../resources/harnessvla` for the research background.

Cosmos Policy, Cosmos3, and Fast-WAM deployment and supported platform pairs are documented in :doc:`../development/add_action_model`. DreamZero remains on the roadmap.
