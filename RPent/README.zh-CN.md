<div align="center">
  <img src="https://github.com/RLinf/misc/raw/main/pic/rpent_logo.png" alt="RPent-logo" width="520"/>
</div>

<div align="center">
<a href="https://arxiv.org/abs/2607.08448"><img src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv"></a>
<a href="https://huggingface.co/RLinf"><img src="https://img.shields.io/badge/HuggingFace-yellow?logo=huggingface&logoColor=white" alt="Hugging Face"></a>
<a href="https://rpent.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/Documentation-Purple?color=8A2BE2&logo=readthedocs"></a>
<a href="https://rpent.readthedocs.io/zh-cn/latest/"><img src="https://img.shields.io/badge/中文文档-red?logo=readthedocs"></a>
<a href="https://github.com/RLinf/misc/blob/main/pic/wechat.jpg?raw=true"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>
</div>

<div align="center">

[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![简体中文](https://img.shields.io/badge/语言-简体中文-red.svg)](README.zh-CN.md)

</div>

<h1 align="center">
  <sub>RPent: 面向物理世界的智能体基础设施</sub>
</h1>

**RPent (Recursive Physical Agent)** 是一个具身智能体开源框架，通过可复用服务连接规划器、感知、动作模型、程序化工具和任务记忆。智能体观察环境、选择工具、检查执行结果，并保存经验供后续运行复用。

<div align="center">
  <img src="https://github.com/RLinf/misc/raw/main/pic/rpent_framework.png" alt="RPent framework"/>
</div>

## 排行榜

对比 RPent 与参考方法在 LIBERO、LIBERO-PRO、RoboCasa365 Target50 和 RoboTwin C2R 上的成功率。排名仅限图中方法及评测范围；套件成绩和模型配置见 [排行榜](https://rpent.readthedocs.io/zh-cn/latest/rst_source/leaderboard/index.html)。

Codex / GPT-6 Astra / low / reasoning 已完成全部八套 LIBERO-PRO，**Overall 92.63%（741/800）**。详见 [套件汇总与 memory 批次说明](https://rpent.readthedocs.io/zh-cn/latest/rst_source/leaderboard/index.html#libero-pro-astra-memory)，其中 Long 与 Spatial/Object/Goal 分别使用各自冻结的 memory 批次。

[![RPent 排行榜](https://rpent.readthedocs.io/zh-cn/latest/_static/leaderboard/leaderboard-zh-light.svg)](https://rpent.readthedocs.io/zh-cn/latest/rst_source/leaderboard/index.html)


## 适用用户

- **具身智能研究者**：比较规划器、动作模型与记忆在操作任务中的表现。
- **在线学习与强化学习研究者**：采集交互数据，研究任务经验如何在运行间复用。
- **机器人应用开发者**：接入机器人硬件，组合任务所需的工具和服务。
- **机器人操作员**：完成平台安装、标定与运动检查后，运行已配置的任务。

## 最新动态

- [2026/10] 🔥 新增可选的 APXInf VLA endpoint 接入，参见 [部署指南](https://rpent.readthedocs.io/zh-cn/latest/rst_source/guides/advanced_deployment.html#apxinf-vla-service-optional)。
- [2026/09] 🔥 新增交互式 RPent 排行榜，覆盖 LIBERO、LIBERO-PRO、RoboCasa365 与 RoboTwin，提供模型对比及套件汇总。查看 [排行榜](https://rpent.readthedocs.io/zh-cn/latest/rst_source/leaderboard/index.html)。
- [2026/09] 🔥 RPent 支持 Franka 单臂与双臂真机扩展。文档：[Franka](https://rpent.readthedocs.io/zh-cn/latest/rst_source/real_world_robots/franka.html) · [Dual Franka](https://rpent.readthedocs.io/zh-cn/latest/rst_source/real_world_robots/dual_franka.html)。
- [2026/08] 🔥 支持 RoboCasa，使用 RLDX-1 作为操作模型。参见 [RoboCasa 安装与 Target50 指南](https://rpent.readthedocs.io/zh-cn/latest/rst_source/simulators/robocasa.html)。
- [2026/08] 🔥 新增非推理模式；历史 LIBERO-PRO Long 评测的平均运行时间降低约 40%。
- [2026/08] 🔥 支持 LIBERO 探索模式。文档：[LIBERO 探索模式](https://rpent.readthedocs.io/zh-cn/latest/rst_source/simulators/libero.html#libero-exploration)。
- [2026/08] 🔥 支持 RoboTwin，使用 LingBot-VLA 处理双臂操作任务。文档：[RoboTwin](https://rpent.readthedocs.io/zh-cn/latest/rst_source/simulators/robotwin.html)。
- [2026/07] 🔥 RPent 首篇论文 [Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents](https://arxiv.org/abs/2607.08448) 发布。

## 功能矩阵

| 组件 | 已有集成 |
| --- | --- |
| [规划器](https://rpent.readthedocs.io/zh-cn/latest/rst_source/guides/configure_planner.html) | 直接 API、Claude Code、Codex；支持[自定义规划器](https://rpent.readthedocs.io/zh-cn/latest/rst_source/development/add_planner.html) |
| [动作模型](https://rpent.readthedocs.io/zh-cn/latest/rst_source/guides/configure_primitives.html) | Pi0.5、RLDX-1、LingBot-VLA；Cosmos Policy / Cosmos3 支持 LIBERO 和 LIBERO-PRO；Fast-WAM 支持 LIBERO、LIBERO-PRO 和 RoboTwin |
| 仿真环境 | [LIBERO / LIBERO-PRO / LIBERO-plus](https://rpent.readthedocs.io/zh-cn/latest/rst_source/simulators/libero.html)、[RoboCasa365](https://rpent.readthedocs.io/zh-cn/latest/rst_source/simulators/robocasa.html)、[RoboTwin](https://rpent.readthedocs.io/zh-cn/latest/rst_source/simulators/robotwin.html)、[RoboDojo](https://rpent.readthedocs.io/zh-cn/latest/rst_source/simulators/robodojo.html) |
| 真实机器人 | [Franka](https://rpent.readthedocs.io/zh-cn/latest/rst_source/real_world_robots/franka.html)、[Dual Franka](https://rpent.readthedocs.io/zh-cn/latest/rst_source/real_world_robots/dual_franka.html)、[YAM](https://rpent.readthedocs.io/zh-cn/latest/rst_source/real_world_robots/yam.html) |
| 计划重放 | [LIBERO Flash Mode](https://rpent.readthedocs.io/zh-cn/latest/rst_source/guides/flash.html)、[Franka 任务卡](https://rpent.readthedocs.io/zh-cn/latest/rst_source/real_world_robots/franka.html#franka-flash) |

RoboDojo 使用程序化运动、夹爪工具与 SAM3。动作模型支持范围取决于平台和 checkpoint，兼容组合见[动作模型指南](https://rpent.readthedocs.io/zh-cn/latest/rst_source/development/add_action_model.html)。DreamZero 与 [SO-101](https://rpent.readthedocs.io/zh-cn/latest/rst_source/real_world_robots/so101.html) 仍在规划中。

## 快速开始

以下示例在 Linux 和 NVIDIA GPU 上运行一个 LIBERO-PRO 任务。先准备 `uv`、兼容的 CUDA 驱动和 Anthropic API Key；完整前提与排障见[快速开始文档](https://rpent.readthedocs.io/zh-cn/latest/rst_source/get_started/quickstart.html)。其他平台使用独立环境，分别按 [RoboCasa365](https://rpent.readthedocs.io/zh-cn/latest/rst_source/simulators/robocasa.html)、[RoboTwin](https://rpent.readthedocs.io/zh-cn/latest/rst_source/simulators/robotwin.html)、[RoboDojo](https://rpent.readthedocs.io/zh-cn/latest/rst_source/simulators/robodojo.html) 或[真机部署](https://rpent.readthedocs.io/zh-cn/latest/rst_source/real_world_robots/index.html)指南安装。

**1. 在独立环境中安装 RPent。**

```bash
git clone https://github.com/RLinf/RPent.git
cd RPent
uv venv --python 3.11
source .venv/bin/activate
# 先安装兼容 CUDA 构建的 torch==2.7.1 和 torchvision==0.22.1。
uv pip install -e ".[libero-pro]"
uv pip install "huggingface_hub>=0.34,<1.0" modelscope
```

按 [PyTorch 安装说明](https://pytorch.org/get-started/previous-versions/)选择 CUDA 构建，并将 `pip install` 换成 `uv pip install`。

**2. 下载仿真资源与模型权重。**

```bash
liberopro-download-assets --skip-existing
hf download RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT \
  --exclude optimizer.pt \
  --local-dir ./checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT
modelscope download --model facebook/sam3 sam3.pt \
  --local_dir ./checkpoints/sam3

export PI05_CHECKPOINT_PATH="$PWD/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT"
export SAM3_CHECKPOINT_PATH="$PWD/checkpoints/sam3/sam3.pt"
```

访问 Hugging Face 较慢时，可在相应下载命令前加 `HF_ENDPOINT=https://hf-mirror.com`。

**3. 检查规划器连接并运行任务。**

```bash
export ANTHROPIC_API_KEY="YOUR_API_KEY"
rpent-check-llm --planner claude_code --model claude-opus-4-8
rpent --robot libero --libero-type pro \
  --suite libero_object_swap --task 2 --seed 0 \
  --planner claude_code --model claude-opus-4-8 \
  --output-dir ./logs/first-libero-pro
```

运行结束后，在输出目录查看 `result.json`、`episode.mp4`、`transcript_*.json` 和 `run.log`。LIBERO 以环境原生信号判定成功，[结果字段说明](https://rpent.readthedocs.io/zh-cn/latest/rst_source/simulators/libero.html#libero-results)给出了具体含义。GPU 选择见 [CLI 设备参数](https://rpent.readthedocs.io/zh-cn/latest/rst_source/guides/cli.html)，其他规划器配置见[规划器指南](https://rpent.readthedocs.io/zh-cn/latest/rst_source/guides/configure_planner.html)，任务经验的生成与复用见[记忆与探索指南](https://rpent.readthedocs.io/zh-cn/latest/rst_source/guides/memory.html)。

### 交互模式

在任务命令中加 `--tui`（`-t`），即可在终端输入指令。Claude Code 和 Codex 在执行期间接收消息，API 规划器在回复后接收后续输入。SDK 控制命令用 `/help` 查看，退出用 `/exit`。详见[交互使用](https://rpent.readthedocs.io/zh-cn/latest/rst_source/guides/dashboard.html#terminal-interaction)。

```bash
rpent --robot libero --suite libero_object_swap --task 2 --seed 0 \
  --planner claude_code --model claude-opus-4-8 --tui
```

### 实时 Dashboard

加上 `--dashboard` 后，会启动本地 Dashboard，并在终端输出访问地址。Session 配置全部来自命令行，页面打开后直接进入实时监控；服务就绪后，通过 `/rpent-task <suite> <task> <seed>` 启动任务。页面会实时显示智能体的推理过程、相机画面和动作时间线，任务结束后可以继续提交下一任务。使用 `--dashboard-language zh-cn` 可切换到中文界面。

```bash
rpent --robot libero --dashboard --dashboard-language zh-cn \
  --planner claude_code --model claude-opus-4-8
```

完整的命令行参数列表见 [命令行参考](https://rpent.readthedocs.io/zh-cn/latest/rst_source/guides/cli.html)。RoboCasa 与 RoboTwin 提供各自的任务参数，参见 [RoboCasa](https://rpent.readthedocs.io/zh-cn/latest/rst_source/simulators/robocasa.html) 与 [RoboTwin](https://rpent.readthedocs.io/zh-cn/latest/rst_source/simulators/robotwin.html) 文档。

更详细的文档请参见 [RPent 中文文档](https://rpent.readthedocs.io/zh-cn/latest/)。

## 参与贡献

欢迎社区贡献。开发环境、必需检查、测试规范和集成检查清单请参见
[CONTRIBUTING_zh.md](CONTRIBUTING_zh.md)。

## 引用与致谢

如果 **RPent** 或 **Harness VLA** 对你的工作有帮助，请引用：

```bibtex
@article{zhang2026harnessvla,
  title={Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents},
  author={Zhang, Yixian and Zhang, Huanming and Gao, Feng and Li, Xiao and Liu, Zhihao and Zhu, Chunyang and Qiu, Jiaxing and Yan, Yuchen and Liu, Jiyuan and Tang, Wenhao and Fang, Zhengru and Nie, Yi and Wei, Changxu and Wang, Yu and Ding, Wenbo and Yu, Chao},
  journal={arXiv preprint arXiv:2607.08448},
  year={2026},
  url={https://arxiv.org/abs/2607.08448}
}
```

RPent 构建于 [RLinf](https://github.com/RLinf/RLinf) 的仿真器、VLA 模型与训练基础设施之上，也得益于更广泛开源社区的 agent SDK — [pydantic-ai](https://ai.pydantic.dev/)、[Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk/overview) 与 OpenAI Codex SDK。感谢 LIBERO、RoboCasa、robosuite、MuJoCo、openpi 背后的团队。
