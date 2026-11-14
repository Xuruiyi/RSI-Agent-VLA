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
  <sub>RPent: Agentic Infrastructure for the Physical World</sub>
</h1>

**RPent (Recursive Physical Agent)** is an open framework for embodied agents. It connects planners, perception, action models, scripted tools, and task memory through reusable services. Agents observe the environment, choose tools, inspect results, and record experience for later runs.

<div align="center">
  <img src="https://github.com/RLinf/misc/raw/main/pic/rpent_framework.png" alt="RPent framework"/>
</div>

## Leaderboard

Compare RPent with reference methods on LIBERO, LIBERO-PRO, RoboCasa365 Target50, and RoboTwin C2R. Rankings apply to the methods and evaluation coverage shown; see the [Leaderboard](https://rpent.readthedocs.io/en/latest/rst_source/leaderboard/index.html) for suite results and model configurations.

Codex / GPT-6 Astra / low / reasoning: **92.63% Overall (741/800)** across all eight LIBERO-PRO suites. See the [suite results and memory-batch explanation](https://rpent.readthedocs.io/en/latest/rst_source/leaderboard/index.html#libero-pro-astra-memory), including the separately frozen Long and Spatial/Object/Goal memory batches.

[![RPent Leaderboard](https://rpent.readthedocs.io/en/latest/_static/leaderboard/leaderboard-en-light.svg)](https://rpent.readthedocs.io/en/latest/rst_source/leaderboard/index.html)


## Who Uses RPent?

- **Embodied intelligence researchers:** compare planners, action models, and memory on manipulation tasks.
- **Online-learning and reinforcement-learning researchers:** collect interaction data and study how task experience transfers across runs.
- **Robotics application developers:** connect robot hardware and compose task-specific tools and services.
- **Robot operators:** run configured tasks after completing the platform's installation, calibration, and motion checks.

## What's NEW!

- [2026/10] 🔥 Added an optional APXInf VLA endpoint. See the [deployment guide](https://rpent.readthedocs.io/en/latest/rst_source/guides/advanced_deployment.html#apxinf-vla-service-optional).
- [2026/09] 🔥 Added the interactive RPent Leaderboard for LIBERO, LIBERO-PRO, RoboCasa365, and RoboTwin, with model comparisons and suite-level results. Explore the [Leaderboard](https://rpent.readthedocs.io/en/latest/rst_source/leaderboard/index.html).
- [2026/09] 🔥 RPent supports Franka single-arm and dual-arm real-robot extensions. Doc: [Franka](https://rpent.readthedocs.io/en/latest/rst_source/real_world_robots/franka.html) · [Dual Franka](https://rpent.readthedocs.io/en/latest/rst_source/real_world_robots/dual_franka.html).
- [2026/08] 🔥 RPent supports RoboCasa with RLDX-1 as manipulation model. See the [RoboCasa setup and Target50 guide](https://rpent.readthedocs.io/en/latest/rst_source/simulators/robocasa.html).
- [2026/08] 🔥 Added non-reasoning mode; the historical LIBERO-PRO Long evaluation measured about 40% lower average runtime.
- [2026/08] 🔥 RPent supports exploration mode for LIBERO. Doc: [LIBERO exploration mode](https://rpent.readthedocs.io/en/latest/rst_source/simulators/libero.html#libero-exploration).
- [2026/08] 🔥 RPent supports RoboTwin with LingBot-VLA for dual-arm manipulation tasks. Doc: [RoboTwin](https://rpent.readthedocs.io/en/latest/rst_source/simulators/robotwin.html).
- [2026/07] 🔥 Our first RPent publication, [Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents](https://arxiv.org/abs/2607.08448), is released.

## Feature Matrix

| Component | Available integrations |
| --- | --- |
| [Planners](https://rpent.readthedocs.io/en/latest/rst_source/guides/configure_planner.html) | Direct API, Claude Code, Codex; [custom planners](https://rpent.readthedocs.io/en/latest/rst_source/development/add_planner.html) |
| [Action models](https://rpent.readthedocs.io/en/latest/rst_source/guides/configure_primitives.html) | Pi0.5, RLDX-1, LingBot-VLA; Cosmos Policy / Cosmos3 for LIBERO and LIBERO-PRO; Fast-WAM for LIBERO, LIBERO-PRO, and RoboTwin |
| Simulators | [LIBERO / LIBERO-PRO / LIBERO-plus](https://rpent.readthedocs.io/en/latest/rst_source/simulators/libero.html), [RoboCasa365](https://rpent.readthedocs.io/en/latest/rst_source/simulators/robocasa.html), [RoboTwin](https://rpent.readthedocs.io/en/latest/rst_source/simulators/robotwin.html), [RoboDojo](https://rpent.readthedocs.io/en/latest/rst_source/simulators/robodojo.html) |
| Real robots | [Franka](https://rpent.readthedocs.io/en/latest/rst_source/real_world_robots/franka.html), [Dual Franka](https://rpent.readthedocs.io/en/latest/rst_source/real_world_robots/dual_franka.html), [YAM](https://rpent.readthedocs.io/en/latest/rst_source/real_world_robots/yam.html) |
| Replay | [LIBERO Flash Mode](https://rpent.readthedocs.io/en/latest/rst_source/guides/flash.html), [Franka task cards](https://rpent.readthedocs.io/en/latest/rst_source/real_world_robots/franka.html#franka-flash) |

RoboDojo uses scripted motion and gripper tools with SAM3. Model support depends on the platform and checkpoint; see the [action-model guide](https://rpent.readthedocs.io/en/latest/rst_source/development/add_action_model.html) for compatible pairs. DreamZero and [SO-101](https://rpent.readthedocs.io/en/latest/rst_source/real_world_robots/so101.html) remain planned.

## Quick Start

This example runs one LIBERO-PRO task on Linux with an NVIDIA GPU. Prepare `uv`, a compatible CUDA driver, and an Anthropic API key. See the [full Quick Start](https://rpent.readthedocs.io/en/latest/rst_source/get_started/quickstart.html) for installation prerequisites and troubleshooting. Use separate environments for other platforms: [RoboCasa365](https://rpent.readthedocs.io/en/latest/rst_source/simulators/robocasa.html), [RoboTwin](https://rpent.readthedocs.io/en/latest/rst_source/simulators/robotwin.html), [RoboDojo](https://rpent.readthedocs.io/en/latest/rst_source/simulators/robodojo.html), and [real robots](https://rpent.readthedocs.io/en/latest/rst_source/real_world_robots/index.html).

**1. Install RPent in an isolated environment.**

```bash
git clone https://github.com/RLinf/RPent.git
cd RPent
uv venv --python 3.11
source .venv/bin/activate
# First install torch==2.7.1 and torchvision==0.22.1 with a compatible CUDA build.
uv pip install -e ".[libero-pro]"
uv pip install "huggingface_hub>=0.34,<1.0" modelscope
```

Choose the CUDA build using the [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/), replacing `pip install` with `uv pip install`.

**2. Download simulator assets and model weights.**

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

For slow Hugging Face downloads, optionally prefix the relevant command with `HF_ENDPOINT=https://hf-mirror.com`.

**3. Check the planner connection and run a task.**

```bash
export ANTHROPIC_API_KEY="YOUR_API_KEY"
rpent-check-llm --planner claude_code --model claude-opus-4-8
rpent --robot libero --libero-type pro \
  --suite libero_object_swap --task 2 --seed 0 \
  --planner claude_code --model claude-opus-4-8 \
  --output-dir ./logs/first-libero-pro
```

Inspect `result.json`, `episode.mp4`, `transcript_*.json`, and `run.log` in the output directory. LIBERO's native success signal determines task success; see [result fields](https://rpent.readthedocs.io/en/latest/rst_source/simulators/libero.html#libero-results). Choose GPUs using the [CLI device options](https://rpent.readthedocs.io/en/latest/rst_source/guides/cli.html), and configure other planners through [Planner Configuration](https://rpent.readthedocs.io/en/latest/rst_source/guides/configure_planner.html). To build or reuse task memory, follow [Memory and Exploration](https://rpent.readthedocs.io/en/latest/rst_source/guides/memory.html).

### Interactive CLI Mode

Add `--tui` (`-t`) to a task command to enter instructions in a terminal. Claude Code and Codex accept messages during execution; the API planner accepts follow-ups after its response. Use `/help` for SDK controls and `/exit` to end the session. See [Interactive Usage](https://rpent.readthedocs.io/en/latest/rst_source/guides/dashboard.html#terminal-interaction).

```bash
rpent --robot libero --suite libero_object_swap --task 2 --seed 0 \
  --planner claude_code --model claude-opus-4-8 --tui
```

### Live Dashboard

Add `--dashboard` to start a local Dashboard and print its URL in the terminal. Session settings come from the CLI, and the page opens directly in the live monitor. Once the services are ready, start a task with `/rpent-task <suite> <task> <seed>`. The page streams agent reasoning, camera views, and the action timeline, and you can submit another task after the current one finishes. Use `--dashboard-language zh-cn` for the Chinese UI.

```bash
rpent --robot libero --dashboard --dashboard-language zh-cn \
  --planner claude_code --model claude-opus-4-8
```

For a complete list of CLI options, see the [Command-Line Reference](https://rpent.readthedocs.io/en/latest/rst_source/guides/cli.html). RoboCasa and RoboTwin provide their own task parameters — see the [RoboCasa](https://rpent.readthedocs.io/en/latest/rst_source/simulators/robocasa.html) and [RoboTwin](https://rpent.readthedocs.io/en/latest/rst_source/simulators/robotwin.html) docs.

For more detailed documentation, see the [RPent documentation](https://rpent.readthedocs.io/en/latest/).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the
development setup, required checks, testing policy, and integration checklists.

## Citation and Acknowledgement

If you find **RPent** or **Harness VLA** helpful, please cite the paper:

```bibtex
@article{zhang2026harnessvla,
  title={Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents},
  author={Zhang, Yixian and Zhang, Huanming and Gao, Feng and Li, Xiao and Liu, Zhihao and Zhu, Chunyang and Qiu, Jiaxing and Yan, Yuchen and Liu, Jiyuan and Tang, Wenhao and Fang, Zhengru and Nie, Yi and Wei, Changxu and Wang, Yu and Ding, Wenbo and Yu, Chao},
  journal={arXiv preprint arXiv:2607.08448},
  year={2026},
  url={https://arxiv.org/abs/2607.08448}
}
```

RPent builds on the simulators, VLA models, and training infrastructure of [RLinf](https://github.com/RLinf/RLinf), and on the agent SDKs of the broader open-source community — [pydantic-ai](https://ai.pydantic.dev/), the [Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk/overview), and the OpenAI Codex SDK. Thanks to the teams behind LIBERO, RoboCasa, robosuite, MuJoCo, and openpi.
