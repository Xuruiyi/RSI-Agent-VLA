# RSI-Agent-VLA

VLA Agent 研究与开发工作区。

- `core/`：研究与执行核心代码、离线测试。
- `config/`、`scripts/`：实验配置和运行脚本。
- `项目文档/`、`问答与自由审查/`：研究方案、设计与审查记录。
- `RPent/`、`RLinf/`、`OpenETA/`、`ENPIRE/`、`gepa/`、`openpi/`、`skydiscover/`：当前使用的项目源码，各目录保留原许可证。

## 提交历史

`main` 保存本地工作区的上传快照，包含上传时尚未提交的源码修改。
原项目历史通过合并父提交连接到 `main`，原有提交 ID、作者和提交信息保持不变。
`history/<项目名>/main` 提供原项目分支入口；这些历史提交中的路径沿用原仓库布局。
RPent 原先记录的上游分支保存在 `history/RPent/remotes/origin/main`。

```bash
git log history/RPent/main
git log --all --graph --oneline
```

## 本地文件

下载的数据、模型权重、运行日志、实验输出、虚拟环境、缓存和本机凭据配置不纳入上传。
原项目已跟踪的示例资源、测试输入和数据处理代码随源码保留。
本地 `config/models.local.sh` 需要在运行机器上单独配置。

各组件的安装与运行方式见其目录中的 README，研究目标和执行计划见 `项目文档/`。
