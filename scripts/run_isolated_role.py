"""Run one publication role with Linux read-only mount boundaries."""

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path


def command(
    task_root,
    role,
    argv,
    *,
    gpu=False,
    scratch_root=None,
    release_root=None,
    protected_paths=(),
):
    task_root = Path(task_root).resolve()
    domains = {
        "operator": task_root / "operator",
        "research": task_root / "state/research",
        "evaluation": task_root / "state/evaluation",
        "release": task_root / "release",
        "runtime": task_root / "runtime",
    }
    if release_root is not None:
        domains["release"] = Path(release_root).resolve(strict=True)
        if (
            domains["release"].is_relative_to(task_root)
            and domains["release"] != task_root / "release"
        ):
            raise ValueError("release override overlaps a different task domain")
    release = domains["release"]
    if any(
        release.is_relative_to(path) or path.is_relative_to(release)
        for name, path in domains.items()
        if name != "release"
    ):
        raise ValueError("release write domain overlaps another role")
    if role not in domains:
        raise ValueError("unsupported restricted role")
    scratch = (
        Path(scratch_root).resolve() if scratch_root else task_root / "scratch" / role
    )
    if not scratch.is_relative_to(task_root / "scratch" / role):
        raise ValueError("scratch escapes the registered role domain")
    scratch.mkdir(parents=True, exist_ok=True)
    args = [
        "/bin/bwrap",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--cap-drop",
        "ALL",
        "--new-session",
        "--die-with-parent",
        "--ro-bind",
        "/",
        "/",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/dev/shm",
        "--bind",
        str(scratch),
        "/tmp",
        "--ro-bind",
        str(task_root),
        str(task_root),
        "--bind",
        str(domains[role]),
        str(domains[role]),
        "--setenv",
        "TMPDIR",
        "/tmp",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        "--setenv",
        "RSI_ISOLATED_ROLE",
        role,
        "--setenv",
        "RSI_ISOLATION_TASK_ROOT",
        str(task_root),
    ]
    if release_root is not None and role != "release":
        args += ["--ro-bind", str(domains["release"]), str(domains["release"])]
    for protected in protected_paths:
        path = Path(protected).resolve(strict=True)
        args += ["--ro-bind", str(path), str(path)]
    if role not in {"runtime", "research"}:
        args += ["--unshare-net"]
    else:
        # Reuse the existing login through read-only links; never copy auth data.
        # All SDK journals and session databases belong to the runtime write domain.
        codex_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
        private_home = scratch / "codex"
        private_home.mkdir(exist_ok=True)
        for name in ("auth.json", "config.toml"):
            source = codex_home / name
            target = private_home / name
            if source.exists() and not target.exists():
                target.symlink_to(source)
        args += [
            "--setenv",
            "CODEX_HOME",
            "/tmp/codex",
            "--setenv",
            "XDG_CACHE_HOME",
            "/tmp/cache",
            "--setenv",
            "MPLCONFIGDIR",
            "/tmp/matplotlib",
            "--setenv",
            "TRITON_CACHE_DIR",
            "/tmp/triton",
            "--setenv",
            "CUDA_CACHE_PATH",
            "/tmp/cuda-cache",
        ]
    if role == "runtime":
        # LIBERO's installed config reader takes a mutable file lock during
        # import. Reuse the exact read-only config and put only its lock in
        # this task's scratch, keeping the user's config directory immutable.
        libero_config = scratch / "liberopro"
        libero_config.mkdir(exist_ok=True)
        config_link = libero_config / "config.yaml"
        if not config_link.exists():
            config_link.symlink_to(Path.home() / ".liberopro/config.yaml")
        args += ["--setenv", "LIBERO_CONFIG_PATH", "/tmp/liberopro"]
        # The existing pinned-memory consumer always opens its coordination
        # lock in append mode, even for a verified offline read. Only that
        # inode is writable; the corpus and replacement parents stay read-only.
        repo = Path(__file__).resolve().parents[1]
        for memory_lock in (repo / "RPent/memory/libero/.versions").glob("*/sync.lock"):
            args += ["--bind", str(memory_lock), str(memory_lock)]
    # flock uses the existing host inode even though each role has private /tmp.
    lock = Path("/tmp/rsi-agent-vla-dynamic.lock")
    if role in {"release", "runtime", "research"}:
        args += ["--bind", str(lock), str(lock)]
    if gpu:
        if role != "runtime":
            raise ValueError("only runtime can access GPU devices")
        for path in glob.glob("/dev/nvidia*") + glob.glob("/dev/dri/renderD*"):
            if Path(path).is_file() or not Path(path).is_dir():
                args += ["--dev-bind", path, path]
        if Path("/dev/nvidia-caps").is_dir():
            args += ["--dev-bind", "/dev/nvidia-caps", "/dev/nvidia-caps"]
    return args + ["--"] + list(argv)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", required=True, type=Path)
    parser.add_argument(
        "--role",
        required=True,
        choices=("operator", "research", "evaluation", "release", "runtime"),
    )
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    if not argv:
        parser.error("a role command is required")
    return subprocess.call(command(args.task_root, args.role, argv, gpu=args.gpu))


if __name__ == "__main__":
    sys.exit(main())
