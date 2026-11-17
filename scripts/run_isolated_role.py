"""Run registered roles with Linux mount, process, network and device boundaries."""

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
    hidden_paths=(),
    read_grants=(),
    request_ref=None,
    write_root=None,
):
    task_root = Path(task_root).resolve()
    domains = {
        "operator": task_root / "operator",
        "research": task_root / "state/research",
        "evaluation": task_root / "state/evaluation",
        "release": task_root / "release",
        "runtime": task_root / "runtime",
        "candidate": task_root / "state/research/code_candidates",
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
    if role == "candidate":
        return _candidate_command(
            task_root, argv, gpu=gpu, scratch_root=scratch_root,
            write_root=write_root, hidden_paths=hidden_paths, read_grants=read_grants,
            release_root=release_root,
        )
    if write_root is not None:
        raise ValueError("write override is restricted to code candidates")
    if domains[role].resolve() != domains[role]:
        raise ValueError("role write domain contains a symlink")
    scratch = (
        Path(scratch_root).resolve() if scratch_root else task_root / "scratch" / role
    )
    if not scratch.is_relative_to(task_root / "scratch" / role):
        raise ValueError("scratch escapes the registered role domain")
    if scratch.resolve() != scratch:
        raise ValueError("scratch contains a symlink")
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
    environment = {
        "PATH", "HOME", "LANG", "LC_ALL", "PYTHONPATH", "VIRTUAL_ENV",
        "CUDA_VISIBLE_DEVICES", "MUJOCO_GL", "PYOPENGL_PLATFORM",
        "HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    }
    if role in {"runtime", "research"}:
        environment |= {
            "OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_API_KEY",
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
            "http_proxy", "https_proxy", "all_proxy", "no_proxy",
        }
    for name in sorted(environment):
        if name in os.environ:
            args += ["--setenv", name, os.environ[name]]
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
    masked_directories = []
    hidden = {Path(value).resolve(strict=True) for value in hidden_paths}
    hidden = sorted(path for path in hidden if not any(
        path != parent and parent.is_dir() and path.is_relative_to(parent)
        for parent in hidden
    ))
    for path in hidden:
        if path == Path("/"):
            raise ValueError("cannot hide the filesystem root")
        if path.is_dir():
            masked_directories.append(path)
        args += ["--tmpfs", str(path)] if path.is_dir() else [
            "--ro-bind", "/dev/null", str(path)
        ]
    for grant in read_grants:
        path = Path(grant).resolve(strict=True)
        if any(other.is_relative_to(path) for other in hidden):
            raise ValueError("read grant would re-expose hidden material through a parent mount")
        if any(path.is_relative_to(other) or other.is_relative_to(path)
               for other in (Path("/dev"), Path("/proc"), Path("/sys"))):
            raise ValueError("read grant cannot expose host devices or process namespaces")
        args += ["--ro-bind", str(path), str(path)]
    if request_ref is not None:
        path = Path(request_ref).resolve(strict=True)
        args += ["--ro-bind", str(path), str(path)]
    for path in reversed(masked_directories):
        args += ["--remount-ro", str(path)]
    return ["/usr/bin/env", "-i", *args, "--", *list(argv)]


def _candidate_command(
    task_root, argv, *, gpu, scratch_root, write_root, hidden_paths,
    read_grants, release_root,
):
    """Mount only registered code/dependencies and output into an empty root."""
    if gpu:
        raise ValueError("code candidates cannot access GPU devices")
    base = (task_root / "state/research/code_candidates").resolve()
    output = Path(write_root).resolve(strict=True) if write_root else None
    if output is None or not output.is_relative_to(base) or output == base:
        raise ValueError("candidate output escapes the registered candidate domain")
    scratch_base = task_root / "scratch/candidate"
    scratch = Path(scratch_root).resolve() if scratch_root else scratch_base
    if not scratch.is_relative_to(scratch_base):
        raise ValueError("scratch escapes the registered candidate domain")
    if scratch.resolve() != scratch:
        raise ValueError("scratch contains a symlink")
    scratch.mkdir(parents=True, exist_ok=True)
    denied = [Path(p).resolve(strict=True) for p in hidden_paths]
    denied += [task_root / "release", task_root / "operator", task_root / "state/evaluation"]
    denied += [Path.home() / name for name in (".codex", ".aws", ".ssh")]
    if os.environ.get("CODEX_HOME"):
        denied.append(Path(os.environ["CODEX_HOME"]).resolve())
    if release_root:
        denied.append(Path(release_root).resolve(strict=True))
    def overlaps(path, other):
        return path.is_relative_to(other) or other.is_relative_to(path)
    if any(overlaps(output, path) for path in denied):
        raise ValueError("candidate output overlaps a protected domain")
    grants = [Path(p).resolve(strict=True) for p in read_grants]
    for path in grants:
        if any(overlaps(path, other) for other in [*denied, output, Path("/dev"), Path("/proc"), Path("/sys")]):
            raise ValueError("candidate read grant overlaps a protected domain")
    args = [
        "/bin/bwrap", "--unshare-all", "--cap-drop", "ALL", "--new-session",
        "--die-with-parent", "--proc", "/proc", "--dev", "/dev",
        "--tmpfs", "/dev/shm", "--bind", str(scratch), "/tmp",
        "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "HOME", "/tmp",
        "--setenv", "TMPDIR", "/tmp", "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
        "--setenv", "CUDA_VISIBLE_DEVICES", "", "--setenv", "RSI_ISOLATED_ROLE", "candidate",
        "--setenv", "RSI_ISOLATION_TASK_ROOT", str(task_root),
    ]
    # System executables and libraries are dependencies, never a host root mount.
    for name in ("/usr", "/bin", "/lib", "/lib64", "/etc/ld.so.cache"):
        if Path(name).exists():
            dependency = Path(name).resolve()
            if any(overlaps(dependency, path) for path in denied):
                raise ValueError("candidate system dependency overlaps a protected domain")
            args += ["--ro-bind", name, name]
    for path in grants:
        args += ["--ro-bind", str(path), str(path)]
    args += ["--bind", str(output), str(output), "--chdir", str(output)]
    return ["/usr/bin/env", "-i", *args, "--", *list(argv)]


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
