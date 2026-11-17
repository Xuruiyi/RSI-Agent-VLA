"""Current installed RLinf/rpent-openpi LIBERO SFT bridge. No model at import."""
import importlib.metadata
import json
import math
import re
import struct
from pathlib import Path
from typing import Any

INSTALL_COMMITS = {
    "rlinf": "2348ece3df113874671478be1360ae3115b346e3",
    "rpent-openpi": "a560f4dd8205b8423ecd4c8a0fabb5f54140b8a0",
}
ACTION_CONTRACT = {
    "backend": "libero", "controller": "OSC_POSE", "action_dim": 7,
    "arm": "normalized_controller_delta_xyz_axis_angle",
    "gripper": "minus1_open_plus1_close", "extra_delta_transform": False,
    "state": "eef_xyz_axis_angle_gripper_qpos", "state_dim": 8,
    "images": ["image", "wrist_image"], "model_action_dim": 32,
    "image_orientation": "native_cameras_rotated_180_degrees",
    "action_horizon": 10, "execution_chunk": 5,
}


def verify_installed_sources():
    result = {}
    for name, commit in INSTALL_COMMITS.items():
        dist = importlib.metadata.distribution(name)
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        actual = direct.get("vcs_info", {}).get("commit_id")
        if actual != commit:
            raise ValueError(f"{name} installed source differs: {actual}")
        result[name] = {"version": dist.version, "commit": actual}
    return result


def validate_dataset_manifest(manifest):
    """Check declared supervision provenance before any native loader call.

    This checks a supplied provenance record, not demonstration quality.
    states/transcript, success booleans and arbitrary 7-D data cannot qualify.
    """
    if manifest.get("source_kind") not in {"public_demonstration", "authorized_demonstration"}:
        raise ValueError("a traceable demonstration source is required")
    if not manifest.get("provenance_ref") or not manifest.get("revision"):
        raise ValueError("dataset provenance_ref and revision are required")
    if manifest.get("action_contract") != ACTION_CONTRACT:
        raise ValueError("dataset LIBERO action/observation contract mismatch")
    root = Path(manifest["dataset_root"]).expanduser().resolve()
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    features = info["features"]
    for key, shape in (("state", [8]), ("actions", [7])):
        if features.get(key, {}).get("shape") != shape:
            raise ValueError(f"dataset feature mismatch: {key}")
    for key in ("image", "wrist_image"):
        if features.get(key, {}).get("dtype") not in {"image", "video"}:
            raise ValueError(f"missing native dataset image: {key}")
    if (type(info.get("fps")) not in (int, float)
            or not math.isfinite(info["fps"]) or info["fps"] <= 0):
        raise ValueError("dataset fps required for horizon timestamps")
    episodes = manifest.get("episodes", [])
    if not episodes or len({e["episode_index"] for e in episodes}) != len(episodes):
        raise ValueError("unique episode/task/source mapping required")
    for episode in episodes:
        if not all(episode.get(k) is not None for k in ("episode_index", "task", "source_ref")):
            raise ValueError("incomplete episode provenance")
    actual = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line]
    if {e["episode_index"] for e in episodes} != {e["episode_index"] for e in actual}:
        raise ValueError("manifest must account for all episodes consumed by native loader")
    if info.get("total_episodes") != len(actual) or not actual:
        raise ValueError("episode metadata count mismatch")
    mapping = {e["episode_index"]: e for e in episodes}
    for episode in actual:
        if episode.get("tasks") != [mapping[episode["episode_index"]]["task"]]:
            raise ValueError("manifest task differs from native episode task metadata")
    if not (root / "meta/tasks.jsonl").is_file() or not any((root / "data").rglob("*.parquet")):
        raise ValueError("local task metadata and parquet data required")
    return {"dataset_root": str(root), "episodes": len(actual),
            "fps": info["fps"], "provenance_checked": "declarations_and_local_metadata"}


def build_recipe(*, dataset_root, model_path, norm_stats_path, output_root, run_id, budget):
    """A one-device microbatch-1 measurement recipe, not a full-scale decision.

    Model/data geometry follows pi05_libero. This partial SFT freezes the
    native VLM, uses native FP32MasterAdamW and a three-step pilot schedule.
    It does not reproduce the original fullshot training recipe.
    """
    if (budget.reserved_devices != 1 or budget.max_steps > 3 or budget.wall_seconds > 900
            or budget.gpu_device_seconds > 900 or budget.artifact_bytes > 64 * 1024**3):
        raise ValueError("first resource pilot is one reserved device, at most three steps")
    return _native_recipe(dataset_root=dataset_root, model_path=model_path,
        norm_stats_path=norm_stats_path, output_root=output_root, run_id=run_id, budget=budget)


def build_learning_recipe(*, dataset_root, model_path, norm_stats_path, output_root,
                          run_id, budget, learning_plan):
    """Build a registered finite action-expert learning job using native SFT.

    The pilot contract stays separate. This recipe saves the final checkpoint
    once and does not select checkpoints using capability evaluation feedback.
    """
    if (budget.reserved_devices != 1 or budget.max_steps > 10000
            or budget.wall_seconds > 28800 or budget.gpu_device_seconds > 28800
            or budget.artifact_bytes > 48 * 1024**3):
        raise ValueError("finite learning requires one device, <=10000 steps/8h/48GiB")
    plan = dict(learning_plan)
    if plan.get("schema") != "rsi-pi05-learning-plan-v1":
        raise ValueError("registered learning plan required")
    for key in ("candidate_id", "protocol_id", "data_admission_ref", "design_source"):
        if not isinstance(plan.get(key), str) or not plan[key]:
            raise ValueError("learning plan missing: " + key)
    if plan.get("stop_rule") != "fixed_optimizer_steps" or plan.get("checkpoint_selection") != "final_step":
        raise ValueError("learning requires fixed steps and final-checkpoint selection")
    if plan.get("max_steps") != budget.max_steps:
        raise ValueError("learning plan and budget step counts differ")
    global_batch = plan.get("global_batch_size")
    warmup = plan.get("warmup_steps")
    seed = plan.get("seed")
    lr = plan.get("learning_rate")
    scheduler = plan.get("lr_scheduler")
    if scheduler not in {"constant", "cosine", "openpi_cosine"}:
        raise ValueError("registered native learning-rate scheduler required")
    if type(global_batch) is not int or global_batch < 1 or plan.get("micro_batch_size") != 1:
        raise ValueError("supported native learning uses microbatch 1 and positive global batch")
    if type(warmup) is not int or not 0 <= warmup < budget.max_steps or type(seed) is not int:
        raise ValueError("learning warmup/seed invalid")
    if type(lr) not in (int, float) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("positive finite learning rate required")
    recipe = _native_recipe(dataset_root=dataset_root, model_path=model_path,
        norm_stats_path=norm_stats_path, output_root=output_root, run_id=run_id, budget=budget)
    recipe["rsi_training"].update(mode="learning", learning_plan=plan)
    recipe["runner"]["logger"]["project_name"] = "rsi-pi05-learning"
    recipe["actor"].update(global_batch_size=global_batch, seed=seed)
    recipe["actor"]["optim"].update(lr=lr, lr_warmup_steps=warmup, lr_scheduler=scheduler)
    return recipe


def _native_recipe(*, dataset_root, model_path, norm_stats_path, output_root, run_id, budget):
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a simple name")
    model = {
        "model_type": "openpi", "model_path": str(Path(model_path).resolve()),
        "precision": None, "pi05": True, "is_lora": False, "lora_rank": 32,
        "num_action_chunks": 5, "action_dim": 7, "use_proprio": True,
        "num_steps": 5, "add_value_head": False,
        "openpi": {"task": "sft", "config_name": "pi05_libero",
                   "model_action_dim": 32, "action_horizon": 10, "action_chunk": 5,
                   "action_env_dim": 7, "num_images_in_input": 2, "num_steps": 5,
                   "paligemma_variant": "gemma_2b", "action_expert_variant": "gemma_300m",
                   "max_token_len": 200, "discrete_state_input": False, "torch_compile": False},
        "openpi_data": {"norm_stats_path": str(Path(norm_stats_path).resolve())},
    }
    fsdp = {"strategy": "fsdp", "sharding_strategy": "no_shard", "use_orig_params": True,
            "gradient_checkpointing": True, "cpu_offload": False,
            "enable_gradient_accumulation": True, "forward_prefetch": False,
            "limit_all_gathers": False, "backward_prefetch": None,
            "save_full_model_weights": True,
            "mixed_precision": {"param_dtype": None, "reduce_dtype": None, "buffer_dtype": None},
            "amp_autocast": {"enabled": False, "precision": "bf16"},
            "grad_scaler": {"enabled": False}}
    return {"rsi_training": {"mode": "resource_pilot", "scope": "action_expert_only", "frozen": ["vision", "paligemma_expert_0", "token_embedder"],
                             "optimizer": "native_FP32MasterAdamW", "full_tuning": False},
            "cluster": {"num_nodes": 1, "component_placement": {"actor,env,rollout": "0-0"}},
            "runner": {"task_type": "sft", "max_epochs": -1, "max_steps": budget.max_steps,
                       "val_check_interval": -1, "save_interval": budget.max_steps,
                       "logger": {"log_path": str(Path(output_root).resolve()),
                                  "experiment_name": run_id, "project_name": "rsi-pi05-pilot",
                                  "logger_backends": ["tensorboard"]}},
            "data": {"train_data_paths": str(Path(dataset_root).resolve()), "num_workers": 0},
            "actor": {"group_name": "ActorGroup", "training_backend": "fsdp",
                      "micro_batch_size": 1, "global_batch_size": 1, "seed": 0,
                      "model": model, "fsdp_config": fsdp,
                      "optim": {"lr": 2.5e-5, "adam_beta1": 0.9, "adam_beta2": 0.95,
                                "adam_eps": 1e-8, "weight_decay": 1e-10, "clip_grad": 1.0,
                                "warmup_style": "cosine", "lr_warmup_steps": 0,
                                "total_training_steps": budget.max_steps,
                                "use_fp32_master_params": True}}}


def inspect_native_config(recipe):
    """Read the actual installed config and processor; never instantiate a model."""
    import os
    tokenizer = Path(os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser() / "big_vision/paligemma_tokenizer.model"
    if not tokenizer.is_file():
        raise FileNotFoundError("local PaliGemma tokenizer is required; implicit downloads are disabled")
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    cfg = recipe["actor"]["model"]
    native = get_openpi_config("pi05_libero", model_path=cfg["model_path"],
                              batch_size=1, repo_id=recipe["data"]["train_data_paths"],
                              data_kwargs=cfg["openpi_data"])
    expected = {"pi05": True, "action_dim": 32, "action_horizon": 10,
                "discrete_state_input": False, "max_token_len": 200}
    if any(getattr(native.model, k) != v for k, v in expected.items()):
        raise ValueError("installed native pi05_libero model geometry changed")
    data = native.data.create(native.assets_dirs, native.model)
    if native.data.extra_delta_transform or data.norm_stats is None:
        raise ValueError("pi05 delta/normalization mismatch")
    return {"model": expected, "extra_delta_transform": False,
            "use_quantile_norm": data.use_quantile_norm,
            "repack": [type(t).__name__ for t in data.repack_transforms.inputs],
            "input_transforms": [type(t).__name__ for t in data.data_transforms.inputs],
            "model_transforms": [type(t).__name__ for t in data.model_transforms.inputs]}


def read_native_batch(recipe, manifest):
    """CPU-only local batch using the same factory consumed by FSDPVlaSftWorker."""
    import os
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("batch check requires HF_HUB_OFFLINE=1 and CUDA_VISIBLE_DEVICES=''")
    validation = validate_dataset_manifest(manifest)
    if Path(recipe["data"]["train_data_paths"]).resolve() != Path(validation["dataset_root"]):
        raise ValueError("native loader input differs from the reviewed dataset manifest")
    from omegaconf import OmegaConf
    from rlinf.data.datasets.openpi import build_openpi_sft_dataloader
    loader, _ = build_openpi_sft_dataloader(OmegaConf.create(recipe), 1, 0,
                                           recipe["data"]["train_data_paths"])
    obs, actions = next(iter(loader))
    import numpy as np
    action_array = np.asarray(actions)
    if action_array.shape != (1, 10, 32) or not np.isfinite(action_array).all():
        raise ValueError("native batch action shape/finite check failed")
    if tuple(obs.state.shape) != (1, 32) or not np.isfinite(np.asarray(obs.state)).all():
        raise ValueError("native batch state shape/finite check failed")
    for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
        if tuple(obs.images[key].shape) != (1, 3, 224, 224):
            raise ValueError(f"native batch image shape mismatch: {key}")
    metadata = json.loads((Path(validation["dataset_root"]) / "meta/info.json").read_text())
    return {"dataset": dataset_identity(manifest), "episodes": validation["episodes"],
            "source_frames": metadata.get("total_frames"),
            "action_shape": list(action_array.shape), "state_shape": list(obs.state.shape),
            "image_shape": [1, 3, 224, 224], "image_layout": "native_pytorch_NCHW",
            "evidence_level": "local_cpu_native_dataloader_batch"}


def inspect_checkpoint(model_path, norm_stats_path):
    """Filesystem/layout check only; actual loading is a separate authorized job."""
    root = Path(model_path)
    candidates = [root / rel for rel in ("actor/model_state_dict/full_weights.pt",
                  "model_state_dict/full_weights.pt", "full_weights.pt")]
    full = next((p for p in candidates if p.is_file() and p.stat().st_size), None)
    safe = root / "model.safetensors"
    selected = full or (safe if safe.is_file() and safe.stat().st_size else None)
    if selected is None:
        raise FileNotFoundError("native-loader model weights missing")
    norm = Path(norm_stats_path)
    if norm.is_dir():
        norm = norm / "norm_stats.json"
    stats = json.loads(norm.read_text())["norm_stats"]
    for key in ("state", "actions"):
        for field in ("mean", "std", "q01", "q99"):
            if (len(stats[key][field]) != 32
                    or any(type(value) not in (int, float) or not math.isfinite(value)
                           for value in stats[key][field])):
                raise ValueError("pi05 normalization dimension mismatch")
    return {"selected_weights": str(selected), "norm_stats": str(norm),
            "loader": "rlinf.models.embodiment.openpi.get_model(task=eval)",
            "weights_bytes": selected.stat().st_size, "actual_model_load": "not_run",
            "evidence_level": "filesystem_layout_only"}


def deployment_handoff(checkpoint_root, norm_stats_path, *, python_executable):
    """Resolve the trainer's actual output for the existing RPent loader."""
    report = inspect_checkpoint(checkpoint_root, norm_stats_path)
    if not report["selected_weights"].endswith("full_weights.pt"):
        raise ValueError("training handoff requires newly saved native full_weights.pt")
    report["load_command"] = [str(python_executable), "-m",
        "rpent.robots.components.pi05_vla_server", "--embodiment", "libero",
        "--model-path", str(Path(checkpoint_root).resolve()), "--norm-stats-path",
        str(Path(norm_stats_path).resolve()), "--transport", "http", "--port", "0"]
    report["task_evaluation"] = "not_run"
    report["model_descriptor"] = {
        "kind": "pi05_libero_native_full_weights",
        "model_path": str(Path(checkpoint_root).resolve()),
        "norm_stats_path": str(Path(norm_stats_path).resolve()),
        "config_name": "pi05_libero", "action_contract": ACTION_CONTRACT,
        "actual_load_verified": False,
    }
    return report


def training_bundle_proposal(base_bundle, *, bundle_id, theta_ref, processor_ref, diff_ref):
    """Create an unaccepted theta/processor-only bundle for the existing Gate."""
    from dataclasses import replace
    if bundle_id == base_bundle.bundle_id or theta_ref == base_bundle.theta_ref or not diff_ref:
        raise ValueError("training proposal needs new bundle/theta and actual diff reference")
    return replace(base_bundle, bundle_id=bundle_id, theta_ref=theta_ref,
                   processor_ref=processor_ref, parent_bundle_id=base_bundle.bundle_id,
                   diff_ref=diff_ref)


def dataset_acquisition_proposal():
    """A bounded future request; no network call or inferred dataset revision."""
    return {
        "repo_id": "physical-intelligence/libero", "repo_type": "dataset",
        "source": "installed rlinf/models/embodiment/openpi/dataconfig/__init__.py:pi05_libero",
        "revision": None, "revision_status": "not_available_in_local_evidence",
        "authorized": False,
        "metadata_stage": {"resolve_revision_first": True, "max_bytes": 8 * 1024**2,
            "files": ["meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl"],
            "action": "resolve immutable dataset revision and inspect episode/task/media sizes"},
        "data_stage": {"max_episodes": 2, "max_download_bytes": 512 * 1024**2,
            "task_selection": "first two demonstration episodes matching salad-dressing-in-basket task metadata",
            "episode_indices": None, "indices_status": "select from pinned metadata, never guess",
            "files": "only selected episode parquet and the two camera assets referenced by pinned metadata",
            "subset": "materialize a self-contained LeRobot v2.1 dataset, preserving actions/images/prompts; reindex metadata and retain original episode IDs in provenance",
            "stop": "missing task, incompatible schema/action provenance, over byte cap, or unable to establish evaluation-data isolation",
            "excluded": "full dataset, model files, dependencies, PRO evaluation trajectories"},
        "role": "interface/resource pilot; two public demos do not establish capability improvement",
    }


def checkpoint_tensor_inventory(path):
    """Read the safetensors JSON header, without loading tensor payloads."""
    with Path(path).open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if size > 32 * 1024 * 1024:
            raise ValueError("unexpected safetensors header size")
        header = json.loads(stream.read(size))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    return {"tensor_count": len(tensors),
            "tensor_elements": sum(math.prod(t["shape"]) for t in tensors.values()),
            "stored_payload_bytes": sum(t["data_offsets"][1] - t["data_offsets"][0] for t in tensors.values()),
            "evidence_level": "safetensors_header_only",
            "training_peak_bytes": None}


def frozen_vlm_key(key):
    return bool(key.startswith(("img.", "llm.embedder.", "llm.final_norms.0.")) or
                re.match(r"^llm\.layers\.\d+\.(?:pre_attention_norms|pre_ffw_norms|mlps|attn\.[qkvo]_proj)\.0\.", key))


def static_training_memory(path):
    """Native key conversion on meta tensors, without model or weight allocation."""
    import torch
    from rlinf.utils.ckpt_convertor.openpi.openpi_pytorch_to_openpi import (
        old_to_new_state_dict,
    )
    with Path(path).open("rb") as stream:
        header = json.loads(stream.read(struct.unpack("<Q", stream.read(8))[0]))
    raw = {k: torch.empty(v["shape"], device="meta") for k, v in header.items() if k != "__metadata__"}
    converted = old_to_new_state_dict(raw) if any(k.startswith("paligemma_with_expert.") for k in raw) else raw
    fp32_selectors = ("img.stem.weight", "img.stem.bias", "img.pos_embedding",
                      "pre_attention_norms", "pre_ffw_norms", "final_norms")
    def element_bytes(key):
        return 4 if not key.startswith(("llm.", "img.")) or any(t in key for t in fp32_selectors) else 2
    model_bytes = sum(t.numel() * element_bytes(k) for k, t in converted.items())
    train = {k: t for k, t in converted.items() if not frozen_vlm_key(k)}
    grad_bytes = sum(t.numel() * element_bytes(k) for k, t in train.items())
    moments = sum(t.numel() * 8 for t in train.values())
    masters = sum(t.numel() * 4 for k, t in train.items() if element_bytes(k) == 2)
    return {"evidence_level": "native_key_conversion_meta_tensors",
        "converted_tensors": len(converted), "converted_elements": sum(t.numel() for t in converted.values()),
        "partial_sft_trainable_elements": sum(t.numel() for t in train.values()),
        "selective_model_parameter_bytes": model_bytes,
        "full_sft_native_adam_parameter_gradient_moments_lower_bound_bytes": 4 * model_bytes,
        "partial_sft_fp32_master_parameter_gradient_optimizer_lower_bound_bytes": model_bytes + grad_bytes + moments + masters,
        "activation_workspace_allocator_bytes": None,
        "fit_on_24gib": "full_sft_excluded_by_state_lower_bound; partial_sft_peak_requires_pilot"}


def _safe_relative(path):
    item = Path(path)
    if item.is_absolute() or ".." in item.parts:
        raise ValueError("dataset metadata contains an unsafe relative path")
    return item


def _fetch_bounded(url, *, remaining, expected_size=None, opener=None, on_bytes=None):
    """Cap downloaded response payload before reading; no automatic retry."""
    from urllib.request import Request, urlopen
    if remaining <= 0 or expected_size is not None and expected_size > remaining:
        raise ValueError("download would exceed the authorized byte limit")
    response = (opener or urlopen)(Request(url), timeout=30)
    with response:
        declared = response.headers.get("Content-Length")
        declared = int(declared) if declared is not None else None
        if declared is not None and declared > remaining:
            raise ValueError("Content-Length exceeds remaining download budget")
        chunks, total = [], 0
        while total < remaining:
            block = response.read(min(64 * 1024, remaining - total))
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if on_bytes is not None:
                on_bytes(url, len(block))
            if declared is not None and total == declared:
                break
        if total == remaining and declared != total:
            raise ValueError("response completeness unknown at the download cap")
        if declared is not None and total != declared:
            raise ValueError("incomplete HTTP response")
        if expected_size is not None and total != expected_size:
            raise ValueError("downloaded size differs from pinned file metadata")
    return b"".join(chunks)


def _download_ledger(root):
    def record(url, count):
        with (root / "download-ledger.jsonl").open("a") as stream:
            stream.write(json.dumps({"url": url, "response_payload_bytes": count}) + "\n")
    return record


def acquire_dataset_metadata(output, permit, *, opener=None):
    """Resolve/pin public source and fetch only metadata, within 8 MiB total."""
    if permit.get("kind") != "pi05-dataset-metadata" or permit.get("authorized") is not True:
        raise PermissionError("separate dataset-metadata authorization required")
    cap = permit.get("max_bytes")
    if type(cap) is not int or not 0 < cap <= 8 * 1024**2:
        raise ValueError("metadata payload cap must be at most 8 MiB")
    root = Path(output).resolve()
    if root.exists():
        raise ValueError("metadata output must be new")
    root.mkdir(parents=True)
    observe = _download_ledger(root)
    repo = "physical-intelligence/libero"
    api = _fetch_bounded(f"https://huggingface.co/api/datasets/{repo}/revision/main?blobs=true",
                         remaining=cap, opener=opener, on_bytes=observe)
    payload = json.loads(api)
    sha = payload["sha"]
    if not re.fullmatch("[0-9a-f]{40}", sha):
        raise ValueError("source API did not resolve an immutable dataset commit")
    sizes = {item["rfilename"]: (item.get("size") if item.get("size") is not None else (item.get("lfs") or {}).get("size"))
             for item in payload["siblings"]}
    fetched, total = {}, len(api)
    names = ["meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl"]
    for name in names:
        if type(sizes.get(name)) is not int:
            raise ValueError(f"pinned metadata file size unavailable: {name}")
        blob = _fetch_bounded(f"https://huggingface.co/datasets/{repo}/resolve/{sha}/{name}",
                              remaining=cap - total, expected_size=sizes[name], opener=opener, on_bytes=observe)
        total += len(blob)
        fetched[name] = blob
    info = json.loads(fetched["meta/info.json"])
    version = info.get("codebase_version")
    if version not in {"v2.0", "v2.1"}:
        raise ValueError(f"installed LeRobot 0.3.3 subset bridge requires v2.0/v2.1, source is {version}")
    stats_name = "meta/stats.json" if version == "v2.0" else "meta/episodes_stats.jsonl"
    if type(sizes.get(stats_name)) is not int:
        raise ValueError("source normalization/episode metadata size unavailable")
    blob = _fetch_bounded(f"https://huggingface.co/datasets/{repo}/resolve/{sha}/{stats_name}",
                         remaining=cap - total, expected_size=sizes[stats_name], opener=opener, on_bytes=observe)
    total += len(blob)
    fetched[stats_name] = blob
    plan = select_demonstration_files(info, fetched["meta/tasks.jsonl"].decode(),
                                       fetched["meta/episodes.jsonl"].decode(), sizes)
    source = {"repo_id": repo, "revision": sha, "downloaded_metadata_bytes": total,
              "metadata_limit_bytes": cap, "file_sizes": sizes, "selection": plan}
    for name, content in fetched.items():
        path = root / _safe_relative(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (root / "source.json").write_text(json.dumps(source, indent=2) + "\n")
    return source


def select_demonstration_files(info, tasks_text, episodes_text, sizes):
    """Select exactly the first two matching public demonstrations, then size them."""
    features = info["features"]
    if features.get("state", {}).get("shape") != [8] or features.get("actions", {}).get("shape") != [7]:
        raise ValueError("public source does not match native LIBERO state/actions")
    def normalize(text):
        return " ".join(re.sub("[^a-z0-9]+", " ", text.lower()).split())
    wanted = normalize("pick_up_the_salad_dressing_and_place_it_in_the_basket")
    tasks = [json.loads(line) for line in tasks_text.splitlines() if line]
    matches = [row for row in tasks if normalize(row["task"]) == wanted]
    if len(matches) != 1:
        raise ValueError("public source does not uniquely identify the salad-dressing-in-basket task")
    task = matches[0]
    episodes = sorted((json.loads(line) for line in episodes_text.splitlines() if line), key=lambda row: row["episode_index"])
    selected = [row for row in episodes if row.get("tasks") == [task["task"]]][:2]
    if len(selected) != 2:
        raise ValueError("two matching public demonstrations are unavailable")
    files = []
    for row in selected:
        index = row["episode_index"]
        formats = {"episode_index": index, "episode_chunk": index // info["chunks_size"]}
        files.append(info["data_path"].format(**formats))
        for camera in ("image", "wrist_image"):
            dtype = features.get(camera, {}).get("dtype")
            if dtype == "video":
                files.append(info["video_path"].format(video_key=camera, **formats))
            elif dtype != "image":
                raise ValueError("source needs both native image cameras")
    files = list(dict.fromkeys(files))
    for name in files:
        _safe_relative(name)
        if type(sizes.get(name)) is not int or sizes[name] < 0:
            raise ValueError(f"pinned file byte size unavailable: {name}")
    count = sum(sizes[name] for name in files)
    if count > 512 * 1024**2:
        raise ValueError("selected two demonstrations exceed the 512 MiB download cap")
    return {"task": task["task"], "task_index": task["task_index"], "episodes": selected,
            "files": files, "download_bytes": count}


def acquire_demonstration_files(metadata_root, output, permit, *, opener=None):
    """Fetch only the pinned selected files; no snapshot_download or full dataset."""
    root = Path(metadata_root)
    source = json.loads((root / "source.json").read_text())
    if (permit.get("kind") != "pi05-dataset-demonstrations" or permit.get("authorized") is not True
            or permit.get("revision") != source["revision"]):
        raise PermissionError("separate pinned demonstration-data authorization required")
    if not permit.get("evaluation_isolation_ref"):
        raise ValueError("independent evaluation-data isolation evidence required")
    cap = permit.get("max_bytes")
    if type(cap) is not int or not 0 < cap <= 512 * 1024**2:
        raise ValueError("demonstration payload cap must be at most 512 MiB")
    info = json.loads((root / "meta/info.json").read_text())
    selection = select_demonstration_files(info, (root / "meta/tasks.jsonl").read_text(),
        (root / "meta/episodes.jsonl").read_text(), source["file_sizes"])
    if selection != source["selection"] or selection["download_bytes"] > cap:
        raise ValueError("selection changed or exceeds authorized download bytes")
    target = Path(output).resolve()
    if target.exists():
        raise ValueError("demonstration download output must be new")
    target.mkdir(parents=True)
    observe = _download_ledger(target)
    total = 0
    for name in selection["files"]:
        data = _fetch_bounded(f"https://huggingface.co/datasets/{source['repo_id']}/resolve/{source['revision']}/{name}",
            remaining=cap - total, expected_size=source["file_sizes"][name], opener=opener, on_bytes=observe)
        total += len(data)
        path = target / _safe_relative(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    result = {**source, "downloaded_demonstration_bytes": total,
              "evaluation_isolation_ref": permit["evaluation_isolation_ref"]}
    (target / "source.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def materialize_demonstration_subset(metadata_root, source_root, output, *, max_frames=4000,
                                      max_output_bytes=2 * 1024**3):
    """Re-export only the two selected demos through installed LeRobot 0.3.3.

    The native writer owns reindexing and statistics. Video/image observations
    are decoded to lossless embedded images; action/state samples are unchanged.
    """
    import os
    import shutil

    import numpy as np
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(source_root).resolve() / "datasets-cache"))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from ..training import artifact_size
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("subset conversion is local CPU only")
    metadata, source_root, output = Path(metadata_root), Path(source_root), Path(output)
    source = json.loads((source_root / "source.json").read_text())
    pinned = json.loads((metadata / "source.json").read_text())
    if source["revision"] != pinned["revision"] or source["selection"] != pinned["selection"]:
        raise ValueError("downloaded source and pinned metadata differ")
    if not source.get("evaluation_isolation_ref"):
        raise ValueError("source demonstration evaluation-isolation record is missing")
    info = json.loads((metadata / "meta/info.json").read_text())
    selected = source["selection"]["episodes"]
    if len(selected) != 2 or sum(row["length"] for row in selected) > max_frames:
        raise ValueError("two-demo conversion frame budget exceeded")
    if output.exists():
        raise ValueError("subset output must be new")
    for name in source["selection"]["files"]:
        item = source_root / _safe_relative(name)
        if not item.is_file() or item.stat().st_size != source["file_sizes"][name]:
            raise ValueError("downloaded demonstration file missing or size changed")
    shutil.copytree(metadata / "meta", source_root / "meta", dirs_exist_ok=False)
    reader = LeRobotDataset(source["repo_id"], root=source_root,
                            episodes=[row["episode_index"] for row in selected],
                            revision=source["revision"], download_videos=False, video_backend="pyav")
    default_keys = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    features = {k: dict(v) for k, v in info["features"].items() if k not in default_keys}
    for feature in features.values():
        feature["shape"] = tuple(feature["shape"])
    for key in ("image", "wrist_image"):
        features[key]["dtype"] = "image"
        features[key].pop("video_info", None)
    writer = LeRobotDataset.create("rsi-local/libero-pilot", root=output, fps=info["fps"],
        robot_type=info.get("robot_type"), features=features, use_videos=False)
    mapping = []
    previous = None
    for index in range(len(reader)):
        sample = reader[index]
        original = int(sample["episode_index"])
        if previous is not None and original != previous:
            writer.save_episode()
        if original != previous:
            mapping.append({"episode_index": len(mapping), "task": source["selection"]["task"],
                "source_episode_index": original,
                "source_ref": f"hf://datasets/{source['repo_id']}@{source['revision']}/episode/{original}"})
        frame = {}
        for key in features:
            value = sample[key].numpy() if hasattr(sample[key], "numpy") else np.asarray(sample[key])
            if key in ("image", "wrist_image"):
                if value.shape[0] == 3:
                    value = value.transpose(1, 2, 0)
                if np.issubdtype(value.dtype, np.floating):
                    value = np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8)
            if key in ("state", "actions") and not np.isfinite(value).all():
                raise ValueError("nonfinite demonstrated state/action sample")
            frame[key] = value
        if sample["task"] != source["selection"]["task"]:
            raise ValueError("downloaded demonstration prompt differs from pinned task")
        writer.add_frame(frame, task=sample["task"], timestamp=float(sample["timestamp"]))
        if artifact_size(output) > max_output_bytes:
            raise ValueError("subset conversion artifact threshold exceeded")
        previous = original
    writer.save_episode()
    if artifact_size(output) > max_output_bytes:
        raise ValueError("serialized subset artifact threshold exceeded")
    if len(mapping) != 2 or len(reader) != sum(row["length"] for row in selected):
        raise ValueError("converted source episode/frame count mismatch")
    manifest = {"source_kind": "public_demonstration", "provenance_ref": str(source_root / "source.json"),
        "revision": source["revision"], "dataset_root": str(output.resolve()),
        "action_contract": ACTION_CONTRACT, "episodes": mapping,
        "evaluation_isolation_ref": source["evaluation_isolation_ref"],
        "conversion": "native LeRobot writer; source actions/states and prompts preserved; images decoded to uint8",
        "source_repo_id": source["repo_id"]}
    validate_dataset_manifest(manifest)
    (output / "demonstration-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def native_verify_load(recipe_path, checkpoint_root, probes_path, report_path):
    """Load through the actual RPent Pi05VLAFacade and decode one prediction."""
    _record_native_start("native_verify_load", recipe_path)
    import numpy as np
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from rpent.robots.components.pi05_vla_server import Pi05VLAFacade
    recipe = json.loads(Path(recipe_path).read_text())
    checkpoint_root = Path(checkpoint_root).resolve()
    norm = Path(recipe["actor"]["model"]["openpi_data"]["norm_stats_path"]).resolve()
    # A load-only context lets the actual service record its effective loader
    # config and model identity. It grants no Runtime or benchmark execution.
    import os

    from ..storage import atomic_json
    context_ref = Path(report_path).parent / "model-load-context.json"
    run_id = recipe["runner"]["logger"]["experiment_name"]
    plan = recipe["rsi_training"].get("learning_plan", {})
    atomic_json(context_ref, {"schema_version": "1.0", "run_id": run_id,
        "episode_id": "native-load-check", "bundle_id": plan.get("candidate_id", "unqualified-" + run_id),
        "protocol_id": plan.get("protocol_id", "resource-pilot"), "profile_ref": str(Path(recipe_path).resolve()),
        "theta_ref": str(checkpoint_root), "processor_ref": str(norm),
        "output_dir": str(Path(report_path).parent.resolve()), "information_scope": "model_load_only"})
    prior_context = os.environ.get("RSI_RUNTIME_CONTEXT")
    os.environ["RSI_RUNTIME_CONTEXT"] = str(context_ref)
    try:
        facade = Pi05VLAFacade(model_path=str(checkpoint_root), embodiment="libero",
                               norm_stats_path=str(norm))
    finally:
        if prior_context is None:
            os.environ.pop("RSI_RUNTIME_CONTEXT", None)
        else:
            os.environ["RSI_RUNTIME_CONTEXT"] = prior_context
    model = facade._model
    expected = torch.load(probes_path, map_location="cpu", weights_only=True)
    named = dict(model.named_parameters())
    equality = {key: torch.equal(named[key].detach().cpu(), tensor) for key, tensor in expected.items()}
    if not equality or not all(equality.values()):
        raise ValueError("actual runtime loader does not reproduce saved trained probes")
    sample = LeRobotDataset(recipe["data"]["train_data_paths"], video_backend="pyav")[0]
    def image(key):
        array = np.asarray(sample[key])
        if array.shape[0] == 3:
            array = array.transpose(1, 2, 0)
        return np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)[None]
    wire = {"main_images": image("image"), "wrist_images": image("wrist_image"),
            "extra_view_images": None, "states": np.asarray(sample["state"], dtype=np.float32)[None],
            "task_descriptions": [sample["task"]]}
    actions = facade.predict(wire, options={"mode": "eval"})
    layout = inspect_checkpoint(checkpoint_root, norm)
    loaded_weights = Path(layout["selected_weights"])
    if actions.shape != (1, 5, 7) or not np.isfinite(actions).all():
        raise ValueError("actual runtime decoded action shape/finite check failed")
    report = {"actual_model_load": "passed", "saved_probe_equal": equality,
              "prediction_shape": list(actions.shape), "prediction_finite": True,
              "loader_config": facade._rsi_model_evidence["effective_config"],
              "service_evidence_ref": str(facade._rsi_model_evidence_path),
              "service_class": type(facade).__module__ + "." + type(facade).__name__,
              "model_class": type(model).__module__ + "." + type(model).__name__,
              "model_path": str(checkpoint_root), "norm_stats_path": str(norm),
              "processor_snapshot": json.loads(norm.read_text()),
              "weights_identity": {"path": str(loaded_weights.resolve()),
                  "bytes": loaded_weights.stat().st_size,
                  "mtime_ns": loaded_weights.stat().st_mtime_ns},
              "decoded_action_contract": ACTION_CONTRACT,
              "prediction": actions.tolist(), "prediction_count": 1,
              "task_evaluation": "not_run", "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "evidence_level": "real_native_loader_and_one_prediction"}
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
    return report


def measured_worker_class():
    """Construct the worker class without model creation or Ray initialization."""
    import resource
    import time

    import torch
    from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker

    class MeasuredSftWorker(FSDPVlaSftWorker):
        def model_provider_func(self):
            model = super().model_provider_func()
            if self.cfg.rsi_training.scope != "action_expert_only":
                raise ValueError("only reviewed action-expert partial SFT is supported")
            model.freeze_vlm()
            parameters = dict(model.named_parameters())
            if any(p.requires_grad for k, p in parameters.items() if frozen_vlm_key(k)):
                raise ValueError("native VLM freeze failed")
            probe_keys = ("action_out_proj.weight", "llm.layers.0.attn.q_proj.1.weight")
            self._update_probes = {k: parameters[k].detach().cpu().clone() for k in probe_keys}
            self._frozen_probes = {k: parameters[k].detach().cpu().clone()
                                   for k in ("img.stem.bias", "llm.final_norms.0.scale")}
            directory = Path(self.cfg.runner.logger.log_path) / self.cfg.runner.logger.experiment_name
            directory.mkdir(parents=True, exist_ok=True)
            torch.save(self._update_probes, directory / "probes-before.pt")
            self._trainable_elements = sum(p.numel() for p in model.parameters() if p.requires_grad)
            census = {"scope": "action_expert_only", "parameters": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype),
                      "trainable": value.requires_grad} for key, value in parameters.items()},
                "trainable_elements": self._trainable_elements,
                "frozen_elements": sum(p.numel() for p in parameters.values() if not p.requires_grad)}
            (directory / "parameter-census.json").write_text(json.dumps(census, indent=2) + "\n")
            return model

        def get_train_model_output(self, batch):
            # Native run_training calls model.train(); keep the frozen image
            # trunk in the eval mode selected by freeze_vlm().
            self.model.img.eval()
            return super().get_train_model_output(batch)

        def _measure(self, phase, started, metrics=None):
            torch.cuda.synchronize()
            row = {"phase": phase, "rank": self._rank, "step": self.global_step,
                   "seconds": time.monotonic() - started,
                   "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                   "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                   "metrics": metrics or {}}
            directory = Path(self.cfg.runner.logger.log_path) / self.cfg.runner.logger.experiment_name
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / f"resources-rank{self._rank}.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")

        def init_worker(self):
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            super().init_worker()
            optimizer = type(self.optimizer).__name__
            if optimizer != "FP32MasterAdamW":
                raise ValueError("native optimizer differs from the declared FP32MasterAdamW")
            directory = Path(self.cfg.runner.logger.log_path) / self.cfg.runner.logger.experiment_name
            (directory / "optimizer-role.json").write_text(json.dumps({
                "class": optimizer, "parameter_groups": len(self.optimizer.param_groups),
                "gradient_accumulation": self.gradient_accumulation,
                "micro_batch_size": self.micro_batch_size, "global_batch_size": self.global_batch_size,
                "world_size": self._world_size}, indent=2) + "\n")
            self._measure("model_optimizer_init", started)

        def save_checkpoint(self, save_path, step=0):
            started = time.monotonic()
            super().save_checkpoint(save_path, step)
            self._measure("native_checkpoint_save", started)

        def run_training(self):
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            metrics = super().run_training()
            metrics = {k: float(v) for k, v in metrics.items()}
            if not all(math.isfinite(v) for v in metrics.values()) or metrics.get("grad_norm", 0) <= 0:
                raise ValueError("nonfinite training metrics or zero gradient")
            if (self.cfg.rsi_training.get("mode") == "learning"
                    and self.global_step + 1 != self.cfg.runner.max_steps):
                self._measure("native_optimizer_step", started, metrics)
                return metrics
            parameters = {k.replace("_fsdp_wrapped_module.", ""): p for k, p in self.model.named_parameters()}
            after = {k: parameters[k].detach().cpu().clone() for k in self._update_probes}
            changes = {k: {"changed_elements": int((after[k] != before).sum()),
                           "max_abs_difference": float((after[k].float() - before.float()).abs().max()),
                           "dtype": str(after[k].dtype)} for k, before in self._update_probes.items()}
            frozen_equal = {key: torch.equal(parameters[key].detach().cpu(), before)
                            for key, before in self._frozen_probes.items()}
            if not all(frozen_equal.values()):
                raise ValueError("sampled frozen VLM parameters changed during native SFT")
            directory = Path(self.cfg.runner.logger.log_path) / self.cfg.runner.logger.experiment_name
            torch.save(after, directory / "probes-after.pt")
            (directory / "weight-update-check.json").write_text(json.dumps({
                "scope": "action_expert_only", "trainable_elements": self._trainable_elements,
                "step": self.global_step + 1, "metrics": metrics, "probes": changes,
                "sampled_frozen_parameters_equal": frozen_equal,
                "nonzero_sampled_weight_update": any(v["changed_elements"] > 0 for v in changes.values())}, indent=2) + "\n")
            if self.global_step + 1 == self.cfg.runner.max_steps and not any(v["changed_elements"] > 0 for v in changes.values()):
                raise ValueError("optimizer completed but sampled model weights did not change")
            self._measure("native_optimizer_step", started, metrics)
            return metrics

    return MeasuredSftWorker


def native_train(recipe_path):
    """Native RLinf launcher with per-step measurement; only called by supervisor."""
    _record_native_start("native_train", recipe_path)
    import torch.multiprocessing as mp
    from omegaconf import OmegaConf
    from rlinf.config import validate_cfg
    from rlinf.runners.sft_runner import SFTRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement

    mp.set_start_method("spawn", force=True)
    import ray
    try:
        cfg = validate_cfg(OmegaConf.load(recipe_path))
        cluster = Cluster(cluster_cfg=cfg.cluster)
        placement = HybridComponentPlacement(cfg, cluster).get_strategy("actor")
        actor = measured_worker_class().create_group(cfg).launch(cluster, name=cfg.actor.group_name,
                                                          placement_strategy=placement)
        runner = SFTRunner(cfg=cfg, actor=actor)
        runner.init_workers()
        runner.run()
    finally:
        ray.shutdown()


def _record_native_start(entrypoint, recipe_path):
    """Publish an exact adapter-process start, including optional-import failures."""
    import os
    from datetime import datetime, timezone

    from ..training import JOB_MARKER
    root = Path(os.environ[JOB_MARKER]).resolve()
    recipe = json.loads(Path(recipe_path).read_text())
    report = {"schema": "rsi-native-training-start-v1", "entrypoint": entrypoint,
              "scope_id": os.environ["RSI_TRAINING_SCOPE_ID"],
              "operation_id": os.environ["RSI_TRAINING_OPERATION_ID"],
              "kind": os.environ["RSI_TRAINING_KIND"],
              "run_id": recipe["runner"]["logger"]["experiment_name"],
              "pid": os.getpid(), "owner_uid": os.geteuid(),
              "started_at": datetime.now(timezone.utc).isoformat(),
              "boundary": "native_adapter_process_entry_before_optional_imports",
              "weight_update_established": False}
    with (root / "native-start.json").open("x") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")


def prepare_cpu_demonstrations(*, metadata_root, output, permit, model_path, norm_stats_path):
    """Acquire two pinned public demos and consume one native CPU batch.

    The caller must run this whole operation under the existing process meter,
    including its output, datasets cache and scratch domains. No weights load.
    An interface batch does not admit a dataset for actual training.
    """
    import os
    import shutil

    from ..training import TrainingBudget
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError("CPU preparation identity already exists; reconcile its receipt")
    for key in ("license_evidence_ref", "action_semantics_ref", "evaluation_isolation_ref"):
        if not permit.get(key) or not Path(permit[key]).is_file():
            raise ValueError("CPU preparation requires local evidence: " + key)
    root.mkdir(parents=True)
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu",
                      HF_DATASETS_CACHE=str(root / "datasets-cache"))
    installed = verify_installed_sources()
    source = acquire_demonstration_files(metadata_root, root / "download", permit)
    manifest = materialize_demonstration_subset(metadata_root, root / "download", root / "subset",
        max_frames=permit["max_subset_frames"], max_output_bytes=permit["max_subset_bytes"])
    manifest.update(license_evidence_ref=permit["license_evidence_ref"],
                    action_semantics_ref=permit["action_semantics_ref"],
                    training_admission="not_admitted_interface_batch_only")
    (root / "subset/demonstration-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    norm = root / "processor/physical-intelligence/libero/norm_stats.json"
    norm.parent.mkdir(parents=True)
    shutil.copyfile(norm_stats_path, norm)
    recipe = build_recipe(dataset_root=root / "subset", model_path=model_path,
        norm_stats_path=norm, output_root=root, run_id="future-training-proposal",
        budget=TrainingBudget(900, 900, 1, 64 * 1024**3, 3))
    native = inspect_native_config(recipe)
    batch = read_native_batch(recipe, manifest)
    report = {"status": "cpu_native_batch_passed", "installed": installed,
        "source": source, "manifest": manifest, "native_config": native, "batch": batch,
        "real_native_batches": 1, "actual_model_load": "not_run", "training_steps": 0,
        "training_admission": "not_admitted_interface_batch_only",
        "capability_evaluation": "not_run", "production_eligible": False}
    (root / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    (root / "cpu-preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def dataset_identity(manifest):
    """Bind audits to every native-consumed episode and its original source."""
    return {"dataset_root": str(Path(manifest["dataset_root"]).resolve()),
            "revision": manifest["revision"], "source_kind": manifest["source_kind"],
            "episodes": sorted(manifest["episodes"], key=lambda row: row["episode_index"])}


def audit_demonstration_payload(manifest):
    """Inspect all local image-demo payloads on CPU; do not admit them for learning.

    This checks decoded RGB, temporal alignment, native numeric ranges and exact
    source preservation. A reviewer must still establish task relevance, camera
    orientation and evaluation isolation using independent evidence.
    """
    import io

    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image

    validation = validate_dataset_manifest(manifest)
    root = Path(validation["dataset_root"])
    info = json.loads((root / "meta/info.json").read_text())
    if any(info["features"][key]["dtype"] != "image" for key in ("image", "wrist_image")):
        raise ValueError("payload audit currently requires embedded RGB image datasets")
    provenance_path = Path(manifest["provenance_ref"])
    source = json.loads(provenance_path.read_text())
    if source["revision"] != manifest["revision"]:
        raise ValueError("source revision differs from demonstration manifest")
    source_root = provenance_path.parent
    source_info = json.loads((source_root / "meta/info.json").read_text())
    reports = []
    errors = []

    def rows_for(directory, metadata, index):
        name = metadata["data_path"].format(episode_chunk=index // metadata["chunks_size"],
                                            episode_index=index)
        return pq.read_table(directory / _safe_relative(name)).to_pylist()

    def decode(value, directory):
        if value.get("bytes") is not None:
            image = Image.open(io.BytesIO(value["bytes"]))
        else:
            image = Image.open(directory / _safe_relative(value["path"]))
        with image:
            if image.mode != "RGB":
                raise ValueError("demonstration camera is not RGB")
            return np.asarray(image).copy()

    for episode in manifest["episodes"]:
        index, original = episode["episode_index"], episode["source_episode_index"]
        rows = rows_for(root, info, index)
        originals = rows_for(source_root, source_info, original)
        checks = {"source_frame_count_equal": len(rows) == len(originals) and len(rows) > 0,
                  "state_action_finite_and_native_shape": True,
                  "normalized_actions_and_binary_gripper": True,
                  "contiguous_frame_and_episode_indices": True,
                  "timestamps_match_fps": True, "source_states_actions_equal": True,
                  "executed_transitions_only": True,
                  "decoded_rgb_shape_and_nonconstant": True, "source_decoded_pixels_equal": True}
        minima, maxima = np.full(7, np.inf), np.full(7, -np.inf)
        for frame, (row, original_row) in enumerate(zip(rows, originals)):
            state, action = np.asarray(row["state"]), np.asarray(row["actions"])
            checks["state_action_finite_and_native_shape"] &= (
                state.shape == (8,) and action.shape == (7,)
                and bool(np.isfinite(state).all()) and bool(np.isfinite(action).all()))
            checks["normalized_actions_and_binary_gripper"] &= (
                action.shape == (7,) and bool((np.abs(action) <= 1.00001).all()) and float(action[6]) in {-1.0, 1.0})
            checks["contiguous_frame_and_episode_indices"] &= (
                row["frame_index"] == frame and row["episode_index"] == index)
            checks["timestamps_match_fps"] &= abs(row["timestamp"] - frame / info["fps"]) < 1e-4
            mask = row.get("execution_mask", True)
            checks["executed_transitions_only"] &= bool(np.asarray(mask).all())
            checks["source_states_actions_equal"] &= (
                bool(np.array_equal(state, original_row["state"]))
                and bool(np.array_equal(action, original_row["actions"])))
            if action.shape == (7,):
                minima, maxima = np.minimum(minima, action), np.maximum(maxima, action)
            for camera in ("image", "wrist_image"):
                image, source_image = decode(row[camera], root), decode(original_row[camera], source_root)
                checks["decoded_rgb_shape_and_nonconstant"] &= (
                    list(image.shape) == info["features"][camera]["shape"] and float(image.std()) > 0)
                checks["source_decoded_pixels_equal"] &= bool(np.array_equal(image, source_image))
        checks = {key: bool(value) for key, value in checks.items()}
        errors.extend(f"episode {index}: {key}" for key, value in checks.items() if not value)
        reports.append({"episode_index": index, "source_episode_index": original,
                        "task": episode["task"], "frames_checked": len(rows),
                        "checks": checks, "action_min": minima.tolist(), "action_max": maxima.tolist()})
    from ..collection import dataset_file_snapshot
    return {"schema": "rsi-demonstration-payload-audit-v1", "dataset": dataset_identity(manifest),
            "payload_snapshot": dataset_file_snapshot(manifest),
            "episodes": reports, "frames_checked": sum(row["frames_checked"] for row in reports),
            "checks_passed": not errors, "errors": errors,
            "training_admission": "pending_independent_quality_and_isolation_review",
            "image_orientation": "source_pixels_preserved; independent_direction_review_required"}


def require_training_data_admission(manifest, *, admission_ref=None, scope_id=None,
                                    protocol_id=None, authority=None):
    """Consume a new reviewed receipt without relabeling the old interface record."""
    validate_dataset_manifest(manifest)
    if not admission_ref or not scope_id or not protocol_id or not authority:
        raise PermissionError("training requires scoped independent demonstration-data admission")
    receipt = json.loads(Path(admission_ref).read_text())
    if (receipt.get("schema") != "rsi-training-data-admission-v1"
            or receipt.get("decision") != "admitted" or receipt.get("scope_id") != scope_id
            or receipt.get("protocol_id") != protocol_id or receipt.get("reviewer_id") != authority):
        raise PermissionError("training data admission identity/version/authority mismatch")
    identity = dataset_identity(manifest)
    if receipt.get("payload_snapshot") is not None:
        from ..collection import dataset_file_snapshot
        if receipt["payload_snapshot"] != dataset_file_snapshot(manifest):
            raise PermissionError("admitted dataset payload changed after review")
    if receipt.get("dataset") != identity:
        raise PermissionError("training admission does not cover all consumed dataset episodes")
    if manifest.get("derivation_kind") == "index_clock_composition":
        from .libero_mixed_data import validate_mixed_manifest
        validate_mixed_manifest(manifest, scope_id=scope_id)
    original = json.loads(Path(receipt["original_manifest_ref"]).read_text())
    if dataset_identity(original) != identity:
        raise PermissionError("training admission original source manifest mismatch")
    audit = json.loads(Path(receipt["payload_audit_ref"]).read_text())
    if (audit.get("schema") != "rsi-demonstration-payload-audit-v1"
            or audit.get("dataset") != identity or audit.get("checks_passed") is not True
            or audit.get("frames_checked", 0) <= 0):
        raise PermissionError("actual payload quality audit missing or mismatched")
    expected_episodes = {row["episode_index"] for row in identity["episodes"]}
    if {row["episode_index"] for row in audit["episodes"]} != expected_episodes:
        raise PermissionError("payload audit does not cover all consumed episodes")
    metadata = {row["episode_index"]: row for row in (
        json.loads(line) for line in (Path(manifest["dataset_root"]) / "meta/episodes.jsonl").read_text().splitlines() if line)}
    if any(not row.get("checks") or not all(value is True for value in row["checks"].values())
           or row["frames_checked"] != metadata[row["episode_index"]]["length"]
           for row in audit["episodes"]):
        raise PermissionError("payload audit quality/frame coverage incomplete")
    tasks = {row["task"] for row in identity["episodes"]}
    if not tasks <= set(receipt.get("approved_training_tasks", [])):
        raise PermissionError("training task coverage was not reviewed")
    for name in ("quality_review", "evaluation_isolation_review"):
        review = json.loads(Path(receipt[name + "_ref"]).read_text())
        schema = ("rsi-demonstration-quality-review-v1" if name == "quality_review"
                  else "rsi-training-evaluation-isolation-v1")
        if (review.get("schema") != schema or review.get("scope_id") != scope_id or review.get("protocol_id") != protocol_id
                or review.get("dataset") != identity or review.get("reviewer_id") != authority
                or review.get("decision") != "accepted"):
            raise PermissionError("independent data review mismatch: " + name)
        if name == "quality_review":
            required = ("image_orientation_verified", "state_action_alignment_verified",
                        "task_relevance_verified", "license_verified")
        else:
            required = ("source_episode_mapping_checked", "development_formal_split_checked",
                        "sealed_labels_not_consumed")
        if any(review.get(key) is not True for key in required) or not review.get("evidence_refs"):
            raise PermissionError("independent actual data review incomplete: " + name)
        for reference in review["evidence_refs"]:
            if not Path(reference).is_file():
                raise PermissionError("review evidence unavailable: " + reference)
    return receipt


def prepare_admitted_training(
    *,
    manifest_ref: str | Path,
    admission_ref: str | Path,
    scope_id: str,
    protocol_id: str,
    reviewer_id: str,
    model_path: str | Path,
    norm_stats_path: str | Path,
    output_root: str | Path,
    run_id: str,
    budget: Any,
    learning_plan: dict[str, Any] | None = None,
    read_batch: bool = True,
) -> dict[str, Any]:
    """Materialize native trainer inputs only from the independently admitted set."""
    import shutil

    from ..storage import atomic_json

    manifest = json.loads(Path(manifest_ref).read_text())
    admission = require_training_data_admission(
        manifest,
        admission_ref=admission_ref,
        scope_id=scope_id,
        protocol_id=protocol_id,
        authority=reviewer_id,
    )
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(
            "training preparation already exists; consume its receipt"
        )
    installed = verify_installed_sources()
    checkpoint = inspect_checkpoint(model_path, norm_stats_path)
    root.mkdir(parents=True)
    import os

    os.environ.update(
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
        JAX_PLATFORMS="cpu",
        HF_DATASETS_CACHE=str(root / "datasets-cache"),
    )
    norm = root / "processor/physical-intelligence/libero/norm_stats.json"
    norm.parent.mkdir(parents=True)
    shutil.copyfile(checkpoint["norm_stats"], norm)
    builder = build_learning_recipe if learning_plan else build_recipe
    kwargs = {"learning_plan": learning_plan} if learning_plan else {}
    if learning_plan and learning_plan["data_admission_ref"] != str(admission_ref):
        raise PermissionError("learning plan binds a different admission")
    recipe = builder(
        dataset_root=manifest["dataset_root"],
        model_path=model_path,
        norm_stats_path=norm,
        output_root=root,
        run_id=run_id,
        budget=budget,
        **kwargs,
    )
    native = inspect_native_config(recipe)
    batch = read_native_batch(recipe, manifest) if read_batch else None
    report = {
        "schema": "rsi-admitted-training-preparation-v1",
        "scope_id": scope_id,
        "protocol_id": protocol_id,
        "run_id": run_id,
        "dataset": admission["dataset"],
        "admission_ref": str(Path(admission_ref).resolve()),
        "manifest_ref": str(Path(manifest_ref).resolve()),
        "recipe_ref": str(root / "recipe.json"),
        "dataset_root": recipe["data"]["train_data_paths"],
        "norm_stats_path": str(norm),
        "processor_snapshot": json.loads(norm.read_text()),
        "native_config": native,
        "installed": installed,
        "batch": batch,
        "training_steps": 0,
        "status": "prepared_pending_training_authorization",
    }
    for name, value in (
        ("recipe.json", recipe),
        ("dataset-manifest.json", manifest),
        ("preparation.json", report),
        (
            "training-permit-proposal.json",
            {
                "kind": "pi05-training-learning"
                if learning_plan
                else "pi05-training-pilot",
                "authorized": False,
                "cuda_device": None,
                "run_id": run_id,
                "dataset_root": recipe["data"]["train_data_paths"],
                "model_path": str(Path(model_path).resolve()),
                "budget": budget.__dict__,
            },
        ),
    ):
        atomic_json(root / name, value)
    return {**report, "artifact_ref": str(root / "preparation.json")}


def resolve_native_training_output(
    *,
    prepared_root: str | Path,
    artifact_root: str | Path,
    permit: dict[str, Any],
    supervision: dict[str, Any],
    python_executable: str,
) -> dict[str, Any]:
    """Resolve completed native saves; partial saves never become a theta export."""
    root, artifact = Path(prepared_root).resolve(), Path(artifact_root).resolve()
    recipe = json.loads((root / "recipe.json").read_text())
    if (
        supervision.get("returncode") != 0
        or supervision.get("stop_reason") is not None
        or supervision.get("cleanup_confirmed") is not True
        or supervision.get("resource_limits_satisfied") is not True
    ):
        raise ValueError(
            "native training process did not complete within its lifecycle limits"
        )
    step = recipe["runner"]["max_steps"]
    if step != permit["budget"]["max_steps"]:
        raise ValueError("trainer recipe and permit step counts differ")
    update_ref = artifact / "weight-update-check.json"
    update = json.loads(update_ref.read_text())
    if (
        update.get("step") != step
        or update.get("nonzero_sampled_weight_update") is not True
        or not update.get("sampled_frozen_parameters_equal")
        or not all(
            value is True
            for value in update["sampled_frozen_parameters_equal"].values()
        )
    ):
        raise ValueError(
            "native final-step weight update or frozen parameter evidence missing"
        )
    # Resolve an existing trainer directory, then verify its native loader layout.
    candidates = [
        path
        for path in (artifact / "checkpoints").glob("global_step_*")
        if path.is_dir() and path.name == f"global_step_{step}"
    ]
    if len(candidates) != 1:
        raise ValueError("completed native final checkpoint is unavailable")
    norm = recipe["actor"]["model"]["openpi_data"]["norm_stats_path"]
    preparation = root / "preparation.json"
    if preparation.exists():
        saved = json.loads(preparation.read_text())
        if (
            saved.get("processor_snapshot") is not None
            and json.loads(Path(norm).read_text()) != saved["processor_snapshot"]
        ):
            raise ValueError("prepared processor changed during training")
    if not candidates[0].resolve().is_relative_to(artifact):
        raise ValueError(
            "native checkpoint path escapes the supervised artifact domain"
        )
    handoff = deployment_handoff(
        candidates[0], norm, python_executable=python_executable
    )
    weights = Path(handoff["selected_weights"])
    handoff.update(
        schema="rsi-native-training-handoff-v1",
        scope_id=permit["scope_id"],
        operation_id=permit["operation_id"],
        run_id=permit["run_id"],
        recipe_ref=str(root / "recipe.json"),
        source_versions=verify_installed_sources(),
        weights_identity={
            "path": str(weights.resolve()),
            "bytes": weights.stat().st_size,
            "mtime_ns": weights.stat().st_mtime_ns,
        },
        processor_snapshot=json.loads(Path(norm).read_text()),
    )
    handoff["model_descriptor"].update(
        training_scope="action_expert_only",
        training_recipe_ref=str(root / "recipe.json"),
        weight_update_ref=str(update_ref),
    )
    return handoff


def verify_native_load_receipt(
    report: dict[str, Any], handoff: dict[str, Any]
) -> dict[str, Any]:
    """Check actual service identity and decoded compatibility independently of gain."""
    descriptor = handoff["model_descriptor"]
    if (
        report.get("actual_model_load") != "passed"
        or report.get("prediction_shape") != [1, 5, 7]
        or report.get("prediction_finite") is not True
        or report.get("decoded_action_contract") != descriptor["action_contract"]
        or not report.get("saved_probe_equal")
        or not all(value is True for value in report["saved_probe_equal"].values())
    ):
        raise ValueError("native loader compatibility/probe verification failed")
    for key in ("model_path", "norm_stats_path"):
        if Path(report.get(key, "")).resolve() != Path(descriptor[key]).resolve():
            raise ValueError("native model/processor path drift: " + key)
    if report.get("processor_snapshot") != handoff.get("processor_snapshot"):
        raise ValueError("native processor content mismatch")
    weights = Path(handoff["selected_weights"])
    identity = {
        "path": str(weights.resolve()),
        "bytes": weights.stat().st_size,
        "mtime_ns": weights.stat().st_mtime_ns,
    }
    if (
        handoff.get("weights_identity") != identity
        or report.get("weights_identity") != identity
    ):
        raise ValueError("native checkpoint changed between save and load")
    return report
