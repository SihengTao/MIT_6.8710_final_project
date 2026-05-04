from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shlex
import socket
import sys
from copy import deepcopy
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


# User-selected BCL11A hg38 gene-centered 1Mb export window.
DEFAULT_BCL11A_HG38_1MB_REGION = "chr2:60005424-61005424"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chromnitron HIC2 LoRA finetuning")
    parser.add_argument("config_positional", nargs="?", help="Path to finetune YAML config")
    parser.add_argument("--config", dest="config", help="Path to finetune YAML config")
    args = parser.parse_args()
    if args.config is None:
        args.config = args.config_positional
    elif args.config_positional is not None and args.config_positional != args.config:
        parser.error("Provide the config path either positionally or with --config, not both.")
    if args.config is None:
        parser.error("A finetune YAML config is required.")
    return args


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(base_dir: Path, raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def add_import_paths() -> None:
    workspace_root = Path(__file__).resolve().parent
    repo_root = Path(__file__).resolve().parents[2]
    for path in (workspace_root, repo_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def setup_distributed() -> tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this finetune script.")

    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        rank = torch.distributed.get_rank()
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0
        rank = 0
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    return distributed, rank, local_rank, world_size, device


def cleanup_distributed(distributed: bool) -> None:
    if distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def set_random_seed(seed: int, rank: int) -> None:
    local_seed = seed + rank
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed_all(local_seed)


def setup_tf32(enabled: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def maybe_barrier(distributed: bool) -> None:
    if distributed:
        torch.distributed.barrier()


def is_rank_zero(rank: int) -> bool:
    return rank == 0


def resource_root(input_dict: dict[str, Any], key: str) -> str:
    return input_dict.get(f"{key}_root", input_dict.get("root", ""))


def resolve_static_resources(config: dict[str, Any], config_path: Path) -> dict[str, str]:
    code_cfg = config.get("code", {})
    atlas_config_path = resolve_path(config_path.parent, code_cfg["atlas_config_path"])
    if atlas_config_path is None:
        raise ValueError("code.atlas_config_path is required.")

    atlas_cfg = load_yaml(atlas_config_path)
    input_cfg = atlas_cfg["input_resource"]
    data_cfg = config["data"]
    assembly = data_cfg["assembly"]

    seq_root = Path(resource_root(input_cfg, "sequence")) / input_cfg.get("sequence", "")
    cap_root = Path(resource_root(input_cfg, "cap")) / input_cfg.get("cap", "")
    input_seq_path = seq_root / f"{assembly}.zarr"
    esm_feature_path = cap_root / f'{config["model"]["target_cap"]}.npz'

    excluded_region_raw = data_cfg.get("excluded_region_path", "auto")
    if excluded_region_raw == "auto":
        excluded_region_path = seq_root / f"{assembly}-blacklist.v2.bed"
    else:
        excluded_region_path = Path(excluded_region_raw)

    return {
        "atlas_config_path": str(atlas_config_path),
        "input_seq_path": str(input_seq_path),
        "esm_feature_path": str(esm_feature_path),
        "excluded_region_path": str(excluded_region_path),
    }


def resolve_manifest_path(config: dict[str, Any], config_path: Path) -> Path:
    manifest_path = resolve_path(config_path.parent, config["data"]["manifest_path"])
    if manifest_path is None:
        raise ValueError("data.manifest_path is required.")
    return manifest_path


def configure_genome_resources(config: dict[str, Any], config_path: Path) -> None:
    chrom_sizes_path = resolve_path(config_path.parent, config["data"].get("chrom_sizes_path"))
    if chrom_sizes_path is not None:
        os.environ["CHROMNITRON_CHR_SIZES_PATH"] = str(chrom_sizes_path)


def build_datasets(
    config: dict[str, Any], config_path: Path
) -> tuple[Any, Any, Any, dict[str, str], Path]:
    from finetune_dataset import ManifestFinetuneDataset

    resources = resolve_static_resources(config, config_path)
    manifest_path = resolve_manifest_path(config, config_path)
    data_cfg = config["data"]
    shared_kwargs = {
        "manifest_path": manifest_path,
        "input_seq_path": resources["input_seq_path"],
        "esm_feature_path": resources["esm_feature_path"],
        "target_cap": config["model"]["target_cap"],
        "excluded_region_file": resources["excluded_region_path"],
        "val_chrs": data_cfg.get("val_chrs", ["chr10"]),
        "test_chrs": data_cfg.get("test_chrs", ["chr20"]),
        "assembly": data_cfg.get("assembly", "hg38"),
        "chunk_size": data_cfg.get("chunk_size", 100000),
        "sample_per_chunk": data_cfg.get("sample_per_chunk", 4),
        "window_size": data_cfg.get("window_size", 8192),
        "atac_log1p": data_cfg.get("atac_log1p", True),
        "target_transform": data_cfg.get("target_transform"),
        "apply_reverse_complement": data_cfg.get("apply_reverse_complement", True),
        "apply_gaussian_noise": data_cfg.get("apply_gaussian_noise", True),
        "cap_embedding_key": data_cfg.get("cap_embedding_key", "embedding"),
        "verbose": data_cfg.get("verbose", False),
    }
    train_dataset = ManifestFinetuneDataset(
        mode="train",
        atac_dropout=deepcopy(config.get("atac_dropout", {})),
        **shared_kwargs,
    )
    val_dataset = ManifestFinetuneDataset(
        mode="val",
        atac_dropout={"enabled": False},
        **shared_kwargs,
    )
    test_dataset = ManifestFinetuneDataset(
        mode="test",
        atac_dropout={"enabled": False},
        **shared_kwargs,
    )
    return train_dataset, val_dataset, test_dataset, resources, manifest_path


def build_dataloaders(
    train_dataset: Any,
    val_dataset: Any,
    test_dataset: Any,
    config: dict[str, Any],
    distributed: bool,
    world_size: int,
) -> tuple[dict[str, DataLoader], dict[str, Any]]:
    training_cfg = config["training"]
    batch_size = int(training_cfg.get("batch_size", 1))
    num_workers = int(training_cfg.get("num_workers", 4))
    prefetch_factor = int(training_cfg.get("prefetch_factor", 2))
    pin_memory = bool(training_cfg.get("pin_memory", True))

    samplers: dict[str, Any] = {"train": None, "val": None, "test": None}
    if distributed:
        samplers["train"] = DistributedSampler(
            train_dataset, shuffle=True, drop_last=False, num_replicas=world_size
        )
        samplers["val"] = DistributedSampler(
            val_dataset, shuffle=False, drop_last=False, num_replicas=world_size
        )
        samplers["test"] = DistributedSampler(
            test_dataset, shuffle=False, drop_last=False, num_replicas=world_size
        )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            shuffle=samplers["train"] is None,
            sampler=samplers["train"],
            **loader_kwargs,
        ),
        "val": DataLoader(
            val_dataset,
            shuffle=False,
            sampler=samplers["val"],
            **loader_kwargs,
        ),
        "test": DataLoader(
            test_dataset,
            shuffle=False,
            sampler=samplers["test"],
            **loader_kwargs,
        ),
    }
    return dataloaders, samplers


def load_state_dict(path: str | Path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        return state["model"]
    if not isinstance(state, dict):
        raise TypeError(f"Expected dict-like state_dict at {path}, got {type(state)!r}")
    return state


def tensor_shape_list(value: Any) -> list[int] | None:
    if torch.is_tensor(value):
        return [int(dim) for dim in value.shape]
    return None


def format_shape_mismatch_preview(
    entries: list[dict[str, Any]],
    limit: int = 3,
) -> str:
    preview_chunks = []
    for entry in entries[:limit]:
        preview_chunks.append(
            f"{entry['key']} checkpoint_shape={entry['checkpoint_shape']} "
            f"model_shape={entry['model_shape']}"
        )
    remaining = len(entries) - min(len(entries), limit)
    if remaining > 0:
        preview_chunks.append(f"+{remaining} more")
    return "; ".join(preview_chunks)


def prepare_warm_start_state_dict(
    model: torch.nn.Module,
    warm_start_sd: dict[str, Any],
    warm_start_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_state = model.state_dict()
    compatible_state: dict[str, Any] = {}
    loaded_keys: list[str] = []
    unexpected_keys: list[str] = []
    shape_mismatches: list[dict[str, Any]] = []
    type_mismatches: list[dict[str, Any]] = []

    for key, value in warm_start_sd.items():
        if key not in model_state:
            unexpected_keys.append(key)
            continue

        model_value = model_state[key]
        if torch.is_tensor(value) != torch.is_tensor(model_value):
            type_mismatches.append(
                {
                    "key": key,
                    "checkpoint_type": type(value).__name__,
                    "model_type": type(model_value).__name__,
                }
            )
            continue

        if torch.is_tensor(value) and tuple(value.shape) != tuple(model_value.shape):
            shape_mismatches.append(
                {
                    "key": key,
                    "checkpoint_shape": tensor_shape_list(value),
                    "model_shape": tensor_shape_list(model_value),
                }
            )
            continue

        compatible_state[key] = value
        loaded_keys.append(key)

    missing_lora_keys = [
        key for key in model_state.keys() if "lora_" in key and key not in warm_start_sd
    ]
    lora_shape_mismatches = [entry for entry in shape_mismatches if "lora_" in entry["key"]]
    skipped_keys = [entry["key"] for entry in shape_mismatches] + [
        entry["key"] for entry in type_mismatches
    ]

    if compatible_state and not skipped_keys and not unexpected_keys and not missing_lora_keys:
        status = "full"
    elif compatible_state:
        status = "partial"
    else:
        status = "empty"

    load_info = {
        "warm_start_checkpoint_path": str(warm_start_path),
        "warm_start_status": status,
        "warm_start_checkpoint_key_count": len(warm_start_sd),
        "warm_start_loaded_key_count": len(loaded_keys),
        "warm_start_skipped_key_count": len(skipped_keys),
        "warm_start_unexpected_key_count": len(unexpected_keys),
        "warm_start_missing_lora_key_count": len(missing_lora_keys),
        "warm_start_loaded_keys": loaded_keys,
        "warm_start_skipped_shape_mismatches": shape_mismatches,
        "warm_start_skipped_type_mismatches": type_mismatches,
        "warm_start_unexpected_keys": unexpected_keys,
        "warm_start_missing_lora_keys": missing_lora_keys,
        "warm_start_lora_shape_mismatch_count": len(lora_shape_mismatches),
        "warm_start_lora_shape_mismatch_keys": [entry["key"] for entry in lora_shape_mismatches],
    }
    return compatible_state, load_info


def disabled_warm_start_load_info(model: torch.nn.Module) -> dict[str, Any]:
    lora_keys = [key for key in model.state_dict().keys() if "lora_" in key]
    return {
        "warm_start_checkpoint_path": "",
        "warm_start_status": "disabled",
        "warm_start_checkpoint_key_count": 0,
        "warm_start_loaded_key_count": 0,
        "warm_start_skipped_key_count": 0,
        "warm_start_unexpected_key_count": 0,
        "warm_start_missing_lora_key_count": len(lora_keys),
        "warm_start_loaded_keys": [],
        "warm_start_skipped_shape_mismatches": [],
        "warm_start_skipped_type_mismatches": [],
        "warm_start_unexpected_keys": [],
        "warm_start_missing_lora_keys": lora_keys,
        "warm_start_lora_shape_mismatch_count": 0,
        "warm_start_lora_shape_mismatch_keys": [],
    }


def build_model(
    config: dict[str, Any],
    device: torch.device,
    local_rank: int,
    distributed: bool,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    from chromnitron.training.finetuning.model.v4_5.chromnitron_models import get_model

    model_cfg = config["model"]
    model_args = {
        "num_genomic_features": model_cfg.get("num_genomic_features", 1),
        "hidden": model_cfg.get("hidden", 384),
        "num_attn_blocks": model_cfg.get("num_attn_blocks", 16),
        "num_of_scale": model_cfg.get("num_of_scale", 4),
        "num_targets": model_cfg.get("num_targets", 1),
        "no_confidence_prediction": model_cfg.get("no_confidence_prediction", False),
        "prot_dim": model_cfg.get("prot_dim", 2560),
        "sample_per_chunk": model_cfg.get("sample_per_chunk", 4),
        "confidence_weight": model_cfg.get("confidence_weight", 0.0),
        "pretrained": {
            "load": True,
            "load_path": model_cfg["base_model_path"],
        },
        "lora": {
            "enabled": True,
            "r": model_cfg["lora_r"],
        },
    }

    torch.cuda.set_device(local_rank)
    model = get_model(model_args)
    warm_start_path = model_cfg.get("warm_start_lora_path")
    if warm_start_path is None or str(warm_start_path).strip() == "":
        load_info = disabled_warm_start_load_info(model)
    else:
        warm_start_sd = load_state_dict(warm_start_path)
        compatible_warm_start_sd, load_info = prepare_warm_start_state_dict(
            model,
            warm_start_sd,
            warm_start_path,
        )
        model.load_state_dict(compatible_warm_start_sd, strict=False)
    trainable_info = configure_trainable_parameters(model, config)
    model = model.to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    return model, load_info, trainable_info


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def count_trainable_parameters(model: torch.nn.Module) -> int:
    base_model = unwrap_model(model)
    return sum(param.numel() for param in base_model.parameters() if param.requires_grad)


def format_parameter_preview(names: list[str], limit: int = 4) -> str:
    preview = names[:limit]
    remaining = len(names) - len(preview)
    if remaining > 0:
        preview.append(f"+{remaining} more")
    return ", ".join(preview) if preview else "none"


def resolve_trainable_mode_prefixes(
    model: torch.nn.Module,
    trainable_mode: str,
) -> list[str]:
    if trainable_mode == "lora":
        return []

    head_prefixes = [
        "decoder.conv_end.",
        "decoder.conv_confidence.",
    ]
    if trainable_mode == "lora_plus_head":
        return head_prefixes

    if trainable_mode == "lora_plus_last_block":
        base_model = unwrap_model(model)
        layers = getattr(getattr(getattr(base_model, "tf", None), "module", None), "layers", None)
        if layers is None:
            raise ValueError(
                "model.trainable_mode=lora_plus_last_block is unsupported because "
                "the model does not expose tf.module.layers."
            )
        if len(layers) == 0:
            raise ValueError(
                "model.trainable_mode=lora_plus_last_block is unsupported because "
                "the transformer block list is empty."
            )
        return head_prefixes + [f"tf.module.layers.{len(layers) - 1}."]

    raise ValueError(
        "Unsupported model.trainable_mode="
        f"{trainable_mode!r}; expected one of "
        "'lora', 'lora_plus_head', or 'lora_plus_last_block'."
    )


def configure_trainable_parameters(
    model: torch.nn.Module,
    config: dict[str, Any],
) -> dict[str, Any]:
    model_cfg = config["model"]
    trainable_mode = str(model_cfg.get("trainable_mode", "lora"))
    baseline_trainable_params = count_trainable_parameters(model)
    extra_prefixes = resolve_trainable_mode_prefixes(model, trainable_mode)

    extra_trainable_names: list[str] = []
    extra_trainable_params = 0
    base_model = unwrap_model(model)
    for name, param in base_model.named_parameters():
        if not any(name.startswith(prefix) for prefix in extra_prefixes):
            continue
        if param.requires_grad:
            continue
        param.requires_grad = True
        extra_trainable_names.append(name)
        extra_trainable_params += param.numel()

    if trainable_mode != "lora" and not extra_trainable_names:
        raise ValueError(
            "model.trainable_mode resolved to extra trainable prefixes but did not match any "
            "currently frozen parameters. Prefixes="
            f"{extra_prefixes!r}"
        )

    total_trainable_params = count_trainable_parameters(model)
    return {
        "trainable_mode": trainable_mode,
        "baseline_lora_trainable_parameters": baseline_trainable_params,
        "extra_trainable_prefixes": extra_prefixes,
        "extra_trainable_tensor_count": len(extra_trainable_names),
        "extra_trainable_parameters": extra_trainable_params,
        "extra_trainable_parameter_names": extra_trainable_names,
        "total_trainable_parameters": total_trainable_params,
        "checkpoint_includes_extra_trainable_parameters": bool(extra_trainable_names),
    }


def build_optimizer_and_scheduler(
    model: torch.nn.Module, config: dict[str, Any]
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau, list[float]]:
    training_cfg = config["training"]
    base_model = unwrap_model(model)
    trainable_params = [param for param in base_model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(training_cfg.get("learning_rate", 1e-5)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    scheduler_cfg = training_cfg.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_cfg.get("mode", "min"),
        factor=float(scheduler_cfg.get("factor", 0.5)),
        patience=int(scheduler_cfg.get("patience", 5)),
        min_lr=float(scheduler_cfg.get("min_lr", 0.0)),
    )
    return optimizer, scheduler, base_lrs


def apply_warmup_lr(
    optimizer: torch.optim.Optimizer,
    base_lrs: list[float],
    current_step: int,
    warmup_steps: int,
) -> None:
    if warmup_steps <= 0 or current_step > warmup_steps:
        return
    scale = current_step / warmup_steps
    for group, base in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base * scale


def move_batch_to_device(
    batch: tuple[torch.Tensor, ...], device: torch.device
) -> tuple[torch.Tensor, ...]:
    return tuple(item.to(device, non_blocking=True) for item in batch)


def compute_loss(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, ...],
    confidence_weight: float,
    no_confidence_prediction: bool,
    loss_cfg: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, float, dict[str, int | float]]:
    seq, input_features, target_features, esm_embeddings = batch
    esm_embeddings = esm_embeddings.float().transpose(-1, -2)

    batch_size, mini_bs, seq_len, seq_dim = seq.shape
    seq = seq.view(batch_size * mini_bs, seq_len, seq_dim).transpose(1, 2).float()
    input_features = (
        input_features.view(batch_size * mini_bs, -1).unsqueeze(2).transpose(1, 2).float()
    )
    _, _, num_target, target_len = target_features.shape
    target_features = target_features.view(batch_size * mini_bs, num_target, target_len).float()

    inputs = (seq, input_features)
    if no_confidence_prediction:
        preds = model(inputs, esm_embeddings)
        confidence = None
    else:
        preds, confidence = model(inputs, esm_embeddings)
    if loss_cfg and loss_cfg.get("name", "mse") != "mse":
        from finetune_losses import compute_configured_loss

        loss, _components = compute_configured_loss(
            preds,
            target_features,
            confidence=confidence,
            confidence_weight=confidence_weight,
            loss_cfg=loss_cfg,
        )
    else:
        loss_map = (preds - target_features) ** 2
        if confidence is not None and confidence_weight > 0:
            rmse = torch.sqrt(loss_map.detach() + 1e-8)
            confidence_loss = (confidence - rmse) ** 2
            loss_map = loss_map + confidence_weight * confidence_loss
        loss = loss_map.mean()
    return loss, float(loss.detach().item()), compute_batch_pearson_stats(preds, target_features)


def compute_batch_pearson_stats(
    preds: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, int | float]:
    preds_flat = preds.detach().float().flatten(start_dim=1)
    target_flat = target.detach().float().flatten(start_dim=1)
    total_samples = int(preds_flat.shape[0])
    if total_samples == 0:
        return {
            "pearson_sum": 0.0,
            "pearson_valid_samples": 0,
            "pearson_total_samples": 0,
            "pearson_nonfinite_samples": 0,
            "pearson_constant_samples": 0,
        }

    finite_mask = torch.isfinite(preds_flat).all(dim=1) & torch.isfinite(target_flat).all(dim=1)
    finite_count = int(finite_mask.sum().item())
    nonfinite_samples = total_samples - finite_count
    if finite_count == 0:
        return {
            "pearson_sum": 0.0,
            "pearson_valid_samples": 0,
            "pearson_total_samples": total_samples,
            "pearson_nonfinite_samples": nonfinite_samples,
            "pearson_constant_samples": 0,
        }

    preds_centered = preds_flat[finite_mask] - preds_flat[finite_mask].mean(dim=1, keepdim=True)
    target_centered = target_flat[finite_mask] - target_flat[finite_mask].mean(dim=1, keepdim=True)
    pred_var = preds_centered.square().sum(dim=1)
    target_var = target_centered.square().sum(dim=1)
    non_constant_mask = (pred_var > eps) & (target_var > eps)
    constant_samples = finite_count - int(non_constant_mask.sum().item())
    if not torch.any(non_constant_mask):
        return {
            "pearson_sum": 0.0,
            "pearson_valid_samples": 0,
            "pearson_total_samples": total_samples,
            "pearson_nonfinite_samples": nonfinite_samples,
            "pearson_constant_samples": constant_samples,
        }

    preds_centered = preds_centered[non_constant_mask]
    target_centered = target_centered[non_constant_mask]
    pred_var = pred_var[non_constant_mask]
    target_var = target_var[non_constant_mask]
    denominator = torch.sqrt((pred_var * target_var).clamp_min(eps))
    corr = (preds_centered * target_centered).sum(dim=1) / denominator
    corr = corr[torch.isfinite(corr)]
    valid_samples = int(corr.numel())
    extra_nonfinite_samples = int(non_constant_mask.sum().item()) - valid_samples
    return {
        "pearson_sum": float(corr.sum().item()) if valid_samples > 0 else 0.0,
        "pearson_valid_samples": valid_samples,
        "pearson_total_samples": total_samples,
        "pearson_nonfinite_samples": nonfinite_samples + extra_nonfinite_samples,
        "pearson_constant_samples": constant_samples,
    }


def reduce_metric_vector(
    values: list[float],
    device: torch.device,
    distributed: bool,
) -> list[float]:
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    if distributed:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return [float(item) for item in tensor.cpu().tolist()]


def empty_split_metrics() -> dict[str, int | float]:
    return {
        "loss": float("nan"),
        "loss_valid_batches": 0,
        "loss_total_batches": 0,
        "pearson": float("nan"),
        "pearson_valid_samples": 0,
        "pearson_total_samples": 0,
        "pearson_nonfinite_samples": 0,
        "pearson_constant_samples": 0,
    }


def finalize_epoch_metrics(
    aggregates: dict[str, float],
    device: torch.device,
    distributed: bool,
) -> dict[str, int | float]:
    reduced = reduce_metric_vector(
        [
            aggregates["loss_sum"],
            aggregates["loss_valid_batches"],
            aggregates["loss_total_batches"],
            aggregates["pearson_sum"],
            aggregates["pearson_valid_samples"],
            aggregates["pearson_total_samples"],
            aggregates["pearson_nonfinite_samples"],
            aggregates["pearson_constant_samples"],
        ],
        device,
        distributed,
    )
    loss_sum = reduced[0]
    loss_valid_batches = int(round(reduced[1]))
    loss_total_batches = int(round(reduced[2]))
    pearson_sum = reduced[3]
    pearson_valid_samples = int(round(reduced[4]))
    pearson_total_samples = int(round(reduced[5]))
    pearson_nonfinite_samples = int(round(reduced[6]))
    pearson_constant_samples = int(round(reduced[7]))

    return {
        "loss": (loss_sum / loss_valid_batches) if loss_valid_batches > 0 else float("nan"),
        "loss_valid_batches": loss_valid_batches,
        "loss_total_batches": loss_total_batches,
        "pearson": (
            pearson_sum / pearson_valid_samples
            if pearson_valid_samples > 0
            else float("nan")
        ),
        "pearson_valid_samples": pearson_valid_samples,
        "pearson_total_samples": pearson_total_samples,
        "pearson_nonfinite_samples": pearson_nonfinite_samples,
        "pearson_constant_samples": pearson_constant_samples,
    }


def add_split_metrics(row: dict[str, Any], split_name: str, metrics: dict[str, Any]) -> None:
    row[f"{split_name}_loss"] = metrics["loss"]
    row[f"{split_name}_loss_valid_batches"] = metrics["loss_valid_batches"]
    row[f"{split_name}_loss_total_batches"] = metrics["loss_total_batches"]
    row[f"{split_name}_pearson"] = metrics["pearson"]
    row[f"{split_name}_pearson_valid_samples"] = metrics["pearson_valid_samples"]
    row[f"{split_name}_pearson_total_samples"] = metrics["pearson_total_samples"]
    row[f"{split_name}_pearson_nonfinite_samples"] = metrics["pearson_nonfinite_samples"]
    row[f"{split_name}_pearson_constant_samples"] = metrics["pearson_constant_samples"]


def format_metric(value: float) -> str:
    return f"{value:.6f}" if np.isfinite(value) else "nan"


def sanitize_for_json(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: sanitize_for_json(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [sanitize_for_json(value) for value in payload]
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, np.integer):
        return int(payload)
    if isinstance(payload, np.floating):
        payload = float(payload)
    if isinstance(payload, float):
        return payload if np.isfinite(payload) else None
    return payload


def ensure_finite_training_loss(
    loss: torch.Tensor,
    loss_value: float,
    *,
    device: torch.device,
    distributed: bool,
    rank: int,
    run_name: str,
    split_name: str,
    epoch_idx: int,
    step_idx: int,
) -> None:
    local_nonfinite = bool((~torch.isfinite(loss.detach()).all()).item() or not np.isfinite(loss_value))
    local_message = ""
    if local_nonfinite:
        local_message = (
            "Non-finite loss detected before backward/optimizer step: "
            f"run_name={run_name} split={split_name} epoch={epoch_idx} "
            f"batch={step_idx} rank={rank} loss={loss_value}"
        )

    if distributed:
        has_nonfinite = torch.tensor([int(local_nonfinite)], device=device, dtype=torch.int32)
        torch.distributed.all_reduce(has_nonfinite, op=torch.distributed.ReduceOp.MAX)
        if int(has_nonfinite.item()):
            gathered_messages = [""] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered_messages, local_message)
            for message in gathered_messages:
                if message:
                    raise RuntimeError(message)
            raise RuntimeError(
                "Non-finite loss detected on at least one rank before backward/optimizer step: "
                f"run_name={run_name} split={split_name} epoch={epoch_idx} batch={step_idx}"
            )
        return

    if local_nonfinite:
        raise RuntimeError(local_message)


def run_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    config: dict[str, Any],
    *,
    training: bool,
    distributed: bool,
    rank: int,
    epoch_idx: int,
    split_name: str,
    lr_state: dict[str, Any] | None = None,
) -> dict[str, int | float]:
    run_name = str(config["experiment"]["run_name"])
    confidence_weight = float(config["model"].get("confidence_weight", 0.0))
    no_confidence_prediction = bool(config["model"].get("no_confidence_prediction", False))
    loss_cfg = config.get("loss", {"name": "mse"})
    gradient_clip_norm = float(config["training"].get("gradient_clip_norm", 0.0))
    log_every_steps = int(config["training"].get("log_every_steps", 50))
    accum_steps = max(1, int(config["training"].get("gradient_accumulation_steps", 1)))
    warmup_steps = int(config["training"].get("warmup_steps", 0))

    if training:
        model.train()
    else:
        model.eval()

    aggregates = {
        "loss_sum": 0.0,
        "loss_valid_batches": 0.0,
        "loss_total_batches": 0.0,
        "pearson_sum": 0.0,
        "pearson_valid_samples": 0.0,
        "pearson_total_samples": 0.0,
        "pearson_nonfinite_samples": 0.0,
        "pearson_constant_samples": 0.0,
    }
    if training:
        optimizer.zero_grad(set_to_none=True)
    num_batches = len(dataloader)
    for step_idx, batch in enumerate(dataloader, start=1):
        batch = move_batch_to_device(batch, device)
        if training:
            loss, loss_value, batch_metrics = compute_loss(
                model, batch, confidence_weight, no_confidence_prediction, loss_cfg
            )
            ensure_finite_training_loss(
                loss,
                loss_value,
                device=device,
                distributed=distributed,
                rank=rank,
                run_name=run_name,
                split_name=split_name,
                epoch_idx=epoch_idx,
                step_idx=step_idx,
            )
            (loss / accum_steps).backward()
            is_last_batch = step_idx == num_batches
            if step_idx % accum_steps == 0 or is_last_batch:
                if gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        unwrap_model(model).parameters(), gradient_clip_norm
                    )
                if lr_state is not None:
                    lr_state["step"] += 1
                    apply_warmup_lr(
                        optimizer,
                        lr_state["base_lrs"],
                        lr_state["step"],
                        warmup_steps,
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        else:
            with torch.no_grad():
                loss, loss_value, batch_metrics = compute_loss(
                    model, batch, confidence_weight, no_confidence_prediction, loss_cfg
                )

        if not training and not np.isfinite(loss_value) and is_rank_zero(rank):
            print(
                f"[{split_name}] step={step_idx} produced non-finite loss={loss_value}; "
                "metrics will exclude this batch from epoch means.",
                flush=True,
            )

        aggregates["loss_total_batches"] += 1.0
        if np.isfinite(loss_value):
            aggregates["loss_sum"] += float(loss_value)
            aggregates["loss_valid_batches"] += 1.0
        aggregates["pearson_sum"] += float(batch_metrics["pearson_sum"])
        aggregates["pearson_valid_samples"] += float(batch_metrics["pearson_valid_samples"])
        aggregates["pearson_total_samples"] += float(batch_metrics["pearson_total_samples"])
        aggregates["pearson_nonfinite_samples"] += float(
            batch_metrics["pearson_nonfinite_samples"]
        )
        aggregates["pearson_constant_samples"] += float(
            batch_metrics["pearson_constant_samples"]
        )

        if is_rank_zero(rank) and training and step_idx % log_every_steps == 0:
            print(f"[{split_name}] step={step_idx} loss={loss_value:.6f}", flush=True)

    return finalize_epoch_metrics(aggregates, device, distributed)


def finetune_state_dict(
    model: torch.nn.Module,
    extra_trainable_parameter_names: list[str] | None = None,
) -> dict[str, Any]:
    import loralib as lora

    state = dict(lora.lora_state_dict(unwrap_model(model)))
    if not extra_trainable_parameter_names:
        return state

    full_state = unwrap_model(model).state_dict()
    for name in extra_trainable_parameter_names:
        state[name] = full_state[name]
    return state


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def append_metrics_row(path: Path, row: dict[str, Any]) -> None:
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def broadcast_run_timestamp(rank: int, distributed: bool) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S") if rank == 0 else None
    if distributed:
        payload = [timestamp]
        torch.distributed.broadcast_object_list(payload, src=0)
        timestamp = payload[0]
    return str(timestamp)


def build_run_dir(config: dict[str, Any], timestamp: str) -> Path:
    run_name = config["experiment"]["run_name"]
    save_root = Path(config["experiment"]["save_root"]).expanduser()
    return save_root / run_name / timestamp


def build_training_state(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epoch_idx: int,
    row: dict[str, Any],
    best_epoch: int,
    best_val_loss: float,
    best_test_loss_at_best_val: float | None,
) -> dict[str, Any]:
    return {
        "epoch": epoch_idx,
        "metrics": row,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_test_loss_at_best_val": best_test_loss_at_best_val,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }


def save_epoch_artifacts(
    run_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epoch_idx: int,
    row: dict[str, Any],
    best_epoch: int,
    best_val_loss: float,
    best_test_loss_at_best_val: float | None,
    extra_trainable_parameter_names: list[str],
) -> None:
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    adapter_state = finetune_state_dict(model, extra_trainable_parameter_names)
    training_state = build_training_state(
        optimizer=optimizer,
        scheduler=scheduler,
        epoch_idx=epoch_idx,
        row=row,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_test_loss_at_best_val=best_test_loss_at_best_val,
    )

    torch.save(adapter_state, checkpoints_dir / f"epoch_{epoch_idx:04d}_adapter.pt")
    torch.save(adapter_state, run_dir / "last_adapter.pt")
    torch.save(training_state, run_dir / "last_training_state.pt")


def parse_finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def build_loss_svg(history_rows: list[dict[str, Any]]) -> str:
    width = 900
    height = 560
    left = 82
    right = 28
    top = 58
    bottom = 76
    plot_width = width - left - right
    plot_height = height - top - bottom
    series_specs = [
        ("train_loss", "#1f77b4", "train"),
        ("val_loss", "#2ca02c", "val"),
        ("test_loss", "#d62728", "test"),
    ]

    epochs = [int(row["epoch"]) for row in history_rows]
    finite_values: list[float] = []
    for row in history_rows:
        for key, _color, _label in series_specs:
            value = parse_finite_float(row.get(key))
            if value is not None:
                finite_values.append(value)

    x_min = min(epochs)
    x_max = max(epochs)
    y_min = min(finite_values) if finite_values else 0.0
    y_max = max(finite_values) if finite_values else 1.0
    if y_min == y_max:
        padding = max(abs(y_min) * 0.05, 1.0)
        y_min -= padding
        y_max += padding
    else:
        padding = (y_max - y_min) * 0.08
        y_min -= padding
        y_max += padding

    def x_pos(epoch: int) -> float:
        if x_min == x_max:
            return left + plot_width / 2
        return left + ((epoch - x_min) / (x_max - x_min)) * plot_width

    def y_pos(value: float) -> float:
        return top + (1.0 - ((value - y_min) / (y_max - y_min))) * plot_height

    def polyline_segments(key: str) -> list[list[tuple[float, float]]]:
        segments: list[list[tuple[float, float]]] = []
        current: list[tuple[float, float]] = []
        for row in history_rows:
            value = parse_finite_float(row.get(key))
            if value is None:
                if current:
                    segments.append(current)
                    current = []
                continue
            current.append((x_pos(int(row["epoch"])), y_pos(value)))
        if current:
            segments.append(current)
        return segments

    def points_attr(points: list[tuple[float, float]]) -> str:
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)

    y_ticks = []
    for idx in range(5):
        value = y_min + (y_max - y_min) * idx / 4
        y_ticks.append((value, y_pos(value)))

    if len(epochs) <= 8:
        x_tick_epochs = epochs
    else:
        x_tick_epochs = sorted(
            {round(x_min + (x_max - x_min) * idx / 4) for idx in range(5)}
        )

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        "<style>"
        "text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}"
        ".axis{stroke:#344054;stroke-width:1.2}"
        ".grid{stroke:#d0d5dd;stroke-width:1;stroke-dasharray:4 4}"
        ".label{font-size:14px}"
        ".tick{font-size:12px;fill:#475467}"
        ".title{font-size:20px;font-weight:700}"
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text class="title" x="82" y="34">Chromnitron HIC2 Finetune Loss</text>',
        (
            f'<line class="axis" x1="{left}" y1="{top + plot_height}" '
            f'x2="{left + plot_width}" y2="{top + plot_height}"/>'
        ),
        (
            f'<line class="axis" x1="{left}" y1="{top}" '
            f'x2="{left}" y2="{top + plot_height}"/>'
        ),
    ]

    for value, y in y_ticks:
        lines.extend(
            [
                (
                    f'<line class="grid" x1="{left}" y1="{y:.2f}" '
                    f'x2="{left + plot_width}" y2="{y:.2f}"/>'
                ),
                f'<text class="tick" x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">'
                f"{value:.4g}</text>",
            ]
        )

    for epoch in x_tick_epochs:
        x = x_pos(int(epoch))
        lines.extend(
            [
                (
                    f'<line stroke="#344054" stroke-width="1" x1="{x:.2f}" '
                    f'y1="{top + plot_height}" x2="{x:.2f}" y2="{top + plot_height + 6}"/>'
                ),
                (
                    f'<text class="tick" x="{x:.2f}" y="{top + plot_height + 24}" '
                    f'text-anchor="middle">{epoch}</text>'
                ),
            ]
        )

    for key, color, label in series_specs:
        for segment in polyline_segments(key):
            if len(segment) > 1:
                lines.append(
                    f'<polyline fill="none" stroke="{color}" stroke-width="2.4" '
                    f'points="{points_attr(segment)}"/>'
                )
            for x, y in segment:
                lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.4" fill="{color}"/>')

    legend_x = left + plot_width - 210
    legend_y = top - 22
    for idx, (_key, color, label) in enumerate(series_specs):
        x = legend_x + idx * 74
        lines.extend(
            [
                f'<line x1="{x}" y1="{legend_y}" x2="{x + 22}" y2="{legend_y}" '
                f'stroke="{color}" stroke-width="3"/>',
                f'<text class="label" x="{x + 28}" y="{legend_y + 5}">{escape(label)}</text>',
            ]
        )

    lines.extend(
        [
            (
                f'<text class="label" x="{left + plot_width / 2:.2f}" '
                f'y="{height - 24}" text-anchor="middle">Epoch</text>'
            ),
            (
                f'<text class="label" transform="translate(24 {top + plot_height / 2:.2f}) '
                'rotate(-90)" text-anchor="middle">Loss</text>'
            ),
        ]
    )
    if not finite_values:
        lines.append(
            f'<text class="label" x="{left + plot_width / 2:.2f}" '
            f'y="{top + plot_height / 2:.2f}" text-anchor="middle">'
            "No finite loss values were available.</text>"
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def write_loss_curve_svg(history_rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.write_text(build_loss_svg(history_rows), encoding="utf-8")
    return output_path


def plot_metrics(history_rows: list[dict[str, Any]], output_path: Path) -> Path | None:
    if not history_rows:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [int(row["epoch"]) for row in history_rows]
        train_loss = [float(row["train_loss"]) for row in history_rows]
        val_loss = [float(row["val_loss"]) for row in history_rows]
        test_loss = [float(row["test_loss"]) for row in history_rows]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_loss, label="train_loss", linewidth=2)
        plt.plot(epochs, val_loss, label="val_loss", linewidth=2)
        plt.plot(epochs, test_loss, label="test_loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Chromnitron HIC2 Finetune Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        return output_path
    except Exception as exc:
        fallback_path = output_path.with_suffix(".svg")
        print(
            "Matplotlib loss plot unavailable; writing fallback SVG "
            f"{fallback_path}: {exc}",
            flush=True,
        )
        try:
            return write_loss_curve_svg(history_rows, fallback_path)
        except Exception as svg_exc:
            print(f"Unable to write fallback loss SVG: {svg_exc}", flush=True)
            return None


def build_hic2_bigwig_export_command_template(run_dir: Path, config: dict[str, Any]) -> str:
    helper_path = Path(__file__).resolve().parent / "scripts" / "export_hic2_best_adapter_bigwig_1mb.py"
    target_cap = str(config["model"].get("target_cap", "HIC2"))
    assembly = str(config["data"].get("assembly", "hg38"))
    command_parts = [
        sys.executable,
        str(helper_path),
        "--run-dir",
        str(run_dir),
        "--config-root",
        "CONFIG_ROOT",
        "--celltype",
        "CELLTYPE",
        "--cap",
        target_cap,
        "--assembly",
        assembly,
    ]
    return " ".join(shlex.quote(part) for part in command_parts)


def write_post_training_export_hint(
    run_dir: Path,
    config: dict[str, Any],
    command_template: str,
) -> Path:
    target_cap = str(config["model"].get("target_cap", "HIC2"))
    target_cap_lower = target_cap.lower()
    hint_path = run_dir / f"post_training_{target_cap_lower}_1mb_bigwig.md"
    hint_path.write_text(
        "\n".join(
            [
                f"# {target_cap} 1Mb bigWig export",
                "",
                "Use this project-local helper after training to run existing Chromnitron "
                "track inference against checkpoints/best_adapter.pt.",
                "",
                "Required values:",
                "- CONFIG_ROOT: directory with infer.yaml, data_config.yaml, and finetune_base.yaml",
                "- CELLTYPE: one inference cell type label accepted by that config",
                "",
                "Default region:",
                "- BCL11A hg38 1Mb gene-centered window: "
                f"{DEFAULT_BCL11A_HG38_1MB_REGION}",
                "- Override with --region chrN:start-end if needed.",
                "",
                "Command template:",
                "",
                "```bash",
                command_template,
                "```",
                "",
                f"The helper stages {target_cap}.pt as a symlink to "
                "checkpoints/best_adapter.pt and writes all inference inputs and outputs "
                "inside this run directory.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return hint_path


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(config_path)
    add_import_paths()
    configure_genome_resources(config, config_path)

    distributed, rank, local_rank, world_size, device = setup_distributed()
    try:
        setup_tf32(bool(config["training"].get("tf32", True)))
        set_random_seed(int(config["training"].get("seed", 42)), rank)

        train_dataset, val_dataset, test_dataset, resources, manifest_path = build_datasets(
            config, config_path
        )
        dataloaders, samplers = build_dataloaders(
            train_dataset, val_dataset, test_dataset, config, distributed, world_size
        )

        model, load_info, trainable_info = build_model(config, device, local_rank, distributed)
        optimizer, scheduler, base_lrs = build_optimizer_and_scheduler(model, config)
        lr_state: dict[str, Any] = {"step": 0, "base_lrs": base_lrs}
        trainable_params = int(trainable_info["total_trainable_parameters"])

        run_timestamp = broadcast_run_timestamp(rank, distributed)
        run_dir = build_run_dir(config, run_timestamp)
        maybe_barrier(distributed)
        if is_rank_zero(rank):
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            save_yaml(run_dir / "resolved_config.yaml", config)
            save_yaml(run_dir / "resolved_resources.yaml", resources)
            (run_dir / "resolved_manifest.csv").write_text(
                Path(manifest_path).read_text(encoding="utf-8"), encoding="utf-8"
            )
            save_yaml(run_dir / "warm_start_load_info.yaml", load_info)
            save_yaml(run_dir / "trainable_setup.yaml", trainable_info)
            save_yaml(
                run_dir / "environment.yaml",
                {
                    "hostname": socket.gethostname(),
                    "world_size": world_size,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                },
            )
            print(f"Saving run outputs to {run_dir}", flush=True)
            print(f"Trainable parameters: {trainable_params}", flush=True)
            print(
                "Trainable setup: "
                f"run_name={config['experiment']['run_name']} "
                f"mode={trainable_info['trainable_mode']} "
                f"baseline_lora_trainable={trainable_info['baseline_lora_trainable_parameters']} "
                f"extra_tensors={trainable_info['extra_trainable_tensor_count']} "
                f"extra_parameters={trainable_info['extra_trainable_parameters']} "
                f"prefixes={trainable_info['extra_trainable_prefixes']}",
                flush=True,
            )
            if trainable_info["extra_trainable_parameter_names"]:
                print(
                    "Trainable override detail: "
                    f"extra_parameter_preview="
                    f"{format_parameter_preview(trainable_info['extra_trainable_parameter_names'])}",
                    flush=True,
                )
            print(
                "Warm-start summary: "
                f"run_name={config['experiment']['run_name']} "
                f"checkpoint={load_info['warm_start_checkpoint_path']} "
                f"current_lora_r={config['model'].get('lora_r')} "
                f"status={load_info['warm_start_status']} "
                f"loaded={load_info['warm_start_loaded_key_count']} "
                f"skipped={load_info['warm_start_skipped_key_count']} "
                f"shape_mismatches={len(load_info['warm_start_skipped_shape_mismatches'])} "
                f"unexpected={load_info['warm_start_unexpected_key_count']} "
                f"missing_lora={load_info['warm_start_missing_lora_key_count']}",
                flush=True,
            )
            lora_shape_mismatches = [
                entry
                for entry in load_info["warm_start_skipped_shape_mismatches"]
                if "lora_" in entry["key"]
            ]
            if lora_shape_mismatches:
                print(
                    "Warm-start warning: "
                    f"run_name={config['experiment']['run_name']} "
                    f"current_lora_r={config['model'].get('lora_r')} "
                    f"checkpoint={load_info['warm_start_checkpoint_path']} "
                    f"skipped_lora_shape_mismatches={len(lora_shape_mismatches)} "
                    f"examples={format_shape_mismatch_preview(lora_shape_mismatches)}",
                    flush=True,
                )
            if load_info["warm_start_status"] == "disabled":
                print(
                    "Warm-start warning: "
                    f"run_name={config['experiment']['run_name']} "
                    "adapter warm-start disabled; LoRA adapters start from scratch.",
                    flush=True,
                )

        best_val_loss = float("inf")
        best_test_loss_at_best_val: float | None = None
        best_epoch = -1
        best_row: dict[str, Any] | None = None
        bad_epochs = 0
        history_rows: list[dict[str, Any]] = []
        history_path = run_dir / "metrics.csv"
        loss_curve_path: Path | None = None
        epochs = int(config["training"].get("epochs", 100))
        early_stopping_patience = int(config["training"].get("early_stopping_patience", 20))

        for epoch_idx in range(1, epochs + 1):
            train_sampler = samplers["train"]
            if train_sampler is not None:
                train_sampler.set_epoch(epoch_idx)

            train_metrics = run_epoch(
                model,
                dataloaders["train"],
                optimizer,
                device,
                config,
                training=True,
                distributed=distributed,
                rank=rank,
                epoch_idx=epoch_idx,
                split_name="train",
                lr_state=lr_state,
            )
            val_metrics = run_epoch(
                model,
                dataloaders["val"],
                optimizer=None,
                device=device,
                config=config,
                training=False,
                distributed=distributed,
                rank=rank,
                epoch_idx=epoch_idx,
                split_name="val",
            )
            if np.isfinite(val_metrics["loss"]):
                scheduler.step(val_metrics["loss"])
            elif is_rank_zero(rank):
                print(
                    f"epoch={epoch_idx} val_loss is non-finite; skipping scheduler step.",
                    flush=True,
                )

            improved = np.isfinite(val_metrics["loss"]) and val_metrics["loss"] < best_val_loss
            if improved:
                test_metrics = run_epoch(
                    model,
                    dataloaders["test"],
                    optimizer=None,
                    device=device,
                    config=config,
                    training=False,
                    distributed=distributed,
                    rank=rank,
                    epoch_idx=epoch_idx,
                    split_name="test",
                )
                test_loss_value: float = test_metrics["loss"]
            else:
                test_metrics = empty_split_metrics()
                test_loss_value = float("nan")

            current_lr = float(optimizer.param_groups[0]["lr"])
            row: dict[str, Any] = {"epoch": epoch_idx, "lr": current_lr}
            add_split_metrics(row, "train", train_metrics)
            add_split_metrics(row, "val", val_metrics)
            add_split_metrics(row, "test", test_metrics)

            if improved:
                best_val_loss = val_metrics["loss"]
                best_test_loss_at_best_val = test_loss_value
                best_epoch = epoch_idx
                best_row = dict(row)
                bad_epochs = 0
            else:
                bad_epochs += 1

            if is_rank_zero(rank):
                history_rows.append(row)
                append_metrics_row(history_path, row)
                save_epoch_artifacts(
                    run_dir=run_dir,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch_idx=epoch_idx,
                    row=row,
                    best_epoch=best_epoch,
                    best_val_loss=best_val_loss,
                    best_test_loss_at_best_val=best_test_loss_at_best_val,
                    extra_trainable_parameter_names=trainable_info[
                        "extra_trainable_parameter_names"
                    ],
                )
                current_loss_curve_path = plot_metrics(history_rows, run_dir / "loss_curve.png")
                if current_loss_curve_path is not None:
                    loss_curve_path = current_loss_curve_path
                test_loss_str = (
                    f"{test_loss_value:.6f}" if np.isfinite(test_loss_value) else "skipped"
                )
                test_pearson_str = (
                    format_metric(float(test_metrics["pearson"]))
                    if improved
                    else "skipped"
                )
                print(
                    f"epoch={epoch_idx} "
                    f"train_loss={format_metric(float(train_metrics['loss']))} "
                    f"train_pearson={format_metric(float(train_metrics['pearson']))} "
                    f"val_loss={format_metric(float(val_metrics['loss']))} "
                    f"val_pearson={format_metric(float(val_metrics['pearson']))} "
                    f"test_loss={test_loss_str} "
                    f"test_pearson={test_pearson_str} "
                    f"lr={current_lr:.8f}",
                    flush=True,
                )

                if improved:
                    adapter_state = finetune_state_dict(
                        model, trainable_info["extra_trainable_parameter_names"]
                    )
                    checkpoints_dir = run_dir / "checkpoints"
                    checkpoints_dir.mkdir(parents=True, exist_ok=True)
                    best_training_state = build_training_state(
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch_idx=epoch_idx,
                        row=row,
                        best_epoch=best_epoch,
                        best_val_loss=best_val_loss,
                        best_test_loss_at_best_val=best_test_loss_at_best_val,
                    )
                    torch.save(adapter_state, run_dir / "best_adapter.pt")
                    torch.save(adapter_state, checkpoints_dir / "best_adapter.pt")
                    torch.save(best_training_state, run_dir / "best_training_state.pt")
                    torch.save(best_training_state, checkpoints_dir / "best_training_state.pt")

            stop_training = bad_epochs >= early_stopping_patience
            if distributed:
                stop_tensor = torch.tensor([int(stop_training)], device=device)
                torch.distributed.broadcast(stop_tensor, src=0)
                stop_training = bool(stop_tensor.item())

            if stop_training:
                if is_rank_zero(rank):
                    print(
                        f"Early stopping at epoch {epoch_idx}; best_epoch={best_epoch} "
                        f"best_val_loss={best_val_loss:.6f}",
                        flush=True,
                    )
                break

        if is_rank_zero(rank):
            loss_curve_png_path = run_dir / "loss_curve.png"
            loss_curve_svg_path = run_dir / "loss_curve.svg"
            post_training_export_command = build_hic2_bigwig_export_command_template(
                run_dir,
                config,
            )
            post_training_export_hint_path = write_post_training_export_hint(
                run_dir,
                config,
                post_training_export_command,
            )
            summary = {
                "target_cap": config["model"]["target_cap"],
                "target_transform": config.get("data", {}).get("target_transform", "legacy_log1p"),
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_test_loss_at_best_val": best_test_loss_at_best_val,
                "best_val_pearson": None if best_row is None else best_row["val_pearson"],
                "best_test_pearson_at_best_val": (
                    None if best_row is None else best_row["test_pearson"]
                ),
                "trainable_mode": trainable_info["trainable_mode"],
                "trainable_parameters": trainable_params,
                "extra_trainable_parameters": trainable_info["extra_trainable_parameters"],
                "run_dir": str(run_dir),
                "metrics_csv": str(history_path),
                "loss_curve": None if loss_curve_path is None else str(loss_curve_path),
                "loss_curve_png": (
                    str(loss_curve_png_path) if loss_curve_png_path.is_file() else None
                ),
                "loss_curve_svg": (
                    str(loss_curve_svg_path) if loss_curve_svg_path.is_file() else None
                ),
                "warm_start_load_info_yaml": str(run_dir / "warm_start_load_info.yaml"),
                "trainable_setup_yaml": str(run_dir / "trainable_setup.yaml"),
                "post_training_hic2_1mb_bigwig_hint": str(post_training_export_hint_path),
                "post_training_hic2_1mb_bigwig_command_template": post_training_export_command,
                "final_metrics": None if not history_rows else history_rows[-1],
                "best_metrics": best_row,
            }
            with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
                json.dump(sanitize_for_json(summary), handle, indent=2)
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
