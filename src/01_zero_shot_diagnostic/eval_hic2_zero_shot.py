#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
FINETUNE_CODE_DIR = REPO_ROOT / "src" / "03_finetuning"
for path in (FINETUNE_CODE_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from run_hic2_finetune import (  # noqa: E402
    add_import_paths,
    build_dataloaders,
    build_datasets,
    build_model,
    cleanup_distributed,
    compute_loss,
    configure_genome_resources,
    disabled_warm_start_load_info,
    finalize_epoch_metrics,
    format_metric,
    is_rank_zero,
    load_yaml,
    maybe_barrier,
    move_batch_to_device,
    sanitize_for_json,
    set_random_seed,
    setup_distributed,
    setup_tf32,
    unwrap_model,
)


MODE_DESCRIPTIONS = {
    "warm_start_lora": (
        "Pretrained base model plus LoRA modules built by run_hic2_finetune.build_model(), "
        "with model.warm_start_lora_path loaded directly and no gradient updates."
    ),
    "base": (
        "Pretrained base model built through the same Chromnitron model API with LoRA disabled "
        "and model.warm_start_lora_path ignored; no adapters are loaded and no gradient updates "
        "are performed."
    ),
}

MODE_CAVEATS = {
    "warm_start_lora": "",
    "base": (
        "This uses get_model(..., lora.enabled=False). Any finetune-only modules absent from "
        "the base checkpoint keep the model API's normal initialization."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Eval-only Chromnitron HIC2 val/test performance without training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Modes: warm_start_lora evaluates the config base model plus the HIC2 LoRA from "
            "model.warm_start_lora_path. base evaluates the same pretrained base model with "
            "LoRA disabled and no adapters loaded; finetune-only modules not present in the "
            "base checkpoint retain the model API's normal initialization. Outputs are "
            "summary.json and metrics.csv in --output-dir."
        ),
    )
    parser.add_argument("--config", required=True, help="Path to the HIC2 finetune YAML config.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=sorted(MODE_DESCRIPTIONS),
        help="Zero-shot baseline mode to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where summary.json and metrics.csv will be written.",
    )
    parser.add_argument(
        "--splits",
        default="val,test",
        help="Comma-separated subset of val,test; also accepts both or all.",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Optional label recorded in summary.json and metrics.csv.",
    )
    return parser.parse_args()


def parse_splits(raw_splits: str) -> list[str]:
    normalized = raw_splits.strip().lower()
    if normalized in {"both", "all"}:
        return ["val", "test"]

    splits: list[str] = []
    for raw_split in normalized.split(","):
        split = raw_split.strip()
        if not split:
            continue
        if split not in {"val", "test"}:
            raise ValueError(
                f"Unsupported split {split!r}; expected val, test, val,test, both, or all."
            )
        if split not in splits:
            splits.append(split)

    if not splits:
        raise ValueError("--splits must include at least one of val or test.")
    return splits


def build_model_args(config: dict[str, Any], *, lora_enabled: bool) -> dict[str, Any]:
    model_cfg = config["model"]
    return {
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
            "enabled": lora_enabled,
            "r": model_cfg["lora_r"],
        },
    }


def count_requires_grad(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in unwrap_model(model).parameters() if param.requires_grad)


def freeze_for_eval(model: torch.nn.Module) -> dict[str, int]:
    before = count_requires_grad(model)
    for param in unwrap_model(model).parameters():
        param.requires_grad_(False)
    return {
        "parameters_requiring_grad_before_eval_freeze": before,
        "parameters_requiring_grad_after_eval_freeze": count_requires_grad(model),
    }


def build_base_model(
    config: dict[str, Any],
    device: torch.device,
    local_rank: int,
    distributed: bool,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    from chromnitron.training.finetuning.model.v4_5.chromnitron_models import get_model

    torch.cuda.set_device(local_rank)
    model = get_model(build_model_args(config, lora_enabled=False))
    load_info = disabled_warm_start_load_info(model)
    trainable_info = {
        "trainable_mode": "eval_base_no_lora",
        "lora_enabled": False,
        "total_trainable_parameters_before_eval_freeze": count_requires_grad(model),
        "checkpoint_includes_extra_trainable_parameters": False,
    }
    model = model.to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    return model, load_info, trainable_info


def build_eval_model(
    config: dict[str, Any],
    *,
    mode: str,
    device: torch.device,
    local_rank: int,
    distributed: bool,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    if mode == "warm_start_lora":
        warm_start_path = config["model"].get("warm_start_lora_path")
        if warm_start_path is None or str(warm_start_path).strip() == "":
            raise ValueError(
                "mode=warm_start_lora requires model.warm_start_lora_path in the config."
            )
        return build_model(config, device, local_rank, distributed)

    if mode == "base":
        base_config = deepcopy(config)
        base_config["model"]["warm_start_lora_path"] = ""
        return build_base_model(base_config, device, local_rank, distributed)

    raise ValueError(f"Unsupported mode: {mode}")


def run_eval_split(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    *,
    distributed: bool,
    split_name: str,
) -> dict[str, int | float]:
    confidence_weight = float(config["model"].get("confidence_weight", 0.0))
    no_confidence_prediction = bool(config["model"].get("no_confidence_prediction", False))
    loss_cfg = config.get("loss", {"name": "mse"})

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

    with torch.inference_mode():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            _loss, loss_value, batch_metrics = compute_loss(
                model,
                batch,
                confidence_weight,
                no_confidence_prediction,
                loss_cfg,
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

    metrics = finalize_epoch_metrics(aggregates, device, distributed)
    metrics["split"] = split_name
    return metrics


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "label",
        "mode",
        "split",
        "loss",
        "loss_valid_batches",
        "loss_total_batches",
        "pearson",
        "pearson_valid_samples",
        "pearson_total_samples",
        "pearson_nonfinite_samples",
        "pearson_constant_samples",
        "target_cap",
        "target_transform",
        "val_chrs",
        "test_chrs",
        "warm_start_status",
        "warm_start_checkpoint_path",
        "warm_start_loaded_key_count",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_metric_rows(
    *,
    label: str,
    mode: str,
    metrics_by_split: dict[str, dict[str, Any]],
    config: dict[str, Any],
    load_info: dict[str, Any],
) -> list[dict[str, Any]]:
    data_cfg = config.get("data", {})
    rows = []
    for split, metrics in metrics_by_split.items():
        rows.append(
            {
                "label": label,
                "mode": mode,
                "split": split,
                "loss": metrics["loss"],
                "loss_valid_batches": metrics["loss_valid_batches"],
                "loss_total_batches": metrics["loss_total_batches"],
                "pearson": metrics["pearson"],
                "pearson_valid_samples": metrics["pearson_valid_samples"],
                "pearson_total_samples": metrics["pearson_total_samples"],
                "pearson_nonfinite_samples": metrics["pearson_nonfinite_samples"],
                "pearson_constant_samples": metrics["pearson_constant_samples"],
                "target_cap": config["model"]["target_cap"],
                "target_transform": data_cfg.get("target_transform", "legacy_log1p"),
                "val_chrs": ",".join(data_cfg.get("val_chrs", ["chr10"])),
                "test_chrs": ",".join(data_cfg.get("test_chrs", ["chr20"])),
                "warm_start_status": load_info["warm_start_status"],
                "warm_start_checkpoint_path": load_info["warm_start_checkpoint_path"],
                "warm_start_loaded_key_count": load_info["warm_start_loaded_key_count"],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    splits = parse_splits(args.splits)
    label = args.label or args.mode

    config = load_yaml(config_path)
    add_import_paths()
    configure_genome_resources(config, config_path)

    distributed, rank, local_rank, world_size, device = setup_distributed()
    try:
        setup_tf32(bool(config["training"].get("tf32", True)))
        set_random_seed(int(config["training"].get("seed", 42)), rank)

        train_dataset, val_dataset, test_dataset, resources, manifest_path = build_datasets(
            config,
            config_path,
        )
        dataloaders, _samplers = build_dataloaders(
            train_dataset,
            val_dataset,
            test_dataset,
            config,
            distributed,
            world_size,
        )
        split_loaders = {
            "val": dataloaders["val"],
            "test": dataloaders["test"],
        }

        model, load_info, trainable_info = build_eval_model(
            config,
            mode=args.mode,
            device=device,
            local_rank=local_rank,
            distributed=distributed,
        )
        eval_freeze_info = freeze_for_eval(model)

        metrics_by_split: dict[str, dict[str, Any]] = {}
        for split in splits:
            metrics_by_split[split] = run_eval_split(
                model,
                split_loaders[split],
                device,
                config,
                distributed=distributed,
                split_name=split,
            )
            if is_rank_zero(rank):
                metrics = metrics_by_split[split]
                print(
                    f"mode={args.mode} split={split} "
                    f"loss={format_metric(float(metrics['loss']))} "
                    f"pearson={format_metric(float(metrics['pearson']))}",
                    flush=True,
                )

        maybe_barrier(distributed)
        if is_rank_zero(rank):
            output_dir.mkdir(parents=True, exist_ok=True)
            metrics_rows = build_metric_rows(
                label=label,
                mode=args.mode,
                metrics_by_split=metrics_by_split,
                config=config,
                load_info=load_info,
            )
            metrics_csv_path = output_dir / "metrics.csv"
            summary_json_path = output_dir / "summary.json"
            write_metrics_csv(metrics_csv_path, metrics_rows)

            summary = {
                "label": label,
                "mode": args.mode,
                "mode_description": MODE_DESCRIPTIONS[args.mode],
                "mode_caveat": MODE_CAVEATS[args.mode],
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "hostname": socket.gethostname(),
                "config_path": str(config_path),
                "output_dir": str(output_dir),
                "summary_json": str(summary_json_path),
                "metrics_csv": str(metrics_csv_path),
                "splits": splits,
                "target_cap": config["model"]["target_cap"],
                "target_transform": config.get("data", {}).get(
                    "target_transform",
                    "legacy_log1p",
                ),
                "val_chrs": config.get("data", {}).get("val_chrs", ["chr10"]),
                "test_chrs": config.get("data", {}).get("test_chrs", ["chr20"]),
                "base_model_path": config["model"]["base_model_path"],
                "configured_warm_start_lora_path": config["model"].get(
                    "warm_start_lora_path",
                    "",
                ),
                "effective_warm_start_lora_path": (
                    config["model"].get("warm_start_lora_path", "")
                    if args.mode == "warm_start_lora"
                    else ""
                ),
                "resources": resources,
                "manifest_path": str(manifest_path),
                "world_size": world_size,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "warm_start_load_info": load_info,
                "trainable_info": trainable_info,
                "eval_freeze_info": eval_freeze_info,
                "metrics": metrics_by_split,
            }
            with open(summary_json_path, "w", encoding="utf-8") as handle:
                json.dump(sanitize_for_json(summary), handle, indent=2)
            print(f"Wrote {summary_json_path}", flush=True)
            print(f"Wrote {metrics_csv_path}", flush=True)
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
