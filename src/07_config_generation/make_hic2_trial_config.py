from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a short HIC2 finetune trial config.")
    parser.add_argument("--base-config", required=True, help="Base YAML config to copy.")
    parser.add_argument("--output", required=True, help="Output YAML config path.")
    parser.add_argument("--run-name", required=True, help="experiment.run_name value.")
    parser.add_argument("--save-root", default=None, help="Optional experiment.save_root override.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None, help="Optional training.seed override.")
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--lora-r", type=int, required=True)
    parser.add_argument(
        "--trainable-mode",
        choices=["lora", "lora_plus_head", "lora_plus_last_block"],
        default=None,
        help="Optional model.trainable_mode override.",
    )
    parser.add_argument(
        "--warm-start-lora-path",
        default=None,
        help="Optional model.warm_start_lora_path override.",
    )
    parser.add_argument(
        "--clear-warm-start-lora-path",
        action="store_true",
        help="Set model.warm_start_lora_path to an empty string for scratch LoRA.",
    )
    parser.add_argument(
        "--pearson-weight",
        type=float,
        default=None,
        help=(
            "Optional loss.pearson_weight value for "
            "loss.name=weighted_multiscale_mse_pearson configs."
        ),
    )
    parser.add_argument(
        "--loss-json",
        default=None,
        help="Optional JSON object for the top-level loss config.",
    )
    parser.add_argument(
        "--target-transform",
        choices=["log1p_clip", "clip", "gamma2"],
        default=None,
        help="Optional data.target_transform override.",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def reanchor_manifest_path(
    config: dict[str, Any],
    *,
    base_config_path: Path,
    output_path: Path,
) -> None:
    data_cfg = config.get("data")
    if not isinstance(data_cfg, dict):
        return

    manifest_path = data_cfg.get("manifest_path")
    if not manifest_path:
        return

    manifest_path_obj = Path(manifest_path).expanduser()
    if manifest_path_obj.is_absolute():
        resolved_manifest = manifest_path_obj
    else:
        resolved_manifest = (base_config_path.parent / manifest_path_obj).resolve()
    data_cfg["manifest_path"] = os.path.relpath(resolved_manifest, start=output_path.parent)


def main() -> None:
    args = parse_args()
    if args.clear_warm_start_lora_path and args.warm_start_lora_path is not None:
        raise ValueError(
            "--clear-warm-start-lora-path cannot be combined with --warm-start-lora-path."
        )

    base_config_path = Path(args.base_config).resolve()
    output_path = Path(args.output).expanduser().resolve()
    config = load_yaml(base_config_path)

    config.setdefault("experiment", {})["run_name"] = args.run_name
    if args.save_root is not None:
        config["experiment"]["save_root"] = args.save_root

    if args.target_transform is not None:
        config.setdefault("data", {})["target_transform"] = args.target_transform

    model_cfg = config.setdefault("model", {})
    model_cfg["lora_r"] = args.lora_r
    if args.trainable_mode is not None:
        model_cfg["trainable_mode"] = args.trainable_mode
    if args.clear_warm_start_lora_path:
        model_cfg["warm_start_lora_path"] = ""
    elif args.warm_start_lora_path is not None:
        model_cfg["warm_start_lora_path"] = args.warm_start_lora_path

    training_cfg = config.setdefault("training", {})
    training_cfg["epochs"] = args.epochs
    training_cfg["early_stopping_patience"] = args.early_stopping_patience
    if args.seed is not None:
        training_cfg["seed"] = args.seed
    training_cfg["learning_rate"] = args.learning_rate

    dropout_cfg = config.setdefault("atac_dropout", {})
    dropout_cfg["enabled"] = False
    dropout_cfg["mask_fraction"] = 0.0
    dropout_cfg["max_spans"] = 0

    if args.loss_json:
        loss_cfg = json.loads(args.loss_json)
        if not isinstance(loss_cfg, dict):
            raise TypeError("--loss-json must decode to a JSON object.")
        config["loss"] = loss_cfg
    loss_cfg = config.get("loss")
    if loss_cfg is None:
        loss_cfg = config.setdefault("loss", {})
    if not isinstance(loss_cfg, dict):
        raise TypeError("Top-level loss config must be a mapping.")
    if (
        args.pearson_weight is not None
        and loss_cfg.get("name") == "weighted_multiscale_mse_pearson"
    ):
        loss_cfg["pearson_weight"] = args.pearson_weight

    reanchor_manifest_path(config, base_config_path=base_config_path, output_path=output_path)
    save_yaml(output_path, config)


if __name__ == "__main__":
    main()
