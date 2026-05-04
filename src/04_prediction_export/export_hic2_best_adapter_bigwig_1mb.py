#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
FINETUNE_CODE_DIR = REPO_ROOT / "src" / "03_finetuning"
for path in (FINETUNE_CODE_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

PROJECT_ROOT = REPO_ROOT
DEFAULT_INFERENCE_SCRIPT = PROJECT_ROOT / "chromnitron" / "inference" / "track_inference" / "inference.py"
WINDOW_SIZE_BP = 1_000_000
MODEL_WINDOW_SIZE_BP = 8_192
MODEL_STEP_SIZE_BP = 4_096
REQUIRED_CONFIG_FILES = ("infer.yaml", "data_config.yaml", "finetune_base.yaml")
# User-selected BCL11A hg38 gene-centered 1Mb export window.
DEFAULT_BCL11A_HG38_1MB_REGION = "chr2:60005424-61005424"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a 1Mb HIC2 prediction bigWig window from a completed finetune "
            "run's best_adapter.pt using the project-local manifest "
            "finetune dataset and model."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Completed run directory with checkpoints/best_adapter.pt or best_adapter.pt.",
    )
    parser.add_argument(
        "--config-root",
        required=True,
        help="Directory containing infer.yaml, data_config.yaml, and finetune_base.yaml.",
    )
    parser.add_argument("--celltype", required=True, help="Single cell type label for inference.")
    parser.add_argument(
        "--region",
        default=DEFAULT_BCL11A_HG38_1MB_REGION,
        help=(
            "Autosomal 1Mb region as chrN:start-end. "
            f"Default BCL11A hg38 1Mb window: {DEFAULT_BCL11A_HG38_1MB_REGION}."
        ),
    )
    parser.add_argument("--cap", default="HIC2", help="CAP name to export; default: HIC2.")
    parser.add_argument(
        "--assembly",
        default=None,
        help="Genome assembly. Defaults to run resolved_config.yaml data.assembly, then hg38.",
    )
    parser.add_argument(
        "--base-model-path",
        default=None,
        help="Base model checkpoint. Defaults to run resolved_config.yaml model.base_model_path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <run-dir>/hic2_1mb_bigwig.",
    )
    parser.add_argument(
        "--inference-script",
        default=str(DEFAULT_INFERENCE_SCRIPT),
        help="Compatibility option retained for older callers; direct HIC2 export does not call it.",
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        default=sys.executable,
        help="Python executable for inference. Defaults to the current interpreter.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacing helper-created files in the output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned command without creating files or running inference.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping at {path}.")
    return payload


def add_project_root_to_sys_path() -> None:
    project_root_str = str(PROJECT_ROOT)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


def resolve_target_transform_mode(*configs: dict[str, Any]) -> str:
    add_project_root_to_sys_path()
    from chromnitron.training.pretraining.data.transforms import (
        validate_target_transform_mode,
    )

    for config in configs:
        data_cfg = config.get("data", {})
        if isinstance(data_cfg, dict) and data_cfg.get("target_transform") is not None:
            return validate_target_transform_mode(data_cfg["target_transform"])
    return "log1p_clip"


def validate_config_root(config_root: Path) -> dict[str, Path]:
    missing = [name for name in REQUIRED_CONFIG_FILES if not (config_root / name).is_file()]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(f"Config root is missing required file(s): {missing_text}")
    return {name: config_root / name for name in REQUIRED_CONFIG_FILES}


def parse_region(region: str) -> tuple[str, int, int]:
    match = re.fullmatch(r"([^:]+):([0-9,]+)-([0-9,]+)", region.strip())
    if match is None:
        raise ValueError(
            "Expected --region in chrN:start-end format, for example chr10:1000000-2000000."
        )
    chrom = match.group(1)
    start = int(match.group(2).replace(",", ""))
    end = int(match.group(3).replace(",", ""))
    if end <= start:
        raise ValueError(f"Region end must be greater than start: {region}")
    if end - start != WINDOW_SIZE_BP:
        raise ValueError(
            f"Expected a {WINDOW_SIZE_BP} bp window, got {end - start} bp for {region}."
        )
    if not chrom.startswith("chr") or not chrom[3:].isdigit():
        raise ValueError(
            "The current track inference reader keeps only autosomal chrN regions; "
            f"got {chrom!r}."
        )
    return chrom, start, end


def path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def resolve_best_adapter_path(run_dir: Path, *, dry_run: bool) -> Path:
    checkpoint_path = run_dir / "checkpoints" / "best_adapter.pt"
    root_path = run_dir / "best_adapter.pt"
    if checkpoint_path.is_file():
        return checkpoint_path
    if root_path.is_file():
        return root_path
    if not dry_run:
        raise FileNotFoundError(
            "Missing best adapter checkpoint; checked "
            f"{checkpoint_path} and {root_path}."
        )
    return checkpoint_path


def write_text_file(path: Path, content: str, *, force: bool) -> None:
    if path.exists():
        old_content = path.read_text(encoding="utf-8")
        if old_content == content:
            return
        if not force:
            raise FileExistsError(f"Refusing to overwrite existing file without --force: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def ensure_symlink(path: Path, target: Path, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        try:
            if path.resolve(strict=True) == target.resolve(strict=True):
                return
        except FileNotFoundError:
            pass
        if not force:
            raise FileExistsError(f"Refusing to replace existing path without --force: {path}")
        if path.is_dir() and not path.is_symlink():
            raise IsADirectoryError(f"Refusing to replace directory: {path}")
        path.unlink()
    path.symlink_to(target)


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def clear_generated_prediction_outputs(output_dir: Path, *, force: bool) -> None:
    prediction_dir = output_dir / "prediction"
    if not force or not (prediction_dir.exists() or prediction_dir.is_symlink()):
        return
    remove_path(prediction_dir)


class InferenceRegion:
    def __init__(self, windows: list[tuple[str, int, int, str]]) -> None:
        self.loci = np.array(windows, dtype=object)

    def __len__(self) -> int:
        return len(self.loci)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.loci[idx]


def resolve_path(base_dir: Path, raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def resource_root(input_dict: dict[str, Any], key: str) -> str:
    return str(input_dict.get(f"{key}_root", input_dict.get("root", "")))


def resolve_static_resources(config: dict[str, Any], config_path: Path) -> dict[str, Path]:
    code_cfg = config.get("code", {})
    atlas_config_path = resolve_path(config_path.parent, code_cfg.get("atlas_config_path"))
    if atlas_config_path is None:
        raise ValueError("code.atlas_config_path is required in finetune_base.yaml.")
    if not atlas_config_path.is_file():
        raise FileNotFoundError(f"Atlas config does not exist: {atlas_config_path}")

    atlas_cfg = load_yaml(atlas_config_path)
    input_cfg = atlas_cfg["input_resource"]
    data_cfg = config["data"]
    assembly = data_cfg["assembly"]
    target_cap = config["model"].get("target_cap", "HIC2")

    seq_root = Path(resource_root(input_cfg, "sequence")) / input_cfg.get("sequence", "")
    cap_root = Path(resource_root(input_cfg, "cap")) / input_cfg.get("cap", "")
    input_seq_path = seq_root / f"{assembly}.zarr"
    esm_feature_path = cap_root / f"{target_cap}.npz"

    excluded_region_raw = data_cfg.get("excluded_region_path", "auto")
    if excluded_region_raw == "auto":
        excluded_region_path = seq_root / f"{assembly}-blacklist.v2.bed"
    else:
        excluded_region_path = Path(str(excluded_region_raw)).expanduser()
        if not excluded_region_path.is_absolute():
            excluded_region_path = (config_path.parent / excluded_region_path).resolve()

    return {
        "atlas_config_path": atlas_config_path,
        "input_seq_path": input_seq_path,
        "esm_feature_path": esm_feature_path,
        "excluded_region_path": excluded_region_path,
    }


def resolve_manifest_path(config: dict[str, Any], config_path: Path) -> Path:
    manifest_path = resolve_path(config_path.parent, config["data"].get("manifest_path"))
    if manifest_path is None:
        raise ValueError("data.manifest_path is required in finetune_base.yaml.")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    return manifest_path


def merge_model_config(
    export_config: dict[str, Any],
    run_config: dict[str, Any],
    *,
    base_model_path: Path,
) -> dict[str, Any]:
    model_cfg = dict(export_config.get("model", {}))
    model_cfg.update(run_config.get("model", {}))
    model_cfg["base_model_path"] = str(base_model_path)
    return model_cfg


def build_model_args(model_cfg: dict[str, Any]) -> dict[str, Any]:
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
            "enabled": True,
            "r": int(model_cfg["lora_r"]),
        },
    }


def load_checkpoint_state_dict(path: Path) -> dict[str, Any]:
    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location="cpu")
    if isinstance(state_dict, dict) and "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected dict-like checkpoint at {path}, got {type(state_dict)!r}")
    return state_dict


def load_adapter_checkpoint(model: torch.nn.Module, adapter_path: Path) -> None:
    state_dict = load_checkpoint_state_dict(adapter_path)
    model_state = model.state_dict()
    compatible_state: dict[str, Any] = {}
    mismatches: list[str] = []
    unexpected: list[str] = []

    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        if key not in model_state:
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            mismatches.append(
                f"{key} checkpoint_shape={list(value.shape)} "
                f"model_shape={list(model_state[key].shape)}"
            )
            continue
        compatible_state[key] = value

    if unexpected or mismatches:
        preview = "; ".join((unexpected[:3] + mismatches[:3])[:6])
        raise ValueError(f"Adapter checkpoint does not match model: {preview}")
    if not compatible_state:
        raise ValueError(f"Adapter checkpoint has no tensor keys loadable by the model: {adapter_path}")

    model.load_state_dict(compatible_state, strict=False)


def make_sliding_windows(
    chrom: str,
    start: int,
    end: int,
    *,
    window_size: int = MODEL_WINDOW_SIZE_BP,
    step_size: int = MODEL_STEP_SIZE_BP,
) -> list[tuple[str, int, int, str]]:
    starts = list(range(start, max(start, end - window_size + 1), step_size))
    last_start = end - window_size
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return [
        (chrom, int(window_start), int(window_start + window_size), f"region_{idx}")
        for idx, window_start in enumerate(starts)
    ]


def configure_dataset_for_region(
    dataset: Any,
    *,
    celltype: str,
    windows: list[tuple[str, int, int, str]],
) -> None:
    selected_indices = [
        idx
        for idx, metadata in dataset.metadata.items()
        if str(metadata.get("sample_id", "")) == celltype
    ]
    if not selected_indices:
        known = ", ".join(str(metadata.get("sample_id", "")) for metadata in dataset.metadata.values())
        raise ValueError(f"Celltype/sample {celltype!r} not found in manifest. Known values: {known}")

    dataset.genomes = [dataset.genomes[idx] for idx in selected_indices]
    dataset.metadata = {
        new_idx: dataset.metadata[old_idx] for new_idx, old_idx in enumerate(selected_indices)
    }
    region = InferenceRegion(windows)
    for new_idx, genome in enumerate(dataset.genomes):
        genome.metadata_key = new_idx
        genome.chunk_size = MODEL_WINDOW_SIZE_BP
        genome.step_size = MODEL_STEP_SIZE_BP
        genome.sample_per_chunk = 1
        genome.region = region


def build_manifest_inference_dataset(
    config: dict[str, Any],
    config_path: Path,
    *,
    celltype: str,
    windows: list[tuple[str, int, int, str]],
) -> Any:
    add_project_root_to_sys_path()
    from finetune_dataset import ManifestFinetuneDataset

    resources = resolve_static_resources(config, config_path)
    manifest_path = resolve_manifest_path(config, config_path)
    data_cfg = config["data"]

    for label, path in resources.items():
        if label == "atlas_config_path":
            continue
        if not path.exists():
            raise FileNotFoundError(f"Required inference resource does not exist: {label}={path}")

    dataset = ManifestFinetuneDataset(
        manifest_path=manifest_path,
        input_seq_path=str(resources["input_seq_path"]),
        esm_feature_path=str(resources["esm_feature_path"]),
        target_cap=str(config["model"].get("target_cap", "HIC2")),
        mode="val",
        excluded_region_file=str(resources["excluded_region_path"]),
        val_chrs=data_cfg.get("val_chrs", ["chr10"]),
        test_chrs=data_cfg.get("test_chrs", ["chr20"]),
        assembly=data_cfg.get("assembly", "hg38"),
        chunk_size=MODEL_WINDOW_SIZE_BP,
        sample_per_chunk=1,
        window_size=data_cfg.get("window_size", MODEL_WINDOW_SIZE_BP),
        atac_log1p=data_cfg.get("atac_log1p", True),
        target_transform=data_cfg.get("target_transform"),
        apply_reverse_complement=False,
        apply_gaussian_noise=False,
        atac_dropout={"enabled": False},
        cap_embedding_key=data_cfg.get("cap_embedding_key", "embedding"),
        verbose=data_cfg.get("verbose", False),
    )
    configure_dataset_for_region(dataset, celltype=celltype, windows=windows)
    return dataset


def run_window_predictions(
    *,
    dataset: Any,
    model_cfg: dict[str, Any],
    adapter_path: Path,
    target_transform: str,
) -> np.ndarray:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for HIC2 bigWig export inference.")

    add_project_root_to_sys_path()
    from chromnitron.training.pretraining.data.transforms import (
        inverse_transform_target_features,
    )
    from chromnitron.training.finetuning.model.v4_5.chromnitron_models import get_model

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model = get_model(build_model_args(model_cfg))
    load_adapter_checkpoint(model, adapter_path)
    model.sample_per_chunk = 1
    model = model.to(device)
    model.eval()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for seq, input_features, _target_features, esm_embeddings in dataloader:
            seq = seq.to(device, non_blocking=True).float()
            input_features = input_features.to(device, non_blocking=True).float()
            esm_embeddings = esm_embeddings.to(device, non_blocking=True).float().transpose(-1, -2)

            batch_size, mini_bs, seq_len, seq_dim = seq.shape
            seq = seq.view(batch_size * mini_bs, seq_len, seq_dim).transpose(1, 2)
            input_features = input_features.view(batch_size * mini_bs, -1).unsqueeze(2).transpose(1, 2)

            outputs = model((seq, input_features), esm_embeddings)
            preds = outputs if model_cfg.get("no_confidence_prediction", False) else outputs[0]
            preds_np = inverse_transform_target_features(
                preds.detach().cpu().float().numpy(),
                target_transform,
            )
            predictions.extend(preds_np[:, 0, :])

    return np.stack(predictions, axis=0).astype(np.float32)


def merge_window_predictions(
    predictions: np.ndarray,
    windows: list[tuple[str, int, int, str]],
    *,
    region_start: int,
    region_end: int,
) -> np.ndarray:
    merged = np.zeros(region_end - region_start, dtype=np.float64)
    weights = np.zeros(region_end - region_start, dtype=np.float64)
    for pred, (_chrom, window_start, window_end, _region_id) in zip(predictions, windows):
        clip_start = max(region_start, window_start)
        clip_end = min(region_end, window_end)
        if clip_end <= clip_start:
            continue
        pred_start = clip_start - window_start
        pred_end = pred_start + (clip_end - clip_start)
        merged_start = clip_start - region_start
        merged_end = clip_end - region_start
        merged[merged_start:merged_end] += pred[pred_start:pred_end]
        weights[merged_start:merged_end] += 1.0

    missing = weights == 0
    if np.any(missing):
        raise RuntimeError(f"Sliding windows left {int(missing.sum())} bp uncovered.")
    return (merged / weights).astype(np.float32)


def read_chrom_size(chrom_sizes_path: Path | None, chrom: str, fallback_size: int) -> int:
    if chrom_sizes_path is not None and chrom_sizes_path.is_file():
        with open(chrom_sizes_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split()
                if len(parts) < 2 or parts[0] != chrom:
                    continue
                if len(parts) >= 3:
                    return int(parts[2])
                return int(parts[1])
    return fallback_size


def write_bigwig(
    output_path: Path,
    *,
    chrom: str,
    chrom_size: int,
    region_start: int,
    values: np.ndarray,
) -> None:
    import pyBigWig

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pyBigWig.open(str(output_path), "w") as bigwig:
        bigwig.addHeader([(chrom, int(chrom_size))])
        chunk_size = 100_000
        for offset in range(0, len(values), chunk_size):
            chunk_values = values[offset : offset + chunk_size]
            starts = list(range(region_start + offset, region_start + offset + len(chunk_values)))
            ends = [start + 1 for start in starts]
            bigwig.addEntries(
                [chrom] * len(starts),
                starts,
                ends=ends,
                values=[float(value) for value in chunk_values],
            )


def export_project_local_bigwig(
    *,
    config_paths: dict[str, Path],
    run_config: dict[str, Any],
    base_model_path: Path,
    adapter_path: Path,
    target_transform: str,
    output_dir: Path,
    celltype: str,
    cap: str,
    chrom: str,
    start: int,
    end: int,
) -> dict[str, Any]:
    export_config = load_yaml(config_paths["finetune_base.yaml"])
    export_config.setdefault("model", {})["target_cap"] = cap
    export_config.setdefault("data", {})["target_transform"] = target_transform
    model_cfg = merge_model_config(export_config, run_config, base_model_path=base_model_path)
    windows = make_sliding_windows(chrom, start, end)
    dataset = build_manifest_inference_dataset(
        export_config,
        config_paths["finetune_base.yaml"],
        celltype=celltype,
        windows=windows,
    )
    predictions = run_window_predictions(
        dataset=dataset,
        model_cfg=model_cfg,
        adapter_path=adapter_path,
        target_transform=target_transform,
    )
    merged_prediction = merge_window_predictions(
        predictions,
        windows,
        region_start=start,
        region_end=end,
    )

    prediction_dir = output_dir / "prediction" / "prediction" / celltype / cap
    bigwig_path = prediction_dir / "bigwigs" / f"{cap}_{celltype}_prediction.bw"
    chrom_sizes_path = resolve_path(
        config_paths["finetune_base.yaml"].parent,
        export_config.get("data", {}).get("chrom_sizes_path"),
    )
    chrom_size = read_chrom_size(chrom_sizes_path, chrom, fallback_size=end)

    prediction_dir.mkdir(parents=True, exist_ok=True)
    np.save(prediction_dir / "data.npy", predictions[:, np.newaxis, :])
    np.save(prediction_dir / "feature_set.npy", np.array([cap]))
    np.save(
        prediction_dir / "loci_data.npy",
        np.array([[window[0], window[1], window[2]] for window in windows], dtype=object),
    )
    write_bigwig(
        bigwig_path,
        chrom=chrom,
        chrom_size=chrom_size,
        region_start=start,
        values=merged_prediction,
    )

    return {
        "backend": "project_local_manifest_bigwig",
        "bigwig_path": str(bigwig_path),
        "chrom_size": chrom_size,
        "window_size": MODEL_WINDOW_SIZE_BP,
        "step_size": MODEL_STEP_SIZE_BP,
        "window_count": len(windows),
        "target_transform": target_transform,
    }


def stage_config_root(
    *,
    source_config_paths: dict[str, Path],
    staged_config_root: Path,
    model_path: Path,
    force: bool,
) -> None:
    infer_config = load_yaml(source_config_paths["infer.yaml"])
    infer_config["model_path"] = str(model_path)
    staged_config_root.mkdir(parents=True, exist_ok=True)
    write_text_file(
        staged_config_root / "infer.yaml",
        yaml.safe_dump(infer_config, sort_keys=False),
        force=force,
    )
    for file_name in ("data_config.yaml", "finetune_base.yaml"):
        write_text_file(
            staged_config_root / file_name,
            source_config_paths[file_name].read_text(encoding="utf-8"),
            force=force,
        )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    config_root = Path(args.config_root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "hic2_1mb_bigwig"
    )
    chrom, start, end = parse_region(args.region)

    if not run_dir.is_dir():
        if not args.dry_run:
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if not config_root.is_dir():
        raise FileNotFoundError(f"Config root does not exist: {config_root}")
    config_paths = validate_config_root(config_root)
    if not path_is_within(output_dir, run_dir):
        raise ValueError(f"Output directory must be inside run-dir: {output_dir}")

    best_adapter_path = resolve_best_adapter_path(run_dir, dry_run=args.dry_run)

    run_config: dict[str, Any] = {}
    resolved_config_path = run_dir / "resolved_config.yaml"
    has_resolved_config = resolved_config_path.is_file()
    if has_resolved_config:
        run_config = load_yaml(resolved_config_path)
    export_config = load_yaml(config_paths["finetune_base.yaml"])
    if has_resolved_config:
        target_transform = resolve_target_transform_mode(run_config)
    else:
        target_transform = resolve_target_transform_mode(export_config)

    base_model_raw = (
        args.base_model_path
        if args.base_model_path is not None
        else run_config.get("model", {}).get(
            "base_model_path",
            export_config.get("model", {}).get("base_model_path"),
        )
    )
    if not base_model_raw:
        raise ValueError(
            "--base-model-path is required when run resolved_config.yaml and "
            "config-root finetune_base.yaml are missing model.base_model_path."
        )
    base_model_path = Path(str(base_model_raw)).expanduser().resolve()
    if not base_model_path.is_file() and not args.dry_run:
        raise FileNotFoundError(f"Base model checkpoint does not exist: {base_model_path}")

    assembly = args.assembly or str(
        run_config.get("data", {}).get(
            "assembly",
            export_config.get("data", {}).get("assembly", "hg38"),
        )
    )
    model_root = output_dir / "model_root"
    inputs_dir = output_dir / "inputs"
    staged_config_root = inputs_dir / "config_root"
    cap_model_path = model_root / "CAPs" / f"{args.cap}.pt"
    base_model_link = model_root / "chromnitron_base.pt"
    celltype_list = inputs_dir / "celltype.txt"
    cap_list = inputs_dir / "cap.txt"
    region_bed = inputs_dir / "region_1mb.bed"
    manifest_path = output_dir / "export_manifest.json"

    command = [
        args.python_executable,
        str(Path(__file__).resolve()),
        "--run-dir",
        str(run_dir),
        "--config-root",
        str(config_root),
        "--celltype",
        args.celltype,
        "--region",
        args.region,
        "--cap",
        args.cap,
        "--assembly",
        assembly,
    ]
    if args.force:
        command.append("--force")
    if args.base_model_path is not None:
        command.extend(["--base-model-path", args.base_model_path])
    if args.output_dir is not None:
        command.extend(["--output-dir", args.output_dir])

    manifest = {
        "run_dir": str(run_dir),
        "best_adapter_path": str(best_adapter_path),
        "base_model_path": str(base_model_path),
        "source_config_root": str(config_root),
        "config_root": str(staged_config_root),
        "output_dir": str(output_dir),
        "model_root": str(model_root),
        "staged_model_path": str(cap_model_path),
        "cap": args.cap,
        "celltype": args.celltype,
        "assembly": assembly,
        "target_transform": target_transform,
        "region": {"chrom": chrom, "start": start, "end": end},
        "bigwig_glob": str(output_dir / "prediction" / "**" / "bigwigs" / "*.bw"),
        "command": command,
        "backend": "project_local_manifest_bigwig",
    }

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    clear_generated_prediction_outputs(output_dir, force=args.force)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_symlink(base_model_link, base_model_path, force=args.force)
    ensure_symlink(cap_model_path, best_adapter_path, force=args.force)
    stage_config_root(
        source_config_paths=config_paths,
        staged_config_root=staged_config_root,
        model_path=cap_model_path,
        force=args.force,
    )
    write_text_file(celltype_list, f"{args.celltype}\n", force=args.force)
    write_text_file(cap_list, f"{args.cap}\n", force=args.force)
    write_text_file(region_bed, f"{chrom}\t{start}\t{end}\n", force=args.force)

    export_result = export_project_local_bigwig(
        config_paths=config_paths,
        run_config=run_config,
        base_model_path=base_model_path,
        adapter_path=best_adapter_path,
        target_transform=target_transform,
        output_dir=output_dir,
        celltype=args.celltype,
        cap=args.cap,
        chrom=chrom,
        start=start,
        end=end,
    )
    manifest.update(export_result)
    write_text_file(manifest_path, json.dumps(manifest, indent=2) + "\n", force=args.force)

    bigwigs = sorted(output_dir.glob("prediction/**/bigwigs/*.bw"))
    if bigwigs:
        print("Generated bigWig files:")
        for path in bigwigs:
            print(path)
    else:
        raise RuntimeError(
            f"Inference finished, but no bigWig files matched {manifest['bigwig_glob']}."
        )


if __name__ == "__main__":
    main()
