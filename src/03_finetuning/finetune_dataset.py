from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from chromnitron.training.infra.origami.base.storages import ZarrStorage
from chromnitron.training.infra.origami.track.tracks import Track
from chromnitron.training.pretraining.data.dataset import (
    GenomeMergedZarrChunkOptimizedDataset,
)
import chromnitron.training.pretraining.data.transforms as transforms


REQUIRED_MANIFEST_COLUMNS = {"sample_id", "atac_path", "target_path"}


def read_manifest(manifest_path: str | Path) -> pd.DataFrame:
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest_df = pd.read_csv(manifest_path)
    missing = REQUIRED_MANIFEST_COLUMNS.difference(manifest_df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Manifest is missing required columns: {missing_str}")
    if manifest_df.empty:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return manifest_df


class DirectTrackChunkedGenomeDataset(GenomeMergedZarrChunkOptimizedDataset):
    def __init__(
        self,
        input_seq_path: str,
        input_features_path: str,
        target_zarr_path: str,
        mode: str,
        excluded_region_file: str,
        *,
        val_chrs: list[str] | None = None,
        test_chrs: list[str] | None = None,
        assembly: str = "hg38",
        chunk_size: int = 100000,
        sample_per_chunk: int = 4,
        window_size: int = 8192,
        atac_log1p: bool = True,
        target_transform: str | None = None,
        apply_reverse_complement: bool = True,
        apply_gaussian_noise: bool = True,
        atac_dropout: dict[str, Any] | None = None,
        verbose: bool = False,
        metadata_key: int | None = None,
    ) -> None:
        self.target_zarr_path = target_zarr_path
        self.apply_reverse_complement = apply_reverse_complement
        self.apply_gaussian_noise = apply_gaussian_noise
        self.atac_dropout = atac_dropout or {}
        self.target_transform = (
            transforms.validate_target_transform_mode(target_transform)
            if target_transform is not None
            else None
        )
        super().__init__(
            input_seq_path=input_seq_path,
            input_features_path=input_features_path,
            target_features_indices=[0],
            target_meta_data=None,
            target_zarr_path=target_zarr_path,
            mode=mode,
            excluded_region_file=excluded_region_file,
            val_chrs=val_chrs or ["chr10"],
            test_chrs=test_chrs or ["chr20"],
            assembly=assembly,
            chunk_size=chunk_size,
            sample_per_chunk=sample_per_chunk,
            window_size=window_size,
            verbose=verbose,
            metadata_key=metadata_key,
            atac_log1p=atac_log1p,
        )
        self.target_mask = np.array([True], dtype=bool)

    def load_data(
        self,
        input_seq_path: str,
        input_features_path: str,
        _target_features_path: list[int],
        assembly: str,
        _verbose: bool,
    ) -> dict[str, Any]:
        data_dict = {
            "seq": Track(ZarrStorage(input_seq_path, assembly)),
            "input_features": self.load_storage_with_paths(
                assembly, "input_features", input_features_path
            ),
            "target_features": Track(ZarrStorage(self.target_zarr_path, assembly)),
        }
        return data_dict

    def get_features_target(
        self,
        _features: Any,
        chrom: str,
        start: int,
        end: int,
    ) -> list[np.ndarray]:
        stored_features = self.data["target_features"].get(chrom, start, end)
        return [np.nan_to_num(stored_features.astype(np.float32))]

    def __getitem__(self, idx: int) -> tuple[np.ndarray, ...]:
        chrom, start_str, end_str, region_id = self.region[idx]
        start, end = int(start_str), int(end_str)
        seq, input_features, target_features, other_features = self.get_data(chrom, start, end)

        if self.aug_bool and self.apply_reverse_complement:
            seq, input_features, target_features = transforms.reverse_features(
                seq, input_features, target_features
            )
        if self.aug_bool and self.apply_gaussian_noise:
            seq, input_features, target_features = transforms.add_gaussian_noise(
                seq, input_features, target_features
            )

        if self.atac_log1p:
            input_features = transforms.log1p_clip_negative(input_features)
            if self.target_transform is None:
                target_features = transforms.log1p_clip_negative(target_features)
        else:
            if self.target_transform is None:
                target_features = transforms.log1p_clip_negative(target_features)
        if self.target_transform is not None:
            target_features = transforms.transform_target_features(
                target_features, self.target_transform
            )

        region_starts, region_ends = self.get_half_overlapping_regions(
            0, seq.shape[0], self.sample_per_chunk
        )
        seq_batch = []
        input_features_batch = []
        target_features_batch = []
        for region_start, region_end in zip(region_starts, region_ends):
            start_aug, end_aug = transforms.subsample_locus(
                self.real_window_size, region_start, region_end, self.aug_bool
            )
            seq_batch.append(seq[start_aug:end_aug])
            input_features_batch.append(input_features[start_aug:end_aug])
            target_features_batch.append(
                [feature[start_aug:end_aug] for feature in target_features]
            )

        seq_batch = np.stack(seq_batch).astype(np.float32)
        input_features_batch = np.stack(input_features_batch).astype(np.float32)
        input_features_batch = self._apply_atac_dropout(input_features_batch)
        target_features_batch = np.stack(target_features_batch).astype(np.float32)

        target_mask_batch = np.stack([self.target_mask for _ in range(self.sample_per_chunk)])
        start_batch = np.array([start] * self.sample_per_chunk)
        end_batch = np.array([end] * self.sample_per_chunk)
        chrom_batch = np.array([int(chrom.split("chr")[1])] * self.sample_per_chunk)
        region_id_batch = np.array([int(region_id.split("region_")[1])] * self.sample_per_chunk)
        metadata_key_batch = np.array([self.metadata_key] * self.sample_per_chunk)

        info_batch = (
            start_batch,
            end_batch,
            chrom_batch,
            region_id_batch,
            target_mask_batch,
            metadata_key_batch,
        )
        return seq_batch, input_features_batch, target_features_batch, info_batch

    def _apply_atac_dropout(self, input_features_batch: np.ndarray) -> np.ndarray:
        if self.mode != "train" or not self.atac_dropout.get("enabled", False):
            return input_features_batch

        mask_fraction = float(self.atac_dropout.get("mask_fraction", 0.1))
        span_length = max(1, int(self.atac_dropout.get("span_length", 32)))
        max_spans = int(self.atac_dropout.get("max_spans", 0))
        if mask_fraction <= 0:
            return input_features_batch

        seq_len = input_features_batch.shape[-1]
        target_masked_bp = max(1, int(round(seq_len * mask_fraction)))
        if max_spans <= 0:
            max_spans = max(1, int(np.ceil(target_masked_bp / span_length)))

        for sample_idx in range(input_features_batch.shape[0]):
            keep_mask = np.ones(seq_len, dtype=np.float32)
            masked_bp = 0
            attempts = 0
            while masked_bp < target_masked_bp and attempts < max_spans * 10:
                cur_span = span_length
                start = np.random.randint(0, max(1, seq_len - cur_span + 1))
                end = min(seq_len, start + cur_span)
                keep_mask[start:end] = 0.0
                masked_bp = int(seq_len - keep_mask.sum())
                attempts += 1
            input_features_batch[sample_idx] *= keep_mask
        return input_features_batch


class ManifestFinetuneDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        input_seq_path: str,
        esm_feature_path: str,
        target_cap: str,
        mode: str,
        excluded_region_file: str,
        val_chrs: list[str] | None = None,
        test_chrs: list[str] | None = None,
        assembly: str = "hg38",
        chunk_size: int = 100000,
        sample_per_chunk: int = 4,
        window_size: int = 8192,
        atac_log1p: bool = True,
        target_transform: str | None = None,
        apply_reverse_complement: bool = True,
        apply_gaussian_noise: bool = True,
        atac_dropout: dict[str, Any] | None = None,
        cap_embedding_key: str = "embedding",
        verbose: bool = False,
    ) -> None:
        manifest_df = read_manifest(manifest_path)
        embedding_npz = np.load(Path(esm_feature_path).expanduser().resolve())
        if cap_embedding_key not in embedding_npz:
            raise KeyError(
                f"Embedding key '{cap_embedding_key}' not found in {esm_feature_path}"
            )
        embedding = np.asarray(embedding_npz[cap_embedding_key], dtype=np.float32)
        if embedding.ndim != 2:
            raise ValueError(
                f"Expected 2D CAP embedding array, got shape {embedding.shape}"
            )

        self.esm_embedding = torch.tensor(embedding[np.newaxis, :, :], dtype=torch.float32)
        self.target_cap = target_cap
        self.targets = [target_cap]
        self.num_targets = 1
        self.metadata = {}
        self.genomes = []

        for idx, row in manifest_df.reset_index(drop=True).iterrows():
            sample_id = str(row["sample_id"])
            genome = DirectTrackChunkedGenomeDataset(
                input_seq_path=input_seq_path,
                input_features_path=str(row["atac_path"]),
                target_zarr_path=str(row["target_path"]),
                mode=mode,
                excluded_region_file=excluded_region_file,
                val_chrs=val_chrs,
                test_chrs=test_chrs,
                assembly=assembly,
                chunk_size=chunk_size,
                sample_per_chunk=sample_per_chunk,
                window_size=window_size,
                atac_log1p=atac_log1p,
                target_transform=target_transform,
                apply_reverse_complement=apply_reverse_complement,
                apply_gaussian_noise=apply_gaussian_noise,
                atac_dropout=atac_dropout,
                verbose=verbose,
                metadata_key=idx,
            )
            self.metadata[idx] = {
                "sample_id": sample_id,
                "target": [target_cap],
                "atac_path": str(row["atac_path"]),
                "target_path": str(row["target_path"]),
            }
            self.genomes.append(genome)

    def __len__(self) -> int:
        return sum(len(genome) for genome in self.genomes)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
        genome_idx = 0
        while idx >= len(self.genomes[genome_idx]):
            idx -= len(self.genomes[genome_idx])
            genome_idx += 1
        seq, input_features, target_features, _metadata = self.genomes[genome_idx][idx]
        return (
            seq.astype(np.float32),
            input_features.astype(np.float32),
            target_features.astype(np.float32),
            self.esm_embedding,
        )
