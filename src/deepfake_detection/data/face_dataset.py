"""Manifest-backed dataset of RetinaFace crops for spatial training."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def validate_manifest(path: Path) -> list[dict[str, str]]:
    """Load and validate a source-aware video manifest before training."""
    required = {"video_path", "label", "source_id", "split"}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    if not required.issubset(rows[0]):
        raise ValueError(f"Manifest needs columns: {sorted(required)}")
    group_splits: dict[str, set[str]] = defaultdict(set)
    split_labels: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        if not row["source_id"].strip() or row["split"] not in {"train", "validation", "test"}:
            raise ValueError("Manifest has empty source IDs or non-final split values")
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError(f"Expected binary label 0/1, received {label}")
        group_splits[row["source_id"]].add(row["split"])
        split_labels[row["split"]].add(label)
    crossing = [group for group, splits in group_splits.items() if len(splits) != 1]
    if crossing:
        raise ValueError(f"Source groups cross splits ({len(crossing)} groups)")
    for split in ("train", "validation", "test"):
        if split_labels.get(split) != {0, 1}:
            raise ValueError(f"Split {split!r} must contain both labels; got {split_labels.get(split)}")
    return rows


class FaceCropVideoDataset(Dataset[tuple[Tensor, Tensor, str]]):
    """Returns fixed-length ``(T, 3, 224, 224)`` normalized face crops per video.

    ``crop_index.csv`` is emitted by ``prepare_faces.py``. Videos with no
    successful face crops are excluded; short clips are deterministically
    padded by repeating their final crop so default collation stays simple.
    """

    def __init__(self, manifest: Path, crop_index: Path, split: str, frames_per_video: int = 10):
        if frames_per_video < 1:
            raise ValueError("frames_per_video must be positive")
        rows = validate_manifest(manifest)
        selected = {row["video_path"]: row for row in rows if row["split"] == split}
        if not selected:
            raise ValueError(f"No manifest videos for split={split!r}")
        self.manifest_video_count = len(selected)
        crops: dict[str, list[Path]] = defaultdict(list)
        with crop_index.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                if row.get("status") == "ok" and row.get("video_path") in selected:
                    crops[row["video_path"]].append(Path(row["crop_path"]))
        self.records = []
        for video_path, row in selected.items():
            crop_paths = sorted(crops.get(video_path, []))
            if crop_paths:
                self.records.append((video_path, int(row["label"]), crop_paths))
        if not self.records:
            raise ValueError(f"No successful face crops found for split={split!r}; run prepare_faces.py")
        self.excluded_video_count = self.manifest_video_count - len(self.records)
        self.frames_per_video = frames_per_video

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str]:
        video_path, label, paths = self.records[index]
        if len(paths) >= self.frames_per_video:
            picks = [round(i * (len(paths) - 1) / (self.frames_per_video - 1))
                     if self.frames_per_video > 1 else len(paths) // 2
                     for i in range(self.frames_per_video)]
            chosen = [paths[i] for i in picks]
        else:
            chosen = paths + [paths[-1]] * (self.frames_per_video - len(paths))
        images = []
        for path in chosen:
            with Image.open(path) as image:
                rgb = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
                images.append(TF.normalize(TF.to_tensor(rgb), IMAGENET_MEAN, IMAGENET_STD))
        return torch.stack(images), torch.tensor(label, dtype=torch.long), video_path
