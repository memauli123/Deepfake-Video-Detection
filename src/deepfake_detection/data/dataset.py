
import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset

from deepfake_detection.data.video_loader import load_video_frames
from deepfake_detection.config import load_config

class DeepfakeVideoDataset(Dataset):

    def __init__(
        self,
        manifest_path,
        split=None,
        num_frames=10,
        image_size=224
    ):
        self.manifest_path = Path(manifest_path)
        self.num_frames = num_frames
        self.image_size = image_size

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
            )

        with self.manifest_path.open(
            "r", newline="", encoding="utf-8-sig"
        ) as file:
            reader = csv.DictReader(file)
            required = {
                "video_path", "label", "manipulation",
                "source_id", "split"
            }

            if not required.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"Manifest must contain columns: {sorted(required)}"
                )

            self.records = [
                row for row in reader
                if split is None or row["split"] == split
            ]

        if not self.records:
            raise ValueError(
                f"No video records found for split: {split}"
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        video_path = Path(record["video_path"])
        if not video_path.is_absolute():
            video_path = Path.cwd() / video_path

        label = int(record["label"])

        if label not in (0, 1):
            raise ValueError("Labels must be 0 (real) or 1 (fake)")

        frames = load_video_frames(
            str(video_path),
            num_frames=self.num_frames,
            image_size=self.image_size
        )

        return frames, torch.tensor(label, dtype=torch.long)

def create_dataset_from_config(
    split=None,
    config_path="configs/config.json"
):
    """
    Create a dataset using settings from config.json.
    """

    config = load_config(config_path)
    dataset_config = config["dataset"]

    dataset = DeepfakeVideoDataset(
        manifest_path=dataset_config["manifest"],
        split=split,
        num_frames=dataset_config["frames_per_video"],
        image_size=dataset_config["image_size"]
    )

    return dataset