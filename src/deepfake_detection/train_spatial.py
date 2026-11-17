"""Train and evaluate a source-split spatial ResNet-18 baseline."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from deepfake_detection.data.face_dataset import FaceCropVideoDataset, validate_manifest
from deepfake_detection.models.spatial_cnn import SpatialCNN


def evaluate(model, loader, device, criterion):
    model.eval()
    all_labels, all_probs, total_loss, total = [], [], 0.0, 0
    with torch.inference_mode():
        for frames, labels, _ in loader:
            batch, time, channels, height, width = frames.shape
            logits = model(frames.to(device).reshape(batch * time, channels, height, width))
            video_logits = logits.reshape(batch, time, -1).mean(dim=1)
            loss = criterion(video_logits, labels.to(device))
            total_loss += float(loss) * batch
            total += batch
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(video_logits.softmax(dim=1)[:, 1].cpu().numpy().tolist())
    predictions = [int(p >= 0.5) for p in all_probs]
    metrics = {"loss": total_loss / max(total, 1), "accuracy": accuracy_score(all_labels, predictions),
               "precision": precision_score(all_labels, predictions, zero_division=0),
               "recall": recall_score(all_labels, predictions, zero_division=0),
               "f1": f1_score(all_labels, predictions, zero_division=0),
               "roc_auc": roc_auc_score(all_labels, all_probs) if len(set(all_labels)) == 2 else None}
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--crop-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frames-per-video", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-layer4", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rows = validate_manifest(args.manifest)
    datasets = {split: FaceCropVideoDataset(args.manifest, args.crop_index, split, args.frames_per_video)
               for split in ("train", "validation", "test")}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for split, dataset in datasets.items():
        actual_labels = {label for _, label, _ in dataset.records}
        if actual_labels != {0, 1}:
            raise ValueError(f"Face-cropped {split} split lost a class; detected label set: {actual_labels}")
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size,
                                 shuffle=split == "train", num_workers=2,
                                 pin_memory=device.type == "cuda")
               for split, dataset in datasets.items()}
    train_counts = np.bincount([label for _, label, _ in datasets["train"].records], minlength=2)
    weights = len(datasets["train"]) / (2 * np.maximum(train_counts, 1))
    model = SpatialCNN(pretrained=not args.no_pretrained,
                       trainable_stages=("layer4",) if args.fine_tune_layer4 else ()).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "spatial_resnet18_best.pt"
    history = []
    best_val = float("inf")
    patience, stale = 3, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for frames, labels, _ in loaders["train"]:
            batch, time, channels, height, width = frames.shape
            optimizer.zero_grad(set_to_none=True)
            logits = model(frames.to(device).reshape(batch * time, channels, height, width))
            video_logits = logits.reshape(batch, time, -1).mean(dim=1)
            loss = criterion(video_logits, labels.to(device))
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * batch; seen += batch
        val = evaluate(model, loaders["validation"], device, criterion)
        epoch_row = {"epoch": epoch, "train_loss": running / max(seen, 1), "validation": val}
        history.append(epoch_row)
        print(json.dumps(epoch_row), flush=True)
        if val["loss"] < best_val:
            best_val, stale = val["loss"], 0
            torch.save({"model_state_dict": model.state_dict(), "feature_dim": 512,
                        "classes": ["real", "fake"], "input_size": 224,
                        "normalization": "ImageNet", "trainable_stages": list(model.trainable_stages),
                        "seed": args.seed, "manifest": str(args.manifest), "epoch": epoch,
                        "validation_metrics": val}, best_path)
        else:
            stale += 1
            if stale >= patience:
                print("Early stopping: validation loss did not improve", flush=True)
                break
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, loaders["test"], device, criterion)
    report = {"best_epoch": checkpoint["epoch"], "best_validation": checkpoint["validation_metrics"],
              "test": test_metrics, "video_counts": {split: len(ds) for split, ds in datasets.items()},
              "manifest_video_counts": {split: ds.manifest_video_count for split, ds in datasets.items()},
              "excluded_no_face_video_counts": {split: ds.excluded_video_count for split, ds in datasets.items()},
              "class_counts_train": train_counts.tolist(), "seed": args.seed,
              "split_rows": {split: sum(row["split"] == split for row in rows) for split in ("train", "validation", "test")},
              "device": str(device), "pretrained": not args.no_pretrained,
              "fine_tuned_stages": list(model.trainable_stages)}
    (output_dir / "spatial_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "spatial_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Best checkpoint: {best_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
