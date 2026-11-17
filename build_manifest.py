"""Build a reproducible source-grouped manifest from the Kaggle CSV archive.

Example:
    python build_manifest.py C:\\Users\\Hp\\Downloads\\faceforensics_metadata_csvs.zip

The output defaults to data/processed/ff_c23_manifest.csv (ignored by Git).
Source groups are inferred from the upstream FF++ and DeepFakeDetection
filename conventions and should be reviewed with the team before training.
"""

from __future__ import annotations

import argparse
import csv
import io
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
import random
import re
import sys
import zipfile


SEED = 42
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
FFPP_METHODS = {
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures",
}
EXPECTED_CATEGORY_COUNTS = {
    "DeepFakeDetection": 1000,
    "Deepfakes": 1000,
    "Face2Face": 1000,
    "FaceShifter": 1000,
    "FaceSwap": 1000,
    "NeuralTextures": 1000,
    "original": 1000,
}


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def read_master_csv(archive: zipfile.ZipFile) -> list[dict[str, str]]:
    matches = [
        name
        for name in archive.namelist()
        if PurePosixPath(name).name.casefold() == "ff++_metadata.csv"
    ]
    if len(matches) != 1:
        raise ValueError("Archive must contain exactly one FF++_Metadata.csv")
    text = archive.read(matches[0]).decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    required = {"File Path", "Label", "Frame Count", "Width", "Height", "Codec", "File Size(MB)"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Master CSV is missing columns: {sorted(required - set(rows[0] if rows else []))}")
    return rows


def source_components(rows: list[dict[str, str]]) -> tuple[dict[str, str], dict[str, str]]:
    """Return per-video source IDs and source ID families (ffpp / dfd)."""
    ff_union, dfd_union = UnionFind(), UnionFind()
    parsed: dict[str, tuple[str, ...]] = {}
    ff_pattern = re.compile(r"^(\d+)_(\d+)$")
    dfd_pattern = re.compile(r"^(\d+)_(\d+)__(.+)__([A-Za-z0-9]+)$")

    for row in rows:
        path = PurePosixPath(row["File Path"])
        method = path.parts[0]
        stem = path.stem
        if method in FFPP_METHODS:
            match = ff_pattern.fullmatch(stem)
            if not match:
                raise ValueError(f"Unexpected FF++ manipulated filename: {row['File Path']}")
            left, right = match.groups()
            ff_union.union(f"ffpp:{left}", f"ffpp:{right}")
            parsed[row["File Path"]] = ("ffpp", left, right)
        elif method == "original":
            if not re.fullmatch(r"\d+", stem):
                raise ValueError(f"Unexpected FF++ original filename: {row['File Path']}")
            ff_union.find(f"ffpp:{stem}")
            parsed[row["File Path"]] = ("ffpp_original", stem)
        elif method == "DeepFakeDetection":
            match = dfd_pattern.fullmatch(stem)
            if not match:
                raise ValueError(f"Unexpected DeepFakeDetection filename: {row['File Path']}")
            target_actor, source_actor, scene, _experiment_id = match.groups()
            target_node = f"dfd:{target_actor}__{scene}"
            source_node = f"dfd:{source_actor}__{scene}"
            dfd_union.union(target_node, source_node)
            parsed[row["File Path"]] = ("dfd", target_node, source_node)
        else:
            raise ValueError(f"Unknown dataset folder in path: {row['File Path']}")

    def stable_component(union: UnionFind, item: str, prefix: str) -> str:
        root = union.find(item)
        members = sorted(member for member in union.parent if union.find(member) == root)
        return prefix + ":" + "+".join(member.split(":", 1)[1] for member in members)

    source_ids: dict[str, str] = {}
    families: dict[str, str] = {}
    for path, parts in parsed.items():
        if parts[0] == "ffpp":
            source_ids[path] = stable_component(ff_union, f"ffpp:{parts[1]}", "ffpp")
            families[path] = "ffpp"
        elif parts[0] == "ffpp_original":
            source_ids[path] = stable_component(ff_union, f"ffpp:{parts[1]}", "ffpp")
            families[path] = "ffpp"
        else:
            source_ids[path] = stable_component(dfd_union, parts[1], "dfd")
            families[path] = "dfd"
    return source_ids, families


def assign_splits(
    rows: list[dict[str, str]],
    source_ids: dict[str, str],
    families: dict[str, str],
) -> dict[str, str]:
    groups_by_family: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        path = row["File Path"]
        groups_by_family[families[path]][source_ids[path]] += 1

    group_split: dict[str, str] = {}
    rng = random.Random(SEED)
    for family, groups in sorted(groups_by_family.items()):
        target_rows = sum(groups.values())
        targets = {name: target_rows * ratio for name, ratio in SPLIT_RATIOS.items()}
        assigned = Counter({name: 0 for name in SPLIT_RATIOS})
        # Largest groups first; seeded random values break ties reproducibly.
        # Match pandas groupby ordering in the Kaggle notebook before consuming
        # seeded tie-breakers, so both generators assign the same groups.
        ordered = sorted(groups)
        tie_break = {group: rng.random() for group in ordered}
        ordered.sort(key=lambda group: (-groups[group], tie_break[group], group))
        for group_id in ordered:
            size = groups[group_id]
            costs = {}
            for split_name, target in targets.items():
                costs[split_name] = sum(
                    ((assigned[name] + (size if name == split_name else 0) - targets[name]) ** 2)
                    / max(targets[name], 1)
                    for name in targets
                )
            chosen = min(costs, key=lambda name: (costs[name], name))
            assigned[chosen] += size
            group_split[group_id] = chosen

    return {path: group_split[source_id] for path, source_id in source_ids.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata_zip", type=Path, help="ZIP containing the metadata CSVs")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/ff_c23_manifest.csv"),
        help="Manifest CSV output path (default: data/processed/ff_c23_manifest.csv)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Optional extracted dataset root; checks every manifest video path",
    )
    args = parser.parse_args()

    if not args.metadata_zip.is_file():
        print(f"Metadata ZIP not found: {args.metadata_zip}", file=sys.stderr)
        return 2

    with zipfile.ZipFile(args.metadata_zip) as archive:
        rows = read_master_csv(archive)

    category_counts = Counter(PurePosixPath(row["File Path"]).parts[0] for row in rows)
    if dict(category_counts) != EXPECTED_CATEGORY_COUNTS:
        raise ValueError(f"Category counts do not match expected inventory: {dict(category_counts)}")

    seen_paths: set[str] = set()
    for row in rows:
        path = row["File Path"]
        if path in seen_paths:
            raise ValueError(f"Duplicate video path in master CSV: {path}")
        seen_paths.add(path)
        expected_label = "REAL" if PurePosixPath(path).parts[0] == "original" else "FAKE"
        if row["Label"].strip().upper() != expected_label:
            raise ValueError(f"Label/folder mismatch for {path}: {row['Label']}")

    source_ids, families = source_components(rows)
    splits = assign_splits(rows, source_ids, families)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "video_path",
        "label",
        "manipulation",
        "source_id",
        "split",
        "frame_count",
        "width",
        "height",
        "codec",
        "file_size_mb",
        "source_group_family",
        "path_exists",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            path = row["File Path"]
            path_exists = (
                args.dataset_root.joinpath(*PurePosixPath(path).parts).is_file()
                if args.dataset_root
                else "not_checked"
            )
            writer.writerow(
                {
                    "video_path": path,
                    "label": 0 if row["Label"].strip().upper() == "REAL" else 1,
                    "manipulation": PurePosixPath(path).parts[0],
                    "source_id": source_ids[path],
                    "split": splits[path],
                    "frame_count": row["Frame Count"],
                    "width": row["Width"],
                    "height": row["Height"],
                    "codec": row["Codec"],
                    "file_size_mb": row["File Size(MB)"],
                    "source_group_family": families[path],
                    "path_exists": path_exists,
                }
            )

    source_splits: dict[str, set[str]] = defaultdict(set)
    split_labels: dict[str, Counter[int]] = defaultdict(Counter)
    for row in rows:
        source_splits[source_ids[row["File Path"]]].add(splits[row["File Path"]])
        split_labels[splits[row["File Path"]]][0 if row["Label"].strip().upper() == "REAL" else 1] += 1
    if any(len(values) != 1 for values in source_splits.values()):
        raise AssertionError("A source group crosses split boundaries")

    print(f"Manifest rows: {len(rows)}")
    print(f"Source groups: {len(set(source_ids.values()))} (FF++ and DeepFakeDetection grouped separately)")
    print(f"Seed: {SEED}; ratios: {SPLIT_RATIOS}")
    print(f"Split label counts: {dict(split_labels)}")
    print(f"Manifest written: {args.output.resolve()}")
    if args.dataset_root:
        missing_paths = sum(
            not args.dataset_root.joinpath(*PurePosixPath(row["File Path"]).parts).is_file()
            for row in rows
        )
        print(f"Missing video paths: {missing_paths}")
    else:
        print("Video paths were not checked; pass --dataset-root to verify them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
