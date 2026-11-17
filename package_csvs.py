"""Package the 10 FaceForensics metadata CSVs without including video files.

Usage:
    python package_csvs.py "D:\\datasets\\ff-c23"

The ZIP is written to the dataset root by default. Pass --output to choose a
different destination. CSV paths inside the archive remain relative to root.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import zipfile


EXPECTED_CSV_NAMES = {
    "DeepFakeDetection.csv",
    "Deepfakes.csv",
    "FF++_Metadata.csv",
    "FF++_Metadata_Shuffled.csv",
    "Face2Face.csv",
    "FaceShifter.csv",
    "FaceSwap.csv",
    "Mean_Data.csv",
    "NeuralTextures.csv",
    "original.csv",
}
OUTPUT_NAME = "faceforensics_metadata_csvs.zip"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Root directory of the downloaded dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="ZIP destination (default: dataset root / " + OUTPUT_NAME + ")",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    if not dataset_dir.is_dir():
        print(f"Dataset directory does not exist: {dataset_dir}", file=sys.stderr)
        return 2

    output_zip = (
        args.output.expanduser().resolve()
        if args.output
        else dataset_dir / OUTPUT_NAME
    )
    found_by_name: dict[str, list[Path]] = {name.casefold(): [] for name in EXPECTED_CSV_NAMES}
    for path in dataset_dir.rglob("*"):
        if path.is_file() and path.name.casefold() in found_by_name:
            found_by_name[path.name.casefold()].append(path)

    missing = sorted(
        name for name in EXPECTED_CSV_NAMES if not found_by_name[name.casefold()]
    )
    duplicates = sorted(
        name for name in EXPECTED_CSV_NAMES if len(found_by_name[name.casefold()]) > 1
    )
    csv_files = sorted(
        matches[0]
        for matches in found_by_name.values()
        if len(matches) == 1
    )

    print(f"Scanning: {dataset_dir}")
    print(f"Named metadata CSVs found: {len(csv_files)} / {len(EXPECTED_CSV_NAMES)}")
    if missing or duplicates:
        if missing:
            print("Missing expected files:", file=sys.stderr)
            for name in missing:
                print(f"  {name}", file=sys.stderr)
        if duplicates:
            print("Expected filenames found more than once:", file=sys.stderr)
            for name in duplicates:
                print(f"  {name}", file=sys.stderr)
        print("No archive was created.", file=sys.stderr)
        return 1

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output_zip, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for csv_path in csv_files:
            relative_path = csv_path.relative_to(dataset_dir)
            archive.write(csv_path, arcname=relative_path.as_posix())
            print(f"Added: {relative_path.as_posix()}")

    print(f"Created archive: {output_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
