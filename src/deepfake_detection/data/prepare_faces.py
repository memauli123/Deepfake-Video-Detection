"""Sample videos, align the highest-confidence RetinaFace and cache RGB crops."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from deepfake_detection.data.face_dataset import validate_manifest
from deepfake_detection.data.landmarks import order_landmarks_by_image_side


TARGET_LANDMARKS = np.asarray(
    # Ordered image-left eye, image-right eye, nose, image-left mouth,
    # image-right mouth. Landmark detections are explicitly reordered by x.
    [[75.0, 95.0], [149.0, 95.0], [112.0, 136.0], [79.0, 177.0], [145.0, 177.0]],
    dtype=np.float32,
)


def make_detector(threshold: float):
    try:
        from retinaface import RetinaFace
    except ImportError as exc:
        raise RuntimeError('Install the optional dependency with: python -m pip install -e ".[retinaface]"') from exc

    def detect(bgr: np.ndarray):
        result = RetinaFace.detect_faces(img_path=bgr, threshold=threshold)
        if not isinstance(result, dict):
            return None
        candidates = []
        for item in result.values():
            points = item.get("landmarks") or {}
            keys = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")
            if not all(key in points for key in keys):
                continue
            score = float(item.get("score", 0))
            area = item.get("facial_area", [0, 0, 0, 0])
            ordered_points = np.asarray(order_landmarks_by_image_side(points), dtype=np.float32)
            candidates.append((score, area, ordered_points))
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item[0], (item[1][2] - item[1][0]) * (item[1][3] - item[1][1])))[2]

    return detect


def sample_indices(frame_count: int, count: int) -> list[int]:
    if frame_count <= 0:
        return []
    return sorted(set(np.linspace(0, frame_count - 1, min(count, frame_count), dtype=int).tolist()))


def align_face(bgr: np.ndarray, detect) -> np.ndarray | None:
    """Return a 224 x 224 aligned RGB crop, or None when detection/alignment fails."""
    landmarks = detect(bgr)
    if landmarks is None:
        return None
    transform, _ = cv2.estimateAffinePartial2D(landmarks, TARGET_LANDMARKS, method=cv2.LMEDS)
    if transform is None:
        return None
    aligned = cv2.warpAffine(bgr, transform, (224, 224), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
    return cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames-per-video", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke run limit; 0 processes all videos")
    args = parser.parse_args()
    rows = validate_manifest(args.manifest)
    if args.frames_per_video < 1 or not 0 < args.threshold <= 1:
        parser.error("frames-per-video must be positive and threshold must be in (0, 1]")
    videos = rows[:args.limit] if args.limit else rows
    missing = [row["video_path"] for row in videos
               if not args.dataset_root.joinpath(*PurePosixPath(row["video_path"]).parts).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest video paths are missing under {args.dataset_root}; first: {missing[0]}")
    detect = make_detector(args.threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "crop_index.csv"
    fields = ["video_path", "frame_index", "timestamp_ms", "crop_path", "status", "detector", "threshold"]
    counts = {"ok": 0, "no_face": 0, "unreadable": 0}
    with index_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for video_number, row in enumerate(videos, start=1):
            relative = PurePosixPath(row["video_path"])
            video_file = args.dataset_root.joinpath(*relative.parts)
            capture = cv2.VideoCapture(str(video_file))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
            digest = hashlib.sha1(row["video_path"].encode("utf-8")).hexdigest()[:16]
            frame_indices = sample_indices(frame_count, args.frames_per_video)
            if not frame_indices:
                counts["unreadable"] += 1
                writer.writerow({"video_path": row["video_path"], "frame_index": -1,
                                 "timestamp_ms": "", "crop_path": "", "status": "unreadable",
                                 "detector": "RetinaFace-5-landmark", "threshold": args.threshold})
            for frame_index in frame_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, bgr = capture.read()
                timestamp = 1000.0 * frame_index / max(capture.get(cv2.CAP_PROP_FPS), 1e-6)
                status, crop_path = "unreadable", ""
                if ok and bgr is not None:
                    rgb = align_face(bgr, detect)
                    if rgb is None:
                        status = "no_face"
                    else:
                        out = args.output_dir / "crops" / digest / f"{frame_index:08d}.jpg"
                        out.parent.mkdir(parents=True, exist_ok=True)
                        if cv2.imwrite(str(out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95]):
                            status, crop_path = "ok", str(out.resolve())
                        else:
                            status = "unreadable"
                counts[status] += 1
                writer.writerow({"video_path": row["video_path"], "frame_index": frame_index,
                                 "timestamp_ms": f"{timestamp:.3f}", "crop_path": crop_path,
                                 "status": status, "detector": "RetinaFace-5-landmark", "threshold": args.threshold})
            capture.release()
            if video_number % 100 == 0 or video_number == len(videos):
                print(f"Processed {video_number}/{len(videos)} videos; crop counts={counts}", flush=True)
    print(f"Crop index: {index_path.resolve()}")
    print(f"Final frame outcomes: {counts}")
    if args.limit:
        print("NOTE: limited run; do not train or report metrics from this crop index")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
