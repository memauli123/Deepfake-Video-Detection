
import cv2
import torch
import numpy as np


def load_video_frames(
    video_path: str,
    num_frames: int = 10,
    image_size: int = 224
) -> torch.Tensor:
    """
    Read and sample frames from a video.

    Args:
        video_path: Path to the input video.
        num_frames: Number of frames to sample.
        image_size: Output height and width.

    Returns:
        Tensor of shape (T, C, H, W), with RGB values
        normalized to the range [0, 1].
    """

    if num_frames <= 0 or image_size <= 0:
        raise ValueError("num_frames and image_size must be positive")

    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        capture.release()
        raise ValueError("Video contains no readable frames")

    frame_indices = np.linspace(
        0, total_frames - 1,
        num=min(num_frames, total_frames),
        dtype=int
    )

    frames = []

    for index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, frame = capture.read()

        if not success:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_size, image_size))

        frame = frame.astype(np.float32) / 255.0
        frame = torch.from_numpy(frame).permute(2, 0, 1)

        frames.append(frame)

    capture.release()

    if not frames:
        raise ValueError("Could not read any frames from video")

    return torch.stack(frames)
