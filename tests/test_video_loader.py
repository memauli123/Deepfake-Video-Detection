
import torch

from deepfake_detection.data.video_loader import load_video_frames


def test_video_loader():
    video_path = "data/sample_video.mp4"

    frames = load_video_frames(
        video_path=video_path,
        num_frames=10,
        image_size=224
    )

    assert frames.ndim == 4
    assert frames.shape[1:] == (3, 224, 224)
    assert frames.dtype == torch.float32
    assert torch.isfinite(frames).all()

    print("Video loader test passed!")
    print("Output shape:", frames.shape)
    print("Data type:", frames.dtype)
    print("Minimum:", frames.min().item())
    print("Maximum:", frames.max().item())


if __name__ == "__main__":
    test_video_loader()
