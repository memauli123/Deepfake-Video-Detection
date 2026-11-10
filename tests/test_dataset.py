
from deepfake_detection.data.dataset import DeepfakeVideoDataset


def test_dataset():
    dataset = DeepfakeVideoDataset(
        manifest_path="data/splits/manifest.csv",
        split="train",
        num_frames=10,
        image_size=224
    )

    frames, label = dataset[0]

    assert frames.shape == (10, 3, 224, 224)
    assert label.item() in (0, 1)

    print("Dataset test passed!")
    print("Number of videos:", len(dataset))
    print("Frames shape:", frames.shape)
    print("Label:", label.item())


if __name__ == "__main__":
    test_dataset()
