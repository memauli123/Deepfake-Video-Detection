
from deepfake_detection.data.dataset import (
    create_dataset_from_config
)


def test_config_dataset():
    dataset = create_dataset_from_config(split="train")

    frames, label = dataset[0]

    print("Config-based dataset test passed!")
    print("Number of videos:", len(dataset))
    print("Frames shape:", frames.shape)
    print("Label:", label.item())

    assert frames.shape == (10, 3, 224, 224)


if __name__ == "__main__":
    test_config_dataset()
