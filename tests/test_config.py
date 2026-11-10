
from deepfake_detection.config import load_config


def test_config():
    config = load_config()

    print("Configuration loaded successfully!")

    print("Dataset:", config["dataset"]["name"])
    print("Dataset path:", config["dataset"]["root"])
    print("Image size:", config["dataset"]["image_size"])
    print("Frames per video:", config["dataset"]["frames_per_video"])

    assert config["labels"]["real"] == 0
    assert config["labels"]["fake"] == 1


if __name__ == "__main__":
    test_config()
