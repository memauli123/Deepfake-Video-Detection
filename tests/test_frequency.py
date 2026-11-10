
import torch
from deepfake_detection.models.frequency import (
    fft_magnitude_spectrum
)


def test_fft_output_shape():
    images = torch.rand(8, 3, 224, 224)

    output = fft_magnitude_spectrum(images)

    assert output.shape == images.shape
    assert torch.isfinite(output).all()
    assert (output >= 0).all()

    print("FFT test passed!")
    print("Input shape:", images.shape)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    test_fft_output_shape()
