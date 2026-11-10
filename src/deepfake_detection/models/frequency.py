
import torch


def fft_magnitude_spectrum(images: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of images into log-scaled FFT magnitude spectra.

    Args:
        images: Float tensor with shape (B, C, H, W).

    Returns:
        Frequency spectrum with shape (B, C, H, W).
    """

    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a PyTorch tensor")

    if images.ndim != 4:
        raise ValueError(
            "Expected input shape (B, C, H, W), "
            f"but received {tuple(images.shape)}"
        )

    if not images.is_floating_point():
        raise TypeError("images must contain floating-point values")

    # Apply 2D FFT across height and width.
    fft_result = torch.fft.fft2(images, dim=(-2, -1))

    # Move zero frequency components to the center.
    fft_shifted = torch.fft.fftshift(fft_result, dim=(-2, -1))

    # Calculate magnitude and apply logarithmic scaling.
    magnitude = torch.abs(fft_shifted)
    log_magnitude = torch.log1p(magnitude)

    return log_magnitude
