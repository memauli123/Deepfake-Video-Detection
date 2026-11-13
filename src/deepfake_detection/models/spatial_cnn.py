"""Spatial CNN branch for frame-level deepfake features and predictions."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


class SpatialCNN(nn.Module):
    """ImageNet-pretrained ResNet-18 spatial encoder with a binary head.

    The input must be an RGB batch of 224 x 224 frames normalized with the
    ImageNet mean and standard deviation. Keep a separate [0, 1] copy of each
    face crop for the frequency branch.

    ``extract_features`` returns one 512-dimensional vector per frame. The
    classification ``forward`` returns one row of class logits per frame.
    This module does not perform video-level temporal modeling.
    """

    FEATURE_DIM = 512
    BACKBONE_STAGES = ("layer1", "layer2", "layer3", "layer4")

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        trainable_stages: Sequence[str] = (),
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(self.FEATURE_DIM, num_classes)
        self._trainable_stages: tuple[str, ...] = ()
        self.set_trainable_stages(*trainable_stages)
        self.train(self.training)

    @property
    def trainable_stages(self) -> tuple[str, ...]:
        """Names of ResNet stages currently enabled for fine-tuning."""
        return self._trainable_stages

    def set_trainable_stages(self, *stage_names: str) -> None:
        """Freeze the backbone, then optionally unfreeze selected ResNet stages.

        For example, ``set_trainable_stages("layer4")`` enables fine-tuning
        only the final ResNet block. The classifier remains trainable.
        """
        invalid = set(stage_names) - set(self.BACKBONE_STAGES)
        if invalid:
            choices = ", ".join(self.BACKBONE_STAGES)
            raise ValueError(
                f"Unknown ResNet stage(s): {sorted(invalid)}. "
                f"Choose from: {choices}"
            )

        # Remove duplicates while preserving the caller's order.
        self._trainable_stages = tuple(dict.fromkeys(stage_names))
        self.backbone.requires_grad_(False)
        for name in self._trainable_stages:
            getattr(self.backbone, name).requires_grad_(True)

    def train(self, mode: bool = True) -> "SpatialCNN":
        """Keep frozen backbone stages, including BatchNorm, in evaluation mode."""
        super().train(mode)
        self.backbone.eval()
        if mode:
            for name in self._trainable_stages:
                getattr(self.backbone, name).train()
        return self

    def _validate_frames(self, frames: Tensor) -> None:
        if not isinstance(frames, Tensor):
            raise TypeError("frames must be a PyTorch tensor")
        if frames.ndim != 4:
            raise ValueError(
                "Expected frames with shape (B, 3, 224, 224), "
                f"received {tuple(frames.shape)}"
            )
        if frames.shape[0] == 0:
            raise ValueError("frames batch must not be empty")
        if frames.shape[1:] != (3, 224, 224):
            raise ValueError(
                "Expected RGB frames with shape (B, 3, 224, 224), "
                f"received {tuple(frames.shape)}"
            )
        if not frames.is_floating_point():
            raise TypeError("frames must contain floating-point values")

    def extract_features(self, frames: Tensor) -> Tensor:
        """Return a 512-dimensional spatial feature vector for each frame."""
        self._validate_frames(frames)
        return self.backbone(frames)

    def forward(self, frames: Tensor) -> Tensor:
        """Return per-frame class logits with shape ``(B, num_classes)``."""
        features = self.extract_features(frames)
        return self.classifier(features)
