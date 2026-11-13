import pytest
import torch

from deepfake_detection.models.spatial_cnn import SpatialCNN


def test_spatial_features_and_logits_have_expected_shapes():
    model = SpatialCNN(pretrained=False)
    frames = torch.randn(2, 3, 224, 224)

    features = model.extract_features(frames)
    logits = model(frames)

    assert features.shape == (2, 512)
    assert logits.shape == (2, 2)
    assert features.dtype == torch.float32
    assert logits.dtype == torch.float32
    assert torch.isfinite(features).all()
    assert torch.isfinite(logits).all()


def test_backbone_is_frozen_by_default_but_classifier_is_trainable():
    model = SpatialCNN(pretrained=False)

    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


def test_selected_stage_can_be_fine_tuned():
    model = SpatialCNN(pretrained=False)
    model.set_trainable_stages("layer4")
    model.train()

    assert model.trainable_stages == ("layer4",)
    assert all(parameter.requires_grad for parameter in model.backbone.layer4.parameters())
    assert not any(parameter.requires_grad for parameter in model.backbone.layer3.parameters())
    assert model.backbone.layer4.training
    assert not model.backbone.layer3.training


def test_invalid_frame_shape_is_rejected():
    model = SpatialCNN(pretrained=False)

    with pytest.raises(ValueError, match="224"):
        model(torch.randn(2, 3, 128, 128))


def test_non_floating_frame_input_is_rejected():
    model = SpatialCNN(pretrained=False)

    with pytest.raises(TypeError, match="floating-point"):
        model(torch.zeros(2, 3, 224, 224, dtype=torch.uint8))


def test_unknown_trainable_stage_is_rejected():
    model = SpatialCNN(pretrained=False)

    with pytest.raises(ValueError, match="Unknown ResNet stage"):
        model.set_trainable_stages("stage5")
