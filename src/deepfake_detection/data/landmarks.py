"""Utilities for mapping face landmarks to the alignment template."""

from __future__ import annotations


def order_landmarks_by_image_side(points: dict[str, list[float]]) -> list[list[float]]:
    """Order RetinaFace points as image-left eye, image-right eye, nose,
    image-left mouth, image-right mouth.

    The template's first eye/mouth points have lower x coordinates. Sort by
    the detected x coordinates instead of inferring image side from anatomical
    landmark names (RetinaFace names such as ``right_eye`` are anatomical).
    """
    eyes = sorted((points["left_eye"], points["right_eye"]), key=lambda point: point[0])
    mouth_corners = sorted((points["mouth_left"], points["mouth_right"]), key=lambda point: point[0])
    return [eyes[0], eyes[1], points["nose"], mouth_corners[0], mouth_corners[1]]
