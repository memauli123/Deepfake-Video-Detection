from deepfake_detection.data.landmarks import order_landmarks_by_image_side


def test_retinaface_landmarks_are_ordered_by_image_coordinates():
    # RetinaFace's anatomical keys need not be in left-to-right image order.
    points = {
        "left_eye": [30.0, 10.0],
        "right_eye": [10.0, 10.0],
        "nose": [20.0, 20.0],
        "mouth_left": [28.0, 30.0],
        "mouth_right": [12.0, 30.0],
    }

    ordered = order_landmarks_by_image_side(points)

    assert ordered == [
        [10.0, 10.0],
        [30.0, 10.0],
        [20.0, 20.0],
        [12.0, 30.0],
        [28.0, 30.0],
    ]
