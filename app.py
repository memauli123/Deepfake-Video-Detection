"""Teacher-facing spatial baseline demo; later phases can extend this page."""

from pathlib import Path
import tempfile

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from deepfake_detection.data.face_dataset import IMAGENET_MEAN, IMAGENET_STD
from deepfake_detection.data.prepare_faces import align_face, make_detector, sample_indices
from deepfake_detection.models.spatial_cnn import SpatialCNN


st.set_page_config(page_title="Deepfake Detection Demo", page_icon="🔎", layout="wide")
st.title("Deepfake Video Detection")
st.caption("Spatial baseline · RetinaFace five-point alignment · ImageNet-pretrained ResNet-18")
st.info("This page demonstrates the frame-level spatial baseline. Frequency fusion, temporal modeling, and explainability can be added as those project phases are completed.")

with st.sidebar:
    st.header("Model")
    checkpoint_path = st.text_input("Checkpoint", "outputs/spatial_resnet18_best.pt")
    frame_count = st.slider("Frames sampled from video", 1, 20, 10)
    threshold = st.slider("RetinaFace confidence", 0.5, 0.99, 0.9, 0.01)

upload = st.file_uploader("Upload a video or face image", type=["mp4", "mov", "avi", "jpg", "jpeg", "png"])
if upload is not None:
    is_image = upload.type.startswith("image/")
    rgb_frames: list[np.ndarray] = []
    if is_image:
        rgb_frames.append(np.asarray(Image.open(upload).convert("RGB")))
    else:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="deepfake_demo_", suffix=Path(upload.name).suffix, delete=False
            ) as temp_file:
                temp_path = Path(temp_file.name)
                temp_file.write(upload.getvalue())
            capture = cv2.VideoCapture(str(temp_path))
            try:
                indices = sample_indices(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), frame_count)
                detector = make_detector(threshold)
                for frame_index in indices:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    success, bgr = capture.read()
                    if success and bgr is not None:
                        aligned = align_face(bgr, detector)
                        if aligned is not None:
                            rgb_frames.append(aligned)
            finally:
                capture.release()
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    if is_image:
        st.warning("Image input is treated as an already selected face crop. Video input uses the shared RetinaFace five-point crop/alignment pipeline.")
    if not rgb_frames:
        st.error("No usable face crops were found. Try a clearer video with a visible face.")
    elif not Path(checkpoint_path).is_file():
        st.warning("No trained checkpoint is available at that path. Train the model first, then set the checkpoint path.")
        st.image(rgb_frames, caption=[f"Usable face crop {i + 1}" for i in range(len(rgb_frames))], width=180)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model = SpatialCNN(pretrained=False, trainable_stages=tuple(saved.get("trainable_stages", []))).to(device)
        model.load_state_dict(saved["model_state_dict"])
        model.eval()
        tensors = []
        for rgb in rgb_frames:
            image = Image.fromarray(rgb).resize((224, 224))
            tensors.append(TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD))
        batch = torch.stack(tensors).to(device)
        with torch.inference_mode():
            frame_probs = model(batch).softmax(dim=1)[:, 1]
        video_fake_probability = float(frame_probs.mean())
        label = "Fake" if video_fake_probability >= 0.5 else "Real"
        left, right = st.columns([1, 2])
        left.metric("Spatial prediction", label)
        left.metric("Mean fake probability", f"{video_fake_probability:.1%}")
        left.caption(f"Aggregated from {len(rgb_frames)} detected frame crop(s). This is a simple mean, not a learned temporal model.")
        right.image(rgb_frames, caption=[f"Frame {i + 1}: fake {float(p):.1%}" for i, p in enumerate(frame_probs)], width=150)
        st.bar_chart({"Fake probability": [float(p) for p in frame_probs]})
else:
    st.markdown("### Demo workflow")
    st.write("1. Train the frozen spatial baseline in Kaggle.  2. Download its best checkpoint.  3. Place it under `outputs/` or select it in the sidebar.  4. Upload a short MP4 to show detected crops and per-frame scores.")
