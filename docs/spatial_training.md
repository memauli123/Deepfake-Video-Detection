# Spatial branch: crop preparation, training, and demo

## Data contract

The manifest supplies video labels and splits. Code refuses empty source IDs, unassigned splits, group overlap, or splits without both classes. Source IDs inferred from filenames are provisional and require manual verification against dataset metadata and files before training. The code can check that its supplied IDs do not cross splits; that check alone does not establish that the split is leakage-free. The crop cache stores aligned 224 x 224 RGB face crops plus a status row for each sampled frame, allowing the frequency branch to reuse exactly the same face pixels. Spatial tensors receive ImageNet normalization; retain the unnormalized cache for frequency input.

Initial preprocessing uses RetinaFace five-landmark detections, confidence threshold 0.90, uniformly spaced frames, and a five-point similarity alignment. It selects the highest-confidence face (face area breaks ties). Failed detections are recorded and never replaced by full frames. Training excludes videos with no successful crops and repeats the final successful crop to pad short videos to ten frames. Report detection failure counts.

## Kaggle run

Attach `xdxd003/ff-c23` to a Kaggle notebook and attach/copy the validated 7,000-row manifest. Upload this repository or clone the agreed spatial branch. Dataset input is read-only; crops, checkpoints, and reports go under `/kaggle/working`. Never upload videos, crops, or checkpoints to GitHub.

The RetinaFace project documents a PyTorch backend extra in its current [installation instructions](https://github.com/serengil/retinaface#installation). From the repository root:

```bash
python -m pip install -e ".[retinaface]"
python -m deepfake_detection.data.prepare_faces \
  --manifest /kaggle/input/<manifest-dataset>/ff_c23_manifest.csv \
  --dataset-root /kaggle/input/datasets/xdxd003/ff-c23/FaceForensics++_C23 \
  --output-dir /kaggle/working/ff_c23_faces --frames-per-video 10 --limit 20
```

The first, limited run checks the detector and paths. Inspect the crops and `crop_index.csv`, then rerun without `--limit` to prepare the full cache. Do not train or report metrics from the limited cache.

Run the frozen-backbone baseline:

```bash
python -m deepfake_detection.train_spatial \
  --manifest /kaggle/input/<manifest-dataset>/ff_c23_manifest.csv \
  --crop-index /kaggle/working/ff_c23_faces/crop_index.csv \
  --output-dir /kaggle/working/spatial_run_frozen --epochs 10 --batch-size 4
```

Compare later-layer fine-tuning in a separate experiment:

```bash
python -m deepfake_detection.train_spatial \
  --manifest /kaggle/input/<manifest-dataset>/ff_c23_manifest.csv \
  --crop-index /kaggle/working/ff_c23_faces/crop_index.csv \
  --output-dir /kaggle/working/spatial_run_layer4 --epochs 10 --batch-size 2 \
  --fine-tune-layer4 --learning-rate 0.0001
```

The training script selects by validation loss, early-stops after three stale epochs, then evaluates the chosen checkpoint once on the held-out test split. It reports video-level accuracy, precision, recall, F1, ROC-AUC, counts, seed, and device. Download `spatial_metrics.json`, `spatial_history.json`, and the checkpoint. Results are provisional until your teammate validates inferred source-group relationships.

## Teacher demo

Install the package with the RetinaFace extra using `python -m pip install -e ".[retinaface]"`, then run `streamlit run app.py`. Set the checkpoint path and upload a short video. The app uses the same RetinaFace alignment and shows detected crops, per-frame scores, and a simple mean video score. This is not a learned temporal or fused model. Add frequency/fusion, Grad-CAM, and experiment history in later phases.
