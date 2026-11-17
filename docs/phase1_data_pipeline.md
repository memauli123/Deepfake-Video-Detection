# Phase 1 data pipeline contract

This document records the agreed interfaces for preparing FaceForensics++ C23 samples from the team's [Kaggle dataset](https://www.kaggle.com/datasets/xdxd003/ff-c23). It does not infer source-video relationships from filenames: those relationships must be checked against the actual Kaggle files and metadata before a manifest is accepted. The upstream FaceForensics++ documentation describes manipulated sequence names as `<target sequence>_<source sequence>`; verify that the Kaggle copy preserves this convention and prefer its metadata CSVs when they identify the original source directly.

## Reported Kaggle inventory (not yet locally verified)

The supplied dataset summary reports 7,000 MP4s: 1,000 originals and 1,000 each in `DeepFakeDetection`, `Deepfakes`, `Face2Face`, `FaceShifter`, `FaceSwap`, and `NeuralTextures`, plus 10 CSV metadata files. The reported archive size is 17.92 GB at C23. These counts are useful integrity checks, but they do not provide per-video labels, original-source mappings, or a train/validation/test split.

Treat `DeepFakeDetection` source relationships as a separate mapping until its CSVs establish otherwise. Upstream documentation gives this subset a different naming scheme from standard FaceForensics++ manipulated videos, and the Kaggle package may curate or rename files. Never merge source IDs across folders based on matching filename fragments alone.

## Metadata inspection and provisional manifest

The ten named CSVs were found in the local Downloads folder and packaged without the unrelated CSV files also present there. `FF++_Metadata.csv` and `FF++_Metadata_Shuffled.csv` contain the same 7,000 rows; the seven category CSVs contain 1,000 rows each and their combined video-path set matches the master CSV. `Mean_Data.csv` is summary-only and is not used to build video rows. The Kaggle notebook also verified that all 7,000 manifest paths exist under the mounted dataset root.

`build_manifest.py` creates the local draft at `data/processed/ff_c23_manifest.csv`. It maps `REAL` to label `0`, `FAKE` to label `1`, and uses the folder as the manipulation label. Its provisional source grouping links the two sequence IDs in each standard FF++ `<target>_<source>` pair, then assigns every connected original/manipulated row to the same source group. For DeepFakeDetection, it links target and source `actor__scene` nodes from the filename convention into connected groups. These mappings are inferred from names because the Kaggle CSVs do not provide a `source_id` column; review them with the teammate before using the split for training.

The current draft uses a seeded `70/15/15` group split (`seed=42`). The local `build_manifest.py` generator reproduces the downloaded Kaggle manifest's source IDs, source-group families, and split assignments exactly, and checks that its inferred source IDs do not cross splits. These checks establish reproducibility and consistency of the inferred grouping only; they do not prove that the source relationships are correct or that the resulting split is leakage-free. Manually verify the source mappings against dataset metadata and files before training. The generated manifest is under the ignored `data/processed/` directory and is not tracked by Git.

## Required manifest

Use one row per video with these columns:

| Column | Meaning |
| --- | --- |
| `video_path` | Path relative to the local dataset root (or a documented absolute path kept local) |
| `label` | `0` for original/real, `1` for manipulated/fake |
| `manipulation` | `original` for real videos; the verified manipulation name for fake videos |
| `source_id` | Verified identity of the original source video shared by all its manipulated derivatives |
| `split` | `train`, `validation`, or `test`; never `unassigned` in a training manifest |

Before use, validate that required values are present, labels agree with the original/manipulated category, every path resolves, all derivatives of one `source_id` have exactly one split, and each requested split contains both classes. A preliminary manifest with empty source IDs or unassigned splits is discovery input only and must not be used for training.

## Split protocol

1. Inspect the downloaded directory tree and metadata CSVs; establish and spot-check the mapping from every manipulated video to its original source ID.
2. Make a deterministic group split by `source_id`, recording the seed and proportions in the manifest-generation command or configuration.
3. Confirm there is no source ID overlap between train, validation, and test. Keep all manipulation variants of a source together.
4. Define the held-out-manipulation experiment separately: reserve the manipulation type(s) before fitting or tuning, and report it as a distinct test protocol.
5. Save the validated manifest under the ignored local data directory. Commit only the schema, generation/validation code, and non-sensitive instructions, never dataset videos or derived crops.

## Frame and face-crop interface

- Sample frames per video using one fixed, documented policy and record the source frame index (and timestamp when available).
- Use RetinaFace with five facial landmarks as the initial detector. Record the implementation/version, confidence threshold, crop expansion, alignment policy, and behavior when no face is found.
- Store crops locally at `224 x 224` in RGB. Both model branches must derive from the same crop for a given video/frame.
- Spatial input: ImageNet-normalized RGB tensor. Frequency input: RGB values in `[0, 1]` before ImageNet normalization.
- Keep the video/frame identifiers alongside extracted crops so results can be traced to the manifest without duplicating labels inconsistently.
- Do not silently replace failed face detections with full frames. Count and report failures; agree on whether to skip the frame or video.

## Current gates before training

- Confirm dataset access, exact C23 subset and applicable FaceForensics++ terms of use.
- Verify source relationships against the downloaded manifest and files; the preliminary 7,000-video manifest described by the teammate has empty `source_id` and `unassigned` splits.
- Agree with the frequency-branch contributor on the face detector, alignment/crop policy, and detection-failure handling before implementing shared preprocessing.
- Generate and validate a source-aware split before any model fitting.
