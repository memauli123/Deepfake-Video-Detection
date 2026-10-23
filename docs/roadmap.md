# Project roadmap

This roadmap is based on the project presentation. It separates features stated in the deck from implementation recommendations; details not specified by the presentation are marked as decisions to make before implementation.

## Project objective

Investigate whether spatial, frequency, and temporal evidence can support deepfake video detection that generalizes to unseen manipulations and can explain its predictions. The presentation also proposes a simplified continual-adaptation demonstration.

## Requirements from the presentation

- Predict real or fake for image/video inputs.
- Phase 1: spatial CNN/transfer-learning features, FFT-based frequency features, and Integrated Gradients explainability.
- Phase 2: add short frame sequences and lightweight temporal features/modeling.
- Phase 3: detect distribution shift, fine-tune on new manipulation samples, and investigate knowledge distillation to reduce forgetting.
- Evaluate across manipulation types, unseen manipulation data, and compression levels.
- Produce a heatmap and numerical explainability score, plus prediction confidence.
- Provide a Streamlit demo/dashboard.
- Proposed tools: Python, PyTorch, OpenCV, Captum, NumPy, Pandas, scikit-learn, Streamlit, and Google Colab.
- Proposed data source: FaceForensics++.

## Phases

### Phase 1 — Spatial and frequency image baseline with explainability

**Objective:** Establish a reproducible, modest image-level baseline before adding video sequence modeling.

**Scope:** Confirm dataset access and terms; document labels and split policy; implement preprocessing; train/evaluate a spatial transfer-learning baseline and FFT feature baseline; combine features; produce prediction confidence and an Integrated Gradients heatmap with a clearly defined numerical summary.

**Suggested files:** `src/deepfake_detection/data/`, `src/deepfake_detection/models/`, `src/deepfake_detection/explainability/`, `src/deepfake_detection/evaluation/`, `configs/`, and `tests/`.

**Validation:** Reproducible split and run instructions; held-out metrics; evidence that related samples do not leak across splits; baseline comparisons; explanation output for a sample; document limitations. Decide the exact metrics, thresholds, and numerical explanation definition before implementation.

**Suggested issues:** dataset/access and split policy; preprocessing and labels; spatial baseline; FFT baseline; feature fusion; explanation output; evaluation protocol and report.

**Exit criteria:** The above behavior runs on an agreed, accessible dataset subset and has documented validation results. No claim of unseen-manipulation generalization is made unless the evaluation split actually withholds manipulation types.

### Phase 2 — Temporal information

**Objective:** Incorporate information across multiple frames.

**Scope:** Define video sampling and face/frame extraction; implement a lightweight temporal model or frame-trend features; combine temporal and Phase 1 features.

**Validation:** Compare the Phase 1 system with the temporal extension on held-out videos and report the effect of temporal features, runtime, and failure cases.

**Dependencies:** Phase 1 data, labels, evaluation protocol, and stable model interface.

**Suggested issues:** frame sampling; temporal feature/model; multimodal fusion; video-level evaluation.

### Phase 3 — Simplified continual adaptation

**Objective:** Demonstrate adaptation to a new manipulation type while measuring retention of prior performance.

**Scope:** Define a staged learning protocol; measure distribution shift; fine-tune with new manipulation samples; evaluate knowledge distillation as a forgetting-reduction method.

**Validation:** Report performance before and after adaptation on both new and previously learned manipulation groups, using a fixed protocol and comparable data.

**Dependencies:** Stable Phase 2 pipeline and a clear definition of new versus previously seen manipulation data.

**Suggested issues:** adaptation protocol; shift signal; fine-tuning; distillation comparison; forgetting evaluation.

### Phase 4 — Streamlit demo and integration

**Objective:** Make the validated model behavior accessible through a small demo.

**Scope:** Upload or select a supported sample, run inference, and show the label, confidence, and explanation view. State supported formats and limitations.

**Validation:** Documented local launch; successful inference on supported samples; clear error handling for unsupported input; no dataset or model weights accidentally committed.

**Dependencies:** A stable inference interface and a selected model artifact from prior phases.

**Suggested issues:** inference wrapper; upload and validation UI; prediction/explanation display; setup and usage docs.

## Non-functional goals and constraints

- Keep modules small and understandable for a student team.
- Make data splits, random seeds, preprocessing, and experiment settings reproducible.
- Avoid committing datasets, private data, credentials, generated artifacts, and model checkpoints.
- Treat confidence and attribution as model outputs, not proof of authenticity or causal explanations.
- Record compute requirements and known limitations with results.

## Decisions needed before Phase 1 implementation

1. Confirm dataset access, terms, and which subset the team can process.
2. Define the held-out split, including how unseen manipulation types are reserved.
3. Select the target image/frame sampling and face extraction policy.
4. Choose evaluation metrics and define the numerical explainability score.
5. Confirm available GPU/Colab compute and how experiments/checkpoints will be shared.
6. Decide whether Phase 1 should include cross-attention or begin with a simpler fusion baseline for comparison.

## Initial team split

- **Contributor A:** dataset access/terms, preprocessing, label and split documentation, spatial baseline.
- **Contributor B:** FFT feature prototype, explanation output prototype, evaluation/reporting scaffold.
- **Shared coordination:** agree on labels, split IDs, input conventions, and output contracts before parallel work. Review changes to shared configuration and interfaces together.

## Phase 1 acceptance criteria

1. Dataset source/access notes, labels, and train/validation/test split are documented.
2. Preprocessing produces the agreed image/face representation from permitted samples.
3. Spatial and FFT baselines each produce predictions.
4. A fused Phase 1 model returns a real/fake prediction and confidence.
5. Integrated Gradients produces a heatmap and documented numerical summary.
6. Held-out evaluation metrics and leakage controls are reported.
7. Setup and execution instructions are documented; data and credentials remain untracked.

The roadmap is a proposal for review. Phase boundaries and acceptance criteria should be agreed by both teammates before implementation begins.
