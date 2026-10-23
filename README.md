# Deepfake Video Detection

Student project: **Continual Spatial-Frequency-Temporal Learning for Generalizable Deepfake Video Detection under Unseen Manipulations**.

The project explores a detector that combines spatial, frequency, and (in a later phase) temporal evidence to classify real and manipulated media. The planned outputs include a confidence estimate and an explainability view. This repository currently contains project documentation and setup scaffolding only; no detection model has been implemented or evaluated yet.

## Project status

Development is planned in phases. Phase 1 is a spatial and frequency image baseline with an initial explainability output. Later phases add temporal modeling, a limited continual adaptation experiment, and a Streamlit demo. See [the roadmap](docs/roadmap.md) for scope and completion criteria.

## Setup

Use Python 3.10 or newer in a virtual environment. Install PyTorch and torchvision using the command appropriate for your operating system and compute device from the official PyTorch installation selector, then install the remaining project dependencies:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead. The repository does not include the FaceForensics++ dataset. Dataset access, permitted use, and the exact download procedure must be confirmed and documented before data preparation begins. Keep downloaded data and model checkpoints out of Git.

## Repository layout

```text
configs/                  Experiment configuration (to be added with Phase 1)
data/                     Dataset setup notes; actual data is not tracked
docs/                     Roadmap and requirement mapping
notebooks/                Exploratory notebooks
src/deepfake_detection/   Application and model package
tests/                    Automated checks added with implementation
```

## Collaboration

Use a descriptive feature branch for each task, keep `main` stable, and open a pull request for substantial changes. Read [CONTRIBUTING.md](CONTRIBUTING.md) before starting work. Coordinate changes to data labels, splits, and model input/output interfaces before parallel implementation.

## Requirements source

The project presentation is the primary requirements source. Planned behavior is not implemented functionality: check the roadmap and future evaluation results before making claims about model performance or generalization.
