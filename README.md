<div align="center">

# MotionPredictionProject

**Experimenting with neural network architectures for future human-motion prediction, built on top of AI4AnimationPy.**

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](Dockerfile)

</div>

This project explores different deep learning approaches for **predicting future human motion** from motion capture data — given a window of past poses, predict what the body will do next. It is built on top of [AI4AnimationPy](https://github.com/sebastianstarke/AI4Animation), a Python framework for AI-driven character animation by Paul Starke and Sebastian Starke, which provides the data pipeline, ECS-style runtime, math/animation utilities, and visualization tooling used throughout this repo.

On top of that framework, this repo adds a series of motion-prediction model architectures, trained checkpoints, and a documented trail of experiments — including approaches that didn't work out, kept around because the dead ends were as informative as the wins.

## Table of Contents

- [Project Structure](#project-structure)
- [Running the Project](#running-the-project)
- [Models](#models)
- [Failed Attempts](#failed-attempts)
- [Demos](#demos)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Project Structure

```
MotionPredictionProject/
├── ai4animation/         # Core framework (ECS, math, rendering, AI building blocks)
├── Demos/                # Interactive demo scenes (Locomotion, IK, MotionEditor, ...)
├── Failed_Attemps/       # Archived architecture experiments that did not pan out
├── test/                 # Minimal example/template scripts
├── Media/                # Demo gifs and diagrams (not embedded in this README)
├── *.pth                 # Trained model checkpoints (see Models below)
├── Dockerfile
├── docker-compose.yml
└── setup.py
```

## Running the Project

There are two ways to run the code in this repo:

### 1. Docker (training only)

The provided Docker setup is **headless** and intended for **training only** — it does not include a display/rendering stack, so the interactive demos and inference scenes (which rely on the framework's Standalone rendering mode) cannot run inside it.

1. Make sure [Docker](https://docs.docker.com/get-docker/) and Docker Compose are installed.
2. From the repo root, build the image:
   ```bash
   docker compose build
   ```
3. Start the container and drop into a shell:
   ```bash
   docker compose run motion-prediction
   ```
4. Inside the container, the `ai4animation` package is already installed (editable mode) and the repo is mounted at `/app`. Run your training script, e.g.:
   ```bash
   python test/network_training.py
   ```

The container is built on `python:3.12-slim` with `MPLBACKEND=Agg` set, so plots are written to disk instead of opened in a window.

### 2. Conda (training + inference)

For running inference, the interactive demos, or anything that needs the Standalone rendering mode, use a local conda environment instead.

1. Create and activate a new environment:
   ```bash
   conda create -n motionprediction python=3.12
   conda activate motionprediction
   ```
2. Install the project and its dependencies:
   ```bash
   pip install -e .
   ```
   This installs the `ai4animation` package along with PyTorch, NumPy, SciPy, scikit-learn, einops, pygltflib, etc. (see [`setup.py`](setup.py) for the full list), and exposes a `convert` CLI for batch motion-data conversion.
3. Run any script directly, including inference and demos:
   ```bash
   python Demos/Locomotion/locomotion.py
   ```

### Quick Start

```python
from ai4animation import Motion

# Load motion capture data (GLB, FBX, or BVH)
motion = Motion.LoadFromBVH("character.bvh", scale=0.01)

# Save to the framework's internal format
motion.SaveToNPZ("character")
```

```bash
# Batch-convert a directory of motion files
convert --input_dir path/to/motions --output_dir path/to/output
```

See [`test/`](test) for runnable examples covering character loading, motion playback, and network training, and [`Demos/`](Demos) for full interactive scenes.

## Models

The repo includes a number of trained checkpoints (`*.pth`) exploring different strategies for future motion prediction:

| Family | Checkpoints | Idea |
|---|---|---|
| **Autoregressive MLP** | `AutoregressiveMLP_model.pth` | Frame-by-frame feed-forward prediction, rolled forward autoregressively |
| **Autoregressive LSTM** | `Autoregressive_LSTM_Model.pth`, `Autoregressive_LSTM_Model_100style.pth`, `Autoregressive_LSTM_Model_Bonus.pth` | Recurrent model conditioned on motion history, with a variant trained on the 100STYLE dataset |
| **Conditional LSTM** | `conditional_lstm_full.pth`, `v1_conditional_lstm_full.pth`, `v2_conditional_lstm_full.pth` | LSTM conditioned on auxiliary signals (e.g. style/action), with iterative versions |
| **Latent LSTM** | `latent_lstm_layernorm_full_model.pth`, `latent_lstm_with_vae_full_model.pth` | Recurrent prediction in a learned latent space rather than raw pose space |
| **VAE** | `vae_full_model.pth`, `vae_full_model_100style.pth` | Variational autoencoder over pose/motion features |
| **Flow Matching** | `flow_matching_raw_model.pth`, `flow_matching_latent_model.pth`, `flow_matching_latent_bonus.pth`, `flow_matching_vae.pth`, `viol_100style_bonus_flow_matching_full.pth` | Generative flow-matching approaches, in raw and latent space, including an AdaLN-conditioned variant |
| **LayerNorm MLP** | `layernorm_full_model.pth` | Feed-forward baseline with layer normalization |

Loss curves for the flow-matching models are saved as `loss_history_FlowMatchingRaw.png` and `loss_history_FlowMatchingLatent.png` in the repo root. See [`test/model_loading.py`](test/model_loading.py) and [`test/network_training.py`](test/network_training.py) for how to load and train these models using the framework's `Tensor`, `DataSampler`, and `Dataset` utilities.

## Failed Attempts

[`Failed_Attemps/`](Failed_Attemps) contains earlier architecture experiments that were tried and ultimately not used for the final models.

## Demos

The underlying framework ships several interactive demo scenes, runnable standalone or headless:

- **Locomotion** — stylized biped locomotion controller
- **Quadruped Locomotion** — gait transitions and action poses
- **Training** — live training visualization
- **Inverse Kinematics** — real-time IK solving
- **Motion Capture Import** — GLB/FBX/BVH/NPZ loading
- **Motion Editor** — animation browsing and feature visualization

Run any demo from [`Demos/`](Demos), e.g.:

```bash
python Demos/Locomotion/locomotion.py
```

## Acknowledgements

This project builds directly on:

- [**AI4AnimationPy**](https://github.com/sebastianstarke/AI4Animation) by [Paul Starke](https://github.com/paulstarke) and [Sebastian Starke](https://github.com/sebastianstarke) — the underlying framework for motion data processing, the ECS/runtime architecture, math and animation modules, and the real-time renderer.
- The original [AI4Animation](https://github.com/sebastianstarke/AI4Animation) research line and its associated SIGGRAPH publications.
- Public motion capture datasets used during development, including Cranberry, 100STYLE, and LaFan — see the upstream [AI4AnimationPy documentation](https://facebookresearch.github.io/ai4animationpy/) for dataset sources and licensing.

Please see [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before opening issues or pull requests.

## License

This project is licensed under the [CC BY-NC 4.0 License](LICENSE) — non-commercial use only, with attribution.
