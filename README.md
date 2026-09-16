# PINN Bubble Profiler

This repository contains a compact PyTorch implementation for training physics-informed neural network models to approximate radial bubble profiles and compare them against benchmark solutions. In the context of vacuum decay and bubble nucleation, the goal is to learn the radial field configuration that minimizes the action while satisfying the corresponding boundary conditions, using a neural network as a differentiable approximation of the profile. The project is intended as a small experimental codebase for studying PINN-based profile solving and evaluating results across a few simple test cases.

## Repository structure

- `main.py`  
  Command-line entry point. Selects which experiment to run via `--exp`.

- `train.py`  
  Training utilities used by both the pretraining and finetuning phases.

- `model.py`  
  Defines the model architecture used by the experiments.

- `compare_benchmark.py`  
  Evaluates trained models against benchmark/reference profiles and saves comparison plots.

- `physics/`  
  Experiment configuration modules.
  - `physics/polynomial.py` — configuration for the polynomial benchmark experiment.
  - `physics/singlet.py` — configuration for the singlet benchmark experiment.

- `saved_models/`  
  Default output directory for trained model checkpoints.

- `outputs/`  
  Default output directory for generated comparison plots.

## Dependencies

The project expects the following Python packages:

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- `cosmoTransitions` for benchmark profile generation and comparison

If you are using a GPU, install the CUDA-enabled PyTorch build that matches your system. CPU-only is also supported.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy matplotlib cosmoTransitions
```

If you want to install PyTorch separately for a specific CUDA version, use the official PyTorch installation instructions for your platform.

## Running experiments

The available experiments are registered in `main.py`:

```bash
python main.py --exp polynomial
python main.py --exp singlet
```

What each run does:

1. Creates the required folders (`saved_models/` and `outputs/`) if not already exist.
2. Loads the selected experiment configuration from `physics/*.py`.
3. Builds the model defined in `model.py`.
4. Runs the pretraining stage.
5. Runs the finetuning stage.
6. Saves the trained weights to a checkpoint in `saved_models/`.
7. Produces a comparison plot in `outputs/`.

## Customizing experiments

Experiment settings live in the `Config` classes inside the modules under `physics/`.

Typical values you may adjust include:

- `pretrain_epochs`
- `finetune_epochs`
- `r_max`
- `device`
- `saveModelPath`
- `saveComparisonPath`
- optimizer and scheduler definitions

This is the main place to change training length, storage paths, and model runtime settings without editing the core training logic.

## Notes

- The project is intended as a compact research prototype rather than a production library.
- Outputs are written to the repository folders `saved_models/` and `outputs/` by default.
- The implementation is intentionally simple and easy to inspect, modify, and extend for thesis work.
