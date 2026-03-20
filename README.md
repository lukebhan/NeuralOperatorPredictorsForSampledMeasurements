# Neural Operator Predictors for Sampled-Data Delay-Compensated Stabilization

Official implementation of **"Neural Operator Predictors for Delay-Compensated Nonlinear Stabilization with Sampled Measurements"**.

---

## Overview

This repository implements learned FNO-based predictors for closed-loop stabilization of nonlinear systems with input delay and sampled measurements. The predictors replace expensive numerical Picard iteration, enabling real-time delay compensation on a 6-DOF xArm manipulator modeled via [Pinocchio](https://github.com/stack-of-tasks/pinocchio).

Two architectures are studied:

- **Case 1 — Single-step FNO:** predicts the system state D seconds ahead from the current measurement and torque history.
- **Case 2 — Multistep FNO:** predicts a full trajectory over the inter-sample horizon, used directly as the internal controller state.

---

## Installation

```bash
git clone https://github.com/<your-repo>/NeuralOperatorPredictorsForSampledMeasurements.git
cd NeuralOperatorPredictorsForSampledMeasurements
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Dependencies: `pinocchio`, `neuralop`, `torch`, `numpy`, `scikit-learn`, `matplotlib`.

---

## Repository Structure

```
NeuralOperatorPredictorsForSampledMeasurements/
├── src/
│   ├── simulate.py               # Simulator, hybrid controller, Picard predictor
│   ├── config.py                 # Experiment configuration
│   ├── case1_dataset_builder.py  # Dataset generation — Case 1
│   ├── case2_dataset_builder.py  # Dataset generation — Case 2
│   ├── case1_fno.py              # FNO model — Case 1
│   ├── case2_fno.py              # FNO model — Case 2
│   ├── case1_trainer.py          # Training utilities — Case 1
│   ├── case2_trainer.py          # Training utilities — Case 2
│   ├── predictors.py             # Predictor factory wrappers
│   └── plot.py                   # Plotting helpers
├── scripts/
│   ├── case1_dataset.py          # Generate single-step dataset
│   ├── case2_dataset.py          # Generate multistep dataset
│   ├── train_case1.py            # Train Case 1 FNO
│   ├── train_case2.py            # Train Case 2 FNO
│   ├── one_example.py            # Run one simulation and plot
│   └── compare_data.py           # Dataset diagnostics
├── dataset/                      # ← place downloaded datasets here
├── models/                       # ← place downloaded models here
├── Evaluation.ipynb              # Reproduces paper results
└── xarm6.urdf
```

---

## Quick Start

Run one closed-loop simulation with the numerical predictor:

```bash
python scripts/one_example.py
```

This runs a 20-second simulation (delay `D = 0.2 s`, sampling `Ts = 0.05 s`) and plots joint tracking results.

---

## Generating Datasets

Datasets are generated in parallel via `ProcessPoolExecutor`. Parameters are set through `make_config` in `src/config.py`.

```bash
python scripts/case1_dataset.py   # single-step dataset
python scripts/case2_dataset.py   # multistep dataset
```

Outputs are saved as compressed `.npz` files in `dataset/`.

---

## Training

```bash
python scripts/train_case1.py   # single-step FNO
python scripts/train_case2.py   # multistep FNO
```

Both use Adam + `ReduceLROnPlateau` and save the best checkpoint to `models/`. Pretrained datasets and models are available on Hugging Face — see [Pretrained Resources](#pretrained-resources).

---

## Evaluation

Open `Evaluation.ipynb` to load pretrained models and reproduce the closed-loop results from the paper.

---

## Configuration

Key parameters in `make_config` (`src/config.py`):

| Parameter             | Default       | Description                                |
| :-------------------- | :-----------: | :----------------------------------------- |
| `dt`                  | `0.001 s`     | Simulation timestep                        |
| `D`                   | `0.2 s`       | Input delay                                |
| `Ts`                  | `0.05 s`      | Measurement sampling period                |
| `T`                   | `20.0 s`      | Total simulation duration                  |
| `tau_max`             | `60.0 Nm`     | Torque saturation limit                    |
| `Kp_val` / `Kd_val`   | `40` / `14`   | PD gain values                             |
| `traj_w` / `traj_amp` | `0.6` / `0.2` | Reference trajectory frequency / amplitude |

---

## Pretrained Resources

Pretrained datasets and models are hosted on Hugging Face:

- **Dataset:** *(link)*
- **Models:** *(link)*

Place the downloaded files in the `dataset/` and `models/` directories.

---

## Citation

```bibtex
@misc{...,
  title  = {Neural Operator Predictors for Delay-Compensated Nonlinear Stabilization with Sampled Measurements},
  author = {Luke Bhan and Peter Quawas and Miroslav Krstic and Yuanyuan Shi},
  year   = {2026},
}
```

---

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
