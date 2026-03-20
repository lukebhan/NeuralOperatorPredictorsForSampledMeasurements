"""Run one simulation example and plot results."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pinocchio as pin

from src.config import make_config
from src.simulate import build_robot, make_reference, make_simulator
from src.plot import plot_results

cfg = make_config(
    urdf="xarm6.urdf",
    dt=0.001,
    T=20.0,
    D=0.2,
    Ts=0.05,
    predictor_tolerance=1e-8,
    max_picard_iters=50,
    inner_predictor_discretization_steps=4,
)

robot = build_robot(cfg["urdf"])
ref = make_reference(robot, cfg)
sim = make_simulator(robot, cfg, ref)

rng = np.random.default_rng(0)

model = robot["model"]

q0 = pin.integrate(
    model,
    pin.neutral(model),
    0.15 * rng.standard_normal(robot["nq"]),
)

v0 = 0.1 * rng.standard_normal(robot["nv"])

out = sim["simulate"](
    q0=q0,
    v0=v0,
    q_meas_noise_std=0.01,
    v_meas_noise_std=0.01,
    use_noisy_measurement_for_reset=True,
    verbose=True,
)

plot_results(out, nq=robot["nq"], nv=robot["nv"])
