"""Configuration helpers for simulation and predictor experiments."""

import numpy as np

def make_config(
    urdf="xarm6.urdf",
    dt=0.01,
    T=20.0,
    D=1.0,
    Ts=0.05,
    tau_max=60.0,
    noise_std=0.0,
    Kp_val=40.0,
    Kd_val=14.0,
    traj_w=0.6,
    traj_amp=0.20,
    predictor_tolerance=1e-6,
    max_picard_iters=20,
    inner_predictor_discretization_steps=1,
):
    """Build and validate an experiment configuration dictionary.

    Computes derived quantities (steps, delay_steps, sample_steps, h_pred)
    and asserts that D and Ts are exact integer multiples of dt.

    Inputs:  all physical and control parameters (see argument defaults above)
    Returns: cfg dict consumed by the simulator and dataset builders
    """
    steps = int(T / dt)
    delay_steps = int(round(D / dt))
    sample_steps = int(round(Ts / dt))

    assert abs(delay_steps * dt - D) < 1e-12, "Require D = N*dt exactly"
    assert abs(sample_steps * dt - Ts) < 1e-12, "Require Ts = M*dt exactly"

    return {
        "urdf": urdf,
        "dt": dt,
        "T": T,
        "steps": steps,
        "D": D,
        "Ts": Ts,
        "delay_steps": delay_steps,
        "sample_steps": sample_steps,
        "N_pred": delay_steps,
        "h_pred": D / delay_steps,
        "tau_max": tau_max,
        "noise_std": noise_std,
        "Kp_val": Kp_val,
        "Kd_val": Kd_val,
        "traj_w": traj_w,
        "traj_amp": traj_amp,
        "predictor_tolerance": predictor_tolerance,
        "max_picard_iters": max_picard_iters,
        "inner_predictor_discretization_steps": inner_predictor_discretization_steps,
    }