"""Plotting helpers for simulation rollouts and predictor comparisons."""

import matplotlib.pyplot as plt
import numpy as np

def plot_results(out, nq=None, nv=None):
    """Generate diagnostic plots from a simulation rollout.

    Inputs:  out dict from sim["simulate"], optional nq / nv (inferred if None)
    Outputs: displays figures via plt.show()
    """
    t = out["t"]
    q = out["q"]
    qz = out["qz"]
    q_sample_pred = out["q_sample_pred"]
    q_des_now = out["q_des_now"]
    q_des_future = out["q_des_future"]
    tau_applied = out["tau_applied"]
    picard = out["picard_iters"]
    predictor_time = out["predictor_time"]
    step_time = out["step_time"]
    sample_flag = out["sample_flag"]

    if nq is None:
        nq = q.shape[1]
    if nv is None:
        nv = tau_applied.shape[1]

    err_now = np.linalg.norm(q - q_des_now, axis=1)
    err_z_future = np.linalg.norm(qz - q_des_future, axis=1)
    err_sample_pred = np.linalg.norm(q_sample_pred - q_des_future, axis=1)

    plt.figure(figsize=(7, 4))
    plt.plot(t, err_now)
    plt.xlabel("time [s]")
    plt.ylabel(r"$\|q(t)-q_d(t)\|$")
    plt.title("Joint-space tracking error")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(t, err_z_future, label=r"$\|Z_q(t)-q_d(t+D)\|$")
    plt.plot(t, err_sample_pred, "--", label=r"$\|P_i(0)-q_d(t+D)\|$")
    plt.xlabel("time [s]")
    plt.ylabel("error")
    plt.title("Hybrid internal state vs future desired")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    num_plot = min(3, nq)
    joint_indices = range(nq - num_plot, nq)

    fig, axes = plt.subplots(num_plot, 1, figsize=(8, 2.5 * num_plot), sharex=True)
    if num_plot == 1:
        axes = [axes]

    for ax, i in zip(axes, joint_indices):
        ax.plot(t, q[:, i], label=f"q[{i}]")
        ax.plot(t, q_des_now[:, i], "--", label=f"q_d[{i}]")
        ax.grid(True)
        ax.legend()
        ax.set_ylabel("rad")

    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Joint tracking (last 3 joints)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    tau_indices = range(nv - min(3, nv), nv)
    for i in tau_indices:
        plt.plot(t, tau_applied[:, i], label=f"tau_applied[{i}]")
    plt.xlabel("time [s]")
    plt.ylabel("torque")
    plt.title("Applied delayed torques (last 3 joints)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    valid = ~np.isnan(picard)
    plt.plot(t[valid], picard[valid])
    plt.xlabel("time [s]")
    plt.ylabel("avg Picard iterations")
    plt.title("Average Picard iterations at sampling instants")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(t, 1000.0 * predictor_time, label="predictor")
    plt.plot(t, 1000.0 * step_time, label="total step")
    plt.xlabel("time [s]")
    plt.ylabel("time [ms]")
    plt.title("Per-step wall-clock timing")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 2.5))
    plt.plot(t, sample_flag.astype(float))
    plt.xlabel("time [s]")
    plt.ylabel("sample")
    plt.title("Sampling instants")
    plt.grid(True)
    plt.tight_layout()
    plt.show()